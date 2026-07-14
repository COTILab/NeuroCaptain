"""
NeuroJSON_MeshLoader.py
NeuroCaptain module for browsing and importing brain meshes from NeuroJSON.io

Uses jdata.neuroj() for listing and jdata.load() for downloading.
Includes patch for jdata 0.9.x load() bug.
Compatible with Blender 3.4.1+
"""

import bpy
import os
import json
import ssl
from bpy.props import StringProperty, EnumProperty, BoolProperty
from bpy.types import PropertyGroup, Operator

# ============================================================================
# DEPENDENCY CHECK + JDATA BUG PATCH
# ============================================================================

HAS_JDATA = False
HAS_NUMPY = False

def _patch_and_check_deps():
    """Check deps and patch jdata 0.9.x bugs."""
    global HAS_JDATA, HAS_NUMPY

    try:
        import numpy
        HAS_NUMPY = True
    except ImportError:
        HAS_NUMPY = False

    try:
        import jdata
        import jdata.jdata as _jd
        import jdata.jfile as _jfile
        import re as _re
        HAS_JDATA = True

        # Patch 1: jdata 0.9.x load(url) returns newdata[0] instead of newdata.
        _orig_load = _jfile.load

        def _patched_load(fname, opt=None, **kwargs):
            if opt is None:
                opt = {}
            if isinstance(fname, str) and _re.match("^https*://", fname):
                newdata, fname, _ = _jfile.downloadlink(fname, opt, **kwargs)
                if newdata is not None:
                    return newdata
            return _orig_load(fname, opt, **kwargs)

        _jfile.load = _patched_load
        jdata.load = _patched_load

        # Patch 2: jdata 0.9.x doesn't resolve _DataLink_ inside _ArrayZipData_.
        # Tet datasets store compressed binary data externally via _DataLink_
        # refs (e.g. "url.bmsh:$.MeshNode._ArrayZipData_"). jdata tries to
        # pass the dict directly to zlib.decompress(), which crashes.
        # Fix: walk the data and resolve _DataLink_ refs before jdata decodes.
        #
        # Must capture the real decode() dispatch/recursion function here,
        # NOT jdatadecode (a one-line alias: `def jdatadecode(obj, **kwargs):
        # return decode(obj, **kwargs)`). jdatadecode has no recursion logic
        # of its own - it just looks up `decode` by name on every call. Once
        # _jd.decode is reassigned below, that lookup permanently resolves to
        # the patched wrapper, so capturing the alias as "_orig_decode" would
        # make it call straight back into the patched wrapper on the exact
        # same object forever (infinite recursion on every input, not just
        # large ones). Capturing the real decode() function object directly
        # means _orig_decode(data) actually runs its dispatch/recursion body.
        _orig_decode = _jd.decode

        def _resolve_datalinks_in_zip(obj):
            """Pre-resolve _DataLink_ refs nested inside _ArrayZipData_ fields."""
            if isinstance(obj, dict):
                if ('_ArrayZipData_' in obj
                        and isinstance(obj['_ArrayZipData_'], dict)
                        and '_DataLink_' in obj['_ArrayZipData_']):
                    obj['_ArrayZipData_'] = _resolve_data(obj['_ArrayZipData_'])
                for v in obj.values():
                    if isinstance(v, (dict, list)):
                        _resolve_datalinks_in_zip(v)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        _resolve_datalinks_in_zip(item)

        # jdata's own decode recurses into nested structures by looking up
        # jdatadecode/decode through the module namespace, which the patch
        # below reassigns globally - so every nested recursive call would
        # re-enter _patched_decode and re-run a full _resolve_datalinks_in_zip
        # scan over its (sub)subtree, compounding into a RecursionError well
        # before jdata finishes decoding any mesh of realistic size. Only run
        # the resolve pass once, on the outermost call, over the whole tree.
        _resolving_datalinks = [False]

        def _patched_decode(data, **kwargs):
            if _resolving_datalinks[0]:
                return _orig_decode(data, **kwargs)
            _resolving_datalinks[0] = True
            try:
                _resolve_datalinks_in_zip(data)
                return _orig_decode(data, **kwargs)
            finally:
                _resolving_datalinks[0] = False

        _jd.jdatadecode = _patched_decode
        _jd.decode = _patched_decode
        jdata.decode = _patched_decode
        jdata.jdatadecode = _patched_decode

    except ImportError as e:
        HAS_JDATA = False
        print(f"[NeuroJSON] Missing modules: {e}")

    return HAS_JDATA and HAS_NUMPY

# Run patch at import time
_patch_and_check_deps()


def _ssl_context():
    """Set unverified SSL for NeuroJSON.io, return previous factory."""
    prev = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    return prev

def _ssl_restore(prev):
    ssl._create_default_https_context = prev


# ============================================================================
# MANUAL JDATA DECODER — fallback when jdata's decoder can't handle
# nested _ArrayZipData_ annotations (common in tet datasets)
# ============================================================================

def _download_and_decode(url):
    """Download NeuroJSON data and decode JData arrays manually."""
    import urllib.request

    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=ctx) as resp:
        raw = json.loads(resp.read())

    return _jdata_walk(raw)


def _fetch_to_cache(url):
    """Download a URL into jdata's local cache, return the file path."""
    import os
    import urllib.request
    from jdata.jfile import jsoncache

    cachepath, filename = jsoncache(url)

    if not isinstance(cachepath, list) and os.path.exists(cachepath):
        return cachepath

    if isinstance(cachepath, list) and cachepath:
        fname = os.path.join(cachepath[0], filename)
        fpath = os.path.dirname(fname)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        ctx = ssl._create_unverified_context()
        rawdata = urllib.request.urlopen(url, context=ctx).read()
        with open(fname, 'wb') as fid:
            fid.write(rawdata)
        return fname

    raise RuntimeError(f"Could not determine cache path for: {url}")


def _resolve_data(val):
    """Resolve _DataLink_ references or nested JData annotations to raw data."""
    import re as _re

    if isinstance(val, dict):
        if '_DataLink_' in val:
            link = val['_DataLink_']
            # _DataLink_ URLs contain a JSONPath suffix after ":$"
            # e.g. "https://…/file.bmsh:$.MeshNode._ArrayZipData_"
            # Split off the JSONPath, download the binary file, parse
            # without JData decoding, and extract via JSONPath.
            match = _re.search(r'^(.+?):\$(.+)$', link)
            if match:
                file_url = match.group(1)
                jpath = '$' + match.group(2)
            else:
                file_url = link
                jpath = None

            prev = _ssl_context()
            try:
                fname = _fetch_to_cache(file_url)
            finally:
                _ssl_restore(prev)

            import bjdata as _bjd
            import numpy as _np
            with open(fname, 'rb') as fid:
                data = _bjd.loadb(fid.read())

            if jpath:
                from jdata.jpath import jsonpath
                data = jsonpath(data, jpath)

            if isinstance(data, _np.ndarray):
                data = data.tobytes()

            return data
        if '_ArrayType_' in val:
            return _jdata_walk(val)
    return val


def _jdata_walk(obj):
    """Recursively decode JData array annotations in a parsed JSON tree."""
    if isinstance(obj, dict):
        if '_ArrayType_' in obj:
            return _jdata_array(obj)
        return {k: _jdata_walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jdata_walk(v) for v in obj]
    return obj


def _jdata_array(d):
    """Decode a single JData array annotation into a numpy array."""
    import numpy as np
    import base64

    DTYPES = {
        'double': np.float64, 'single': np.float32,
        'int8': np.int8, 'uint8': np.uint8,
        'int16': np.int16, 'uint16': np.uint16,
        'int32': np.int32, 'uint32': np.uint32,
        'int64': np.int64, 'uint64': np.uint64,
    }

    dtype = DTYPES.get(d.get('_ArrayType_', 'double'), np.float64)
    shape = d.get('_ArraySize_')

    # JData _ArrayOrder_: "c"/"col"/"column" means column-major → numpy 'F'
    # When absent, jdata defaults to row-major (numpy 'C')
    if '_ArrayOrder_' in d:
        ao = d['_ArrayOrder_'].lower()
        order = 'F' if ao in ('c', 'col', 'column') else 'C'
    else:
        order = 'C'

    # --- Compressed data path ---
    if '_ArrayZipData_' in d:
        raw_data = _resolve_data(d['_ArrayZipData_'])
        zip_type = d.get('_ArrayZipType_', 'zlib')

        if isinstance(raw_data, np.ndarray):
            raw_data = raw_data.tobytes()
        elif isinstance(raw_data, str):
            raw_data = base64.b64decode(raw_data)

        if zip_type in ('zlib', 'deflate'):
            import zlib
            raw_data = zlib.decompress(raw_data)
        elif zip_type == 'gzip':
            import gzip
            raw_data = gzip.decompress(raw_data)
        elif zip_type in ('lzma', 'lzip'):
            import lzma
            raw_data = lzma.decompress(raw_data)

        arr = np.frombuffer(raw_data, dtype=dtype).copy()

    # --- Uncompressed data path ---
    elif '_ArrayData_' in d:
        arr_data = _resolve_data(d['_ArrayData_'])

        if isinstance(arr_data, (bytes, bytearray)):
            arr = np.frombuffer(arr_data, dtype=dtype).copy()
        elif isinstance(arr_data, str):
            arr = np.frombuffer(base64.b64decode(arr_data), dtype=dtype).copy()
        elif isinstance(arr_data, (list, tuple)):
            arr = np.array(arr_data, dtype=dtype).flatten()
        elif isinstance(arr_data, np.ndarray):
            arr = arr_data.astype(dtype).flatten()
        else:
            raise ValueError(f"Unsupported _ArrayData_ type: {type(arr_data)}")
    else:
        raise ValueError("JData array has no _ArrayData_ or _ArrayZipData_")

    if shape and len(shape) >= 2:
        arr = arr.reshape(shape, order=order)

    return arr


# ============================================================================
# SETTINGS
# ============================================================================

class NJ_Settings(PropertyGroup):
    """NeuroJSON loader settings, registered as scene.neurojson_settings"""

    download_folder: StringProperty(
        name="Download To",
        description="Folder to save downloaded meshes (leave empty to skip saving)",
        default="",
        subtype='DIR_PATH'
    )
    search_text: StringProperty(
        name="Filter",
        description="Filter datasets by name",
        default=""
    )
    import_mode: EnumProperty(
        name="Mode",
        items=[
            ('MESH', "Create Mesh", "Import as Blender mesh object"),
            ('INFO', "Info Only", "Print mesh info to console without creating objects"),
        ],
        default='MESH',
    )
    dataset_cache: StringProperty(
        name="Dataset Cache",
        default="[]",
        options={'HIDDEN'}
    )
    selected_subject: StringProperty(
        name="Selected Subject",
        default="",
        options={'HIDDEN'}
    )
    show_datasets: BoolProperty(
        name="Show Datasets",
        description="Show or hide the dataset list",
        default=True,
    )
    # Info display fields
    info_name: StringProperty(default="")
    info_verts: StringProperty(default="")
    info_faces: StringProperty(default="")
    info_type: StringProperty(default="")


def _get_datasets(context):
    try:
        return json.loads(context.scene.neurojson_settings.dataset_cache)
    except Exception:
        return []

def _set_datasets(context, datasets):
    context.scene.neurojson_settings.dataset_cache = json.dumps(datasets)


# ============================================================================
# OPERATORS
# ============================================================================

class NJ_OT_refresh_datasets(Operator):
    """Fetch the full dataset list from NeuroJSON.io"""
    bl_idname = "neurocaptain.nj_refresh_datasets"
    bl_label = "Refresh NeuroJSON Datasets"

    def execute(self, context):
        if not HAS_JDATA:
            self.report({'ERROR'}, "Missing dependencies: jdata and/or numpy. Install via Dependencies tab.")
            return {'CANCELLED'}

        prev = _ssl_context()
        try:
            import jdata as jd
            result = jd.neuroj('list', 'brainmeshlibrary')
            datasets = result.get('dataset', [])
            _set_datasets(context, datasets)
            msg = f"NeuroJSON: Found {len(datasets)} datasets"
            self.report({'INFO'}, msg)
            return {'FINISHED'}

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"NeuroJSON fetch error: {e}")
            return {'CANCELLED'}
        finally:
            _ssl_restore(prev)


class NJ_OT_select_subject(Operator):
    """Expand or collapse a subject folder in the dataset browser"""
    bl_idname = "neurocaptain.nj_select_subject"
    bl_label = "Select Subject"

    subject: StringProperty()

    def execute(self, context):
        settings = context.scene.neurojson_settings
        if settings.selected_subject == self.subject:
            settings.selected_subject = ""
        else:
            settings.selected_subject = self.subject
        return {'FINISHED'}


class NJ_OT_load_dataset(Operator):
    """Download and import a dataset from NeuroJSON.io"""
    bl_idname = "neurocaptain.nj_load_dataset"
    bl_label = "Load NeuroJSON Dataset"
    bl_options = {'REGISTER', 'UNDO'}

    dataset_key: StringProperty(name="Key")

    def execute(self, context):
        if not HAS_JDATA:
            self.report({'ERROR'}, "Missing dependencies: jdata and/or numpy")
            return {'CANCELLED'}

        settings = context.scene.neurojson_settings
        prev = _ssl_context()

        try:
            import jdata as jd
            import numpy

            # Ensure HOME is set (Blender on Windows sometimes lacks it)
            if not os.environ.get('HOME'):
                os.environ['HOME'] = os.path.expanduser('~')

            url = 'https://neurojson.io:7777/brainmeshlibrary/' + self.dataset_key
            try:
                data = jd.load(url)
            except Exception as e_jd:
                print(f"[NeuroJSON] jdata.load failed ({e_jd}), using manual decoder...")
                data = _download_and_decode(url)

            # Save to disk
            self._save_to_disk(data, settings.download_folder)

            # Import or show info
            if settings.import_mode == 'MESH':
                self._import_mesh(context, data)
            else:
                self._show_info(context, data)

            return {'FINISHED'}

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Load error: {e}")
            return {'CANCELLED'}
        finally:
            _ssl_restore(prev)

    # ------------------------------------------------------------------
    def _save_to_disk(self, data, folder):
        """Save downloaded data as .jmsh file."""
        try:
            import jdata as jd
            folder = folder.strip()
            if not folder:
                return
            folder = bpy.path.abspath(folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            fname = self.dataset_key.replace('--', '_') + '.jmsh'
            path = os.path.join(folder, fname)
            jd.save(data, path)
        except Exception as e:
            print(f"[NeuroJSON] Save error (non-fatal): {e}")

    # ------------------------------------------------------------------
    def _import_mesh(self, context, data):
        """Create a Blender mesh object from the NeuroJSON data."""
        import numpy

        verts_raw = data.get('MeshNode')
        if verts_raw is None:
            self.report({'ERROR'}, "No vertex data (MeshNode missing)")
            return

        # Determine face type
        if 'MeshSurf' in data:
            faces_raw = data['MeshSurf']
            mesh_type = 'surf'
        elif 'MeshElem' in data:
            faces_raw = data['MeshElem']
            mesh_type = 'tet'
        else:
            faces_raw = None
            mesh_type = 'none'

        # Convert vertices
        verts = verts_raw.tolist() if isinstance(verts_raw, numpy.ndarray) else list(verts_raw)

        # Convert faces
        if faces_raw is None:
            faces = []
        elif isinstance(faces_raw, numpy.ndarray):
            faces = faces_raw.tolist()
        else:
            faces = list(faces_raw)

        if mesh_type == 'surf':
            faces_bl = [[int(f[0]) - 1, int(f[1]) - 1, int(f[2]) - 1] for f in faces]

        elif mesh_type == 'tet':
            self._import_tet_layered(context, data)
            return

        else:
            faces_bl = []

        # Create Blender mesh
        name = self.dataset_key.replace('--', '_')
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, [], faces_bl)
        mesh.update()

        obj = bpy.data.objects.new(name, mesh)
        context.scene.collection.objects.link(obj)
        context.view_layer.objects.active = obj
        obj.select_set(True)

        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)

        # Store dataset key as custom property
        obj['neurojson_dataset_key'] = self.dataset_key

        msg = f"Imported {name}: {len(verts)}v {len(faces_bl)}f"
        self.report({'INFO'}, msg)

    # ------------------------------------------------------------------
    def _show_info(self, context, data):
        """Display mesh info without creating objects."""
        import numpy
        settings = context.scene.neurojson_settings

        verts_raw = data.get('MeshNode')
        n_verts = len(verts_raw) if verts_raw is not None else 0

        if 'MeshSurf' in data:
            n_faces = len(data['MeshSurf'])
            mtype = "Surface (triangular)"
        elif 'MeshElem' in data:
            n_faces = len(data['MeshElem'])
            mtype = "Volume (tetrahedral)"
        else:
            n_faces = 0
            mtype = "Unknown"

        settings.info_name = self.dataset_key.replace('--', ' / ')
        settings.info_verts = str(n_verts)
        settings.info_faces = str(n_faces)
        settings.info_type = mtype

        self.report({'INFO'}, f"{n_verts}v  {n_faces}f  {mtype}")

    # ------------------------------------------------------------------
    def _import_tet_layered(self, context, data):
        """Import tetrahedral mesh as layered head model, storing tissue data
        and displaying as head surface + cortical surface."""
        import numpy as np
        from . import layered_mesh_manager as lmm

        verts_raw = data['MeshNode']
        elems_raw = data['MeshElem']

        node = np.array(verts_raw, dtype=np.float64)
        if node.shape[1] > 3:
            node = node[:, :3]

        elem = np.array(elems_raw, dtype=np.int32)

        # Tissue labels from column 5 (NeuroJSON uses 1-indexed convention)
        if elem.shape[1] >= 5:
            tissue_labels = elem[:, 4].astype(np.int32)
        else:
            tissue_labels = np.ones(len(elem), dtype=np.int32)

        # Analyze layers
        unique_labels = np.unique(tissue_labels)
        num_layers = len(unique_labels)

        layer_names = {
            1: "Scalp", 2: "Skull", 3: "CSF",
            4: "Gray Matter", 5: "White Matter"
        }

        layer_info = {}
        for label in unique_labels:
            count = int(np.sum(tissue_labels == label))
            layer_info[int(label)] = {
                'name': layer_names.get(int(label), f"Tissue {label}"),
                'count': count
            }

        # Extract surfaces
        if lmm.ISO2MESH_AVAILABLE:
            scalp_faces = lmm.extract_outer_surface_only(
                node, elem, tissue_labels, target_tissue=1
            )
            cortex_faces = lmm.extract_cortex_surface(node, elem, tissue_labels)
        else:
            base = 1 if elem[:, :4].min() >= 1 else 0
            elem_0 = elem[:, :4].astype(np.int32) - base
            scalp_faces = self._boundary_parity(elem_0)
            brain_mask = np.isin(tissue_labels, [4, 5])
            if brain_mask.any():
                cortex_faces = self._boundary_parity(elem_0[brain_mask])
            else:
                cortex_faces = np.zeros((0, 3), dtype=np.int32)

        # Mesh centroid from scalp surface vertices
        scalp_vert_idx = np.unique(scalp_faces.flatten())
        mesh_centroid = node[scalp_vert_idx].mean(axis=0)

        # Create / clear FiveLayer_Visualization collection
        coll_name = "FiveLayer_Visualization"
        if coll_name in bpy.data.collections:
            coll = bpy.data.collections[coll_name]
            for obj in list(coll.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
        else:
            coll = bpy.data.collections.new(coll_name)
            bpy.context.scene.collection.children.link(coll)

        # --- Head surface (scalp) ---
        head_obj = lmm.create_mesh_object("Head_Surface_5L", node, scalp_faces, coll)

        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = head_obj
        head_obj.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
        head_obj.location = (0, 0, 0)
        head_obj.rotation_euler = (0.0, 0.0, 0.0)

        ref = bpy.data.objects.get('headmesh')
        if ref:
            head_obj.scale = ref.scale

        mat_head = bpy.data.materials.new("HeadSurface_Mat")
        mat_head.use_nodes = True
        if bpy.app.version < (4, 0, 0):
            mat_head.blend_method = 'BLEND'
        mat_nodes = mat_head.node_tree.nodes
        mat_nodes.clear()
        bsdf = mat_nodes.new('ShaderNodeBsdfPrincipled')
        output = mat_nodes.new('ShaderNodeOutputMaterial')
        bsdf.inputs['Base Color'].default_value = (1.0, 0.9, 0.8, 1.0)
        bsdf.inputs['Alpha'].default_value = 0.3
        mat_head.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        head_obj.data.materials.append(mat_head)
        head_obj['neurojson_dataset_key'] = self.dataset_key

        # --- Brain cortex ---
        cortex_obj = None
        if len(cortex_faces) > 0:
            cortex_obj = lmm.create_mesh_object(
                "Brain_Cortex_5L", node, cortex_faces, coll
            )

            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = cortex_obj
            cortex_obj.select_set(True)
            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
            cortex_obj.location = (0, 0, 0)
            cortex_obj.rotation_euler = (0.0, 0.0, 0.0)
            cortex_obj.scale = head_obj.scale

            mat_cortex = bpy.data.materials.new("Cortex_Mat")
            mat_cortex.use_nodes = True
            mat_nodes = mat_cortex.node_tree.nodes
            mat_nodes.clear()
            bsdf = mat_nodes.new('ShaderNodeBsdfPrincipled')
            output = mat_nodes.new('ShaderNodeOutputMaterial')
            bsdf.inputs['Base Color'].default_value = (0.8, 0.7, 0.7, 1.0)
            mat_cortex.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
            cortex_obj.data.materials.append(mat_cortex)
            cortex_obj['neurojson_dataset_key'] = self.dataset_key

        # Store in global LAYERED_MESH for simulations
        lmm.LAYERED_MESH.mesh_path = f"neurojson://{self.dataset_key}"
        lmm.LAYERED_MESH.nodes = node
        lmm.LAYERED_MESH.elems = elem
        lmm.LAYERED_MESH.tissue_labels = tissue_labels
        lmm.LAYERED_MESH.num_layers = num_layers
        lmm.LAYERED_MESH.layer_info = layer_info
        lmm.LAYERED_MESH.mesh_centroid = mesh_centroid
        lmm.LAYERED_MESH.head_surface_obj = head_obj
        lmm.LAYERED_MESH.cortex_obj = cortex_obj

        bpy.ops.object.select_all(action='DESELECT')

        msg = f"Imported layered tet mesh: {len(node):,}v, {len(elem):,} elems, {num_layers} layers"
        self.report({'INFO'}, msg)

    # ------------------------------------------------------------------
    @staticmethod
    def _boundary_parity(elems_0indexed):
        """Extract boundary faces from 0-indexed tet connectivity via parity trick."""
        import numpy as np
        seen = set()
        for tet in elems_0indexed:
            n0, n1, n2, n3 = int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])
            for tf in [
                tuple(sorted([n0, n1, n2])),
                tuple(sorted([n0, n1, n3])),
                tuple(sorted([n0, n2, n3])),
                tuple(sorted([n1, n2, n3])),
            ]:
                if tf in seen:
                    seen.discard(tf)
                else:
                    seen.add(tf)
        if seen:
            return np.array([list(tf) for tf in seen], dtype=np.int32)
        return np.zeros((0, 3), dtype=np.int32)


class NJ_OT_set_folder(Operator):
    """Browse for a download folder"""
    bl_idname = "neurocaptain.nj_set_folder"
    bl_label = "Browse"

    directory: StringProperty(subtype='DIR_PATH')

    def execute(self, context):
        context.scene.neurojson_settings.download_folder = self.directory
        abs_dir = bpy.path.abspath(self.directory)
        if not os.path.exists(abs_dir):
            os.makedirs(abs_dir)
        self.report({'INFO'}, f"Download folder: {abs_dir}")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


# ============================================================================
# PANEL DRAW FUNCTION — called from ui.py panels
# ============================================================================

def draw_neurojson_browser(layout, context):
    """
    Draw the NeuroJSON browser UI. Call from any NeuroCaptain panel.

    Usage in panel draw():
        from .NeuroJSON_MeshLoader import draw_neurojson_browser
        draw_neurojson_browser(layout, context)
    """
    settings = context.scene.neurojson_settings

    if not HAS_JDATA:
        box = layout.box()
        box.label(text="NeuroJSON requires: jdata, numpy", icon='ERROR')
        box.label(text="Install via Dependencies tab")
        return

    # Header
    box = layout.box()
    box.label(text="NeuroJSON.io Brain Mesh Library", icon='URL')

    # Import mode + refresh
    row = box.row(align=True)
    row.prop(settings, "import_mode", text="")
    row.operator("neurocaptain.nj_refresh_datasets", text="Refresh", icon='FILE_REFRESH')

    # Info display
    if settings.import_mode == 'INFO' and settings.info_name:
        info_box = box.box()
        info_box.scale_y = 0.8
        info_box.label(text=f"Name:  {settings.info_name}")
        info_box.label(text=f"Type:  {settings.info_type}")
        info_box.label(text=f"Verts: {settings.info_verts}")
        info_box.label(text=f"Faces: {settings.info_faces}")

    # Download folder
    row = box.row(align=True)
    row.prop(settings, "download_folder", text="")
    row.operator("neurocaptain.nj_set_folder", text="", icon='FILEBROWSER')

    # Filter
    box.prop(settings, "search_text", text="Filter", icon='VIEWZOOM')

    # Dataset list
    datasets = _get_datasets(context)
    if not datasets:
        box.label(text="Press Refresh to load dataset list", icon='INFO')
        return

    search = settings.search_text.lower()

    # Group by subject
    subject_map = {}
    for item in datasets:
        if not isinstance(item, dict):
            continue
        str_key = str(item.get('key', ''))
        if not str_key:
            continue
        key_lower = str_key.lower()
        if '--surf' not in key_lower and '--tet' not in key_lower:
            continue
        if search and search not in str_key.lower():
            continue
        parts = str_key.split('--')
        subject = parts[0] + '--' + parts[1] if len(parts) >= 2 else parts[0]
        subject_map.setdefault(subject, []).append(str_key)

    total = sum(len(v) for v in subject_map.values())

    row = box.row(align=True)
    icon = 'TRIA_DOWN' if settings.show_datasets else 'TRIA_RIGHT'
    row.prop(settings, "show_datasets", text="", icon=icon, emboss=False)
    row.label(text=f"{total} meshes in {len(subject_map)} subjects")

    if not settings.show_datasets:
        return

    col = box.column(align=True)
    for subject in sorted(subject_map.keys()):
        ds_list = subject_map[subject]
        display_name = subject.replace('--', ' / ')
        is_open = (settings.selected_subject == subject)

        icon = 'TRIA_DOWN' if is_open else 'TRIA_RIGHT'
        op = col.operator(
            "neurocaptain.nj_select_subject",
            text=f"{display_name}  ({len(ds_list)})",
            icon=icon
        )
        op.subject = subject

        if is_open:
            sub_col = col.column(align=True)
            for ds_key in sorted(ds_list):
                parts = ds_key.split('--')
                short = '--'.join(parts[2:]) if len(parts) > 2 else ds_key
                is_tet = '--tet' in ds_key.lower()
                op2 = sub_col.operator(
                    "neurocaptain.nj_load_dataset",
                    text="    " + short,
                    icon='MESH_CUBE' if is_tet else 'IMPORT'
                )
                op2.dataset_key = ds_key


# ============================================================================
# REGISTRATION
# ============================================================================

classes = (
    NJ_Settings,
    NJ_OT_refresh_datasets,
    NJ_OT_select_subject,
    NJ_OT_load_dataset,
    NJ_OT_set_folder,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.neurojson_settings = bpy.props.PointerProperty(type=NJ_Settings)


def unregister():
    del bpy.types.Scene.neurojson_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)