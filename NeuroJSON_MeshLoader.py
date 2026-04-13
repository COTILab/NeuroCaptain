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
    """Check deps and patch jdata.load() bug (returns newdata[0] instead of newdata)."""
    global HAS_JDATA, HAS_NUMPY

    try:
        import numpy
        HAS_NUMPY = True
    except ImportError:
        HAS_NUMPY = False

    try:
        import jdata
        import jdata.jfile as _jfile
        import re as _re
        HAS_JDATA = True

        # Patch jdata 0.9.x bug: load(url) returns newdata[0] instead of newdata
        def _patched_load(fname, opt=None, **kwargs):
            if opt is None:
                opt = {}
            if _re.match("^https*://", fname):
                newdata, fname, _ = _jfile.downloadlink(fname, opt, **kwargs)
                if newdata is not None:
                    return newdata  # fix: original returns newdata[0]
            spl = os.path.splitext(fname)
            ext = spl[1].lower()
            if ext in _jfile.jext["t"]:
                return _jfile.loadt(fname, opt, **kwargs)
            elif ext in _jfile.jext["b"]:
                return _jfile.loadb(fname, opt, **kwargs)
            else:
                raise Exception("JData", "file extension is not recognized")

        _jfile.load = _patched_load
        jdata.load = _patched_load
        print("[NeuroJSON] jdata loaded + load() bug patched")

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
            print("\n[NeuroJSON] Fetching dataset list...")
            result = jd.neuroj('list', 'brainmeshlibrary')
            datasets = result.get('dataset', [])
            _set_datasets(context, datasets)
            msg = f"NeuroJSON: Found {len(datasets)} datasets"
            print(f"[NeuroJSON] {msg}")
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

            print("\n" + "=" * 60)
            print(f"[NeuroJSON] LOADING: {self.dataset_key}")
            print("=" * 60)

            # Ensure HOME is set (Blender on Windows sometimes lacks it)
            if not os.environ.get('HOME'):
                os.environ['HOME'] = os.path.expanduser('~')

            # Use patched jd.load(url) — auto-decodes JData arrays
            url = 'https://neurojson.io:7777/brainmeshlibrary/' + self.dataset_key
            print(f"[NeuroJSON] Loading: {url}")
            data = jd.load(url)
            print(f"[NeuroJSON] Loaded keys: {list(data.keys())}")

            # Save to disk
            self._save_to_disk(data, settings.download_folder)

            # Import or show info
            if settings.import_mode == 'MESH':
                self._import_mesh(context, data)
            else:
                self._show_info(context, data)

            print("=" * 60 + "\n")
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
                print("[NeuroJSON] No download folder set, skipping save")
                return
            folder = bpy.path.abspath(folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            fname = self.dataset_key.replace('--', '_') + '.jmsh'
            path = os.path.join(folder, fname)
            print(f"[NeuroJSON] Saving: {path}")
            jd.save(data, path)
            print(f"[NeuroJSON] Saved OK")
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
            print("[NeuroJSON] Using MeshSurf (triangular surface)")
        elif 'MeshElem' in data:
            faces_raw = data['MeshElem']
            mesh_type = 'tet'
            print("[NeuroJSON] Using MeshElem (tetrahedral, extracting surface)")
        else:
            faces_raw = None
            mesh_type = 'none'
            print("[NeuroJSON] No face data found")

        # Convert vertices
        verts = verts_raw.tolist() if isinstance(verts_raw, numpy.ndarray) else list(verts_raw)

        # Convert faces
        if faces_raw is None:
            faces = []
        elif isinstance(faces_raw, numpy.ndarray):
            faces = faces_raw.tolist()
        else:
            faces = list(faces_raw)

        print(f"[NeuroJSON] Verts: {len(verts)}, Faces/Elems: {len(faces)}")

        if mesh_type == 'surf':
            faces_bl = [[int(f[0]) - 1, int(f[1]) - 1, int(f[2]) - 1] for f in faces]

        elif mesh_type == 'tet':
            # Extract boundary faces via parity trick
            # MeshElem has 5 columns: [n0, n1, n2, n3, region_label]
            seen = set()
            for tet in faces:
                try:
                    n0 = int(tet[0]) - 1
                    n1 = int(tet[1]) - 1
                    n2 = int(tet[2]) - 1
                    n3 = int(tet[3]) - 1
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
                except Exception:
                    pass
            faces_bl = [list(tf) for tf in seen]
            print(f"[NeuroJSON] Extracted {len(faces_bl)} boundary faces")

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

        # Store dataset key as custom property
        obj['neurojson_dataset_key'] = self.dataset_key

        msg = f"Imported {name}: {len(verts)}v {len(faces_bl)}f"
        print(f"[NeuroJSON] {msg}")
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

        print(f"\n[NeuroJSON] Mesh Info:")
        print(f"  Name    : {settings.info_name}")
        print(f"  Type    : {mtype}")
        print(f"  Vertices: {n_verts}")
        print(f"  Faces   : {n_faces}")
        self.report({'INFO'}, f"{n_verts}v  {n_faces}f  {mtype}")


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
        # Only show surface meshes (tet decoding not yet reliable)
        if '--surf' not in str_key.lower():
            continue
        if search and search not in str_key.lower():
            continue
        parts = str_key.split('--')
        subject = parts[0] + '--' + parts[1] if len(parts) >= 2 else parts[0]
        subject_map.setdefault(subject, []).append(str_key)

    total = sum(len(v) for v in subject_map.values())
    box.label(text=f"{total} surface meshes in {len(subject_map)} subjects")

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
                op2 = sub_col.operator(
                    "neurocaptain.nj_load_dataset",
                    text="    " + short,
                    icon='IMPORT'
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
    print("[NeuroJSON] Module registered")


def unregister():
    del bpy.types.Scene.neurojson_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    print("[NeuroJSON] Module unregistered")