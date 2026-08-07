import bpy
import numpy as np
import scipy.io as sio
from scipy import ndimage
from pathlib import Path
import json
import os
import sys
import time

from .utils import recenter_on_volume_centroid, compute_volume_centroid_world

try:
    import iso2mesh as i2m
    ISO2MESH_AVAILABLE = True
except ImportError:
    ISO2MESH_AVAILABLE = False

try:
    import redbirdpy as rb
    REDBIRD_AVAILABLE = True
except ImportError:
    REDBIRD_AVAILABLE = False


def _patch_iso2mesh_extractloops_bug():
    """Patch a bug in iso2mesh.trait.extractloops (still present in the
    latest PyPI release, 0.6.5, as of this writing) that crashes
    slicesurf/slicesurf3 (used for brain hemisphere/landmark slicing and
    mesh cropping) with "operands could not be broadcast together" whenever
    a loop needs to be re-joined at both ends.

    Root cause: `loops` is a plain Python list, but `lp = np.flip(loops)`
    turns it into a numpy array; `loops[:n] + lp[:m]` then triggers numpy
    element-wise addition (broadcasting) instead of the intended sequence
    concatenation, which only "works" by coincidence when both slices
    happen to have equal length. Fix: use np.concatenate explicitly.

    modify.py imports extractloops by name at its own module load time
    (`from iso2mesh.trait import ... extractloops`), so slicesurf/
    slicesurf3 resolve it from iso2mesh.modify's own namespace at call
    time - patching iso2mesh.trait.extractloops alone would not affect
    them; both must be patched.
    """
    import numpy as _np

    def extractloops(edges):
        loops = []
        edges = edges[edges[:, 0] != edges[:, 1], :]
        if len(edges) == 0:
            return loops

        loops.extend(edges[0, :])
        loophead = edges[0, 0]
        loopend = edges[0, 1]
        edges = _np.delete(edges, 0, axis=0)

        while edges.size > 0:
            idx = _np.concatenate(
                [_np.where(edges[:, 0] == loopend)[0], _np.where(edges[:, 1] == loopend)[0]]
            )
            if len(idx) > 1:
                idx = idx[0]
            if not isinstance(idx, _np.ndarray):
                idx = _np.array(idx)

            if idx.size == 0:
                idx_head = _np.concatenate(
                    [
                        _np.where(edges[:, 0] == loophead)[0],
                        _np.where(edges[:, 1] == loophead)[0],
                    ]
                )
                if len(idx_head) == 0:
                    loops.append(_np.nan)
                    loops.extend(edges[0, :])
                    loophead = edges[0, 0]
                    loopend = edges[0, 1]
                    edges = _np.delete(edges, 0, axis=0)
                else:
                    loophead, loopend = loopend, loophead
                    lp = _np.flip(loops)
                    seg = _np.where(_np.isnan(lp))[0]
                    if len(seg) == 0:
                        loops = lp.tolist()
                    else:
                        # Fixed: np.concatenate instead of list + ndarray,
                        # which triggered numpy broadcasting instead of
                        # concatenation.
                        loops = _np.concatenate(
                            [loops[: len(loops) - seg[0]], lp[: seg[0]]]
                        ).tolist()
                continue

            if idx.size == 1:
                ed = edges[idx, :].flatten()
                ed = ed[ed != loopend]
                newend = ed[0]
                if newend == loophead:
                    loops.extend([loophead, _np.nan])
                    edges = _np.delete(edges, idx, axis=0)
                    if edges.size == 0:
                        break
                    loops.extend(edges[0, :])
                    loophead = edges[0, 0]
                    loopend = edges[0, 1]
                    edges = _np.delete(edges, 0, axis=0)
                    continue
                else:
                    loops.append(newend)
                loopend = newend
                edges = _np.delete(edges, idx, axis=0)

        return _np.array(loops)

    i2m.trait.extractloops = extractloops
    i2m.modify.extractloops = extractloops


def _fix_iso2mesh_windows_cork_binary():
    """surfboolean() invokes plain cork.exe, an old 32-bit build missing a
    DLL on this machine; cork_x86-64.exe works. Point ISO2MESH_SURFBOOLEAN
    at it if present, without touching iso2mesh's own code."""
    if not sys.platform.startswith("win") or "ISO2MESH_SURFBOOLEAN" in os.environ:
        return
    candidate = i2m.mcpath("cork_x86-64", i2m.getexeext())
    if os.path.isfile(candidate):
        os.environ["ISO2MESH_SURFBOOLEAN"] = "cork_x86-64"


if ISO2MESH_AVAILABLE:
    try:
        _patch_iso2mesh_extractloops_bug()
    except AttributeError:
        # Some environments have a top-level "iso2mesh" module importable
        # (so the plain `import iso2mesh` above succeeds) that isn't actually
        # the real py-iso2mesh package - e.g. missing its trait/modify
        # submodules entirely. Treat that the same as "not available" rather
        # than crashing this module's import (and therefore the whole
        # add-on's registration) - iso2mesh-dependent features below already
        # have fallback implementations for this case.
        ISO2MESH_AVAILABLE = False

if ISO2MESH_AVAILABLE:
    try:
        _fix_iso2mesh_windows_cork_binary()
    except AttributeError:
        pass  # same "not really iso2mesh" case handled above


# =============================================================================
# LAYERED MESH DATA STRUCTURE
# =============================================================================

class LayeredMeshData:
    """Store layered mesh data for later use in simulations"""
    
    def __init__(self):
        self.mesh_path = None
        self.nodes = None
        self.elems = None
        self.tissue_labels = None
        self.num_layers = 0
        self.layer_info = {}
        
        # Coordinate alignment
        # Centroid of the scalp surface vertices in original mesh coords.
        # Used as the Blender→mesh translation (Blender centers Head_Surface_5L
        # on this point, so mesh_coords = blender_coords + mesh_centroid).
        self.mesh_centroid = None

        # Blender objects
        self.head_surface_obj = None
        self.cortex_obj = None
        
    def is_loaded(self):
        return self.mesh_path is not None and self.nodes is not None
    
    def get_info_string(self):
        if not self.is_loaded():
            return "No mesh loaded"
        
        info = [
            f"Mesh: {Path(self.mesh_path).name}",
            f"Nodes: {len(self.nodes):,}",
            f"Elements: {len(self.elems):,}",
            f"Layers: {self.num_layers}",
            ""
        ]
        
        for layer_id, layer_data in sorted(self.layer_info.items()):
            info.append(f"  Layer {layer_id}: {layer_data['name']} ({layer_data['count']:,} elems)")
        
        return "\n".join(info)


# Global instance to store mesh data
LAYERED_MESH = LayeredMeshData()


# =============================================================================
# SAVE/RELOAD PERSISTENCE
# =============================================================================
# LAYERED_MESH only ever lives in Python memory, so it resets to empty every
# time Blender restarts - even though the visible Head_Surface_5L/
# Brain_Cortex_5L meshes it was built from ARE saved in the .blend file (real
# bpy.data objects). Without this, reopening a saved file leaves the
# simulation tools greyed out ("Import 5-layer mesh first") despite the
# geometry being visibly right there, forcing a full re-import from the
# original external file. Mirrors the property names lightsim_neurocaptain.
# import_five_layer_mesh already uses for its own (older, disconnected)
# import path, so load_mesh_and_register_optodes's existing "Priority 2"
# fallback recognizes the same keys too.

def _persist_layered_mesh_to_object():
    """Stash LAYERED_MESH's data as custom properties on its head surface
    object, so it survives a normal Blender save/reopen."""
    obj = LAYERED_MESH.head_surface_obj
    if obj is None or LAYERED_MESH.nodes is None:
        return
    obj['volumetric_nodes'] = LAYERED_MESH.nodes.tolist()
    obj['volumetric_elements'] = LAYERED_MESH.elems.tolist()
    obj['tissue_labels'] = LAYERED_MESH.tissue_labels.tolist()
    obj['mesh_filepath'] = LAYERED_MESH.mesh_path
    obj['num_layers'] = LAYERED_MESH.num_layers
    obj['layer_info_json'] = json.dumps(LAYERED_MESH.layer_info)
    obj['cortex_object_name'] = (
        LAYERED_MESH.cortex_obj.name if LAYERED_MESH.cortex_obj else ""
    )
    if LAYERED_MESH.mesh_centroid is not None:
        # .tolist(), not list() - the latter yields raw numpy scalars, which
        # crash Blender's ID-property assignment.
        obj['mesh_centroid'] = np.asarray(LAYERED_MESH.mesh_centroid).tolist()


def restore_layered_mesh_from_saved_data():
    """Reconstruct LAYERED_MESH from custom properties saved on a head
    surface object, if any exist in the current file (e.g. right after
    opening a .blend saved by an earlier session). Returns True if
    LAYERED_MESH was restored, False if there was nothing to restore.

    Safe to call even when nothing was ever persisted (older .blend files
    saved before this existed) - it just finds no matching object and
    leaves LAYERED_MESH untouched, same as today.
    """
    if LAYERED_MESH.is_loaded():
        return False  # already populated this session, don't clobber it

    obj = next(
        (o for o in bpy.data.objects if 'volumetric_nodes' in o.keys()),
        None,
    )
    if obj is None:
        return False

    try:
        LAYERED_MESH.nodes = np.array(obj['volumetric_nodes'], dtype=np.float64)
        LAYERED_MESH.elems = np.array(obj['volumetric_elements'], dtype=np.int32)
        LAYERED_MESH.tissue_labels = np.array(obj['tissue_labels'], dtype=np.int32)
        LAYERED_MESH.mesh_path = obj.get('mesh_filepath')
        LAYERED_MESH.num_layers = obj.get('num_layers', 0)
        layer_info_json = obj.get('layer_info_json')
        LAYERED_MESH.layer_info = (
            {int(k): v for k, v in json.loads(layer_info_json).items()}
            if layer_info_json else {}
        )
        centroid = obj.get('mesh_centroid')
        LAYERED_MESH.mesh_centroid = np.array(centroid) if centroid else None
        LAYERED_MESH.head_surface_obj = obj
        cortex_name = obj.get('cortex_object_name')
        LAYERED_MESH.cortex_obj = bpy.data.objects.get(cortex_name) if cortex_name else None
    except Exception as e:
        print(f"Warning: found saved layered-mesh data but failed to restore it: {e}")
        LAYERED_MESH.__init__()
        return False

    print(f"Restored layered mesh from saved data ({LAYERED_MESH.num_layers} layers, "
          f"{len(LAYERED_MESH.nodes):,} nodes)")
    return True


# =============================================================================
# SURFACE EXTRACTION : scalp and cotext (gm+wm)
# =============================================================================

def _ensure_1indexed(conn):
    """Ensure element connectivity is 1-indexed (iso2mesh/MATLAB convention)."""
    if conn.min() == 0:
        return conn + 1
    return conn


def _ensure_0indexed(faces):
    """Ensure face indices are 0-indexed (Blender convention)."""
    if faces.min() >= 1:
        return faces - 1
    return faces


def extract_outer_surface_only(node, elem, tissue_labels, target_tissue=1):
    """
    Extract the outermost surface for a given tissue using i2m.volface.

    """
    print(f"    Extracting outer surface for tissue {target_tissue}...")

    tet_conn = _ensure_1indexed(elem[:, :4].astype(int))

    openface, elemid = i2m.volface(tet_conn)

    # Convert to 0-indexed for Blender
    openface = _ensure_0indexed(openface.astype(int))

    # keep only faces whose parent element is the target tissue
    if target_tissue is not None:
        elemid = elemid.flatten().astype(int)
        # elemid from volface is 1-indexed
        mask = tissue_labels[elemid - 1] == target_tissue
        openface = openface[mask]

    print(f"      Found {len(openface):,} exterior faces")
    return openface


def extract_cortex_surface(node, elem, tissue_labels):
    """
    Extract cortex surface using i2m.volface on just the brain elements (gray + white matter) 

    """
    print(f"    Extracting cortex surface via volface on brain elements...")

    # Select brain elements (gray matter = 4, white matter = 5)
    brain_mask = np.isin(tissue_labels, [4, 5]) # hard coded!!!!
    brain_elems = elem[brain_mask][:, :4].astype(int)

    brain_elems = _ensure_1indexed(brain_elems)

    openface, elemid = i2m.volface(brain_elems)

    # Convert to 0-indexed for Blender
    openface = _ensure_0indexed(openface.astype(int))

    print(f"      Found {len(openface):,} cortex faces")
    return openface


def extract_layer_shell(node, elem, tissue_labels, target_labels):
    """
    Extract the closed shell surface of one or more tissue labels via i2m.volface,
    for visualizing an individual layer of an arbitrary-layer-count model.
    """
    mask = np.isin(tissue_labels, target_labels)
    layer_elems = elem[mask][:, :4].astype(int)
    layer_elems = _ensure_1indexed(layer_elems)

    openface, _ = i2m.volface(layer_elems)
    openface = _ensure_0indexed(openface.astype(int))
    return openface


def create_mesh_object(name, vertices, faces, collection):
    """Create Blender mesh object in specified collection"""
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    
    mesh.from_pydata(vertices.tolist(), [], [list(f) for f in faces])
    mesh.update()
    
    # link to scene temporarily for editing. Deselect everything else first -
    # Blender's multi-object edit mode would otherwise pull in whatever else
    # happens to still be selected (e.g. headmesh, if the user clicked it in
    # the outliner earlier), and remove_doubles/normals_make_consistent below
    # would silently mutate that object's geometry too.
    bpy.context.collection.objects.link(obj)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # clean mesh using blender
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    if bpy.app.version >= (4, 1, 0):
        bpy.ops.object.shade_smooth_by_angle()
    else:
        bpy.ops.object.shade_smooth()
    
    # move to target collection
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    collection.objects.link(obj)
    
    return obj


# =============================================================================
# FLEXIBLE (N-LAYER) IMPORT: generic loading + layer-definition discovery
# =============================================================================

# Field/key names checked (case-insensitive) for an embedded per-label name list.
_MAT_LAYER_NAME_FIELDS = (
    'layername', 'layer_name', 'layernames', 'tissuename', 'tissue_name',
    'tissuenames', 'seg_name', 'segname', 'labelname', 'label_name', 'names',
)
_JMESH_LAYER_NAME_KEYS = (
    'MeshTissueName', 'TissueName', 'MeshLayerName', 'LayerName',
    'MeshGroupName', 'GroupName',
)

_ROLE_COLORS = {
    'scalp':        (1.00, 0.90, 0.80, 1.0),
    'skull':        (0.90, 0.85, 0.70, 1.0),
    'csf':          (0.70, 0.85, 1.00, 1.0),
    'gray_matter':  (0.80, 0.70, 0.70, 1.0),
    'white_matter': (0.95, 0.95, 0.90, 1.0),
}
_FALLBACK_PALETTE = [
    (1.00, 0.90, 0.80, 1.0), (0.90, 0.85, 0.70, 1.0), (0.70, 0.85, 1.00, 1.0),
    (0.80, 0.70, 0.70, 1.0), (0.95, 0.95, 0.90, 1.0), (0.75, 0.60, 0.90, 1.0),
    (0.60, 0.90, 0.75, 1.0), (0.90, 0.60, 0.60, 1.0),
]


def _coerce_name_list(raw):
    """Normalize a MATLAB cell array / numpy object array / list into plain strings."""
    try:
        flat = np.array(raw, dtype=object).flatten()
    except Exception:
        return None

    names = []
    for item in flat:
        if isinstance(item, np.ndarray):
            item = item.item() if item.size == 1 else str(item)
        s = str(item).strip()
        if not s:
            return None
        names.append(s)
    return names if names else None


def _names_to_definitions(names, unique_labels):
    if not names or len(names) < len(unique_labels):
        return None
    return {
        label: {'name': names[i], 'role': names[i].lower().replace(' ', '_')}
        for i, label in enumerate(unique_labels)
    }


def _extract_embedded_layer_names(field_lookup, field_keys, unique_labels):
    """Search a dict of struct/file fields (checked case-insensitively) for a
    per-label name list. Returns {label: {'name', 'role'}} or None."""
    lower_map = {str(k).lower(): k for k in field_lookup.keys()}
    for candidate in field_keys:
        actual_key = lower_map.get(candidate.lower())
        if actual_key is None:
            continue
        names = _coerce_name_list(field_lookup[actual_key])
        defs = _names_to_definitions(names, unique_labels)
        if defs:
            return defs
    return None


def _load_sidecar_layer_json(mesh_path, unique_labels):
    """Look for a same-named .json (or <stem>_layers.json) next to the mesh file
    holding {"<label>": "Name"} or {"<label>": {"name":.., "role":..}} entries."""
    candidates = [
        mesh_path.with_suffix('.json'),
        mesh_path.with_name(mesh_path.stem + '_layers.json'),
    ]
    for candidate in candidates:
        if candidate == mesh_path or not candidate.exists():
            continue
        try:
            with open(candidate, 'r') as f:
                raw = json.load(f)
        except Exception:
            continue

        entries = raw.get('layers', raw) if isinstance(raw, dict) else None
        if not isinstance(entries, dict):
            continue

        layer_definitions = {}
        for label in unique_labels:
            entry = entries.get(str(label), entries.get(label))
            if entry is None:
                continue
            if isinstance(entry, str):
                layer_definitions[label] = {'name': entry, 'role': entry.lower().replace(' ', '_')}
            elif isinstance(entry, dict):
                name = entry.get('name', f"Layer {label}")
                layer_definitions[label] = {'name': name, 'role': entry.get('role', name.lower().replace(' ', '_'))}

        if len(layer_definitions) == len(unique_labels):
            return layer_definitions

    return None


def _load_mat_mesh(mesh_path):
    """Load node/elem/tissue_labels from a .mat file, and search the struct's
    own fields for an embedded layer-name list."""
    mat_data = sio.loadmat(str(mesh_path))
    mesh_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
    mesh = mat_data[mesh_key][0, 0]

    node = np.array(mesh['node'], dtype=np.float64)
    if node.ndim > 2:
        node = node.squeeze()
    if node.shape[1] > 3:
        node = node[:, :3]

    elem = np.array(mesh['elem'], dtype=np.int32)
    if elem.ndim > 2:
        elem = elem.squeeze()

    tissue_labels = elem[:, 4].astype(np.int32) if elem.shape[1] >= 5 else np.ones(len(elem), dtype=np.int32)
    unique_labels = sorted(int(l) for l in np.unique(tissue_labels))

    struct_fields = {name: mesh[name] for name in (mesh.dtype.names or ())}
    embedded_names = _extract_embedded_layer_names(struct_fields, _MAT_LAYER_NAME_FIELDS, unique_labels)

    return node, elem, tissue_labels, embedded_names


def _load_jmesh_mesh(mesh_path):
    """Load node/elem/tissue_labels from a .jmsh/.bmsh/.json (JNIfTI-style mesh)
    file via jdata, and search top-level keys for an embedded layer-name list."""
    import jdata as jd
    data = jd.load(str(mesh_path))

    node = np.array(data['MeshNode'], dtype=np.float64)
    if node.shape[1] > 3:
        node = node[:, :3]

    elem = np.array(data['MeshElem'], dtype=np.int32)
    tissue_labels = elem[:, 4].astype(np.int32) if elem.shape[1] >= 5 else np.ones(len(elem), dtype=np.int32)
    unique_labels = sorted(int(l) for l in np.unique(tissue_labels))

    embedded_names = _extract_embedded_layer_names(data, _JMESH_LAYER_NAME_KEYS, unique_labels)

    return node, elem, tissue_labels, embedded_names


def load_layered_mesh_source(mesh_path):
    """
    Load a layered mesh from .mat/.jmsh/.bmsh/.json and search for layer
    definitions (label -> name/role), in order: fields embedded in the mesh
    file itself, then a sidecar JSON next to it. Neither found means the
    caller should prompt the user (or fall back to "Layer N" defaults).

    Returns:
        dict with 'node', 'elem', 'tissue_labels', 'unique_labels',
        'layer_definitions' (dict or None if not found anywhere)
    """
    mesh_path = Path(mesh_path)
    suffix = mesh_path.suffix.lower()

    if suffix == '.mat':
        node, elem, tissue_labels, embedded_names = _load_mat_mesh(mesh_path)
    elif suffix in ('.jmsh', '.bmsh', '.json'):
        node, elem, tissue_labels, embedded_names = _load_jmesh_mesh(mesh_path)
    else:
        raise ValueError(f"Unsupported mesh format: {suffix} (expected .mat/.jmsh/.bmsh/.json)")

    unique_labels = sorted(int(l) for l in np.unique(tissue_labels))

    layer_definitions = embedded_names
    if layer_definitions is None:
        layer_definitions = _load_sidecar_layer_json(mesh_path, unique_labels)

    return {
        'node': node,
        'elem': elem,
        'tissue_labels': tissue_labels,
        'unique_labels': unique_labels,
        'layer_definitions': layer_definitions,
    }


# =============================================================================
# NIFTI SEGMENTED-VOLUME IMPORT (single-pass CGAL multi-domain meshing via
# iso2mesh.cgalv2m() - see _mesh_labeled_volume())
# =============================================================================

# Canonical tissue-ID convention: 1-Scalp, 2-Skull, 3-CSF, 4-GM, 5-WM, 6-air.
_BRAIN2MESH_LAYER_DEFINITIONS = {
    1: {'name': 'Scalp', 'role': 'scalp'},
    2: {'name': 'Skull', 'role': 'skull'},
    3: {'name': 'CSF', 'role': 'csf'},
    4: {'name': 'Gray Matter', 'role': 'gray_matter'},
    5: {'name': 'White Matter', 'role': 'white_matter'},
    6: {'name': 'Air Pocket', 'role': 'other'},
}

# Filename keywords used to guess a per-file tissue role (case-insensitive).
_NIFTI_FILENAME_TISSUE_KEYWORDS = {
    'scalp': ('scalp', 'skin'),
    'skull': ('skull', 'bone'),
    'csf':   ('csf',),
    'gm':    ('graymatter', 'greymatter', 'gray_matter', 'grey_matter', 'gray', 'grey', 'gm'),
    'wm':    ('whitematter', 'white_matter', 'white', 'wm'),
}

_BRAIN2MESH_REQUIRED_KEYS = ('wm', 'gm')

# Matches _BRAIN2MESH_LAYER_DEFINITIONS' numbering.
_SEG_KEY_TO_CANONICAL_ID = {'scalp': 1, 'skull': 2, 'csf': 3, 'gm': 4, 'wm': 5}
# Outer-to-inner so the more specific/inner tissue wins where masks overlap.
_CANONICAL_TISSUE_APPLY_ORDER = ('scalp', 'skull', 'csf', 'gm', 'wm')

# Beyond this, a file is almost certainly a scan/probability map, not a segmentation.
_MAX_PLAUSIBLE_SEGMENTATION_LABELS = 100

_ROLE_TO_BRAIN2MESH_KEY = {
    'scalp': 'scalp', 'skull': 'skull', 'csf': 'csf',
    'gray_matter': 'gm', 'white_matter': 'wm',
}

# Pre-fills the role dialog only; user always confirms/edits.
_NIFTI_OUTER_TO_INNER_GUESS = ('scalp', 'skull', 'csf', 'gray_matter', 'white_matter')


_vendored_jnifti_module = None


def _vendored_jnifti():
    """Load this add-on's own vendored jdata.jnifti directly by path, so a
    plain `import jdata` from another add-on can't shadow it with a
    different copy."""
    global _vendored_jnifti_module
    if _vendored_jnifti_module is None:
        import importlib.util
        jnifti_path = Path(__file__).resolve().parent / "_libs" / "jdata" / "jnifti.py"
        spec = importlib.util.spec_from_file_location("_neurocaptain_vendored_jnifti", jnifti_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _vendored_jnifti_module = module
    return _vendored_jnifti_module


def _read_nifti_volume(path, downsample=1):
    """Read a .nii/.nii.gz into a plain 3D numpy array (squeezed to drop any
    singleton time/channel axis). downsample keeps every Nth voxel per axis
    (strided, not averaged - this is a discrete label volume)."""
    jnii = _vendored_jnifti().nii2jnii(str(path))
    volume = np.squeeze(np.asarray(jnii['NIFTIData']))
    if downsample > 1:
        volume = volume[::downsample, ::downsample, ::downsample]
    return volume


def _read_nifti_voxel_size_mm(path):
    """Read a .nii/.nii.gz file's per-axis voxel size in mm."""
    jnii = _vendored_jnifti().nii2jnii(str(path))
    voxel_size = np.asarray(jnii['NIFTIHeader']['VoxelSize'], dtype=np.float64)
    return voxel_size[:3]


def guess_nifti_tissue_roles(unique_labels):
    """Best-guess outer-to-inner role for each raw label detected in a single
    segmented volume, for pre-filling (never silently applying) the
    role-assignment dialog. Labels beyond the 5 known roles default to 'other'."""
    return {
        label: (_NIFTI_OUTER_TO_INNER_GUESS[i] if i < len(_NIFTI_OUTER_TO_INNER_GUESS) else 'other')
        for i, label in enumerate(unique_labels)
    }


def _smooth_binary_mask(mask, sigma):
    """Gaussian-blur a binary mask then re-threshold at 0.5, rounding off
    downsampling's staircase edges. sigma <= 0 is a no-op."""
    if sigma <= 0:
        return mask
    blurred = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return blurred > 0.5


def match_nifti_filename_to_tissue(filename):
    """Match a per-layer NIfTI filename against known tissue keywords.
    Returns the brain2mesh seg-dict key ('scalp'/'skull'/'csf'/'gm'/'wm') or
    None if no keyword matched."""
    stem = Path(filename).stem.lower()
    for key, keywords in _NIFTI_FILENAME_TISSUE_KEYWORDS.items():
        for kw in sorted(keywords, key=len, reverse=True):
            if kw in stem:
                return key
    return None


def build_seg_dict_from_files_with_roles(filepaths, file_roles, downsample=1, mask_smooth=0.0):
    """Build brain2mesh's seg dict from multiple per-tissue NIfTI files using
    an explicit {str(filepath): role} mapping. Files sharing a role are OR'd
    together; 'other'/unassigned files are dropped."""
    seg = {}
    for path in filepaths:
        role = file_roles.get(str(path))
        key = _ROLE_TO_BRAIN2MESH_KEY.get(role)
        if key is None:  # 'other' / unassigned
            continue
        mask = _read_nifti_volume(path, downsample=downsample) > 0
        mask = _smooth_binary_mask(mask, mask_smooth)
        seg[key] = mask if key not in seg else (seg[key] | mask)

    missing = [k for k in _BRAIN2MESH_REQUIRED_KEYS if k not in seg]
    if missing:
        raise ValueError(f"At least one file must be assigned to each of: {', '.join(missing)} (gray and white matter are required)")
    empty = [k for k, mask in seg.items() if not mask.any()]
    if empty:
        raise ValueError(
            f"Tissue(s) {', '.join(empty)} have no voxels left after downsampling by {downsample} "
            "and smoothing - try a smaller downsample factor or mask_smooth"
        )
    return seg


def build_seg_dict_from_single_nifti(filepath, label_roles, downsample=1, mask_smooth=0.0):
    """Build brain2mesh's seg dict from one multi-label NIfTI volume plus a
    {label: role} mapping. Labels sharing a role are OR'd together."""
    volume = _read_nifti_volume(filepath, downsample=downsample)
    seg = {}
    for label, role in label_roles.items():
        key = _ROLE_TO_BRAIN2MESH_KEY.get(role)
        if key is None:  # 'other' / unassigned
            continue
        mask = (volume == label)
        seg[key] = mask if key not in seg else (seg[key] | mask)

    seg = {key: _smooth_binary_mask(mask, mask_smooth) for key, mask in seg.items()}

    missing = [k for k in _BRAIN2MESH_REQUIRED_KEYS if k not in seg]
    if missing:
        raise ValueError(f"At least one label must be assigned to each of: {', '.join(missing)} (gray and white matter are required)")
    empty = [k for k, mask in seg.items() if not mask.any()]
    if empty:
        raise ValueError(
            f"Tissue(s) {', '.join(empty)} have no voxels left after downsampling by {downsample} "
            "and smoothing - try a smaller downsample factor or mask_smooth"
        )
    return seg


def _combine_seg_masks_to_labeled_volume(seg):
    """Combine per-tissue binary masks into one uint8 volume using the
    canonical tissue IDs, applied outer-to-inner so the more specific/inner
    tissue wins where masks overlap."""
    shape = next(iter(seg.values())).shape
    combined = np.zeros(shape, dtype=np.uint8)
    for key in _CANONICAL_TISSUE_APPLY_ORDER:
        if key in seg:
            combined[seg[key]] = _SEG_KEY_TO_CANONICAL_ID[key]
    return combined


def _mesh_labeled_volume(volume, opt=None, maxvol=100, voxel_size_mm=(1.0, 1.0, 1.0)):
    """Tetrahedralize a labeled uint8 volume (0=background, 1-5 canonical
    tissue IDs) via iso2mesh.cgalv2m()'s single-pass CGAL multi-domain
    mesher. voxel_size_mm scales its output from raw voxel-index units to
    real mm (cgalv2m() has no way to be told the real voxel size).

    Returns node (x/y/z only, scaled to mm), elem (last column = tissue ID).
    """
    if opt is None:
        opt = {}
    node, elem, _face = i2m.cgalv2m(volume, opt, maxvol)
    if node.shape[1] > 3:
        node = node[:, :3]
    node = node * np.asarray(voxel_size_mm, dtype=np.float64)
    return node, elem.astype(np.int32)


def load_layered_mesh_source_from_nifti(filepaths, label_roles=None, file_roles=None, downsample=1, mask_smooth=0.0, maxvol=100, **cfg):
    """
    Load a layered mesh source from segmented NIfTI volume(s) via
    _mesh_labeled_volume(), in the same node/elem/tissue_labels shape
    load_layered_mesh_source() produces from a .mat/.jmsh file.

    label_roles ({label: role}) is required for the single-file case;
    file_roles ({str(filepath): role}) for the multi-file case. Pass None
    for whichever applies to get back {'needs_roles'/'needs_file_roles':
    True, ...} so the caller can prompt the user, then call again with it
    filled in.

    Returns a dict with 'node', 'elem', 'tissue_labels', 'unique_labels',
    'layer_definitions', or a needs_roles/needs_file_roles dict as above.
    """
    if not ISO2MESH_AVAILABLE:
        raise RuntimeError("iso2mesh is required for NIfTI import. Install with: pip install iso2mesh")

    if len(filepaths) > 1 and file_roles is None:
        return {'needs_file_roles': True, 'filenames': [str(p) for p in filepaths]}

    if len(filepaths) == 1 and label_roles is None:
        volume = _read_nifti_volume(filepaths[0], downsample=downsample)
        unique_labels = sorted(int(l) for l in np.unique(volume) if l != 0)
        if len(unique_labels) < 2:
            raise ValueError(
                f"Only {len(unique_labels)} distinct tissue label(s) found in {Path(filepaths[0]).name} - "
                "brain2mesh needs at least 2 separate regions (gray matter and white matter, at "
                "minimum) to build a layered head model, and no role assignment can make one region "
                "satisfy two required roles. If this is a single-region head/background mask rather "
                "than a multi-tissue segmentation, use 'Headmesh from NIfTI Mask' (under Head Model "
                "& Landmark Geometry) instead - that only needs one region."
            )
        if len(unique_labels) > _MAX_PLAUSIBLE_SEGMENTATION_LABELS:
            raise ValueError(
                f"{len(unique_labels)} distinct nonzero values found in {Path(filepaths[0]).name} - "
                "that's far too many to be a tissue segmentation (a handful to a few dozen labels is "
                "expected, e.g. scalp/skull/csf/gray/white matter). This file is likely a continuous "
                "intensity or probability-map volume rather than a discrete labeled segmentation - "
                "check that you selected the hard/discrete label map, not the raw scan or a tissue "
                "probability map."
            )
        return {'needs_roles': True, 'unique_labels': unique_labels}

    if len(filepaths) == 1:
        seg = build_seg_dict_from_single_nifti(filepaths[0], label_roles, downsample=downsample, mask_smooth=mask_smooth)
    else:
        seg = build_seg_dict_from_files_with_roles(filepaths, file_roles, downsample=downsample, mask_smooth=mask_smooth)

    combined_volume = _combine_seg_masks_to_labeled_volume(seg)
    voxel_size_mm = _read_nifti_voxel_size_mm(filepaths[0]) * downsample
    print(f"   Running single-pass CGAL multi-domain meshing on {len(seg)} tissue mask(s) "
          f"(downsample={downsample}, mask_smooth={mask_smooth}, maxvol={maxvol}, "
          f"voxel_size_mm={voxel_size_mm.tolist()}): {', '.join(seg.keys())}...")
    node, elem = _mesh_labeled_volume(combined_volume, opt=cfg.get('opt'), maxvol=maxvol, voxel_size_mm=voxel_size_mm)
    tissue_labels = elem[:, 4]
    unique_labels = sorted(int(l) for l in np.unique(tissue_labels))

    return {
        'node': node,
        'elem': elem,
        'tissue_labels': tissue_labels,
        'unique_labels': unique_labels,
        'layer_definitions': {
            l: _BRAIN2MESH_LAYER_DEFINITIONS.get(l, {'name': f'Layer {l}', 'role': 'other'})
            for l in unique_labels
        },
    }


def import_layered_head_model_from_nifti(filepaths, label_roles=None, file_roles=None, reference_obj_name='headmesh', downsample=1, mask_smooth=0.0, **cfg):
    """
    Import a layered head model from segmented NIfTI volume(s) - see
    load_layered_mesh_source_from_nifti() for the file-count/label_roles/
    file_roles/downsample/mask_smooth/maxvol contract. Bridges the resulting
    mesh straight into import_layered_head_model_flexible() without
    re-reading/re-meshing.
    """
    filepaths = [Path(p) for p in filepaths]
    for p in filepaths:
        if not p.exists():
            return {'success': False, 'message': f"File not found: {p}", 'info': None}

    try:
        source = load_layered_mesh_source_from_nifti(
            filepaths, label_roles=label_roles, file_roles=file_roles,
            downsample=downsample, mask_smooth=mask_smooth, **cfg)
    except Exception as e:
        return {'success': False, 'message': f"Failed to load/mesh NIfTI volume(s): {e}", 'info': None}

    if source.get('needs_roles'):
        return {
            'success': False, 'needs_roles': True, 'unique_labels': source['unique_labels'],
            'message': 'Tissue role assignment required', 'info': None,
        }
    if source.get('needs_file_roles'):
        return {
            'success': False, 'needs_file_roles': True, 'filenames': source['filenames'],
            'message': 'Per-file tissue role assignment required', 'info': None,
        }

    display_path = filepaths[0] if len(filepaths) == 1 else filepaths[0].parent / f"{len(filepaths)}_nifti_layer_files"
    return import_layered_head_model_flexible(
        display_path, reference_obj_name=reference_obj_name,
        layer_definitions=source['layer_definitions'], source=source,
    )


def import_layered_head_model_flexible(mesh_path, reference_obj_name='headmesh', layer_definitions=None, source=None):
    """
    Import a layered head model with an arbitrary number of tissue layers
    (fewer than the standard 5, or more), unlike import_layered_head_model()
    which assumes exactly the scalp/skull/CSF/gray/white 5-layer convention.

    Layer names/roles are resolved in this order: the explicit
    layer_definitions argument (typically supplied by the
    NEUROCAPTAIN_OT_define_layers prompt after the user fills it in), then
    whatever load_layered_mesh_source() found embedded in the mesh file or in
    a sidecar JSON, then a "Layer {label}" / role 'other' default for any
    label still unresolved.

    Every detected layer is visualized as its own closed shell object (an
    "exploded"/onion view), rather than only extracting a head surface and a
    single cortex surface as the 5-layer import does. The layer whose role is
    'scalp' (or, absent that, the lowest tissue label) is used for orientation/
    scale matching against reference_obj_name and stored as head_surface_obj;
    the outermost layer with a gray/white-matter-ish role (if any) is stored
    as cortex_obj, for compatibility with the existing MMC/redbird pipeline.

    Args:
        mesh_path: Path to .mat/.jmsh/.bmsh/.json mesh file (only read from
            disk if `source` is not supplied; otherwise used just for display)
        reference_obj_name: Name of reference object for scale matching
        layer_definitions: optional {label: {'name':.., 'role':..}} override
        source: optional pre-built {'node','elem','tissue_labels',
            'unique_labels','layer_definitions'} dict, e.g. already produced
            by a NIfTI->brain2mesh meshing pass - skips load_layered_mesh_source()
            so that (comparatively expensive) meshing step isn't repeated.

    Returns:
        dict: Status dictionary with 'success', 'message', and 'info' keys
    """
    global LAYERED_MESH
    t0 = time.time()
    print("\n" + "=" * 70)
    print("IMPORTING LAYERED HEAD MODEL (flexible layer count)")
    print("=" * 70)

    mesh_path = Path(mesh_path)
    if not ISO2MESH_AVAILABLE:
        return {
            'success': False,
            'message': "iso2mesh is required for surface extraction. Install with: pip install iso2mesh",
            'info': None,
        }

    if source is None:
        if not mesh_path.exists():
            return {'success': False, 'message': f"File not found: {mesh_path}", 'info': None}
        print(f"\n1. Loading mesh from: {mesh_path.name}")
        try:
            source = load_layered_mesh_source(mesh_path)
        except Exception as e:
            return {'success': False, 'message': f"Failed to load mesh: {e}", 'info': None}
    else:
        print(f"\n1. Using pre-built mesh source ({mesh_path.name})")

    node = source['node']
    elem = source['elem']
    tissue_labels = source['tissue_labels']
    unique_labels = source['unique_labels']
    num_layers = len(unique_labels)
    print(f"   Nodes: {len(node):,}  Elements: {len(elem):,}  Layers: {num_layers}")

    if layer_definitions is None:
        layer_definitions = source['layer_definitions']

    print("\n2. Resolving layer names/roles...")
    layer_info = {}
    for label in unique_labels:
        count = int(np.sum(tissue_labels == label))
        defn = (layer_definitions or {}).get(label, {})
        name = defn.get('name') or f"Layer {label}"
        role = defn.get('role') or 'other'
        layer_info[label] = {'name': name, 'role': role, 'count': count}
        print(f"   Layer {label}: {name} [{role}] ({count:,} elements)")

    # Create/get collection (kept separate from the fixed 5-layer collection)
    print("\n3. Extracting + creating per-layer shell surfaces...")
    coll_name = "LayeredModel_Visualization"
    if coll_name in bpy.data.collections:
        coll = bpy.data.collections[coll_name]
        for obj in list(coll.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    else:
        coll = bpy.data.collections.new(coll_name)
        bpy.context.scene.collection.children.link(coll)

    ref = bpy.data.objects.get(reference_obj_name)
    scalp_label = next((l for l in unique_labels if layer_info[l]['role'] == 'scalp'), unique_labels[0])

    layer_objects = {}
    try:
        for i, label in enumerate(unique_labels):
            faces = extract_layer_shell(node, elem, tissue_labels, [label])
            if len(faces) == 0:
                print(f"   ! Layer {label} ({layer_info[label]['name']}) produced no faces, skipping")
                continue

            safe_name = layer_info[label]['name'].replace(' ', '_')
            obj_name = f"Layer{label}_{safe_name}"
            obj = create_mesh_object(obj_name, node, faces, coll)

            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            recenter_on_volume_centroid(obj)
            obj.rotation_euler = (0.0, 0.0, 0.0)
            if ref:
                obj.scale = ref.scale

            mat = bpy.data.materials.new(f"{obj_name}_Mat")
            mat.use_nodes = True
            is_scalp = (label == scalp_label)
            if is_scalp and bpy.app.version < (4, 0, 0):
                mat.blend_method = 'BLEND'
            mat_nodes = mat.node_tree.nodes
            mat_nodes.clear()
            bsdf = mat_nodes.new('ShaderNodeBsdfPrincipled')
            output = mat_nodes.new('ShaderNodeOutputMaterial')
            color = _ROLE_COLORS.get(layer_info[label]['role'], _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)])
            bsdf.inputs['Base Color'].default_value = color
            if is_scalp:
                bsdf.inputs['Alpha'].default_value = 0.3
            mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
            obj.data.materials.append(mat)

            layer_objects[label] = obj
            print(f"   OK {obj_name} ({'semi-transparent' if is_scalp else 'opaque'})")
    except Exception as e:
        return {'success': False, 'message': f"Failed to extract/create layer surfaces: {e}", 'info': None}

    if not layer_objects:
        return {'success': False, 'message': "No layer surfaces could be extracted", 'info': None}

    # Snap the whole assembly onto headmesh's volume centroid (not headmesh's
    # own pivot, which is vertex-mean-centered to match the landmark files -
    # see recenter_on_vertex_mean) by translating every layer with the same
    # rigid offset (anchored on the scalp layer). Volume-centroid-to-volume-
    # centroid matching is what actually lines up two independently-meshed
    # files of the same physical head (verified: ~1mm apart), unlike vertex
    # mean, which drifts with how densely each file happens to sample the
    # surface. Each layer's recenter_on_volume_centroid() call above already
    # recentered its own pivot independently, so without this they'd each
    # have been left at their own separate volume-centroid location instead
    # of lining up with each other and with headmesh.
    scalp_obj = layer_objects.get(scalp_label)
    if scalp_obj is not None:
        target_location = tuple(compute_volume_centroid_world(ref)) if ref else (0.0, 0.0, 0.0)
        offset = tuple(t - s for t, s in zip(target_location, scalp_obj.location))
        for obj in layer_objects.values():
            obj.location = tuple(l + o for l, o in zip(obj.location, offset))

    head_obj = layer_objects.get(scalp_label)
    scalp_faces = extract_layer_shell(node, elem, tissue_labels, [scalp_label])
    scalp_vertex_idx = np.unique(scalp_faces.flatten())
    mesh_centroid = node[scalp_vertex_idx].mean(axis=0)

    brain_roles = ('gray_matter', 'white_matter', 'brain', 'cortex')
    brain_labels = [l for l in unique_labels if layer_info[l]['role'] in brain_roles]
    cortex_obj = None
    for l in brain_labels:
        if l in layer_objects:
            cortex_obj = layer_objects[l]  # innermost brain-ish layer wins

    # Store data in global structure (used in redbird or mmc sim)
    LAYERED_MESH.mesh_path = str(mesh_path)
    LAYERED_MESH.nodes = node
    LAYERED_MESH.elems = elem
    LAYERED_MESH.tissue_labels = tissue_labels
    LAYERED_MESH.num_layers = num_layers
    LAYERED_MESH.layer_info = layer_info
    LAYERED_MESH.mesh_centroid = mesh_centroid
    LAYERED_MESH.head_surface_obj = head_obj
    LAYERED_MESH.cortex_obj = cortex_obj
    _persist_layered_mesh_to_object()

    bpy.ops.object.select_all(action='DESELECT')
    print(f"\nOK layered import ({num_layers} layers): {time.time()-t0:.1f} seconds")
    print("\n" + "=" * 70)
    print("OK IMPORT COMPLETE")
    print("=" * 70)

    return {
        'success': True,
        'message': f'Layered head model imported successfully ({num_layers} layers)',
        'info': LAYERED_MESH.get_info_string(),
    }


# =============================================================================
# MAIN IMPORT FUNCTION
# =============================================================================

def import_layered_head_model(mesh_path, reference_obj_name='headmesh'):
    """
    Import a multi-layer head model from .mat file
    
    Args:
        mesh_path: Path to .mat file containing mesh data
        reference_obj_name: Name of reference object for orientation (default: 'headmesh')
    
    Returns:
        dict: Status dictionary with 'success', 'message', and 'info' keys
    """
    global LAYERED_MESH
    t0 = time.time()
    print("\n" + "=" * 70)
    print("IMPORTING LAYERED HEAD MODEL")
    print("=" * 70)
    
    # Validate path
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        return {
            'success': False,
            'message': f"File not found: {mesh_path}",
            'info': None
        }
    
    if not mesh_path.suffix.lower() == '.mat':
        return {
            'success': False,
            'message': "File must be a MATLAB .mat file",
            'info': None
        }
    
    # Check iso2mesh is available (needed for volface)
    if not ISO2MESH_AVAILABLE:
        return {
            'success': False,
            'message': "iso2mesh is required for surface extraction. Install with: pip install iso2mesh",
            'info': None
        }
    
    # load mesh data
    print(f"\n1. Loading mesh from: {mesh_path.name}")
    try:
        mat_data = sio.loadmat(str(mesh_path))
        mesh_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
        mesh = mat_data[mesh_key][0, 0]
        
        node = np.array(mesh['node'], dtype=np.float64)
        if node.ndim > 2:
            node = node.squeeze()
        if node.shape[1] > 3:
            node = node[:, :3]
        
        elem = np.array(mesh['elem'], dtype=np.int32)
        if elem.ndim > 2:
            elem = elem.squeeze()
        
        tissue_labels = elem[:, 4].astype(np.int32) if elem.shape[1] >= 5 else np.ones(len(elem), dtype=np.int32)
        
        print(f"   ✓ Nodes: {len(node):,}")
        print(f"   ✓ Elements: {len(elem):,}")
        
    except Exception as e:
        return {
            'success': False,
            'message': f"Failed to load mesh: {str(e)}",
            'info': None
        }
    
    # analyze layers
    print("\n2. Analyzing tissue layers...")
    unique_labels = np.unique(tissue_labels)
    num_layers = len(unique_labels)
    
    # standard 5-layer naming
    layer_names = {
        1: "Scalp",
        2: "Skull",
        3: "CSF",
        4: "Gray Matter",
        5: "White Matter"
    }
    
    layer_info = {}
    for label in unique_labels:
        count = np.sum(tissue_labels == label)
        layer_info[int(label)] = {
            'name': layer_names.get(int(label), f"Tissue {label}"),
            'count': int(count)
        }
        print(f"   Layer {label}: {layer_info[int(label)]['name']} ({count:,} elements)")
    
    # extract surfaces using volface
    print("\n3. Extracting surfaces (via iso2mesh volface)...")
    try:
        scalp_faces = extract_outer_surface_only(node, elem, tissue_labels, target_tissue=1)
        cortex_faces = extract_cortex_surface(node, elem, tissue_labels)

        # Centroid of scalp surface vertices in original mesh coords.
        # Blender will center Head_Surface_5L on this point, so this is the
        # correct Blender→mesh translation offset.
        scalp_vertex_idx = np.unique(scalp_faces.flatten())
        mesh_centroid = node[scalp_vertex_idx].mean(axis=0)
        print(f"   Scalp centroid (mesh coords): {mesh_centroid}")
    except Exception as e:
        return {
            'success': False,
            'message': f"Failed to extract surfaces: {str(e)}",
            'info': None
        }
    
    # Create/get collection
    print("\n4. Creating Blender objects...")
    coll_name = "FiveLayer_Visualization"
    
    if coll_name in bpy.data.collections:
        coll = bpy.data.collections[coll_name]
        # Clear existing objects
        for obj in list(coll.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    else:
        coll = bpy.data.collections.new(coll_name)
        bpy.context.scene.collection.children.link(coll)
    
    print(f"   ✓ Collection: {coll_name}")
    
    # Create head surface (scalp)
    head_obj = create_mesh_object("Head_Surface_5L", node, scalp_faces, coll)
    
    # Set origin, then snap onto headmesh's volume centroid - not headmesh's
    # own pivot, which is vertex-mean-centered to match the landmark files
    # (see recenter_on_vertex_mean). Volume-centroid-to-volume-centroid
    # matching is what actually lines up two independently-meshed files of
    # the same physical head (verified: ~1mm apart), unlike vertex mean,
    # which drifts with how densely each file happens to sample the
    # surface.
    ref = bpy.data.objects.get(reference_obj_name)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = head_obj
    head_obj.select_set(True)
    recenter_on_volume_centroid(head_obj)
    target_location = tuple(compute_volume_centroid_world(ref)) if ref else (0.0, 0.0, 0.0)
    offset = tuple(t - h for t, h in zip(target_location, head_obj.location))
    head_obj.location = target_location

    # Match scale from reference if available, but always use zero rotation.
    # MAT/iso2mesh meshes are in RAS coordinates (same Z-up as Blender), so
    # no rotation transform is needed. Copying rotation from headmesh would
    # propagate any pre-existing rotation error on that object.
    if ref:
        head_obj.scale = ref.scale
    head_obj.rotation_euler = (0.0, 0.0, 0.0)
    
    # Semi-transparent material
    mat_head = bpy.data.materials.new("HeadSurface_Mat")
    mat_head.use_nodes = True
    if bpy.app.version < (4, 0, 0):
        mat_head.blend_method = 'BLEND'
    nodes = mat_head.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    output = nodes.new('ShaderNodeOutputMaterial')
    bsdf.inputs['Base Color'].default_value = (1.0, 0.9, 0.8, 1.0)
    bsdf.inputs['Alpha'].default_value = 0.3
    mat_head.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    head_obj.data.materials.append(mat_head)
    
    print("   ✓ Head_Surface_5L (semi-transparent)")
    
    # Create brain cortex
    cortex_obj = create_mesh_object("Brain_Cortex_5L", node, cortex_faces, coll)
    
    # Match head orientation. Apply the same rigid offset used for head_obj
    # (rather than independently recentering to world zero) so the cortex
    # keeps its real position relative to the head surface.
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = cortex_obj
    cortex_obj.select_set(True)
    recenter_on_volume_centroid(cortex_obj)
    cortex_obj.location = tuple(c + o for c, o in zip(cortex_obj.location, offset))
    cortex_obj.rotation_euler = (0.0, 0.0, 0.0)
    cortex_obj.scale = head_obj.scale
    
    # Cortex material
    mat_cortex = bpy.data.materials.new("Cortex_Mat")
    mat_cortex.use_nodes = True
    nodes = mat_cortex.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    output = nodes.new('ShaderNodeOutputMaterial')
    bsdf.inputs['Base Color'].default_value = (0.8, 0.7, 0.7, 1.0)
    mat_cortex.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    cortex_obj.data.materials.append(mat_cortex)
    
    print("   ✓ Brain_Cortex_5L (for sensitivity visualization)")
    
    # Store data in global structure (used in redbird or mmc sim)
    LAYERED_MESH.mesh_path = str(mesh_path)
    LAYERED_MESH.nodes = node
    LAYERED_MESH.elems = elem
    LAYERED_MESH.tissue_labels = tissue_labels
    LAYERED_MESH.num_layers = num_layers
    LAYERED_MESH.layer_info = layer_info
    LAYERED_MESH.mesh_centroid = mesh_centroid
    LAYERED_MESH.head_surface_obj = head_obj
    LAYERED_MESH.cortex_obj = cortex_obj
    _persist_layered_mesh_to_object()
    
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')
    print(f"\n✓ 5 layer import: {time.time()-t0:.1f} seconds")
    print("\n" + "=" * 70)
    print("✓ IMPORT COMPLETE")
    print("=" * 70)
    
    return {
        'success': True,
        'message': 'Layered head model imported successfully',
        'info': LAYERED_MESH.get_info_string()
    }


# =============================================================================
# REDBIRD CONFIGURATION
# =============================================================================

class RedbirdConfig:
    """Store Redbird simulation configuration"""
    
    # Default optical properties (5-layer head model)
    DEFAULT_OPTICAL_PROPERTIES = {
        1: {'mua': 0.018, 'mus': 7.8, 'g': 0.90, 'n': 1.37},   # Scalp
        2: {'mua': 0.016, 'mus': 9.0, 'g': 0.90, 'n': 1.37},   # Skull
        3: {'mua': 0.004, 'mus': 0.3, 'g': 0.90, 'n': 1.33},   # CSF
        4: {'mua': 0.036, 'mus': 8.4, 'g': 0.90, 'n': 1.37},   # Gray Matter
        5: {'mua': 0.018, 'mus': 11.0, 'g': 0.90, 'n': 1.37}   # White Matter
    }
    
    def __init__(self):
        self.mode = 'CW'  # 'CW' or 'FD'
        self.frequency = 70.0  # MHz (for FD mode)
        self.sd_max_distance = 60.0  # mm
        self.crop_margin = 10.0  # mm
        self.min_depth = 2.0  # mm
        self.smooth_iterations = 5
        self.max_iter = 10              # CG solver max iterations
        self.regularization_lambda = 1e-6  # Tikhonov regularization for FEM system

        # Copy default optical properties
        self.optical_properties = {
            k: v.copy() for k, v in self.DEFAULT_OPTICAL_PROPERTIES.items()
        }
    
    def set_optical_property(self, layer, property_name, value):
        """Set optical property for a specific layer"""
        if layer not in self.optical_properties:
            self.optical_properties[layer] = {}
        self.optical_properties[layer][property_name] = value
    
    def get_config_string(self):
        """Get human-readable configuration summary"""
        lines = [
            f"Mode: {self.mode}",
            f"Frequency: {self.frequency} MHz" if self.mode == 'FD' else "",
            f"SD Max Distance: {self.sd_max_distance} mm",
            f"Crop Margin: {self.crop_margin} mm",
            f"Min Depth: {self.min_depth} mm",
            f"Solver: max_iter={self.max_iter}, lambda={self.regularization_lambda:.2e}",
            f"Smooth Iterations: {self.smooth_iterations}",
            "",
            "Optical Properties:"
        ]
        
        layer_names = {1: "Scalp", 2: "Skull", 3: "CSF", 4: "Gray Matter", 5: "White Matter"}
        
        for layer in sorted(self.optical_properties.keys()):
            props = self.optical_properties[layer]
            name = layer_names.get(layer, f"Layer {layer}")
            lines.append(f"  {name}:")
            lines.append(f"    μa={props.get('mua', 0):.3f}, μs={props.get('mus', 0):.1f}, g={props.get('g', 0):.2f}, n={props.get('n', 0):.2f}")
        
        return "\n".join([l for l in lines if l])


# Global config instance
REDBIRD_CONFIG = RedbirdConfig()


def setup_redbird_config(**kwargs):
    """
    Configure Redbird simulation parameters
    
    Keyword Args:
        mode: 'CW' or 'FD'
        frequency: Modulation frequency in MHz (for FD mode)
        sd_max_distance: Maximum source-detector distance in mm
        crop_margin: mm to extend beyond optode bounding box on all sides (default 10)
        min_depth: Minimum optode depth in mm
        smooth_iterations: Number of smoothing iterations for visualization
        optical_properties: Dict of optical properties per layer
    
    Returns:
        dict: Status dictionary
    """
    global REDBIRD_CONFIG
    
    # Update configuration
    if 'mode' in kwargs:
        REDBIRD_CONFIG.mode = kwargs['mode']
    if 'frequency' in kwargs:
        REDBIRD_CONFIG.frequency = kwargs['frequency']
    if 'sd_max_distance' in kwargs:
        REDBIRD_CONFIG.sd_max_distance = kwargs['sd_max_distance']
    if 'crop_margin' in kwargs:
        REDBIRD_CONFIG.crop_margin = kwargs['crop_margin']
    if 'min_depth' in kwargs:
        REDBIRD_CONFIG.min_depth = kwargs['min_depth']
    if 'smooth_iterations' in kwargs:
        REDBIRD_CONFIG.smooth_iterations = kwargs['smooth_iterations']
    if 'max_iter' in kwargs:
        REDBIRD_CONFIG.max_iter = kwargs['max_iter']
    if 'regularization_lambda' in kwargs:
        REDBIRD_CONFIG.regularization_lambda = kwargs['regularization_lambda']

    # Update optical properties
    if 'optical_properties' in kwargs:
        for layer, props in kwargs['optical_properties'].items():
            for prop_name, value in props.items():
                REDBIRD_CONFIG.set_optical_property(layer, prop_name, value)
    
    print("\nRedbird configuration updated:")
    print(REDBIRD_CONFIG.get_config_string())
    
    return {
        'success': True,
        'message': 'Redbird configuration updated',
        'config': REDBIRD_CONFIG.get_config_string()
    }


def get_layered_mesh_info():
    """Get information about currently loaded mesh"""
    global LAYERED_MESH
    return LAYERED_MESH.get_info_string()


def get_redbird_config_info():
    """Get current Redbird configuration"""
    global REDBIRD_CONFIG
    return REDBIRD_CONFIG.get_config_string()


def is_mesh_loaded():
    """Check if a layered mesh is currently loaded"""
    global LAYERED_MESH
    return LAYERED_MESH.is_loaded()


def check_dependencies():
    """Check if required dependencies are available"""
    return {
        'iso2mesh': ISO2MESH_AVAILABLE,
        'redbirdpy': REDBIRD_AVAILABLE,
        'all_available': ISO2MESH_AVAILABLE and REDBIRD_AVAILABLE
    }