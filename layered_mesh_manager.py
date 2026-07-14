import bpy
import numpy as np
import scipy.io as sio
from pathlib import Path
import json
import time

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
    
    # link to scene temporarily for editing
    bpy.context.collection.objects.link(obj)
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


def import_layered_head_model_flexible(mesh_path, reference_obj_name='headmesh', layer_definitions=None):
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
        mesh_path: Path to .mat/.jmsh/.bmsh/.json mesh file
        reference_obj_name: Name of reference object for scale matching
        layer_definitions: optional {label: {'name':.., 'role':..}} override

    Returns:
        dict: Status dictionary with 'success', 'message', and 'info' keys
    """
    global LAYERED_MESH
    t0 = time.time()
    print("\n" + "=" * 70)
    print("IMPORTING LAYERED HEAD MODEL (flexible layer count)")
    print("=" * 70)

    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        return {'success': False, 'message': f"File not found: {mesh_path}", 'info': None}

    if not ISO2MESH_AVAILABLE:
        return {
            'success': False,
            'message': "iso2mesh is required for surface extraction. Install with: pip install iso2mesh",
            'info': None,
        }

    print(f"\n1. Loading mesh from: {mesh_path.name}")
    try:
        source = load_layered_mesh_source(mesh_path)
    except Exception as e:
        return {'success': False, 'message': f"Failed to load mesh: {e}", 'info': None}

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
            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
            obj.location = (0, 0, 0)
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
    
    # Set origin and position
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = head_obj
    head_obj.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
    head_obj.location = (0, 0, 0)
    
    # Match scale from reference if available, but always use zero rotation.
    # MAT/iso2mesh meshes are in RAS coordinates (same Z-up as Blender), so
    # no rotation transform is needed. Copying rotation from headmesh would
    # propagate any pre-existing rotation error on that object.
    ref = bpy.data.objects.get(reference_obj_name)
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
    
    # Match head orientation
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = cortex_obj
    cortex_obj.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
    cortex_obj.location = (0, 0, 0)
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