import sys
user_site = '/home/users/mccann.as/.local/lib/python3.10/site-packages'
if user_site not in sys.path:
    sys.path.insert(0, user_site)

import bpy
import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
import time
import gc
import os

try:
    import redbirdpy as rb
    REDBIRD_AVAILABLE = True
except ImportError:
    REDBIRD_AVAILABLE = False
    print("Warning: redbirdpy not available. Install with: pip install redbirdpy")

try:
    import pmmc
    MMC_AVAILABLE = True
except ImportError:
    MMC_AVAILABLE = False
    print("Warning: pmmc not available")

# =============================================================================
# OPTICAL PROPERTIES (shared between solvers)
# =============================================================================
OPTICAL_PROPERTIES = {
    1: {'mua': 0.018, 'mus': 7.8, 'g': 0.90, 'n': 1.37},   # scalp
    2: {'mua': 0.016, 'mus': 9.0, 'g': 0.90, 'n': 1.37},   # skull
    3: {'mua': 0.004, 'mus': 0.3, 'g': 0.90, 'n': 1.33},   # CSF
    4: {'mua': 0.036, 'mus': 8.4, 'g': 0.90, 'n': 1.37},   # gray matter
    5: {'mua': 0.018, 'mus': 11.0, 'g': 0.90, 'n': 1.37}   # white matter
}

MIN_DEPTH = 2.0  # Minimum projection depth into mesh


# =============================================================================
# FIVE-LAYER MESH IMPORT FUNCTIONS
# =============================================================================

def load_mesh_from_file(filepath):
    """
    Load mesh from various formats and return nodes, elements, tissue_labels.
    Supports: .mat, .jmsh, .bmsh
    """
    import os
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext == '.mat':
        return load_mesh_from_mat(filepath)
    elif file_ext in ['.jmsh', '.bmsh']:
        return load_mesh_from_jmsh(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .mat, .jmsh, or .bmsh")


def load_mesh_from_mat(filepath):
    """Load mesh from MATLAB .mat file."""
    mat_data = sio.loadmat(filepath)
    
    if 'mesh1' in mat_data:
        mesh1 = mat_data['mesh1'][0, 0]
        nodes = np.array(mesh1['node'], dtype=np.float64)[:, :3]
        elems = np.array(mesh1['elem'], dtype=np.int32)
    elif 'node' in mat_data and 'elem' in mat_data:
        nodes = np.array(mat_data['node'], dtype=np.float64)[:, :3]
        elems = np.array(mat_data['elem'], dtype=np.int32)
    else:
        raise ValueError("MAT file must contain 'mesh1' or 'node'/'elem' fields")
    
    # Extract tissue labels (5th column)
    if elems.shape[1] >= 5:
        tissue_labels = elems[:, 4].astype(np.int32)
    else:
        tissue_labels = np.ones(len(elems), dtype=np.int32)
    
    return nodes, elems, tissue_labels


def load_mesh_from_jmsh(filepath):
    """Load mesh from JSON mesh format (.jmsh or .bmsh) - pure Python."""
    import json
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract mesh data from JSON structure
        if 'MeshVertex3' in data:
            nodes = np.array(data['MeshVertex3'], dtype=np.float64)
        else:
            raise ValueError("JMSH file missing 'MeshVertex3' field")
        
        # Try to get tetrahedral elements
        if 'MeshTet4' in data:
            elems = np.array(data['MeshTet4'], dtype=np.int32)
        elif 'MeshTetrahedron' in data:
            elems = np.array(data['MeshTetrahedron'], dtype=np.int32)
        else:
            raise ValueError("JMSH file missing tetrahedral element data")
        
        # Get tissue labels
        if 'MeshTissue' in data:
            tissue_labels = np.array(data['MeshTissue'], dtype=np.int32).flatten()
        elif 'TissueLabel' in data:
            tissue_labels = np.array(data['TissueLabel'], dtype=np.int32).flatten()
        else:
            tissue_labels = np.ones(len(elems), dtype=np.int32)
        
        return nodes, elems, tissue_labels
        
    except json.JSONDecodeError:
        # Try binary format (BMSH)
        raise NotImplementedError("Binary BMSH format requires additional parsing. Please use .mat or text .jmsh format.")


def verify_five_layers(tissue_labels):
    """Verify mesh has 5 tissue layers."""
    unique_tissues = np.unique(tissue_labels[tissue_labels > 0])
    
    if len(unique_tissues) != 5:
        print(f"Warning: Expected 5 tissue layers, found {len(unique_tissues)}: {unique_tissues}")
        return False
    
    return True


def create_blender_mesh_from_surface(nodes, faces, name):
    """Create a Blender mesh object from nodes and faces."""
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    
    bpy.context.collection.objects.link(obj)
    
    # Create mesh from nodes and faces
    mesh.from_pydata(nodes.tolist(), [], faces.tolist())
    mesh.update()
    
    return obj


def extract_surface_from_volume(nodes, elems, tissue_labels):
    """Extract outer surface from volumetric mesh."""
    # Find boundary faces (faces that belong to only one tetrahedron)
    face_dict = {}
    
    for elem_idx, elem in enumerate(elems[:, :4]):
        # Each tetrahedron has 4 faces
        faces = [
            tuple(sorted([elem[0], elem[1], elem[2]])),
            tuple(sorted([elem[0], elem[1], elem[3]])),
            tuple(sorted([elem[0], elem[2], elem[3]])),
            tuple(sorted([elem[1], elem[2], elem[3]])),
        ]
        
        for face in faces:
            if face in face_dict:
                face_dict[face] += 1
            else:
                face_dict[face] = 1
    
    # Boundary faces appear only once
    boundary_faces = [face for face, count in face_dict.items() if count == 1]
    
    # Convert to 0-indexed if needed
    if len(boundary_faces) > 0:
        min_idx = min(min(face) for face in boundary_faces)
        if min_idx == 1:
            boundary_faces = [[v - 1 for v in face] for face in boundary_faces]
    
    return boundary_faces


def create_layer_materials(obj, tissue_labels):
    """Create materials for visualizing different tissue layers."""
    # Define colors for each tissue type
    tissue_colors = {
        1: (1.0, 0.8, 0.6, 1.0),  # Scalp - peachy
        2: (0.9, 0.9, 0.8, 1.0),  # Skull - bone white
        3: (0.3, 0.5, 0.8, 0.5),  # CSF - light blue, transparent
        4: (0.8, 0.7, 0.7, 1.0),  # Gray matter - pinkish gray
        5: (0.9, 0.9, 0.9, 1.0),  # White matter - white
    }
    
    tissue_names = {
        1: "Scalp",
        2: "Skull", 
        3: "CSF",
        4: "Gray Matter",
        5: "White Matter"
    }
    
    # Create materials for each tissue type
    for tissue_id, color in tissue_colors.items():
        mat_name = f"Tissue_{tissue_names.get(tissue_id, tissue_id)}"
        mat = bpy.data.materials.get(mat_name)
        
        if not mat:
            mat = bpy.data.materials.new(mat_name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            nodes.clear()
            
            # Create material nodes
            output = nodes.new('ShaderNodeOutputMaterial')
            bsdf = nodes.new('ShaderNodeBsdfPrincipled')
            
            bsdf.inputs['Base Color'].default_value = color
            if len(color) > 3 and color[3] < 1.0:  # If transparent
                bsdf.inputs['Alpha'].default_value = color[3]
                mat.blend_method = 'BLEND'
            
            mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        # Add material to object if not already present
        if mat.name not in [m.name for m in obj.data.materials]:
            obj.data.materials.append(mat)


def import_five_layer_mesh(filepath, context):
    """
    Import a 5-layer head mesh and create visualization in Blender.
    Returns True if successful.
    """
    print("\n" + "=" * 70)
    print("IMPORTING 5-LAYER HEAD MESH")
    print("=" * 70)
    
    try:
        # Load mesh data
        nodes, elems, tissue_labels = load_mesh_from_file(filepath)
        print(f"  Loaded: {len(nodes):,} nodes, {len(elems):,} elements")
        
        # Verify 5 layers
        unique_tissues = np.unique(tissue_labels[tissue_labels > 0])
        print(f"  Tissue layers found: {len(unique_tissues)} - {unique_tissues}")
        
        if not verify_five_layers(tissue_labels):
            print(f"  WARNING: Expected 5 tissue layers, continuing anyway...")
        
        # Extract outer surface for visualization
        print("  Extracting surface...")
        surface_faces = extract_surface_from_volume(nodes, elems, tissue_labels)
        print(f"  Surface: {len(surface_faces):,} faces")
        
        # Create Blender mesh object
        print("  Creating Blender object...")
        obj = create_blender_mesh_from_surface(nodes, surface_faces, "FiveLayerHead")
        
        # Store volumetric mesh data in object custom properties
        # Convert to lists for Blender storage
        obj['volumetric_nodes'] = nodes.tolist()
        obj['volumetric_elements'] = elems.tolist()
        obj['tissue_labels'] = tissue_labels.tolist()
        obj['mesh_filepath'] = filepath
        
        # Orient mesh — always zero rotation.
        # MAT/iso2mesh meshes are in RAS coordinates (same Z-up as Blender),
        # so no rotation is required. Scale is matched from headmesh if present.
        reference_obj = bpy.data.objects.get('headmesh')
        if reference_obj:
            print("  Matching scale to 'headmesh' (zero rotation)...")
            obj.location = reference_obj.location
            obj.scale = reference_obj.scale
        else:
            print("  Centering at origin (zero rotation)...")
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')
        obj.rotation_euler = (0.0, 0.0, 0.0)
        
        # Create layer visualization (optional)
        create_layer_materials(obj, tissue_labels)
        
        print("  ✓ 5-layer mesh imported successfully")
        print("=" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error importing mesh: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# HELPER FUNCTIONS (shared between solvers)
# =============================================================================
def get_blender_vertices(obj):
    """Get world-space vertices from Blender object."""
    return np.array([(obj.matrix_world @ v.co).to_tuple() for v in obj.data.vertices])


def get_surface_normal_inward(pos, mesh_obj):
    """Get inward-pointing normal at surface position."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(mesh_obj, depsgraph)
    location, normal, _, _ = bvh.find_nearest(Vector(pos))
    if location is None:
        verts = get_blender_vertices(mesh_obj)
        direction = verts.mean(axis=0) - pos
        return direction / np.linalg.norm(direction)
    normal_in = -np.array(normal)
    return normal_in / np.linalg.norm(normal_in)


def find_element_containing_point(point, nodes, elems, tree=None, centroids=None):
    """Find tetrahedral element containing a point using barycentric coordinates."""
    if tree is None:
        centroids = nodes[elems[:, :4] - 1].mean(axis=1)
        tree = cKDTree(centroids)
    _, candidates = tree.query(point, k=200)
    for ei in candidates:
        v0, v1, v2, v3 = nodes[elems[ei, :4] - 1]
        T = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
        if abs(np.linalg.det(T)) < 1e-12:
            continue
        bary = np.linalg.solve(T, point - v0)
        if np.all(bary >= -1e-6) and np.sum(bary) <= 1 + 1e-6:
            return ei, tree, centroids
    return -1, tree, centroids


def project_into_mesh(surface_pos, normal_in, nodes, elems, tree=None, centroids=None):
    """Project surface position into mesh along normal direction."""
    if tree is None:
        centroids = nodes[elems[:, :4] - 1].mean(axis=1)
        tree = cKDTree(centroids)
    
    # Try incrementally deeper positions
    for depth in np.arange(MIN_DEPTH, 25.0, 0.5):
        test_pos = surface_pos + normal_in * depth
        elem_idx, tree, centroids = find_element_containing_point(test_pos, nodes, elems, tree, centroids)
        if elem_idx >= 0:
            return test_pos, elem_idx, depth, tree, centroids
    
    # Fallback: move toward nearest element centroid
    _, nearest_elem = tree.query(surface_pos)
    direction = centroids[nearest_elem] - surface_pos
    for frac in [0.3, 0.5, 0.7]:
        test_pos = surface_pos + direction * frac
        elem_idx, tree, centroids = find_element_containing_point(test_pos, nodes, elems, tree, centroids)
        if elem_idx >= 0:
            return test_pos, elem_idx, np.linalg.norm(direction) * frac, tree, centroids
    
    return None, -1, 0, tree, centroids


def smooth_on_mesh(values, nodes, n_iterations=3, k_neighbors=10):
    """Smooth values on mesh by averaging with nearest neighbors."""
    tree = cKDTree(nodes)
    smoothed = values.copy()
    
    for iteration in range(n_iterations):
        new_values = np.zeros_like(smoothed)
        distances, indices = tree.query(nodes, k=k_neighbors)
        
        for i in range(len(nodes)):
            neighbor_vals = smoothed[indices[i]]
            weights = 1.0 / (distances[i] + 0.1)
            weights /= weights.sum()
            new_values[i] = np.sum(neighbor_vals * weights)
        
        smoothed = new_values
    
    return smoothed


# =============================================================================
# MMC-SPECIFIC FUNCTIONS
# =============================================================================
def run_single_mmc(cfg_dict):
    """Run a single MMC simulation with fresh config."""
    gc.collect()

    use_gpu = cfg_dict.get('use_gpu', True)
    gpuid   = cfg_dict.get('gpuid', '01')

    cfg = {
        'nphoton':       int(cfg_dict['nphoton']),
        'tstart':        0,
        'tend':          5e-9,
        'tstep':         5e-9,
        'srctype':       'isotropic',
        'node':          cfg_dict['node'],
        'elem':          cfg_dict['elem'],
        'elemprop':      cfg_dict['elemprop'],
        'prop':          cfg_dict['prop'],
        'isreflect':     1,
        'issavedet':     0,
        'autopilot':     1 if use_gpu else 0,
        'seed':          int(cfg_dict['seed']),
        'srcpos':        list(cfg_dict['srcpos']),
        'srcdir':        list(cfg_dict['srcdir']) + [0.0] if len(cfg_dict['srcdir']) == 3 else list(cfg_dict['srcdir']),
        'e0':            int(cfg_dict['e0']),
        'minenergy':     1e-4,
        'maxjumpdebug':  1000000,
    }
    # Only add gpuid when running on GPU; omitting it lets pmmc default to CPU
    if use_gpu and gpuid:
        cfg['gpuid'] = gpuid

    result = pmmc.run(cfg)
    del cfg
    gc.collect()

    return result


def sample_flux_at_nodes(flux_vol, nodes):
    """Sample volumetric flux at mesh node positions using interpolation."""
    flux = flux_vol.squeeze()
    if flux.ndim < 3:
        return np.zeros(len(nodes))
    
    nx, ny, nz = flux.shape[:3]
    node_min, node_max = nodes.min(axis=0), nodes.max(axis=0)
    padding = (node_max - node_min) * 0.05
    grid_min = node_min - padding
    grid_size = (node_max + padding) - grid_min
    
    normalized = (nodes - grid_min) / grid_size
    coords = normalized * np.array([nx-1, ny-1, nz-1])
    coords = np.clip(coords, 0, [nx-1, ny-1, nz-1])
    
    node_flux = map_coordinates(flux, coords.T, order=1, mode='nearest')
    
    return node_flux


def run_sensitivity_mmc(data, settings):
    """MMC-based sensitivity analysis using settings from Blender UI."""
    print("\n" + "=" * 70)
    print("RUNNING MMC SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    nodes = data['nodes']
    ns, nd, nn = len(data['sources']), len(data['detectors']), len(nodes)
    
    nphoton     = settings.get('nphoton', 1000)
    sd_max_dist = settings.get('sd_max_distance', 60.0)
    use_gpu     = settings.get('use_gpu', False)
    gpuid       = settings.get('gpuid', settings.get('gpu_id', None))

    print(f"  Sources: {ns}, Detectors: {nd}, Nodes: {nn:,}")
    print(f"  Photons: {nphoton:,}, GPU: {'yes (id=' + str(gpuid) + ')' if use_gpu else 'no (CPU)'}")
    
    # Compute SD mask
    sd_mask = np.zeros(ns * nd, dtype=bool)
    idx = 0
    for si in range(ns):
        for di in range(nd):
            dist = np.linalg.norm(data['sources'][si] - data['detectors'][di])
            sd_mask[idx] = 10.0 <= dist <= sd_max_dist
            idx += 1
    print(f"  Valid channels: {sd_mask.sum()}/{ns*nd}")
    
    # Prepare base data
    node_list = nodes.tolist()
    elem_list = data['elems_for_mmc'].tolist()
    elemprop_list = data['tissue_labels'].tolist()
    
    phi_fwd = np.zeros((nn, ns))
    phi_adj = np.zeros((nn, nd))
    
    # Forward simulations
    print("\n  Forward simulations...")
    for si in range(ns):
        src = data['sources'][si]
        sdir = data['source_dirs'][si]
        e0 = int(data['source_elem_idx'][si] + 1)
        
        print(f"    S{si+1}/{ns} e0={e0}...", end='', flush=True)
        
        try:
            t0 = time.time()
            res = run_single_mmc({
                'nphoton':  nphoton,
                'node':     node_list,
                'elem':     elem_list,
                'elemprop': elemprop_list,
                'prop':     data['properties'],
                'seed':     si * 1000 + 42,
                'srcpos':   src.tolist(),
                'srcdir':   sdir.tolist(),
                'e0':       e0,
                'use_gpu':  use_gpu,
                'gpuid':    gpuid,
            })
            elapsed = time.time() - t0
            
            if 'flux' in res:
                phi_fwd[:, si] = sample_flux_at_nodes(np.array(res['flux']), nodes)
                print(f" ✓ max={phi_fwd[:, si].max():.2e} ({elapsed:.1f}s)")
            else:
                print(f" ✗ no flux ({elapsed:.1f}s)")
            
            del res
            time.sleep(0.5)
            
        except Exception as e:
            print(f" ✗ {str(e)[:30]}")
    
    # Adjoint simulations
    print("\n  Adjoint simulations...")
    for di in range(nd):
        det = data['detectors'][di]
        ddir = data['detector_dirs'][di]
        e0 = int(data['detector_elem_idx'][di] + 1)
        
        print(f"    D{di+1}/{nd} e0={e0}...", end='', flush=True)
        
        try:
            t0 = time.time()
            res = run_single_mmc({
                'nphoton':  nphoton,
                'node':     node_list,
                'elem':     elem_list,
                'elemprop': elemprop_list,
                'prop':     data['properties'],
                'seed':     (ns + di) * 1000 + 42,
                'srcpos':   det.tolist(),
                'srcdir':   ddir.tolist(),
                'e0':       e0,
                'use_gpu':  use_gpu,
                'gpuid':    gpuid,
            })
            elapsed = time.time() - t0
            
            if 'flux' in res:
                phi_adj[:, di] = sample_flux_at_nodes(np.array(res['flux']), nodes)
                print(f" ✓ max={phi_adj[:, di].max():.2e} ({elapsed:.1f}s)")
            else:
                print(f" ✗ no flux ({elapsed:.1f}s)")
            
            del res
            time.sleep(0.5)
            
        except Exception as e:
            print(f" ✗ {str(e)[:30]}")
    
    # Check results
    fwd_ok = np.sum(phi_fwd.sum(axis=0) > 0)
    adj_ok = np.sum(phi_adj.sum(axis=0) > 0)
    print(f"\n  Valid: {fwd_ok}/{ns} forward, {adj_ok}/{nd} adjoint")
    
    if fwd_ok == 0 or adj_ok == 0:
        print("  ✗ Insufficient data for sensitivity")
        return None
    
    # Compute sensitivity
    print("\n  Computing sensitivity...")
    sensitivity = np.zeros((nn, ns * nd))
    idx = 0
    for si in range(ns):
        for di in range(nd):
            sensitivity[:, idx] = phi_fwd[:, si] * phi_adj[:, di]
            idx += 1
    
    sel = np.where(sd_mask)[0]
    sens_map = sensitivity[:, sel].sum(axis=1)
    sens_log = np.log10(sens_map + 1e-20)
    
    valid = sens_map > 0
    if valid.any():
        print(f"  Sens range: {sens_map[valid].min():.2e} to {sens_map.max():.2e}")
        print(f"  Active nodes: {valid.sum():,} / {len(sens_map):,}")

    return {'sens_map': sens_map, 'sens_log': sens_log, 'phi_fwd': phi_fwd, 'phi_adj': phi_adj}


# =============================================================================
# REDBIRD-SPECIFIC FUNCTIONS
# =============================================================================
def prepare_redbird_mesh(nodes, elems, tissue_labels):
    """Prepare mesh for Redbird (ensure 1-indexing)."""
    face = np.array([])  # Redbird can work without explicit faces
    
    # Ensure 1-indexed for Redbird
    elems_1indexed = elems.copy()
    if elems_1indexed.min() == 0:
        elems_1indexed += 1
    
    return nodes, face, elems_1indexed[:, :4], tissue_labels


def build_redbird_prop_array(tissue_labels):
    """Build Redbird property array: [mua, mus, g, n] for each tissue type."""
    max_label = max(tissue_labels.max(), 5)
    props = [[0, 0, 1, 1]]  # Background
    
    for label in range(1, max_label + 1):
        p = OPTICAL_PROPERTIES.get(label, {'mua': 0.01, 'mus': 1.0, 'g': 0.9, 'n': 1.37})
        props.append([p['mua'], p['mus'], p['g'], p['n']])
    
    return np.array(props)


def run_redbird_forward(nodes, face, elem, seg, srcpos, srcdir, detpos, detdir, prop, omega):
    """Run a single Redbird forward simulation."""
    cfg = {
        "node": nodes,
        "face": face,
        "elem": elem,
        "seg": seg,
        "srcpos": srcpos.reshape(1, -1) if srcpos.ndim == 1 else srcpos,
        "detpos": detpos.reshape(1, -1) if detpos.ndim == 1 else detpos,
        "srcdir": srcdir,
        "detdir": detdir,
        "prop": prop,
        "omega": omega,
    }
    
    # Prepare mesh and solve
    cfg, sd = rb.meshprep(cfg)
    detphi, phi = rb.run(cfg)
    
    return phi, detphi


def run_sensitivity_redbird(data, settings):
    """Redbird-based sensitivity analysis using settings from Blender UI."""
    print("\n" + "=" * 70)
    print("RUNNING REDBIRD SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    nodes = data['nodes']
    ns, nd, nn = len(data['sources']), len(data['detectors']), len(nodes)
    
    # Get settings from UI
    mode = settings.get('mode', 'CW')
    frequency_mhz = settings.get('frequency', 70.0)
    sd_max_dist = settings.get('sd_max_distance', 60.0)
    
    # Calculate omega
    omega = 0 if mode == 'CW' else 2 * np.pi * frequency_mhz * 1e6
    
    print(f"  Sources: {ns}, Detectors: {nd}, Nodes: {nn:,}")
    print(f"  Mode: {mode}, Omega: {omega:.2e} rad/s")
    
    # Prepare mesh for Redbird
    rb_nodes, rb_face, rb_elem, rb_seg = prepare_redbird_mesh(
        nodes, data['elems_for_mmc'], data['tissue_labels']
    )
    
    prop = build_redbird_prop_array(data['tissue_labels'])
    
    # Compute SD mask
    sd_mask = np.zeros(ns * nd, dtype=bool)
    idx = 0
    for si in range(ns):
        for di in range(nd):
            dist = np.linalg.norm(data['sources'][si] - data['detectors'][di])
            sd_mask[idx] = 10.0 <= dist <= sd_max_dist
            idx += 1
    print(f"  Valid channels: {sd_mask.sum()}/{ns*nd}")
    
    phi_fwd = np.zeros((nn, ns), dtype=complex)
    phi_adj = np.zeros((nn, nd), dtype=complex)
    
    # Create dummy detector for simulations
    dummy_det = data['sources'][0] + np.array([0, 0, 100])
    
    # Forward simulations (sources)
    print("\n  Forward simulations...")
    for si in range(ns):
        print(f"    S{si+1}/{ns}...", end='', flush=True)
        
        try:
            t0 = time.time()
            phi, detphi = run_redbird_forward(
                nodes=rb_nodes,
                face=rb_face,
                elem=rb_elem,
                seg=rb_seg,
                srcpos=data['sources'][si],
                srcdir=data['source_dirs'][si],
                detpos=dummy_det,
                detdir=[0, 0, -1],
                prop=prop,
                omega=omega
            )
            elapsed = time.time() - t0
            
            phi_fwd[:, si] = phi.flatten()
            max_val = np.abs(phi_fwd[:, si]).max()
            print(f" ✓ max={max_val:.2e} ({elapsed:.1f}s)")
            
        except Exception as e:
            print(f" ✗ {str(e)[:50]}")
    
    # Adjoint simulations (detectors as sources)
    print("\n  Adjoint simulations...")
    for di in range(nd):
        print(f"    D{di+1}/{nd}...", end='', flush=True)
        
        try:
            t0 = time.time()
            phi, detphi = run_redbird_forward(
                nodes=rb_nodes,
                face=rb_face,
                elem=rb_elem,
                seg=rb_seg,
                srcpos=data['detectors'][di],
                srcdir=data['detector_dirs'][di],
                detpos=dummy_det,
                detdir=[0, 0, -1],
                prop=prop,
                omega=omega
            )
            elapsed = time.time() - t0
            
            phi_adj[:, di] = phi.flatten()
            max_val = np.abs(phi_adj[:, di]).max()
            print(f" ✓ max={max_val:.2e} ({elapsed:.1f}s)")
            
        except Exception as e:
            print(f" ✗ {str(e)[:50]}")
    
    # Check results
    fwd_ok = np.sum(np.abs(phi_fwd).sum(axis=0) > 0)
    adj_ok = np.sum(np.abs(phi_adj).sum(axis=0) > 0)
    print(f"\n  Valid: {fwd_ok}/{ns} forward, {adj_ok}/{nd} adjoint")
    
    if fwd_ok == 0 or adj_ok == 0:
        print("  ✗ Insufficient data for sensitivity")
        return None
    
    # Compute sensitivity (use absolute value for complex fluence)
    print("\n  Computing sensitivity...")
    sensitivity = np.zeros((nn, ns * nd))
    idx = 0
    for si in range(ns):
        for di in range(nd):
            sensitivity[:, idx] = np.abs(phi_fwd[:, si]) * np.abs(phi_adj[:, di])
            idx += 1
    
    sel = np.where(sd_mask)[0]
    sens_map = sensitivity[:, sel].sum(axis=1)
    sens_log = np.log10(sens_map + 1e-20)
    
    valid = sens_map > 0
    if valid.any():
        print(f"  Sens range: {sens_map[valid].min():.2e} to {sens_map.max():.2e}")
        print(f"  Active nodes: {valid.sum():,} / {len(sens_map):,}")

    return {'sens_map': sens_map, 'sens_log': sens_log, 'phi_fwd': phi_fwd, 'phi_adj': phi_adj}


# =============================================================================
# MAIN WORKFLOW FUNCTIONS
# =============================================================================
def load_mesh_and_register_optodes(mesh_path=None, optical_properties=None):
    """
    Load mesh and register optode positions from Blender.

    Priority order for mesh data:
      1. layered_mesh_manager (lmm) – populated by the 'From File (.mat)' import button
      2. FiveLayerHead Blender object with stored volumetric data (legacy)
      3. mesh_path file on disk

    Args:
        mesh_path: Path to .mat/.jmsh file (used only if lmm and FiveLayerHead are unavailable)
        optical_properties: Optional dict {layer: {mua, mus, g, n}} to override defaults
    """
    from . import layered_mesh_manager as lmm_module

    print("=" * 70)
    print("LOADING MESH & REGISTERING OPTODES")
    print("=" * 70)

    ref_obj = None

    # Priority 1: layered mesh manager (lmm) — used by the UI 'From File (.mat)' button
    if lmm_module.is_mesh_loaded():
        print("  Using mesh from layered mesh manager (lmm)")
        nodes        = lmm_module.LAYERED_MESH.nodes.copy()
        elems        = lmm_module.LAYERED_MESH.elems.copy()
        tissue_labels = lmm_module.LAYERED_MESH.tissue_labels.copy()
        ref_obj = lmm_module.LAYERED_MESH.head_surface_obj
        if ref_obj is None:
            ref_obj = (bpy.data.objects.get('Head_Surface_5L') or
                       bpy.data.objects.get('headmesh'))

    else:
        # Priority 2: FiveLayerHead in scene (legacy workflow)
        five_layer_obj = bpy.data.objects.get('FiveLayerHead')

        if five_layer_obj and 'volumetric_nodes' in five_layer_obj:
            print("  Using FiveLayerHead from Blender scene")
            nodes         = np.array(five_layer_obj['volumetric_nodes'],   dtype=np.float64)
            elems         = np.array(five_layer_obj['volumetric_elements'], dtype=np.int32)
            tissue_labels = np.array(five_layer_obj['tissue_labels'],       dtype=np.int32)
            mesh_path = five_layer_obj.get('mesh_filepath', mesh_path)
            ref_obj = five_layer_obj

        elif mesh_path:
            print(f"  Loading mesh from file: {mesh_path}")
            nodes, elems, tissue_labels = load_mesh_from_file(mesh_path)
            ref_obj = bpy.data.objects.get('headmesh')

        else:
            raise RuntimeError("No mesh available. Import a 5-layer mesh using 'From File (.mat)'.")

    print(f"  Mesh: {len(nodes):,} nodes, {len(elems):,} elements")

    # Ensure tissue labels are valid
    tissue_labels[tissue_labels < 0] = 0

    # Get reference surface for translation
    if ref_obj is None:
        raise RuntimeError(
            "No reference surface found. Need 'Head_Surface_5L', 'FiveLayerHead', or 'headmesh' object."
        )
    
    bl_verts = get_blender_vertices(ref_obj)

    # Use the scalp surface centroid stored at import time.
    # nodes.mean() is the centroid of ALL volume nodes — it sits deep inside
    # the brain and causes optodes to be projected inward, reversing the
    # sensitivity gradient (deep sulci appear more sensitive than surface).
    # mesh_centroid is the mean of scalp surface vertices only, matching how
    # Blender centred Head_Surface_5L.
    if lmm_module.is_mesh_loaded() and lmm_module.LAYERED_MESH.mesh_centroid is not None:
        translation = lmm_module.LAYERED_MESH.mesh_centroid - bl_verts.mean(axis=0)
        print(f"  Translation (scalp centroid): [{translation[0]:.1f}, {translation[1]:.1f}, {translation[2]:.1f}]")
    else:
        translation = nodes.mean(axis=0) - bl_verts.mean(axis=0)
        print(f"  Translation (volume mean fallback): [{translation[0]:.1f}, {translation[1]:.1f}, {translation[2]:.1f}]")

    # Use ref object for surface projection
    headmesh = ref_obj
    mesh_center = nodes.mean(axis=0)   # interior reference for normal direction checks
    
    # Precompute search structures
    tree, centroids = None, None
    
    # Register sources
    print("\n  Registering sources...")
    sources, source_dirs, source_elem_idx = [], [], []
    
    if 'Sources' not in bpy.data.collections:
        print("    WARNING: 'Sources' collection not found")
    else:
        for obj in bpy.data.collections['Sources'].objects:
            bl_pos = np.array(obj.matrix_world.translation)
            bl_normal = get_surface_normal_inward(bl_pos, headmesh)
            mat_pos = bl_pos + translation
            
            inside_pos, elem_idx, depth, tree, centroids = project_into_mesh(
                mat_pos, bl_normal, nodes, elems, tree, centroids
            )
            
            if inside_pos is not None and elem_idx >= 0:
                direction = mesh_center - inside_pos
                direction = direction / np.linalg.norm(direction)
                
                sources.append(inside_pos)
                source_dirs.append(direction)
                source_elem_idx.append(elem_idx)
                print(f"    {obj.name}: elem={elem_idx}, depth={depth:.1f}mm ✓")
            else:
                print(f"    {obj.name}: FAILED ✗")
    
    # Register detectors
    print("\n  Registering detectors...")
    detectors, detector_dirs, detector_elem_idx = [], [], []
    
    if 'Detectors' not in bpy.data.collections:
        print("    WARNING: 'Detectors' collection not found")
    else:
        for obj in bpy.data.collections['Detectors'].objects:
            bl_pos = np.array(obj.matrix_world.translation)
            bl_normal = get_surface_normal_inward(bl_pos, headmesh)
            mat_pos = bl_pos + translation
            
            inside_pos, elem_idx, depth, tree, centroids = project_into_mesh(
                mat_pos, bl_normal, nodes, elems, tree, centroids
            )
            
            if inside_pos is not None and elem_idx >= 0:
                direction = mesh_center - inside_pos
                direction = direction / np.linalg.norm(direction)
                
                detectors.append(inside_pos)
                detector_dirs.append(direction)
                detector_elem_idx.append(elem_idx)
                print(f"    {obj.name}: elem={elem_idx}, depth={depth:.1f}mm ✓")
            else:
                print(f"    {obj.name}: FAILED ✗")
    
    print(f"\n  Registered: {len(sources)} sources, {len(detectors)} detectors")
    
    if len(sources) == 0 or len(detectors) == 0:
        raise RuntimeError("Need at least one source and one detector")
    
    # Build optical properties (use caller-supplied override if given)
    opt_props = optical_properties if optical_properties else OPTICAL_PROPERTIES
    max_label = max(tissue_labels.max(), 5)
    props = [[0, 0, 1, 1]]
    for label in range(1, max_label + 1):
        p = opt_props.get(label, OPTICAL_PROPERTIES.get(label, {'mua': 0.01, 'mus': 1.0, 'g': 0.9, 'n': 1.37}))
        props.append([p['mua'], p['mus'], p['g'], p['n']])
    
    # Prepare element array
    elems_for_mmc = elems[:, :4].copy()
    if elems_for_mmc.min() == 0:
        elems_for_mmc += 1
    
    return {
        'nodes': nodes,
        'elems_for_mmc': elems_for_mmc,
        'tissue_labels': tissue_labels,
        'sources': np.array(sources),
        'source_dirs': np.array(source_dirs),
        'source_elem_idx': np.array(source_elem_idx, dtype=np.int32),
        'detectors': np.array(detectors),
        'detector_dirs': np.array(detector_dirs),
        'detector_elem_idx': np.array(detector_elem_idx, dtype=np.int32),
        'properties': props,
        'translation': translation,
    }


def visualize_on_cortex(results, data, smooth_iterations=5):
    """Visualize sensitivity map on Brain_Cortex object."""
    print("\n" + "=" * 70)
    print("VISUALIZING ON CORTEX")
    print("=" * 70)
    
    cortex = (bpy.data.objects.get('Brain_Cortex') or
              bpy.data.objects.get('Brain_Cortex_5L'))
    if not cortex:
        print("  ✗ Brain_Cortex / Brain_Cortex_5L object not found")
        return
    
    bl_verts = get_blender_vertices(cortex)
    mat_verts = bl_verts + data['translation']

    FILL_RADIUS   = 20.0   # mm — cortex vertices within this radius of any
                           # sensitive node get an interpolated value

    # ── Identify sensitive nodes in the FULL volumetric mesh ─────────────────
    all_sens_log  = results['sens_log']          # (N_nodes,)
    all_nodes     = data['nodes']                # (N_nodes, 3)
    valid_mask    = all_sens_log > -19           # truly non-zero sensitivity
    valid_log     = all_sens_log[valid_mask]
    valid_pos     = all_nodes[valid_mask]

    if valid_mask.sum() == 0:
        print("  WARNING: No valid sensitivity values in mesh.")
        return

    # Distribution diagnostics
    pcts     = [50, 75, 90, 95, 99, 99.9]
    pct_vals = [float(np.percentile(valid_log, p)) for p in pcts]
    print(f"  Valid mesh nodes (log>-19): {valid_mask.sum():,}  abs_max={valid_log.max():.2f}")
    print(f"  Distribution: " + "  ".join(f"p{p}={v:.1f}" for p, v in zip(pcts, pct_vals)))

    vmax = valid_log.max()
    vmin = valid_log.min()
    print(f"  Display window: log10 [{vmin:.1f}, {vmax:.1f}]  (true min/max)")

    # ── Distance-weighted interpolation onto cortex surface ───────────────────
    # For each cortex vertex, find the k nearest SENSITIVE mesh nodes within
    # FILL_RADIUS and compute an inverse-distance-weighted average log value.
    # This produces a smooth continuous heatmap — no speckle from zero-value
    # nearest-node lookups, no artificial background inflation.
    k_near = min(8, valid_mask.sum())
    valid_tree = cKDTree(valid_pos)
    dists, near_idx = valid_tree.query(mat_verts, k=k_near, workers=-1)

    if k_near == 1:
        dists   = dists[:, np.newaxis]
        near_idx = near_idx[:, np.newaxis]

    # Vertices with no sensitive node within FILL_RADIUS → background
    in_region = dists[:, 0] <= FILL_RADIUS

    weights = 1.0 / (dists + 0.5)          # inverse-distance weights
    weights[dists > FILL_RADIUS] = 0.0     # exclude out-of-radius nodes
    w_sum = weights.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum > 0, w_sum, 1.0)
    weights /= w_sum

    sens_log_interp = (weights * valid_log[near_idx]).sum(axis=1)
    sens_log_interp[~in_region] = -20.0

    print(f"  Cortex vertices:       {len(bl_verts):,}")
    print(f"  In region (≤{FILL_RADIUS:.0f}mm):   {in_region.sum():,}")
    print(f"  Background:            {(~in_region).sum():,}")

    if in_region.sum() == 0:
        print("  WARNING: No cortex vertices within fill radius — "
              "check translation / mesh alignment or increase FILL_RADIUS.")
        return

    # ── Normalise over true min/max window ───────────────────────────────────
    dynamic_range = vmax - vmin if vmax > vmin else 1.0
    norm = np.zeros(len(bl_verts))
    norm[in_region] = (
        np.clip(sens_log_interp[in_region], vmin, vmax) - vmin
    ) / dynamic_range

    n_vals = norm[in_region]
    print(f"  Norm distribution: min={n_vals.min():.3f}  max={n_vals.max():.3f}  "
          f"mean={n_vals.mean():.3f}  median={np.median(n_vals):.3f}")

    # Apply vertex colors — always delete and recreate to avoid stale data
    mesh = cortex.data
    if 'Sensitivity' in mesh.vertex_colors:
        mesh.vertex_colors.remove(mesh.vertex_colors['Sensitivity'])
    vc = mesh.vertex_colors.new(name='Sensitivity')

    def _colormap(v):
        """Blue→cyan→green→yellow→red, alpha=1 (sensitive node)."""
        if v < 0.25:
            t = v / 0.25;      return (0.0, t,       1.0,       1.0)
        elif v < 0.5:
            t = (v-0.25)/0.25; return (0.0, 1.0,     1.0 - t,   1.0)
        elif v < 0.75:
            t = (v-0.5)/0.25;  return (t,   1.0,     0.0,       1.0)
        else:
            t = (v-0.75)/0.25; return (1.0, 1.0 - t, 0.0,       1.0)

    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            if in_region[vi]:
                vc.data[li].color = _colormap(norm[vi])
            else:
                vc.data[li].color = _colormap(0.0)  # lowest heatmap colour — no black background

    mesh.vertex_colors.active = vc

    # Material: Mix Shader — alpha=0 → original brain gray, alpha=1 → heatmap
    mat = bpy.data.materials.get("SensMat") or bpy.data.materials.new("SensMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out  = nt.nodes.new('ShaderNodeOutputMaterial')
    mix  = nt.nodes.new('ShaderNodeMixShader')
    orig = nt.nodes.new('ShaderNodeBsdfPrincipled')   # brain gray (background)
    heat = nt.nodes.new('ShaderNodeBsdfPrincipled')   # heatmap color (sensitive)
    vcn  = nt.nodes.new('ShaderNodeVertexColor')
    vcn.layer_name = 'Sensitivity'
    orig.inputs['Base Color'].default_value = (0.70, 0.65, 0.62, 1.0)
    nt.links.new(vcn.outputs['Color'], heat.inputs['Base Color'])
    nt.links.new(vcn.outputs['Alpha'], mix.inputs['Fac'])   # 0=brain, 1=heatmap
    nt.links.new(orig.outputs['BSDF'],  mix.inputs[1])
    nt.links.new(heat.outputs['BSDF'],  mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    mesh.materials.clear()
    mesh.materials.append(mat)
    
    # Set viewport shading
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'SOLID'
                    space.shading.color_type = 'VERTEX'
    
    print("  ✓ Done")


def save_results(results, data, output_path, solver_type):
    """Save sensitivity results to npz file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path,
             sens_map=results['sens_map'],
             sens_log=results['sens_log'],
             phi_fwd=results['phi_fwd'],
             phi_adj=results['phi_adj'],
             sources=data['sources'],
             detectors=data['detectors'],
             nodes=data['nodes'],
             translation=data['translation'],
             solver=solver_type)
    print(f"  Saved: {output_path}")


# =============================================================================
# STANDALONE EXECUTION (for testing)
# =============================================================================
if __name__ == "__main__":
    # Example standalone execution
    BASE_DIR = '/drives/buzhou1/users/mccann.as/Projects/NeuroCaptain_V2'
    MESH_PATH = os.path.join(BASE_DIR, 'mesh1_7_14.mat')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'sensitivity_output')
    
    SOLVER = 'redbird'  # or 'mmc'
    
    print("\n" + "=" * 70)
    print(f"SENSITIVITY ANALYSIS - {SOLVER.upper()} SOLVER")
    print("=" * 70 + "\n")
    
    # Check solver availability
    if SOLVER == 'redbird' and not REDBIRD_AVAILABLE:
        print("✗ Redbird not available. Install with: pip install redbirdpy")
        sys.exit(1)
    elif SOLVER == 'mmc' and not MMC_AVAILABLE:
        print("✗ MMC (pmmc) not available")
        sys.exit(1)
    
    try:
        data = load_mesh_and_register_optodes(MESH_PATH)
        
        if SOLVER == 'redbird':
            settings = {'mode': 'CW', 'frequency': 70.0, 'sd_max_distance': 60.0}
            results = run_sensitivity_redbird(data, settings)
        else:
            settings = {'nphoton': 1000, 'gpu_id': '01', 'sd_max_distance': 60.0}
            results = run_sensitivity_mmc(data, settings)
        
        if results:
            output_path = os.path.join(OUTPUT_DIR, f'sensitivity_{SOLVER}.npz')
            save_results(results, data, output_path, SOLVER)
            visualize_on_cortex(results, data, smooth_iterations=5)
            
            print("\n" + "=" * 70)
            print(f"✓ COMPLETE ({SOLVER.upper()})")
            print("=" * 70)
        else:
            print("\n✗ FAILED")
            
    except Exception as e:
        import traceback
        print(f"\n✗ {e}")
        traceback.print_exc()