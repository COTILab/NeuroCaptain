"""
Redbird Simulation Runner for NeuroCaptain and  visualization of sensitivity on cortex
"""

import bpy
import numpy as np
from scipy.spatial import cKDTree
import time
import pickle 
import os 

try:
    import redbirdpy as rb
    from redbirdpy import forward
    REDBIRD_AVAILABLE = True
except ImportError:
    REDBIRD_AVAILABLE = False
    print("ERROR: redbirdpy not available")

try:
    import iso2mesh as i2m
    ISO2MESH_AVAILABLE = True
except ImportError:
    ISO2MESH_AVAILABLE = False
    print("ERROR: iso2mesh not available")

from . import layered_mesh_manager as lmm


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

_normals_fixed = set()  # track which objects have had normals fixed


def get_blender_vertices(obj):
    """World-space vertices from Blender object"""
    return np.array([(obj.matrix_world @ v.co).to_tuple() for v in obj.data.vertices])


def get_inward_normal_at_point(pos, mesh_obj):
    """
    Get inward-pointing normal at closest surface point for Redbird srcdir/detdir.
    Returns: inward normal (numpy array, unit vector)
    """
    import bmesh
    from mathutils import Vector

    # Ensure consistent outward normals once per object.
    # Use bmesh so we don't need bpy.ops / an active object in context.
    if mesh_obj.name not in _normals_fixed:
        bm = bmesh.new()
        bm.from_mesh(mesh_obj.data)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(mesh_obj.data)
        bm.free()
        if bpy.app.version < (4, 1, 0):
            mesh_obj.data.calc_normals()
        _normals_fixed.add(mesh_obj.name)
        print(f"  Normals recalculated (outward) for: {mesh_obj.name}")
    
    # closest point in local space
    local_pos = mesh_obj.matrix_world.inverted() @ Vector(pos)
    result, location, normal, face_index = mesh_obj.closest_point_on_mesh(local_pos)
    
    if not result:
        # fallback: point toward mesh center
        verts = get_blender_vertices(mesh_obj)
        direction = verts.mean(axis=0) - np.array(pos)
        return direction / np.linalg.norm(direction)
    
    # Transform normal to world space
    normal_mat = mesh_obj.matrix_world.to_3x3().inverted().transposed()
    world_normal = (normal_mat @ normal).normalized()
    
    # Negate outward → inward 
    inward = -np.array(world_normal)
    return inward / np.linalg.norm(inward)


def register_optodes(headmesh, translation, mesh_center):
    """
    Register all sources and detectors using head surface inward normals.
    Returns: sources, source_dirs, detectors, detector_dirs (all numpy arrays)
    """
    sources, source_dirs = [], []
    if 'Sources' in bpy.data.collections:
        for obj in bpy.data.collections['Sources'].objects:
            bl_pos = np.array(obj.matrix_world.translation)
            inward_normal = get_inward_normal_at_point(bl_pos, headmesh)
            mat_pos = bl_pos + translation
            
            sources.append(mat_pos)
            source_dirs.append(inward_normal)
    
    detectors, detector_dirs = [], []
    if 'Detectors' in bpy.data.collections:
        for obj in bpy.data.collections['Detectors'].objects:
            bl_pos = np.array(obj.matrix_world.translation)
            inward_normal = get_inward_normal_at_point(bl_pos, headmesh)
            mat_pos = bl_pos + translation
            
            detectors.append(mat_pos)
            detector_dirs.append(inward_normal)
    
    sources = np.array(sources, dtype=np.float64)
    source_dirs = np.array(source_dirs, dtype=np.float64)
    detectors = np.array(detectors, dtype=np.float64)
    detector_dirs = np.array(detector_dirs, dtype=np.float64)
      
    return sources, source_dirs, detectors, detector_dirs


def crop_mesh_around_optodes(nodes, elems, tissue_labels, all_optodes, crop_margin):
    """
    Crop volumetric mesh to axis-aligned bounding box around optodes + margin.

    Parameters
    ----------
    crop_margin : float
        mm to extend beyond the optode bounding box on all sides.
    """
    print(f"\nCropping mesh (margin={crop_margin}mm)...")
    print(f"  Original: {len(nodes):,} nodes, {len(elems):,} elements")

    elem_nodes_idx = elems[:, :4].copy()
    if elem_nodes_idx.min() == 1:
        elem_nodes_idx -= 1
    centroids = nodes[elem_nodes_idx].mean(axis=1)

    bb_min = all_optodes.min(axis=0) - crop_margin
    bb_max = all_optodes.max(axis=0) + crop_margin

    inside_mask = np.all((centroids >= bb_min) & (centroids <= bb_max), axis=1)
    print(f"  Bbox: x=[{bb_min[0]:.1f},{bb_max[0]:.1f}]  "
          f"y=[{bb_min[1]:.1f},{bb_max[1]:.1f}]  "
          f"z=[{bb_min[2]:.1f},{bb_max[2]:.1f}]")

    cropped_elems_raw = elems[inside_mask, :4].copy()
    cropped_seg       = tissue_labels[inside_mask].copy()

    pct = 100.0 * len(cropped_elems_raw) / max(len(elems), 1)
    print(f"  Elements in region: {len(cropped_elems_raw):,}  ({pct:.1f}% of full mesh)")

    print("  Removing isolated nodes...")
    try:
        result = i2m.removeisolatednode(nodes, cropped_elems_raw)

        if len(result) == 2:
            node_cropped, elem_cropped = result
        elif len(result) == 3:
            node_cropped, elem_cropped, _ = result
        else:
            node_cropped, elem_cropped = result[0], result[1]

        node_cropped = np.ascontiguousarray(node_cropped, dtype=np.float64)
        elem_cropped = np.ascontiguousarray(elem_cropped, dtype=np.int32)
        seg_cropped  = cropped_seg[:len(elem_cropped)]

        reduction = 100.0 * (1 - len(elem_cropped) / max(len(elems), 1))
        print(f"  Cropped: {len(node_cropped):,} nodes, {len(elem_cropped):,} elements  "
              f"({reduction:.1f}% reduction)")

        return node_cropped, elem_cropped, seg_cropped

    except Exception as e:
        print(f"  ERROR in removeisolatednode: {e}")
        return None, None, None


def smooth_on_mesh(values, nodes, n_iterations=3, k_neighbors=10):
    """Spatially smooth values on mesh nodes--> for visualizaton"""
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
# MAIN SIMULATION RUNNER
# =============================================================================

def run_redbird_simulation():
    """
    Run complete Redbird forward simulation and visualize sensitivity on cortex
    
    """
    
    if not REDBIRD_AVAILABLE or not ISO2MESH_AVAILABLE:
        return {
            'success': False,
            'message': 'Need redbirdpy and iso2mesh'
        }
    
    # Check if mesh is loaded
    if not lmm.is_mesh_loaded():
        return {
            'success': False,
            'message': 'No mesh loaded'
        }
    
    print("\n" + "=" * 70)
    print("REDBIRD SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Get mesh data
    mesh_data = lmm.LAYERED_MESH
    nodes = mesh_data.nodes.copy()
    elems = mesh_data.elems.copy()
    tissue_labels = mesh_data.tissue_labels.copy()
    
    full_nodes = nodes.copy()
    
    print(f"\nFull mesh: {len(nodes):,} nodes, {len(elems):,} elements")
    
    # Get reference head surface (look up by name — stored references go stale)
    headmesh = bpy.data.objects.get('Head_Surface_5L')
    if not headmesh:
        headmesh = bpy.data.objects.get('headmesh')
    print(f"  Reference surface: {headmesh.name if headmesh else 'NONE'}")
    
    if not headmesh:
        return {
            'success': False,
            'message': 'No head surface found'
        }

    # run_redbird_simulation()/visualize_on_cortex() hide this object
    # (hide_viewport=True) at the end of a previous successful run, purely
    # so the cortex heatmap displays cleanly - but a hidden object has no
    # evaluated depsgraph data, which get_inward_normal_at_point()'s
    # closest_point_on_mesh() call below needs. Resetting hide_viewport alone
    # isn't enough to fix a second run: closest_point_on_mesh() reads the
    # object via DEG_get_evaluated_object(), and nothing forces the
    # depsgraph to actually re-evaluate the now-visible object before that
    # call runs (unlike interactive use, where a viewport redraw does this
    # automatically) - so it can still report "no evaluated mesh data" even
    # with hide_viewport already False. view_layer.update() forces that
    # re-evaluation immediately.
    if headmesh.hide_viewport:
        headmesh.hide_viewport = False
    bpy.context.view_layer.update()

    bl_verts = get_blender_vertices(headmesh)

    # Head_Surface_5L is created with ALL volumetric nodes as vertices,
    # so Blender's ORIGIN_CENTER_OF_MASS sets the origin at nodes.mean()
    # (the center of ALL volume nodes).  The translation must match that
    # origin so Blender→mesh coordinate conversion is exact.
    translation = nodes.mean(axis=0) - bl_verts.mean(axis=0)
    print(f"  Translation (volume center): [{translation[0]:.1f}, {translation[1]:.1f}, {translation[2]:.1f}]")

    mesh_center = nodes.mean(axis=0)  # kept as brain interior reference for normal checks
    
   # Register optodes
    print("\nRegistering optodes...")
    config = lmm.REDBIRD_CONFIG
    
    sources, source_dirs, detectors, detector_dirs = register_optodes(
        headmesh, translation, mesh_center
    )
    
    ns, nd = len(sources), len(detectors)
    
    if ns == 0 or nd == 0:
        return {'success': False, 'message': 'Need sources and detectors'}
    
    ns, nd = len(sources), len(detectors)
    print(f"  Registered: {ns} sources, {nd} detectors")

    if ns == 0 or nd == 0:
        return {'success': False, 'message': 'Need sources and detectors'}

    # Diagnostic: verify sources are within mesh bounding box
    nmin, nmax = nodes.min(axis=0), nodes.max(axis=0)
    print(f"\n  Mesh bbox: x=[{nmin[0]:.1f},{nmax[0]:.1f}] y=[{nmin[1]:.1f},{nmax[1]:.1f}] z=[{nmin[2]:.1f},{nmax[2]:.1f}]")
    for i in range(min(3, ns)):
        s = sources[i]
        inside = np.all(s >= nmin) and np.all(s <= nmax)
        print(f"    S{i+1}: [{s[0]:.1f},{s[1]:.1f},{s[2]:.1f}] {'inside bbox ✓' if inside else 'OUTSIDE bbox ✗'}")
    for i in range(min(3, nd)):
        d = detectors[i]
        inside = np.all(d >= nmin) and np.all(d <= nmax)
        print(f"    D{i+1}: [{d[0]:.1f},{d[1]:.1f},{d[2]:.1f}] {'inside bbox ✓' if inside else 'OUTSIDE bbox ✗'}")

    # Diagnostic: verify normals point inward --> get NaN phi values otherwise
    print("\n  Direction sanity check:")
    for i in range(min(3, ns)):
        dot = np.dot(source_dirs[i], mesh_center - sources[i])
        print(f"    S{i+1}: normal dot (center-pos) = {dot:.2f} {'(inward ✓)' if dot > 0 else '(OUTWARD ✗)'}")
    for i in range(min(3, nd)):
        dot = np.dot(detector_dirs[i], mesh_center - detectors[i])
        print(f"    D{i+1}: normal dot (center-pos) = {dot:.2f} {'(inward ✓)' if dot > 0 else '(OUTWARD ✗)'}")

    
    all_optodes = np.vstack([sources, detectors])
    cropped_nodes, cropped_elems, cropped_seg = crop_mesh_around_optodes(
        nodes, elems, tissue_labels, all_optodes,
        crop_margin=config.crop_margin,
    )
    
    if cropped_nodes is None:
        return {
            'success': False,
            'message': 'Mesh cropping failed'
        }
    
   # Map optodes to cropped mesh
    print("\nMapping optodes to cropped mesh...")
    tree_crop = cKDTree(cropped_nodes)
    _, src_idx = tree_crop.query(sources)
    _, det_idx = tree_crop.query(detectors)
    
    sources_crop = cropped_nodes[src_idx]
    detectors_crop = cropped_nodes[det_idx]
    
    # Prepare for Redbird
    print("\n" + "=" * 70)
    print("PREPARING REDBIRD")
    print("=" * 70)
    
    rb_elem = cropped_elems[:, :4].copy()
    if rb_elem.min() == 0:
        rb_elem += 1

    rb_elem = np.ascontiguousarray(rb_elem, dtype=np.int32)
    
    rb_seg = np.ascontiguousarray(cropped_seg, dtype=np.int32)
    
    print("  Extracting faces...")
    try:
        t0 = time.time()
        face_result = i2m.volface(rb_elem)
        
        if isinstance(face_result, tuple):
            rb_face = face_result[0]
        else:
            rb_face = face_result
        
        if isinstance(rb_face, list):
            tri_faces = [f for f in rb_face if len(f) == 3]
            rb_face = np.array(tri_faces, dtype=np.int32)
        else:
            rb_face = np.ascontiguousarray(rb_face, dtype=np.int32)
        
        print(f"  Faces: {len(rb_face):,} ({time.time()-t0:.1f}s)")
        
    except Exception as e:
        print(f"  WARNING: {e}")
        rb_face = None
    
    # Optical properties
    max_label = max(cropped_seg.max(), 5)
    props = [[0, 0, 1, 1]]  # Background
    for label in range(1, max_label + 1):
        if label in config.optical_properties:
            p = config.optical_properties[label]
            props.append([p['mua'], p['mus'], p['g'], p['n']])
        else:
            props.append([0.01, 1.0, 0.9, 1.37])
    
    prop = np.ascontiguousarray(np.array(props), dtype=np.float64)
    
    # Omega for CW vs FD
    omega = 0 if config.mode == 'CW' else 2 * np.pi * config.frequency * 1e6
    
    # SD mask (only channels within distance range)
    sd_mask = np.zeros(ns * nd, dtype=bool)
    idx = 0
    for si in range(ns):
        for di in range(nd):
            dist = np.linalg.norm(sources[si] - detectors[di])
            sd_mask[idx] = 10.0 <= dist <= config.sd_max_distance
            idx += 1
    print(f"  Valid channels: {sd_mask.sum()}/{ns*nd}")
    
    # Run simulation: all sources then all detectors (adjoint) --> compute sensitivity map
    print("\n" + "=" * 70)
    print("RUNNING SENSITIVITY (ADJOINT METHOD)")
    print("=" * 70)
    
    nn_crop = len(cropped_nodes)
    
    print("\n" + "=" * 70)
    print("RUNNING FORWARD + JACOBIAN")
    print("=" * 70)
    
    nn_crop = len(cropped_nodes)
    
    # Build config
    cfg = {
        "node":   cropped_nodes,
        "elem":   rb_elem,
        "seg":    rb_seg,
        "srcpos": sources_crop,
        "detpos": detectors_crop,
        "srcdir": source_dirs.tolist(),
        "detdir": detector_dirs.tolist(),
        "prop":   prop,
        "omega":   omega,
        "maxiter": config.max_iter,
        "lambda":  config.regularization_lambda,
    }
    if rb_face is not None and len(rb_face) > 0:
        cfg["face"] = rb_face
    
    # Mesh prep (computes deldotdel, evol, etc.)
    t0 = time.time()
    cfg, sd = rb.meshprep(cfg)
    print(f"  Mesh prep: {time.time()-t0:.1f}s")

    # meshreorient (called inside meshprep) correctly fixes element winding
    # but returns the ORIGINAL signed volumes, not the corrected ones.
    # For meshes that needed reorientation (e.g. NeuroJSON), evol/nvol/deldotdel
    # are all computed from those stale negative volumes.  Delete them and let
    # the second meshprep pass recompute with unsigned elemvolume (always positive).
    if (cfg['evol'] < 0).any():
        n_neg = int((cfg['evol'] < 0).sum())
        print(f"  Fixing stale signed volumes: {n_neg:,}/{len(cfg['evol']):,} negative")
        for key in ['evol', 'nvol', 'deldotdel']:
            cfg.pop(key, None)
        cfg, sd = rb.meshprep(cfg)
        print(f"  Recomputed: evol min={cfg['evol'].min():.6f} max={cfg['evol'].max():.6f} neg={int((cfg['evol']<0).sum())}")
    
    # Diagnostic: check mesh data going into RedBird
    print(f"\n  RedBird cfg check:")
    print(f"    node: shape={cfg['node'].shape} dtype={cfg['node'].dtype} range=[{cfg['node'].min():.1f},{cfg['node'].max():.1f}]")
    print(f"    elem: shape={cfg['elem'].shape} dtype={cfg['elem'].dtype} range=[{cfg['elem'].min()},{cfg['elem'].max()}]")
    print(f"    seg:  shape={cfg['seg'].shape} dtype={cfg['seg'].dtype} unique={np.unique(cfg['seg']).tolist()}")
    print(f"    srcpos: shape={cfg['srcpos'].shape}")
    print(f"    evol: min={cfg['evol'].min():.6f} max={cfg['evol'].max():.6f} neg={int((cfg['evol']<0).sum())}")
    if 'face' in cfg:
        print(f"    face: shape={cfg['face'].shape} dtype={cfg['face'].dtype} range=[{cfg['face'].min()},{cfg['face'].max()}]")

    # Diagnostic: verify source/detector points actually land inside a
    # cropped-mesh tetrahedron. redbirdpy's femrhs() silently zeroes out an
    # optode's RHS column (rather than raising) if its inward-displaced
    # position (srcpos/detpos + srcdir/detdir * 1/mu_tr) falls outside every
    # element of cfg['elem'] - if that happens for all sources at once, the
    # forward solve returns exactly phi=0.0 everywhere with no other error.
    try:
        _, loc_diag, _, optode_diag = forward.femrhs(cfg, sd)
        n_missed = int(np.isnan(loc_diag).sum())
        print(f"    femrhs: {len(loc_diag) - n_missed}/{len(loc_diag)} optodes landed inside a mesh element ({n_missed} missed)")
        if n_missed > 0:
            missed_idx = np.where(np.isnan(loc_diag))[0]
            print(f"    missed optode positions: {optode_diag[missed_idx].tolist()}")
    except Exception as e:
        print(f"    femrhs diagnostic failed: {e}")

    # Forward solve — phi shape: (nn_crop, ns)
    print("\n1. Forward simulation...")
    t0 = time.time()
    detphi, phi = forward.runforward(cfg, sd=sd, verbose=True)
    print(f"  Forward: {time.time()-t0:.1f}s")
    print(f"    phi shape: {phi.shape}")
    print(f"    phi range: {np.abs(phi).min():.2e} to {np.abs(phi).max():.2e}")
    
    if np.abs(phi).max() == 0:
        print("  ERROR: Forward fluence is zero!")
        print(f"    prop:\n{cfg['prop'][:6]}")
        return {'success': False, 'message': 'Forward fluence is zero'}
    
    # Jacobian — Jmua shape: (ns*nd, nn_crop), handles adjoint internally
    print("\n2. Computing Jacobian (sensitivity)...")
    t0 = time.time()
    Jmua, _ = forward.jac(sd, phi, cfg["deldotdel"], cfg["elem"], cfg["evol"])
    print(f"  Jacobian: {time.time()-t0:.1f}s")
    print(f"    Jmua shape: {Jmua.shape}  (SD pairs x nodes)")
    
    # 3. Filter by SD distance
    print("\n3. Filtering by SD distance...")
    sd_mask = np.zeros(ns * nd, dtype=bool)
    idx = 0
    for si in range(ns):
        for di in range(nd):
            dist = np.linalg.norm(sources[si] - detectors[di])
            sd_mask[idx] = 10.0 <= dist <= 60.0
            idx += 1
    
    sel = np.where(sd_mask)[0]
    print(f"  Valid SD pairs: {len(sel)}/{ns*nd}")
    
    # 4. Collapse channels: sum absolute Jacobian across valid SD pairs
    # Jmua rows = SD pairs, cols = nodes → transpose to match MATLAB convention
    sens_map = np.abs(Jmua[sel, :]).sum(axis=0)  # shape: (nn_crop,)
    sens_log_crop = np.log10(sens_map + 1e-20)
    
    valid = sens_map > 0
    if valid.any():
        sens_log_crop[~valid] = sens_log_crop[valid].min()
        print(f"  Sensitivity range: {sens_map[valid].min():.2e} to {sens_map.max():.2e}")
    else:
        print("  ERROR: No valid sensitivity")
        return {'success': False, 'message': 'No valid sensitivity values computed'}

    # Smooth sensitivity on the volumetric mesh nodes before projecting to cortex
    if config.smooth_iterations > 0:
        print(f"\n  Smoothing sensitivity ({config.smooth_iterations} iterations)...")
        sens_log_crop = smooth_on_mesh(sens_log_crop, cropped_nodes,
                                       n_iterations=config.smooth_iterations, k_neighbors=10)

    # validation checks
    print("\n" + "=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)
    
    print("\nTop 5 sensitivity locations:")
    top_indices = np.argsort(sens_log_crop)[-5:][::-1]
    for i, idx in enumerate(top_indices):
        pos = cropped_nodes[idx]
        sens_val = sens_log_crop[idx]
        
        dist_to_srcs = np.linalg.norm(sources_crop - pos, axis=1)
        dist_to_dets = np.linalg.norm(detectors_crop - pos, axis=1)
        
        print(f"  {i+1}. log_sens={sens_val:.2f}, nearest_src={dist_to_srcs.min():.1f}mm, nearest_det={dist_to_dets.min():.1f}mm")
    
    print("\nSD distance vs total sensitivity:") #check if longer channels have lower sensitivity
    idx = 0
    for si in range(min(3, ns)):
        for di in range(min(3, nd)):
            if idx < len(sd_mask) and sd_mask[idx]:
                dist = np.linalg.norm(sources[si] - detectors[di])
                total_sens = Jmua[idx, :].sum()
                print(f"  S{si+1}-D{di+1}: dist={dist:.1f}mm, total_sens={total_sens:.2e}")
            idx += 1
    
    # Visualize on cortex
    print("\n" + "=" * 70)
    print("VISUALIZING ON BRAIN CORTEX")
    print("=" * 70)
    
    cortex = bpy.data.objects.get('Brain_Cortex_5L')
    
    if not cortex:
        return {
            'success': False,
            'message': 'Brain_Cortex_5L not found'
        }
    
    print(f"  Using: {cortex.name}")
    
    # get cortex vertices in world space (map to extracted cortex)
    bl_verts = get_blender_vertices(cortex)
    mat_verts = bl_verts + translation
    
    tree_crop = cKDTree(cropped_nodes)
    distances_crop, idx_crop = tree_crop.query(mat_verts)
    
    #  vertices in simulated region
    in_simulated_region = distances_crop < 5.0
    
    print(f"  Cortex vertices: {len(bl_verts):,}")
    print(f"  In simulated region: {in_simulated_region.sum():,}")
    print(f"  Outside region: {(~in_simulated_region).sum():,}")
    
    # initially set all to background
    norm = np.zeros(len(bl_verts))

    # sensitivity for vertices in simulated region
    sens_in_region = sens_log_crop[idx_crop[in_simulated_region]]

    # normalize only simulated region
    if in_simulated_region.sum() > 0:
        valid_sens = sens_in_region[sens_in_region > -19]
        if len(valid_sens) > 0:
            # Check for custom range from GUI
            nc = getattr(bpy.context.scene, "neurocaptain_settings", None)
            if nc and getattr(nc, "viz_custom_range", False):
                vmin = float(nc.viz_vmin)
                vmax = float(nc.viz_vmax)
                print(f"  Sensitivity range: [{vmin:.1f}, {vmax:.1f}]  (custom range)")
            else:
                vmin = float(np.percentile(valid_sens, 5))
                vmax = float(np.percentile(valid_sens, 95))
                print(f"  Sensitivity range: [{vmin:.1f}, {vmax:.1f}]  (auto 5th-95th percentile)")

            if vmax > vmin:
                sens_clipped = np.clip(sens_in_region, vmin, vmax)
                norm[in_simulated_region] = (sens_clipped - vmin) / (vmax - vmin)
    
    # apply vertex colors
    mesh = cortex.data
    
    # clear old sensitivity
    if bpy.app.version >= (4, 0, 0):
        if 'Sensitivity' in mesh.color_attributes:
            mesh.color_attributes.remove(mesh.color_attributes['Sensitivity'])
        vc = mesh.color_attributes.new(name='Sensitivity', type='BYTE_COLOR', domain='CORNER')
    else:
        if 'Sensitivity' in mesh.vertex_colors:
            mesh.vertex_colors.remove(mesh.vertex_colors['Sensitivity'])
        vc = mesh.vertex_colors.new(name='Sensitivity')
    
    def blue_red_colormap(v):
        """Blue → cyan → green → yellow → red heatmap."""
        if v < 0.25:
            t = v / 0.25
            return (0.0, t,       1.0,       1.0)
        elif v < 0.5:
            t = (v - 0.25) / 0.25
            return (0.0, 1.0,     1.0 - t,   1.0)
        elif v < 0.75:
            t = (v - 0.5) / 0.25
            return (t,   1.0,     0.0,       1.0)
        else:
            t = (v - 0.75) / 0.25
            return (1.0, 1.0 - t, 0.0,       1.0)
    
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            vc.data[li].color = blue_red_colormap(norm[vi])
    
    if bpy.app.version >= (4, 0, 0):
        mesh.color_attributes.active_color = vc
    else:
        mesh.vertex_colors.active = vc

    for hide_name in ('Head_Surface_5L', 'headmesh'):
        obj = bpy.data.objects.get(hide_name)
        if obj:
            obj.hide_viewport = True
            obj.hide_render = True

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'SOLID'
                    space.shading.color_type = 'VERTEX'

    print("\n" + "=" * 70)
    print("✓ SIMULATION COMPLETE")
    print("=" * 70)

    return {
        'success': True,
        'message': 'Redbird simulation completed successfully'
    }