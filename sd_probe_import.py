import bpy
import numpy as np
from scipy.io import loadmat
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy.types import Operator
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper

def create_anchor_ring_material():
    mat_name = "Anchor_Ring_Material"
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    mat.diffuse_color = (1.0, 0.85, 0.0, 1.0)
    nodes = mat.node_tree.nodes
    nodes.clear()
    node_emission = nodes.new(type='ShaderNodeEmission')
    node_emission.inputs[0].default_value = (1.0, 0.85, 0.0, 1.0)
    node_emission.inputs[1].default_value = 5.0
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(node_emission.outputs[0], node_output.inputs[0])
    return mat

def get_or_create_collection(collection_name):
    collection = bpy.data.collections.get(collection_name)
    if not collection:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    return collection

def add_to_collection(obj, collection):
    for col in obj.users_collection:
        col.objects.unlink(obj)
    collection.objects.link(obj)

class NEUROCAPTAIN_OT_import_sd_probe(Operator, ImportHelper):
    """Import fNIRS probe from Homer/AtlasViewer SD file"""
    bl_idname = "neurocaptain.import_sd_probe"
    bl_label = "Import SD Probe"
    bl_description = "Import and register fNIRS probe from SD file to head mesh"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".SD"
    filter_glob: StringProperty(default="*.SD;*.sd", options={'HIDDEN'})

    def execute(self, context):
        try:
            sd_file_path       = self.filepath
            head_mesh_name     = 'headmesh'
            landmark_mesh_name = 'LandmarkMesh'
            source_color       = (1.0, 0.0, 0.0, 1.0)
            detector_color     = (0.0, 0.0, 0.0, 1.0)

            sd_data = self.load_sd_file(sd_file_path)

            if landmark_mesh_name not in bpy.data.objects:
                self.report({'ERROR'}, f"'{landmark_mesh_name}' not found.")
                return {'CANCELLED'}
            landmarks_3d = self.get_landmark_positions_with_labels(landmark_mesh_name)

            anchors_3d, matched_labels = self.match_anchors_to_landmarks(
                sd_data['anchor_labels'], landmarks_3d
            )
            if len(anchors_3d) < 3:
                self.report({'ERROR'}, f"Only {len(anchors_3d)} anchors matched.")
                return {'CANCELLED'}

            matched_indices        = [i for i, label in enumerate(sd_data['anchor_labels'])
                                      if label in matched_labels]
            anchors_2d_matched     = sd_data['anchors_2d'][matched_indices]
            matched_anchor_indices = [sd_data['anchor_indices'][i] for i in matched_indices]

            if head_mesh_name not in bpy.data.objects:
                self.report({'ERROR'}, f"Head mesh '{head_mesh_name}' not found")
                return {'CANCELLED'}
            head_mesh = bpy.data.objects[head_mesh_name]

            optode_diameter, optode_thickness = self.get_head_scale(head_mesh)

            all_positions_3d = self.register_with_springs(
                sd_data['all_positions_2d'],
                anchors_2d_matched,
                anchors_3d,
                matched_anchor_indices,
                sd_data['spring_list'],
                head_mesh
            )

            n_srcs       = sd_data['n_srcs']
            n_dets       = sd_data['n_dets']
            src_pos_3d   = all_positions_3d[:n_srcs]
            det_pos_3d   = all_positions_3d[n_srcs:n_srcs + n_dets]
            dummy_pos_3d = all_positions_3d[n_srcs + n_dets:]
            anchor_set   = set(matched_anchor_indices)  # dummy anchor indices

            sources_collection   = get_or_create_collection("Sources")
            detectors_collection = get_or_create_collection("Detectors")
            anchors_collection   = get_or_create_collection("Anchor Indicators")
            ring_mat             = create_anchor_ring_material()

            # Build set of source/detector indices directly connected to an anchor dummy
            anchor_adjacent = set()
            if sd_data['spring_list'] is not None:
                for spring in sd_data['spring_list']:
                    i1 = int(spring[0]) - 1
                    i2 = int(spring[1]) - 1
                    if i1 in anchor_set and i2 not in anchor_set and i2 < n_srcs + n_dets:
                        anchor_adjacent.add(i2)
                    if i2 in anchor_set and i1 not in anchor_set and i1 < n_srcs + n_dets:
                        anchor_adjacent.add(i1)
            for i, pos in enumerate(src_pos_3d):
                snapped_pos, normal = self.snap_to_mesh_surface(pos, head_mesh)
                optode = self.create_optode_disc(snapped_pos, normal, f"Source_{i+1}",
                                                 source_color, optode_diameter, optode_thickness)
                add_to_collection(optode, sources_collection)
                if i in anchor_adjacent:
                    optode["is_anchor"] = 1
                    bpy.ops.mesh.primitive_torus_add(location=(0, 0, 0),
                                                     major_radius=2.5, minor_radius=0.3)
                    ind = context.active_object
                    add_to_collection(ind, anchors_collection)
                    ind.name = f"Anchor_Ring_Source_{i+1}"
                    ind.parent = optode
                    ind.location = (0, 0, 0)
                    ind.hide_render = True
                    ind.data.materials.append(ring_mat)
                else:
                    optode["is_anchor"] = 0

            for i, pos in enumerate(det_pos_3d):
                snapped_pos, normal = self.snap_to_mesh_surface(pos, head_mesh)
                optode = self.create_optode_disc(snapped_pos, normal, f"Detector_{i+1}",
                                                 detector_color, optode_diameter, optode_thickness)
                add_to_collection(optode, detectors_collection)
                anchor_idx = n_srcs + i
                if anchor_idx in anchor_adjacent:
                    optode["is_anchor"] = 1
                    bpy.ops.mesh.primitive_torus_add(location=(0, 0, 0),
                                                     major_radius=2.5, minor_radius=0.3)
                    ind = context.active_object
                    add_to_collection(ind, anchors_collection)
                    ind.name = f"Anchor_Ring_Detector_{i+1}"
                    ind.parent = optode
                    ind.location = (0, 0, 0)
                    ind.hide_render = True
                    ind.data.materials.append(ring_mat)
                else:
                    optode["is_anchor"] = 0

            # Dummies used internally only — not shown in viewport

            # ── Connection mesh with spring type visualization ──────────
            self.create_connection_mesh(
                context, src_pos_3d, det_pos_3d,
                sd_data['spring_list'], n_srcs, n_dets
            )

            self.report({'INFO'},
                f"Created {len(src_pos_3d)} sources + {len(det_pos_3d)} detectors "
                f"({len(matched_anchor_indices)} anchors)")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

    # ------------------------------------------------------------------
    # SD file loading
    # ------------------------------------------------------------------
    def load_sd_file(self, filepath):
        mat_data = loadmat(filepath)
        SD  = mat_data['SD']
        sd  = SD[0, 0]

        src_pos     = sd['SrcPos']
        det_pos     = sd['DetPos']
        dummy_pos   = sd['DummyPos']
        anchor_list = sd['AnchorList']
        spring_list = sd['SpringList'] if 'SpringList' in sd.dtype.names else None
        n_srcs = int(sd['nSrcs'][0, 0])
        n_dets = int(sd['nDets'][0, 0])

        all_positions_2d = np.vstack([src_pos, det_pos, dummy_pos])

        anchors_2d, anchor_labels, anchor_indices = [], [], []
        for anchor in anchor_list:
            idx   = int(anchor[0][0, 0]) - 1
            label = str(anchor[1][0])
            if idx < len(all_positions_2d):
                pos = all_positions_2d[idx]
                anchors_2d.append([float(pos[0]), float(pos[1])])
                anchor_labels.append(label)
                anchor_indices.append(idx)

        return {
            'all_positions_2d': all_positions_2d.astype(float),
            'anchors_2d':       np.array(anchors_2d, dtype=float),
            'anchor_labels':    anchor_labels,
            'anchor_indices':   anchor_indices,
            'spring_list':      spring_list,
            'n_srcs':           n_srcs,
            'n_dets':           n_dets,
        }

    # ------------------------------------------------------------------
    # Landmark helpers
    # ------------------------------------------------------------------
    def get_landmark_positions_with_labels(self, landmark_mesh_name):
        landmark_obj  = bpy.data.objects[landmark_mesh_name]
        num_landmarks = len(landmark_obj.data.vertices)

        # --- Primary: use labels stored by brain1020mesh at generation time ---
        stored_labels = landmark_obj.get("landmark_labels", None)
        if stored_labels is not None:
            landmark_labels = list(stored_labels)
            _standard_check = {"Nz", "Iz", "Cz", "C3", "C4", "Fpz", "Oz",
                               "F7", "F8", "T7", "T8", "T3", "T4"}
            if not any(lbl in _standard_check for lbl in landmark_labels):
                stored_labels = None
        if stored_labels is None:
            if num_landmarks >= 88:
                landmark_labels = [
                    "Nz","Iz","Lpa","Rpa","Cz",
                    "T7","C5","C3","C1","Cz","C2","C4","C6","T8",
                    "Fpz","AFz","Fz","FCz","Cz","CPz","Pz","POz","Oz",
                    "FT7","F7","AF7","Fp1","TP7","P7","PO7","O1",
                    "FT8","F8","AF8","Fp2","TP8","P8","PO8","O2",
                    "FC1","FC3","FC5","FC2","FC4","FC6",
                    "F1","F3","F5","F2","F4","F6",
                    "","AF3","","","AF4","",
                    "CP1","CP3","CP5","CP2","CP4","CP6",
                    "P1","P3","P5","P2","P4","P6",
                    "","PO3","","","PO4","",
                    "FT9","F9","","","TP9","P9","PO9","O9",
                    "FT10","F10","","","TP10","P10","PO10","O10",
                ]
            elif num_landmarks >= 77:
                landmark_labels = [
                    "Nz","Iz","Lpa","Rpa","Cz",
                    "T7","C5","C3","C1","Cz","C2","C4","C6","T8",
                    "Fpz","AFz","Fz","FCz","Cz","CPz","Pz","POz","Oz",
                    "FT7","F7","AF7","Fp1","TP7","P7","PO7","O1",
                    "FT8","F8","AF8","Fp2","TP8","P8","PO8","O2",
                    "FC1","FC3","FC5","FC2","FC4","FC6",
                    "F1","F3","F5","F2","F4","F6","AF3","AF4",
                    "CP1","CP3","CP5","CP2","CP4","CP6",
                    "P1","P3","P5","P2","P4","P6","PO3","PO4",
                    "FT9","F9","","","TP9","P9","PO9","O9",
                    "FT10","F10","","","TP10","P10","PO10","O10",
                ]
            elif num_landmarks >= 27:
                landmark_labels = [
                    "Nz","Iz","Lpa","Rpa","Cz",
                    "T3","C3","Cz","C4","T4",
                    "Fpz","Fz","Cz","Pz","Oz",
                    "F7","Fp1","T5","O1","F8","Fp2","T6","O2",
                    "F3","F4","P3","P4",
                ]
            else:
                landmark_labels = [f"Landmark_{i+1}" for i in range(num_landmarks)]

        # Build label→position dict
        landmarks_3d = {}
        for i, vert in enumerate(landmark_obj.data.vertices):
            v_global = landmark_obj.matrix_world @ vert.co
            if i < len(landmark_labels) and landmark_labels[i]:
                landmarks_3d[landmark_labels[i]] = np.array(
                    [v_global.x, v_global.y, v_global.z])

        # Fallback: unit-vector nearest-neighbor for any missing anchor labels
        ref_units = {
            "F7":  np.array([-0.769,  0.638,  0.122]),
            "F8":  np.array([ 0.780,  0.640,  0.112]),
            "FC5": np.array([-0.823,  0.330,  0.471]),
            "FC6": np.array([ 0.838,  0.327,  0.461]),
            "CP3": np.array([-0.633, -0.381,  0.758]),
            "CP4": np.array([ 0.648, -0.388,  0.750]),
            "P3":  np.array([-0.528, -0.688,  0.627]),
            "P4":  np.array([ 0.533, -0.681,  0.611]),
            "PO3": np.array([-0.340, -0.855,  0.400]),
            "PO4": np.array([ 0.341, -0.860,  0.398]),
        }
        missing = [lb for lb in ref_units if lb not in landmarks_3d]
        if missing:
            all_verts = np.array([
                list(landmark_obj.matrix_world @ v.co)
                for v in landmark_obj.data.vertices
            ])
            norms = np.linalg.norm(all_verts, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1.0
            unit_verts = all_verts / norms
            for lb in missing:
                ref  = ref_units[lb]
                best = int(np.argmax(unit_verts @ ref))
                landmarks_3d[lb] = all_verts[best]

        return landmarks_3d

    def match_anchors_to_landmarks(self, anchor_labels, landmarks_3d):
        anchors_3d, matched_labels = [], []
        for label in anchor_labels:
            if label in landmarks_3d:
                anchors_3d.append(landmarks_3d[label])
                matched_labels.append(label)
            else:
                self.report({'WARNING'}, f"Anchor '{label}' not found in LandmarkMesh")
        return np.array(anchors_3d, dtype=float), matched_labels

    # ------------------------------------------------------------------
    # Seed: barycentric from 2D SD anchor positions -> 3D atlas positions
    # ------------------------------------------------------------------
    def _seed_positions_affine(self, all_positions_2d, anchor_indices, anchors_3d, n_points):
        """
        Per-patch affine seed — translation of gen_xform_from_pts.m applied
        separately to each hemisphere patch.

        A global affine fails for bilateral probes because the 2D flat layout
        is hemisphere-flipped (F7 is on the RIGHT side of flat space but maps
        to the LEFT hemisphere in 3D). A single affine cannot fold space.

        Fix: split anchors into 2 patches by K-means on 2D X coordinate,
        compute a separate least-squares affine per patch (pinv([p1,1])*p2),
        assign each free optode to its nearest patch centroid, then apply
        that patch's affine transform.
        """
        anc2d = all_positions_2d[list(anchor_indices), :2].astype(float)
        anc3d = anchors_3d.astype(float)
        n_anc = len(anchor_indices)

        def fit_affine(p1_2d, p2_3d):
            """pinv([p1,1]) * p2 per output dim — gen_xform_from_pts.m"""
            A = np.column_stack([p1_2d, np.ones(len(p1_2d))])
            T = np.zeros((3, 3))
            for ii in range(3):
                T[ii, :] = np.linalg.pinv(A) @ p2_3d[:, ii]
            return T

        def apply_affine(T, pts_2d):
            pts_h = np.column_stack([pts_2d[:, :2], np.ones(len(pts_2d))])
            return (T @ pts_h.T).T

        anchor_set  = set(anchor_indices)
        anchor_list = list(anchor_indices)
        pos3d = np.zeros((n_points, 3), dtype=float)

        # K-means split on X coordinate — separates left/right SD patches
        from scipy.cluster.vq import kmeans2
        _, labels = kmeans2(anc2d[:, 0:1].astype(float), 2, seed=42, minit='points')
        g0 = np.where(labels == 0)[0]
        g1 = np.where(labels == 1)[0]
        # If either patch has < 2 anchors, a per-patch affine is degenerate.
        # Fall back to a single global affine across all anchors.
        if len(g0) < 2 or len(g1) < 2:
            T_global = fit_affine(anc2d, anc3d)
            for i in range(n_points):
                if i in anchor_set:
                    k = anchor_list.index(i)
                    pos3d[i] = anc3d[k]
                else:
                    pos3d[i] = apply_affine(T_global, all_positions_2d[i:i+1, :2])[0]
        else:
            T0 = fit_affine(anc2d[g0], anc3d[g0])
            T1 = fit_affine(anc2d[g1], anc3d[g1])

            # Ensure patch0 maps to negative-X hemisphere (left) and patch1 to positive-X.
            if anc3d[g0, 0].mean() > anc3d[g1, 0].mean():
                T0, T1 = T1, T0
                g0, g1 = g1, g0

            c0 = anc2d[g0].mean(axis=0)
            c1 = anc2d[g1].mean(axis=0)

            n0 = n1 = 0
            for i in range(n_points):
                if i in anchor_set:
                    k = anchor_list.index(i)
                    pos3d[i] = anc3d[k]
                    continue
                pt = all_positions_2d[i, :2].astype(float)
                if np.linalg.norm(pt - c0) <= np.linalg.norm(pt - c1):
                    pos3d[i] = apply_affine(T0, pt.reshape(1, 2))[0]
                    n0 += 1
                else:
                    pos3d[i] = apply_affine(T1, pt.reshape(1, 2))[0]
                    n1 += 1

        return pos3d

    # ------------------------------------------------------------------
    # Main registration: AtlasViewer 1mm-step mechanics
    # Aasted et al. 2015 Neurophotonics 2(2):020801 Section 3.1
    # ------------------------------------------------------------------
    def register_with_springs(self, all_positions_2d, anchors_2d, anchors_3d,
                               anchor_indices, spring_list, head_mesh):
        """
        Faithful Python translation of AtlasViewer's positionprobe.m / optodes2bnd.

        Key mechanics (from source code):
          1. optodes2bnd: shoots a ray from each free optode TOWARD the head
             center (posAttract) until it hits the air-tissue surface.
             This is a ray cast toward center — NOT nearest-surface snap.
          2. Spring forces: Jacobian-style, globally normalized so the max
             displacement component across ALL optodes = 1 unit per iteration.
          3. Anchors: dpos = [0,0,0] — never moved by optodes2bnd or springs.
        """
        n_points     = len(all_positions_2d)
        anchor_set   = set(anchor_indices)
        free_indices = [i for i in range(n_points) if i not in anchor_set]
        anchor_pos   = {aidx: anchors_3d[k].copy()
                        for k, aidx in enumerate(anchor_indices)}

        # Build BVH once
        bvh = BVHTree.FromObject(head_mesh, bpy.context.evaluated_depsgraph_get())

        # --- Affine seed (gen_positionprobe_dat.m / gen_xform_from_pts.m) ---
        # Least-squares affine from flat 2D anchor positions -> 3D atlas positions,
        # applied to all optodes. Direct translation of AtlasViewer's initialization.
        positions = self._seed_positions_affine(
            all_positions_2d, anchor_indices, anchors_3d, n_points
        )
        for aidx, apos in anchor_pos.items():
            positions[aidx] = apos.copy()

        def snap_to_surface(positions):
            """Snap each free optode to nearest surface point."""
            for i in free_indices:
                loc, _, _, _ = bvh.find_nearest(Vector(positions[i]))
                if loc is not None:
                    positions[i] = np.array(loc)
            for aidx, apos in anchor_pos.items():
                positions[aidx] = apos.copy()
            return positions

        # Initial surface placement
        positions = snap_to_surface(positions)

        # --- Parse springs ---
        spring_data = []
        if spring_list is not None:
            nf = ns = 0
            for spring in spring_list:
                i1 = int(spring[0]) - 1
                i2 = int(spring[1]) - 1
                if i1 >= n_points or i2 >= n_points:
                    continue
                s = float(spring[2])
                if s == -1.0:
                    rest = np.linalg.norm(positions[i1] - positions[i2])
                    ks   = 1e-4
                    nf  += 1
                else:
                    rest = s
                    ks   = 1.0
                    ns  += 1
                spring_data.append((i1, i2, rest, ks))

        def total_energy():
            return sum(0.5 * ks * (np.linalg.norm(positions[i1] - positions[i2]) - rest) ** 2
                       for i1, i2, rest, ks in spring_data)

        N_ITER   = 400
        TOL      = 0.5
        E_window = []

        for it in range(N_ITER):

            # Spring forces — Jacobian style (from positionprobe.m)
            forces = np.zeros((n_points, 3))
            for (i1, i2, rest, ks) in spring_data:
                rsepvec = positions[i1] - positions[i2]
                rsep    = np.linalg.norm(rsepvec)
                if rsep < 1e-10:
                    continue
                rd = rsep - rest
                J  = ks * (rsepvec * rd) / rsep
                if i1 not in anchor_set:
                    forces[i1] += J
                if i2 not in anchor_set:
                    forces[i2] -= J

            # Global normalization — from positionprobe.m: / max(abs(delta_r))
            delta = -forces
            for aidx in anchor_set:
                delta[aidx] = 0.0
            max_delta = np.max(np.abs(delta))
            if max_delta > 1e-10:
                delta = delta / max_delta

            for i in free_indices:
                positions[i] += delta[i]

            # Snap to surface after spring displacement
            positions = snap_to_surface(positions)

            if (it + 1) % 50 == 0:
                E = total_energy()
                E_window.append(E)
                if len(E_window) >= 2 and abs(E_window[-1] - E_window[-2]) < TOL:
                    break

        return positions

    def create_connection_mesh(self, context, src_pos_3d, det_pos_3d,
                               spring_list, n_srcs, n_dets):
        """
        Build an Optode_Connections mesh from the SD SpringList.
        Stiff springs (length > 0) → blue material.
        Flexible springs (length == -1) → orange material.
        spring_states stored on object for JSON export compatibility.
        """
        import bmesh

        # Remove any existing connection mesh
        if "Optode_Connections" in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects["Optode_Connections"], do_unlink=True)

        # Build ordered optode list matching SD unified index scheme
        # (sources first, then detectors — dummies excluded)
        srcs = sorted([o for o in bpy.data.objects if o.name.startswith("Source_")],
                      key=lambda x: int(x.name.split("_")[1]))
        dets = sorted([o for o in bpy.data.objects if o.name.startswith("Detector_")],
                      key=lambda x: int(x.name.split("_")[1]))
        all_optodes = srcs + dets
        optode_names = [o.name for o in all_optodes]
        n_real = len(all_optodes)

        if n_real == 0:
            return

        mesh    = bpy.data.meshes.new("Optode_Connections")
        conn_obj = bpy.data.objects.new("Optode_Connections", mesh)
        context.collection.objects.link(conn_obj)

        bm = bmesh.new()
        verts_bm = [bm.verts.new(o.location) for o in all_optodes]
        bm.verts.ensure_lookup_table()

        spring_states = {}
        n_stiff = n_flex = 0

        if spring_list is not None:
            for spring in spring_list:
                i1 = int(spring[0]) - 1
                i2 = int(spring[1]) - 1
                s  = float(spring[2])

                # Only connect real optodes (sources + detectors), skip dummies
                if i1 >= n_real or i2 >= n_real:
                    continue

                is_flexible = (s == -1.0)
                rest_length = (all_optodes[i1].location -
                               all_optodes[i2].location).length

                try:
                    bm.edges.new([verts_bm[i1], verts_bm[i2]])
                except ValueError:
                    pass  # edge already exists

                opt1, opt2 = optode_names[i1], optode_names[i2]
                pair = sorted([opt1, opt2])
                key  = f"{pair[0]}_{pair[1]}"
                spring_states[key] = {
                    "optode1":    opt1,
                    "optode2":    opt2,
                    "rest_length": rest_length if s == -1.0 else abs(s),
                    "pull":        0.3 if is_flexible else 0.9,
                    "push":        0.3 if is_flexible else 0.9,
                    "is_flexible": is_flexible,
                }
                if is_flexible:
                    n_flex += 1
                else:
                    n_stiff += 1

        bm.to_mesh(mesh)
        bm.free()

        conn_obj["spring_states"]       = spring_states
        conn_obj["optode_names_ordered"] = optode_names
        conn_obj["optode_names"]         = optode_names
        conn_obj["is_optode_connections"] = True

        # Materials — blue = stiff, orange = flexible
        def get_or_make_mat(name, color):
            mat = bpy.data.materials.get(name)
            if not mat:
                mat = bpy.data.materials.new(name=name)
                mat.use_nodes = True
                mat.diffuse_color = color
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs['Base Color'].default_value = color
            return mat

        mat_stiff    = get_or_make_mat("Spring_Stiff",    (0.2, 0.5, 1.0, 1.0))  # blue
        mat_flexible = get_or_make_mat("Spring_Flexible", (1.0, 0.6, 0.0, 1.0))  # orange
        conn_obj.data.materials.append(mat_stiff)
        conn_obj.data.materials.append(mat_flexible)

        # Assign material per edge via bevel weight layer (stiff=0, flexible=1)
        # Use edge crease to visually distinguish flexible springs
        for edge in conn_obj.data.edges:
            i1, i2 = edge.vertices[0], edge.vertices[1]
            if i1 < len(optode_names) and i2 < len(optode_names):
                opt1, opt2 = optode_names[i1], optode_names[i2]
                pair = sorted([opt1, opt2])
                key  = f"{pair[0]}_{pair[1]}"
                if key in spring_states and spring_states[key]["is_flexible"]:
                    # Mark flexible edges with edge crease for visual distinction
                    conn_obj.data.edges[edge.index].crease = 1.0

        # Soft body modifier (consistent with JSON import)
        sb_mod = conn_obj.modifiers.new(name="Softbody", type='SOFT_BODY')
        sb     = sb_mod.settings
        sb.use_edges  = True
        sb.use_goal   = True
        sb.pull       = 0.9
        sb.push       = 0.9
        sb.damping    = 0.5

        print(f"Imported connection mesh: {n_stiff} stiff springs, {n_flex} flexible springs")

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def get_head_scale(self, head_mesh):
        verts = head_mesh.data.vertices
        bbox_min = Vector([min(v.co[i] for v in verts) for i in range(3)])
        bbox_max = Vector([max(v.co[i] for v in verts) for i in range(3)])
        dims = (head_mesh.matrix_world @ bbox_max) - (head_mesh.matrix_world @ bbox_min)
        sz   = max(dims)
        return sz * 0.02, sz * 0.02 * 0.15

    def snap_to_mesh_surface(self, position, head_mesh):
        bvh = BVHTree.FromObject(head_mesh, bpy.context.evaluated_depsgraph_get())
        loc, normal, _, _ = bvh.find_nearest(Vector(position))
        if loc:
            return np.array(loc), np.array(normal)
        return position, np.array([0.0, 0.0, 1.0])

    def create_optode_disc(self, position, normal, name, color, diameter, thickness):
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32, radius=diameter / 2.0, depth=thickness,
            enter_editmode=False, align='WORLD', location=(0, 0, 0)
        )
        optode = bpy.context.active_object
        optode.name = name
        z_axis = Vector((0, 0, 1))
        optode.rotation_euler = z_axis.rotation_difference(Vector(normal)).to_euler()
        optode.location = Vector(position)
        mat  = bpy.data.materials.new(name=f"{name}_mat")
        mat.use_nodes = True
        mat.diffuse_color = color
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Metallic'].default_value   = 0.3
            bsdf.inputs['Roughness'].default_value  = 0.4
        optode.data.materials.append(mat)
        if bpy.app.version >= (4, 1, 0):
            bpy.ops.object.shade_smooth()
        else:
            for face in optode.data.polygons:
                face.use_smooth = True
        return optode


def register():
    bpy.utils.register_class(NEUROCAPTAIN_OT_import_sd_probe)

def unregister():
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_import_sd_probe)

if __name__ == "__main__":
    register()