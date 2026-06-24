"""
Import optode probe JSON and relax using Blender's cloth simulation.

Optodes are placed via barycentric registration from the probe JSON.
A rest shape key encodes per-edge target lengths from the JSON export.
Blender's cloth solver relaxes the probe toward those rest lengths while
anchor optodes stay pinned.  Final positions are snapped to the head surface.
"""

import bpy
import numpy as np
from collections import defaultdict
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .optode_connect import (
    get_or_create_collection,
    add_to_collection,
    create_anchor_ring_material,
    create_connection_material,
    setup_optode_hooks,
)
from .landmark_labels import get_landmark_labels


class NEUROCAPTAIN_OT_import_optode_json_blender_goal(bpy.types.Operator):
    """Import probe JSON and relax using Blender cloth simulation"""
    bl_idname = "neurocaptain.import_optode_json_blender_goal"
    bl_label = "Import Probe (Blender Goals)"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})

    simulation_frames: bpy.props.IntProperty(
        name="Simulation Frames",
        description="Number of frames to run the cloth simulation",
        default=100,
        min=10,
        max=500,
    )

    spring_stiffness: bpy.props.FloatProperty(
        name="Spring Stiffness",
        description="Structural spring stiffness (tension and compression)",
        default=15.0,
        min=0.1,
        max=100.0,
    )

    pin_stiffness: bpy.props.FloatProperty(
        name="Pin Stiffness",
        description="How strongly pinned (anchor) vertices resist movement",
        default=1.0,
        min=0.01,
        max=10.0,
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        import json

        SRC_COLOR = (1.0, 0.0, 0.0, 1.0)
        DET_COLOR = (0.0, 0.0, 0.0, 1.0)

        if "headmesh" not in bpy.data.objects:
            self.report({'ERROR'}, "No 'headmesh' found in scene")
            return {'CANCELLED'}
        headmesh = bpy.data.objects["headmesh"]

        landmark_mesh, landmark_labels, label_to_indices = self._load_landmarks()
        num_target_landmarks = len(landmark_mesh.data.vertices) if landmark_mesh else 0

        try:
            with open(self.filepath, 'r') as f:
                config = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load JSON: {e}")
            return {'CANCELLED'}

        if config.get("format") != "NeuroCaptain Probe Configuration":
            self.report({'ERROR'}, "Invalid JSON format")
            return {'CANCELLED'}

        json_lm_count = config.get("metadata", {}).get("num_landmarks", 0)
        same_landmark_system = (json_lm_count == num_target_landmarks and num_target_landmarks > 0)
        print(f"Landmark system check: JSON={json_lm_count}, scene={num_target_landmarks}, same={same_landmark_system}")

        optode_objects = {}
        optode_diameter, optode_thickness = self._get_head_scale(headmesh)
        placement_stats = defaultdict(int)

        bpy.ops.object.select_all(action='DESELECT')

        # ── Place optodes ────────────────────────────────────────────────
        for optode_data in config.get("optodes", []):
            name = optode_data["name"]
            optode_type = optode_data["type"]
            is_anchor = optode_data.get("is_anchor", False)

            initial_pos, method = self._resolve_position(
                optode_data, landmark_mesh, landmark_labels,
                label_to_indices, same_landmark_system, num_target_landmarks,
            )
            placement_stats[method] += 1
            snapped_pos, surface_normal = self._snap_to_surface(
                initial_pos, headmesh
            )

            color = SRC_COLOR if optode_type == "source" else DET_COLOR
            coll_name = "Sources" if optode_type == "source" else "Detectors"
            coll = get_or_create_collection(coll_name)

            optode = self._create_optode_disc(
                snapped_pos, surface_normal, name, color,
                optode_diameter, optode_thickness,
            )
            add_to_collection(optode, coll)
            optode["is_anchor"] = 1 if is_anchor else 0

            if is_anchor:
                optode["anchor_pos"] = list(snapped_pos)
                anchors_coll = get_or_create_collection("Anchor Indicators")
                ring_mat = create_anchor_ring_material()
                bpy.ops.mesh.primitive_torus_add(
                    location=(0, 0, 0), major_radius=2.5, minor_radius=0.3,
                )
                indicator = context.active_object
                add_to_collection(indicator, anchors_coll)
                indicator.name = f"Anchor_{name}"
                indicator.parent = optode
                indicator.location = (0, 0, 0)
                indicator.hide_render = True
                indicator.data.materials.append(ring_mat)

            cst = optode.constraints.new(type='SHRINKWRAP')
            cst.target = headmesh
            cst.shrinkwrap_type = 'NEAREST_SURFACE'
            cst.distance = 0.0
            cst.influence = 1.0

            optode_objects[name] = optode

        if not optode_objects:
            self.report({'WARNING'}, "No optodes found")
            return {'CANCELLED'}

        print(f"\nPlacement summary:")
        print(f"  Barycentric (index):  {placement_stats['barycentric_index']}")
        print(f"  Barycentric (label):  {placement_stats['barycentric_label']}")
        print(f"  Nearest landmark:     {placement_stats['nearest_lm']}")
        print(f"  Raw position:         {placement_stats['raw']}")

        # ── Build connection mesh ────────────────────────────────────────
        mesh_data = bpy.data.meshes.new("Optode_Connections")
        conn_obj = bpy.data.objects.new("Optode_Connections", mesh_data)
        context.collection.objects.link(conn_obj)

        names_ordered = list(optode_objects.keys())
        name_to_idx = {n: i for i, n in enumerate(names_ordered)}

        vertices = [optode_objects[n].location.copy() for n in names_ordered]
        edges = []
        spring_states = {}

        n_opt = len(names_ordered)
        vertex_pull_sum = np.zeros(n_opt, dtype=np.float64)
        vertex_edge_count = np.zeros(n_opt, dtype=np.float64)
        vertex_has_fixed = np.zeros(n_opt, dtype=bool)
        connections_stiff = []
        n_stiff_edges = 0
        n_flexible_edges = 0

        for cd in config.get("connections", []):
            n1, n2 = cd["optode1"], cd["optode2"]
            if n1 in name_to_idx and n2 in name_to_idx:
                i1, i2 = name_to_idx[n1], name_to_idx[n2]
                edges.append((i1, i2))
                ek = f"{min(n1, n2)}_{max(n1, n2)}"
                spring_states[ek] = {
                    "optode1": n1, "optode2": n2,
                    "rest_length": cd["rest_length"],
                    "pull": cd["pull"], "push": cd["push"],
                    "is_flexible": cd["is_flexible"],
                    "fix_distance": cd.get("fix_distance", False),
                }
                pull = float(cd.get("pull", 0.9))
                vertex_pull_sum[i1] += pull
                vertex_pull_sum[i2] += pull
                vertex_edge_count[i1] += 1
                vertex_edge_count[i2] += 1

                if cd.get("fix_distance", False):
                    vertex_has_fixed[i1] = True
                    vertex_has_fixed[i2] = True

                if not cd.get("is_flexible", False):
                    n_stiff_edges += 1
                    connections_stiff.append({
                        'optode1': n1, 'optode2': n2,
                        'rest_length': float(cd['rest_length']),
                    })
                else:
                    n_flexible_edges += 1

        print(f"Springs — Stiff: {n_stiff_edges}   Flexible: {n_flexible_edges}")

        # Find triangles in the connection graph to give the cloth sim faces
        neighbors = defaultdict(set)
        for i1, i2 in edges:
            neighbors[i1].add(i2)
            neighbors[i2].add(i1)

        tri_faces = []
        seen_tris = set()
        for i1 in range(len(names_ordered)):
            for i2 in neighbors[i1]:
                if i2 <= i1:
                    continue
                for i3 in (neighbors[i1] & neighbors[i2]):
                    if i3 <= i2:
                        continue
                    tri = (i1, i2, i3)
                    if tri not in seen_tris:
                        seen_tris.add(tri)
                        tri_faces.append(tri)

        # Edges not already implied by triangle faces
        face_edge_set = set()
        for f in tri_faces:
            for j in range(3):
                a, b = f[j], f[(j + 1) % 3]
                face_edge_set.add((min(a, b), max(a, b)))
        extra_edges = [(a, b) for a, b in edges
                       if (min(a, b), max(a, b)) not in face_edge_set]

        mesh_data.from_pydata(vertices, extra_edges, tri_faces)
        mesh_data.update()
        print(f"\nConnection mesh: {len(vertices)} verts, {len(edges)} edges, "
              f"{len(tri_faces)} triangle faces")

        conn_obj["spring_states"] = spring_states
        conn_obj["optode_names_ordered"] = names_ordered

        mat = bpy.data.materials.get("Connection_Material")
        if not mat:
            mat = bpy.data.materials.new(name="Connection_Material")
            mat.use_nodes = True
            mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (
                0.8, 0.8, 0.8, 1,
            )
        if conn_obj.data.materials:
            conn_obj.data.materials[0] = mat
        else:
            conn_obj.data.materials.append(mat)

        # ── Record initial positions ─────────────────────────────────────
        initial_np = np.array([list(v) for v in vertices])

        # ── Shape keys for cloth rest lengths ────────────────────────────
        # Only stiff edges drive the rest shape — flexible edges get whatever
        # length falls out naturally, so they don't impose a distance target.
        conn_obj.shape_key_add(name="Basis", from_mix=False)

        anchor_mask = np.array(
            [bool(optode_objects[n].get("is_anchor", 0)) for n in names_ordered],
            dtype=bool,
        )
        rest_positions = self._compute_rest_shape(
            vertices, name_to_idx, connections_stiff, anchor_mask,
        )
        rest_key = conn_obj.shape_key_add(name="Rest", from_mix=False)
        for i, rpos in enumerate(rest_positions):
            rest_key.data[i].co = rpos

        # Log a few edge length comparisons
        rest_np = np.array([list(r) for r in rest_positions])
        sample_conns = [cd for cd in config.get("connections", [])
                        if cd["optode1"] in name_to_idx and cd["optode2"] in name_to_idx]
        for cd in sample_conns[:5]:
            i1, i2 = name_to_idx[cd["optode1"]], name_to_idx[cd["optode2"]]
            cur = np.linalg.norm(initial_np[i2] - initial_np[i1])
            tgt = float(cd["rest_length"])
            rst = np.linalg.norm(rest_np[i2] - rest_np[i1])
            print(f"  {cd['optode1']}-{cd['optode2']}: "
                  f"current={cur:.3f}  json_rest={tgt:.3f}  rest_shape={rst:.3f}")

        # ── Pin group (anchors pinned, non-anchors free) ─────────────────
        pin_group = conn_obj.vertex_groups.new(name="Pins")
        for i, name in enumerate(names_ordered):
            is_anch = bool(optode_objects[name].get("is_anchor", 0))
            pin_group.add([i], 1.0 if is_anch else 0.0, 'REPLACE')

        n_anchors = int(anchor_mask.sum())
        print(f"\nPins: {n_anchors} anchors pinned, "
              f"{len(names_ordered) - n_anchors} non-anchors free")

        # ── Structural stiffness group (flexible edges resist less) ──────
        vertex_edge_count[vertex_edge_count == 0] = 1
        vertex_weights = np.clip(vertex_pull_sum / vertex_edge_count / 0.9,
                                 0.01, 1.0)
        vertex_weights[vertex_has_fixed] = 1.0

        stiffness_group = conn_obj.vertex_groups.new(name="Stiffness")
        for i in range(n_opt):
            stiffness_group.add([i], float(vertex_weights[i]), 'REPLACE')

        n_fixed = int(vertex_has_fixed.sum())
        print(f"Stiffness weights: min={vertex_weights.min():.3f} "
              f"max={vertex_weights.max():.3f} mean={vertex_weights.mean():.3f} "
              f"({n_fixed} vertices locked by fix_distance)")

        # ── Cloth simulation ─────────────────────────────────────────────
        cloth_mod = conn_obj.modifiers.new(name="Cloth", type='CLOTH')
        settings = cloth_mod.settings

        settings.rest_shape_key = rest_key
        settings.vertex_group_mass = "Pins"
        settings.pin_stiffness = self.pin_stiffness

        settings.tension_stiffness = self.spring_stiffness
        settings.compression_stiffness = self.spring_stiffness
        settings.vertex_group_structural_stiffness = "Stiffness"
        settings.shear_stiffness = 0.0
        settings.bending_stiffness = 0.0
        settings.tension_damping = 5.0
        settings.compression_damping = 5.0

        settings.mass = 0.3
        settings.air_damping = 1.0
        settings.quality = 5
        settings.time_scale = 1.0

        # ── Bake simulation (gravity disabled) ───────────────────────────
        scene = context.scene
        orig_frame = scene.frame_current
        orig_start = scene.frame_start
        orig_end = scene.frame_end
        orig_gravity = scene.use_gravity

        scene.use_gravity = False
        scene.frame_start = 1
        scene.frame_end = self.simulation_frames

        bpy.ops.object.select_all(action='DESELECT')
        conn_obj.select_set(True)
        context.view_layer.objects.active = conn_obj

        print(f"\nBaking cloth simulation ({self.simulation_frames} frames, "
              f"stiffness={self.spring_stiffness}, pin={self.pin_stiffness}, "
              f"gravity OFF)...")

        for frame in range(1, self.simulation_frames + 1):
            scene.frame_set(frame)
            if frame % 20 == 0:
                print(f"  Frame {frame}/{self.simulation_frames}")

        # Read final evaluated positions
        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = conn_obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()

        final_positions = []
        for vert in eval_mesh.vertices:
            final_positions.append(
                Vector(conn_obj.matrix_world @ vert.co)
            )
        eval_obj.to_mesh_clear()

        # Restore scene state
        scene.use_gravity = orig_gravity
        scene.frame_start = orig_start
        scene.frame_end = orig_end
        scene.frame_set(orig_frame)

        # Remove cloth modifier and shape keys
        conn_obj.modifiers.remove(cloth_mod)
        context.view_layer.objects.active = conn_obj
        if conn_obj.data.shape_keys:
            bpy.ops.object.shape_key_remove(all=True)

        # ── Apply results (snap to head surface) ────────────────────────
        head_bvh = BVHTree.FromObject(headmesh, context.evaluated_depsgraph_get())
        conn_inv = conn_obj.matrix_world.inverted()
        for i, name in enumerate(names_ordered):
            loc, normal, _, _ = head_bvh.find_nearest(final_positions[i])
            if loc is not None:
                final_positions[i] = Vector(loc)
            optode_objects[name].location = final_positions[i]
            conn_obj.data.vertices[i].co = conn_inv @ final_positions[i]
        conn_obj.data.update()

        # ── Displacement report ──────────────────────────────────────────
        final_np = np.array([list(p) for p in final_positions])
        displacements = np.linalg.norm(final_np - initial_np, axis=1)

        print(f"\n{'='*60}")
        print("DISPLACEMENT SUMMARY  (Blender cloth simulation)")
        print(f"{'='*60}")
        for cat_name, mask in [
            ("Anchors",     anchor_mask),
            ("Non-anchors", ~anchor_mask),
        ]:
            idx = np.where(mask)[0]
            if len(idx) > 0:
                d = displacements[idx]
                print(f"  {cat_name:12s}  N={len(idx):2d}  "
                      f"min={d.min():.4f}  max={d.max():.4f}  "
                      f"mean={d.mean():.4f}")
            else:
                print(f"  {cat_name:12s}  N= 0  (none)")
        print(f"{'='*60}\n")

        # ── Rebuild connection mesh as edges-only (like Delaunay) ────────
        # Discard the cloth sim mesh entirely and rebuild from evaluated
        # optode positions so hooks attach correctly.
        context.view_layer.update()

        import bmesh
        old_mesh = conn_obj.data
        new_mesh = bpy.data.meshes.new("Optode_Connections_Mesh")
        conn_obj.data = new_mesh
        bpy.data.meshes.remove(old_mesh)

        bm = bmesh.new()
        depsgraph = context.evaluated_depsgraph_get()

        optode_to_vert = {}
        for idx, oname in enumerate(names_ordered):
            optode_eval = optode_objects[oname].evaluated_get(depsgraph)
            vert = bm.verts.new(optode_eval.matrix_world.translation)
            optode_to_vert[oname] = vert
            vert.index = idx

        bm.verts.ensure_lookup_table()

        for i1, i2 in edges:
            bm.edges.new([bm.verts[i1], bm.verts[i2]])

        bm.to_mesh(new_mesh)
        bm.free()

        conn_obj["optode_names"] = names_ordered
        conn_obj["spring_states"] = spring_states
        conn_obj["is_optode_connections"] = True
        conn_obj.display_type = 'WIRE'

        create_connection_material(conn_obj, "Optode_Connection_Stiff", (0.2, 0.5, 1.0, 1.0))

        optodes_ordered = [optode_objects[n] for n in names_ordered]
        setup_optode_hooks(conn_obj, optodes_ordered)
        context.view_layer.update()

        self.report(
            {'INFO'},
            f"Imported {len(optode_objects)} optodes, {len(edges)} connections  "
            f"(bary: {placement_stats['barycentric_index']+placement_stats['barycentric_label']}, "
            f"fallback: {placement_stats['nearest_lm']+placement_stats['raw']}, "
            f"cloth sim: {self.simulation_frames} frames)",
        )
        return {'FINISHED'}

    # ── Position resolution (ported from optode_connect.py) ────────────

    def _resolve_position(self, optode_data, landmark_mesh, landmark_labels,
                          label_to_indices, same_landmark_system, num_target_landmarks):
        """
        Return (Vector position, method_string).

        Resolution priority:
          1. Barycentric via vertex INDICES  (exact when same landmark system)
          2. Barycentric via vertex LABELS   (cross-system fallback)
          3. Nearest-landmark + offset       (legacy / partial data)
          4. Raw stored position             (last resort)
        """
        name = optode_data["name"]

        reg = optode_data.get("registration") or optode_data.get("anchor_info")

        if reg and landmark_mesh and reg.get("method") == "barycentric":
            bary = reg.get("barycentric_coords")
            stored_indices = reg.get("vertex_indices")
            vlabels = reg.get("vertex_labels", [])

            if bary and len(bary) == 3:

                # Strategy 1: vertex indices (same landmark system)
                if same_landmark_system and stored_indices and len(stored_indices) == 3:
                    all_valid = all(0 <= idx < num_target_landmarks for idx in stored_indices)
                    if all_valid:
                        pos = self._reconstruct_from_indices(
                            landmark_mesh, stored_indices, bary
                        )
                        if pos is not None:
                            print(f"  {name}: BARY-INDEX  tri=[{stored_indices}]  w={[f'{b:.3f}' for b in bary]}")
                            return pos, "barycentric_index"

                # Strategy 2: vertex labels (cross-system)
                if vlabels and len(vlabels) == 3:
                    resolved = self._resolve_labels_to_indices(
                        vlabels, stored_indices, label_to_indices, num_target_landmarks
                    )
                    if resolved is not None:
                        pos = self._reconstruct_from_indices(
                            landmark_mesh, resolved, bary
                        )
                        if pos is not None:
                            print(f"  {name}: BARY-LABEL  labels={vlabels} -> idx={resolved}  w={[f'{b:.3f}' for b in bary]}")
                            return pos, "barycentric_label"

                self.report({'WARNING'}, f"{name}: barycentric labels/indices failed, trying fallback")

        # Strategy 3: nearest landmark + offset
        if reg and landmark_mesh:
            nl = reg.get("nearest_landmark")
            if nl:
                lm_name = nl.get("landmark", "")
                offset = nl.get("offset")
                candidates = label_to_indices.get(lm_name, [])
                if candidates and offset:
                    lm_idx = candidates[0]
                    lm_world = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[lm_idx].co
                    pos = lm_world + Vector(offset)
                    print(f"  {name}: NEAREST-LM  landmark={lm_name}  idx={lm_idx}")
                    return pos, "nearest_lm"

        # Strategy 4: raw position
        pos = Vector(optode_data["position"])
        is_anchor = optode_data.get("is_anchor", False)
        if is_anchor:
            self.report({'WARNING'}, f"Anchor {name}: using RAW position -- cross-atlas will be wrong!")
        print(f"  {name}: RAW POSITION FALLBACK")
        return pos, "raw"

    @staticmethod
    def _reconstruct_from_indices(landmark_mesh, indices, bary):
        """Reconstruct world position from vertex indices + barycentric weights."""
        try:
            v0 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[indices[0]].co
            v1 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[indices[1]].co
            v2 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[indices[2]].co
            return bary[0] * v0 + bary[1] * v1 + bary[2] * v2
        except (IndexError, KeyError):
            return None

    @staticmethod
    def _resolve_labels_to_indices(vlabels, stored_indices, label_to_indices, num_verts):
        """
        Resolve three vertex labels to indices on the target LandmarkMesh.

        Handles duplicate labels (e.g. "Cz" at indices 4, 9, 18) by:
          - If label has exactly one candidate -> use it.
          - If label has multiple candidates -> prefer the stored_index if it matches
            one of the candidates; otherwise use the first candidate.
        """
        resolved = []
        for i, lbl in enumerate(vlabels):
            candidates = label_to_indices.get(lbl, [])
            if not candidates:
                return None

            if len(candidates) == 1:
                resolved.append(candidates[0])
            else:
                hint = stored_indices[i] if (stored_indices and i < len(stored_indices)) else None
                if hint is not None and hint in candidates and hint < num_verts:
                    resolved.append(hint)
                else:
                    resolved.append(candidates[0])

        if all(0 <= idx < num_verts for idx in resolved):
            return resolved
        return None

    # ── Rest shape computation ───────────────────────────────────────────

    @staticmethod
    def _compute_rest_shape(vertices, name_to_idx, connections, anchor_mask):
        """
        Find vertex positions where edge lengths approximate JSON rest lengths.

        Iteratively adjusts positions via spring forces (unconstrained 3D,
        no surface projection).  Anchors stay fixed so the rest shape is
        properly aligned with the head.
        """
        n = len(vertices)
        pos = np.array([list(v) for v in vertices], dtype=np.float64)

        edge_list = []
        for cd in connections:
            n1, n2 = cd["optode1"], cd["optode2"]
            if n1 in name_to_idx and n2 in name_to_idx:
                edge_list.append((
                    name_to_idx[n1], name_to_idx[n2],
                    float(cd["rest_length"]),
                ))

        if not edge_list:
            return [Vector(list(v)) for v in vertices]

        for _ in range(500):
            forces = np.zeros_like(pos)
            for i1, i2, target in edge_list:
                diff = pos[i2] - pos[i1]
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    continue
                error = dist - target
                direction = diff / dist
                f = direction * error * 0.5
                forces[i1] += f
                forces[i2] -= f

            forces[anchor_mask] = 0.0
            pos += forces * 0.05

        return [Vector(p.tolist()) for p in pos]

    # ── Geometry helpers ─────────────────────────────────────────────────

    @staticmethod
    def _load_landmarks():
        if "LandmarkMesh" not in bpy.data.objects:
            return None, [], {}
        lm = bpy.data.objects["LandmarkMesh"]
        labels = get_landmark_labels(lm)

        label_to_indices = defaultdict(list)
        for i, lbl in enumerate(labels):
            if lbl:
                label_to_indices[lbl].append(i)

        return lm, labels, label_to_indices

    @staticmethod
    def _get_head_scale(head_mesh):
        bbox_min = Vector(
            [min(v.co[i] for v in head_mesh.data.vertices) for i in range(3)]
        )
        bbox_max = Vector(
            [max(v.co[i] for v in head_mesh.data.vertices) for i in range(3)]
        )
        dims = (
            (head_mesh.matrix_world @ bbox_max)
            - (head_mesh.matrix_world @ bbox_min)
        )
        hs = max(dims)
        return hs * 0.02, hs * 0.02 * 0.15

    @staticmethod
    def _snap_to_surface(position, head_mesh):
        bvh = BVHTree.FromObject(
            head_mesh, bpy.context.evaluated_depsgraph_get(),
        )
        location, normal, _index, _dist = bvh.find_nearest(Vector(position))
        if location is None:
            return Vector(position), Vector((0, 0, 1))
        return Vector(location), Vector(normal)

    @staticmethod
    def _create_optode_disc(position, normal, name, color, diameter, thickness):
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32, radius=diameter / 2.0, depth=thickness,
            enter_editmode=False, align='WORLD', location=(0, 0, 0),
        )
        optode = bpy.context.active_object
        optode.name = name
        rot_q = Vector((0, 0, 1)).rotation_difference(Vector(normal))
        optode.rotation_euler = rot_q.to_euler()
        optode.location = Vector(position)

        mat = bpy.data.materials.new(name=f"{name}_mat")
        mat.use_nodes = True
        mat.diffuse_color = color
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Metallic'].default_value = 0.3
            bsdf.inputs['Roughness'].default_value = 0.4
        optode.data.materials.append(mat)
        if bpy.app.version >= (4, 1, 0):
            bpy.ops.object.shade_smooth()
        else:
            for face in optode.data.polygons:
                face.use_smooth = True
        return optode


def _reset_connection_hooks(conn_obj):
    """Reset all Hook modifiers on conn_obj after optode positions change.

    Directly writing mod.matrix is the equivalent of bpy.ops.object.hook_reset
    but works from any context (not just edit mode).
    """
    conn_world = conn_obj.matrix_world
    n = 0
    for mod in conn_obj.modifiers:
        if mod.type == 'HOOK' and mod.object:
            mod.matrix = mod.object.matrix_world.inverted() @ conn_world
            n += 1
    if n:
        print(f"  Reset {n} hook modifier matrices on {conn_obj.name}")


def _compute_rest_shape_standalone(vertices, name_to_idx, connections, anchor_mask):
    n = len(vertices)
    pos = np.array([list(v) for v in vertices], dtype=np.float64)

    edge_list = []
    for cd in connections:
        n1, n2 = cd["optode1"], cd["optode2"]
        if n1 in name_to_idx and n2 in name_to_idx:
            edge_list.append((
                name_to_idx[n1], name_to_idx[n2],
                float(cd["rest_length"]),
            ))

    if not edge_list:
        return [Vector(list(v)) for v in vertices]

    for _ in range(500):
        forces = np.zeros_like(pos)
        for i1, i2, target in edge_list:
            diff = pos[i2] - pos[i1]
            dist = np.linalg.norm(diff)
            if dist < 1e-8:
                continue
            error = dist - target
            direction = diff / dist
            f = direction * error * 0.5
            forces[i1] += f
            forces[i2] -= f

        forces[anchor_mask] = 0.0
        pos += forces * 0.05

    return [Vector(p.tolist()) for p in pos]


def run_cloth_relaxation(conn_obj, optode_objects, simulation_frames=100,
                         spring_stiffness=15.0, pin_stiffness=1.0,
                         report_callback=None):
    """
    Cloth-based relaxation using Blender's built-in cloth solver.
    Rest shape key encodes per-edge target lengths, anchors are pinned,
    gravity is off, and final positions are snapped to the head surface.
    """
    if "optode_names_ordered" not in conn_obj:
        if report_callback:
            report_callback({'ERROR'}, "Missing optode data")
        return None

    if "headmesh" not in bpy.data.objects:
        if report_callback:
            report_callback({'ERROR'}, "No 'headmesh' found in scene")
        return None

    optode_names = list(conn_obj["optode_names_ordered"])
    spring_states = dict(conn_obj.get("spring_states", {}))
    n_verts = len(optode_names)

    print(f"\n{'='*70}")
    print(f"CLOTH RELAXATION")
    print(f"{'='*70}")
    print(f"Optodes: {n_verts}   Springs: {len(spring_states)}")
    print(f"Stiffness: {spring_stiffness}   Pin: {pin_stiffness}   "
          f"Frames: {simulation_frames}")

    is_anchor = np.zeros(n_verts, dtype=bool)
    positions = np.zeros((n_verts, 3), dtype=np.float64)

    scene_anchor_names: set = set()
    anchor_coll = bpy.data.collections.get("Anchor Indicators")
    if anchor_coll:
        optode_name_set = set(optode_names)
        for indicator in anchor_coll.objects:
            if indicator.parent and indicator.parent.name in optode_name_set:
                scene_anchor_names.add(indicator.parent.name)
                indicator.parent["is_anchor"] = 1
                if indicator.parent.get("anchor_pos") is None:
                    indicator.parent["anchor_pos"] = list(indicator.parent.location)
        print(f"Anchor Indicators: {len(scene_anchor_names)} anchor(s): "
              f"{sorted(scene_anchor_names)}")

    for i, name in enumerate(optode_names):
        if name not in optode_objects:
            if report_callback:
                report_callback({'ERROR'}, f"Optode '{name}' not found")
            return None
        optode = optode_objects[name]
        is_anchor[i] = (optode.get("is_anchor", 0) == 1) or (name in scene_anchor_names)

        if is_anchor[i]:
            stored = optode.get("anchor_pos")
            if stored is not None:
                pos = np.array(list(stored), dtype=np.float64)
                optode.location = Vector(pos)
            else:
                pos = np.array(optode.location, dtype=np.float64)
                optode["anchor_pos"] = list(pos)
            positions[i] = pos
        else:
            positions[i] = np.array(optode.location, dtype=np.float64)

    initial_positions = positions.copy()
    anchor_mask = is_anchor.copy()
    name_to_idx = {n: i for i, n in enumerate(optode_names)}

    print(f"Anchors: {is_anchor.sum()}   Free: {(~is_anchor).sum()}")

    for mod in list(conn_obj.modifiers):
        if mod.type == 'HOOK':
            conn_obj.modifiers.remove(mod)

    conn_obj.vertex_groups.clear()

    bpy.context.view_layer.objects.active = conn_obj
    if conn_obj.data.shape_keys:
        bpy.ops.object.shape_key_remove(all=True)

    edges = []
    for edge in conn_obj.data.edges:
        idx1, idx2 = edge.vertices[:]
        if idx1 < n_verts and idx2 < n_verts:
            edges.append((idx1, idx2))

    connections_stiff = []
    vertex_pull_sum = np.zeros(n_verts, dtype=np.float64)
    vertex_edge_count = np.zeros(n_verts, dtype=np.float64)
    vertex_has_fixed = np.zeros(n_verts, dtype=bool)
    n_stiff = 0
    n_flexible = 0

    for key, ss in spring_states.items():
        n1, n2 = ss['optode1'], ss['optode2']
        if n1 in name_to_idx and n2 in name_to_idx:
            i1, i2 = name_to_idx[n1], name_to_idx[n2]
            pull = float(ss.get('pull', 0.9))
            vertex_pull_sum[i1] += pull
            vertex_pull_sum[i2] += pull
            vertex_edge_count[i1] += 1
            vertex_edge_count[i2] += 1

            if ss.get('fix_distance', False):
                vertex_has_fixed[i1] = True
                vertex_has_fixed[i2] = True

            if not ss.get('is_flexible', False):
                n_stiff += 1
                connections_stiff.append({
                    'optode1': n1, 'optode2': n2,
                    'rest_length': float(ss['rest_length']),
                })
            else:
                n_flexible += 1

    print(f"Springs — Stiff: {n_stiff}   Flexible: {n_flexible}")

    if not edges:
        if report_callback:
            report_callback({'ERROR'}, "No edges found on connection mesh")
        return None

    neighbors = defaultdict(set)
    for i1, i2 in edges:
        neighbors[i1].add(i2)
        neighbors[i2].add(i1)

    tri_faces = []
    seen_tris = set()
    for i1 in range(n_verts):
        for i2 in neighbors[i1]:
            if i2 <= i1:
                continue
            for i3 in (neighbors[i1] & neighbors[i2]):
                if i3 <= i2:
                    continue
                tri = (i1, i2, i3)
                if tri not in seen_tris:
                    seen_tris.add(tri)
                    tri_faces.append(tri)

    face_edge_set = set()
    for f in tri_faces:
        for j in range(3):
            a, b = f[j], f[(j + 1) % 3]
            face_edge_set.add((min(a, b), max(a, b)))
    extra_edges = [(a, b) for a, b in edges
                   if (min(a, b), max(a, b)) not in face_edge_set]

    vertices_vec = [Vector(positions[i].tolist()) for i in range(n_verts)]
    conn_obj.data.clear_geometry()
    conn_obj.data.from_pydata(vertices_vec, extra_edges, tri_faces)
    conn_obj.data.update()

    print(f"Mesh rebuilt: {n_verts} verts, {len(edges)} edges, "
          f"{len(tri_faces)} triangle faces")

    conn_obj.shape_key_add(name="Basis", from_mix=False)

    rest_positions = _compute_rest_shape_standalone(
        vertices_vec, name_to_idx, connections_stiff, anchor_mask,
    )
    rest_key = conn_obj.shape_key_add(name="Rest", from_mix=False)
    for i, rpos in enumerate(rest_positions):
        rest_key.data[i].co = rpos

    pin_group = conn_obj.vertex_groups.new(name="Pins")
    for i in range(n_verts):
        pin_group.add([i], 1.0 if is_anchor[i] else 0.0, 'REPLACE')

    n_anchors = int(anchor_mask.sum())
    print(f"Pins: {n_anchors} anchors pinned, {n_verts - n_anchors} free")

    vertex_edge_count[vertex_edge_count == 0] = 1
    vertex_weights = np.clip(vertex_pull_sum / vertex_edge_count / 0.9,
                             0.01, 1.0)
    vertex_weights[vertex_has_fixed] = 1.0

    stiffness_group = conn_obj.vertex_groups.new(name="Stiffness")
    for i in range(n_verts):
        stiffness_group.add([i], float(vertex_weights[i]), 'REPLACE')

    n_fixed = int(vertex_has_fixed.sum())
    print(f"Stiffness weights: min={vertex_weights.min():.3f} "
          f"max={vertex_weights.max():.3f} mean={vertex_weights.mean():.3f} "
          f"({n_fixed} vertices locked by fix_distance)")

    cloth_mod = conn_obj.modifiers.new(name="Cloth", type='CLOTH')
    settings = cloth_mod.settings

    settings.rest_shape_key = rest_key
    settings.vertex_group_mass = "Pins"
    settings.pin_stiffness = pin_stiffness

    settings.tension_stiffness = spring_stiffness
    settings.compression_stiffness = spring_stiffness
    settings.vertex_group_structural_stiffness = "Stiffness"
    settings.shear_stiffness = 0.0
    settings.bending_stiffness = 0.0
    settings.tension_damping = 5.0
    settings.compression_damping = 5.0

    settings.mass = 0.3
    settings.air_damping = 1.0
    settings.quality = 5
    settings.time_scale = 1.0

    scene = bpy.context.scene
    orig_frame = scene.frame_current
    orig_start = scene.frame_start
    orig_end = scene.frame_end
    orig_gravity = scene.use_gravity

    scene.use_gravity = False
    scene.frame_start = 1
    scene.frame_end = simulation_frames

    bpy.ops.object.select_all(action='DESELECT')
    conn_obj.select_set(True)
    bpy.context.view_layer.objects.active = conn_obj

    print(f"\nBaking cloth simulation ({simulation_frames} frames, "
          f"stiffness={spring_stiffness}, pin={pin_stiffness}, gravity OFF)...")

    for frame in range(1, simulation_frames + 1):
        scene.frame_set(frame)
        if frame % 20 == 0:
            print(f"  Frame {frame}/{simulation_frames}")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = conn_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()

    final_positions = []
    for vert in eval_mesh.vertices:
        final_positions.append(Vector(conn_obj.matrix_world @ vert.co))
    eval_obj.to_mesh_clear()

    scene.use_gravity = orig_gravity
    scene.frame_start = orig_start
    scene.frame_end = orig_end
    scene.frame_set(orig_frame)

    conn_obj.modifiers.remove(cloth_mod)
    bpy.context.view_layer.objects.active = conn_obj
    if conn_obj.data.shape_keys:
        bpy.ops.object.shape_key_remove(all=True)

    headmesh = bpy.data.objects["headmesh"]
    head_bvh = BVHTree.FromObject(headmesh, bpy.context.evaluated_depsgraph_get())
    conn_inv = conn_obj.matrix_world.inverted()

    for i, name in enumerate(optode_names):
        loc, normal, _, _ = head_bvh.find_nearest(final_positions[i])
        if loc is not None:
            final_positions[i] = Vector(loc)
        optode_objects[name].location = final_positions[i]
        conn_obj.data.vertices[i].co = conn_inv @ final_positions[i]
    conn_obj.data.update()

    bpy.ops.object.select_all(action='DESELECT')
    conn_obj.select_set(True)
    bpy.context.view_layer.objects.active = conn_obj

    for i, name in enumerate(optode_names):
        vg = conn_obj.vertex_groups.new(name=f"VG_{name}")
        vg.add([i], 1.0, 'REPLACE')

    for i, name in enumerate(optode_names):
        optode = optode_objects[name]
        bpy.ops.object.mode_set(mode='OBJECT')
        hm = conn_obj.modifiers.new(name=f"Hook_{name}", type='HOOK')
        hm.object = optode
        hm.vertex_group = f"VG_{name}"
        for v in conn_obj.data.vertices:
            v.select = False
        conn_obj.data.vertices[i].select = True
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.object.hook_assign(modifier=hm.name)
        bpy.ops.object.hook_reset(modifier=hm.name)
        bpy.ops.object.hook_recenter(modifier=hm.name)

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()

    final_np = np.array([list(p) for p in final_positions])
    displacements = np.linalg.norm(final_np - initial_positions, axis=1)

    anchor_indices = np.where(is_anchor)[0].tolist()
    free_indices = np.where(~is_anchor)[0].tolist()

    def _stats(indices, label):
        if not indices:
            print(f"  {label:40s}  N=0  (none)")
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'count': 0}
        d = displacements[np.array(indices)]
        print(f"  {label:40s}  N={len(indices):2d}"
              f"  min={d.min():.4f}  max={d.max():.4f}  mean={d.mean():.4f}")
        return {'min': float(d.min()), 'max': float(d.max()),
                'mean': float(d.mean()), 'count': len(indices)}

    print(f"\n{'='*60}")
    print(f"  DISPLACEMENT SUMMARY  (Blender cloth simulation)")
    print(f"{'='*60}")
    a_s = _stats(anchor_indices, "Anchors   (should be ~0.0)")
    f_s = _stats(free_indices,   "Free      (cloth-relaxed)")
    print(f"{'='*60}\n")

    return {
        'anchor_displacement':      a_s['max'],
        'anchor_displacement_min':  a_s['min'],
        'anchor_displacement_mean': a_s['mean'],
        'anchor_count':             a_s['count'],
        'free_displacement_max':    f_s['max'],
        'free_displacement_min':    f_s['min'],
        'free_displacement_mean':   f_s['mean'],
        'free_count':               f_s['count'],
    }


class NEUROCAPTAIN_OT_relax_probe_manual(bpy.types.Operator):
    """Re-run cloth relaxation on the current probe using Blender's cloth solver"""
    bl_idname = "neurocaptain.relax_probe_manual"
    bl_label = "Relax Probe (Cloth)"
    bl_options = {'REGISTER', 'UNDO'}

    simulation_frames: bpy.props.IntProperty(
        name="Simulation Frames",
        description="Number of frames to run the cloth simulation",
        default=100,
        min=10,
        max=500,
    )

    spring_stiffness: bpy.props.FloatProperty(
        name="Spring Stiffness",
        description="Structural spring stiffness (tension and compression)",
        default=15.0,
        min=0.1,
        max=100.0,
    )

    pin_stiffness: bpy.props.FloatProperty(
        name="Pin Stiffness",
        description="How strongly pinned (anchor) vertices resist movement",
        default=1.0,
        min=0.01,
        max=10.0,
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=350)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "simulation_frames")
        layout.prop(self, "spring_stiffness")
        layout.prop(self, "pin_stiffness")

    def execute(self, context):
        if "Optode_Connections" not in bpy.data.objects:
            self.report({'ERROR'}, "No 'Optode_Connections' object found. Create connections first.")
            return {'CANCELLED'}

        conn_obj = bpy.data.objects["Optode_Connections"]

        if "optode_names_ordered" in conn_obj:
            optode_names = list(conn_obj["optode_names_ordered"])
        elif "optode_names" in conn_obj:
            optode_names = list(conn_obj["optode_names"])
            conn_obj["optode_names_ordered"] = optode_names
        else:
            self.report({'ERROR'}, "Optode_Connections missing optode name data.")
            return {'CANCELLED'}

        if not conn_obj.get("spring_states"):
            self.report({'ERROR'},
                "No spring properties found. Define springs via "
                "'Set Flexible Spring Properties' or 'Make Springs Stiff' first.")
            return {'CANCELLED'}

        optode_objects = {}
        missing = []
        for name in optode_names:
            if name in bpy.data.objects:
                optode_objects[name] = bpy.data.objects[name]
            else:
                missing.append(name)

        if missing:
            self.report({'ERROR'},
                f"Optodes not found: {', '.join(missing[:5])}"
                f"{'…' if len(missing) > 5 else ''}")
            return {'CANCELLED'}

        try:
            stats = run_cloth_relaxation(
                conn_obj=conn_obj,
                optode_objects=optode_objects,
                simulation_frames=self.simulation_frames,
                spring_stiffness=self.spring_stiffness,
                pin_stiffness=self.pin_stiffness,
                report_callback=self.report,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Cloth relaxation error: {e}")
            return {'CANCELLED'}

        if not stats:
            self.report({'ERROR'}, "Cloth relaxation failed — check console")
            return {'CANCELLED'}

        self.report({'INFO'},
            f"Cloth relaxation done — "
            f"Anchor max={stats['anchor_displacement']:.4f}, "
            f"Free min={stats['free_displacement_min']:.4f} "
            f"max={stats['free_displacement_max']:.4f} "
            f"mean={stats['free_displacement_mean']:.4f}")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(NEUROCAPTAIN_OT_import_optode_json_blender_goal)
    bpy.utils.register_class(NEUROCAPTAIN_OT_relax_probe_manual)


def unregister():
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_relax_probe_manual)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_import_optode_json_blender_goal)
