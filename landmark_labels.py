import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


# ═══════════════════════════════════════════════════════════════════════
#  Canonical landmark label lists — brain1020 output order
#
#  Each list follows the struct-field order from brain1020.m:
#    reference → cm → sm → aal → apl → aar → apr →
#    cal_1,car_1 → cal_2,car_2 → … →
#    cpl_1,cpr_1 → cpl_2,cpr_2 → … →
#    paal → papl → paar → papr  (baseplane, perc2 ≤ 10 only)
#
#  Within each group, points go from the midpoint of the arc outward
#  toward the endpoint (e.g. cm goes T7→…→Cz→…→T8, L→R).
#
#  Cz appears 3 times: reference[4], cm midpoint, sm midpoint.
# ═══════════════════════════════════════════════════════════════════════

# 10-20 system  (p1=10, p2=20) — 27 total, 25 unique
LANDMARK_LABELS_1020 = [
    # reference (5)
    "Nz", "Iz", "Lpa", "Rpa", "Cz",
    # cm: coronal medial [10:20:90] (5)
    "T3", "C3", "Cz", "C4", "T4",
    # sm: sagittal medial [10:20:90] (5)
    "Fpz", "Fz", "Cz", "Pz", "Oz",
    # aal: anterior axial left, step=40 (2)
    "F7", "Fp1",
    # apl: posterior axial left, step=40 (2)
    "T5", "O1",
    # aar: anterior axial right, step=40 (2)
    "F8", "Fp2",
    # apr: posterior axial right, step=40 (2)
    "T6", "O2",
    # cal_1: anterior coronal left, step=50 (1)
    "F3",
    # car_1: anterior coronal right, step=50 (1)
    "F4",
    # cpl_1: posterior coronal left, step=50 (1)
    "P3",
    # cpr_1: posterior coronal right, step=50 (1)
    "P4",
]

# 10-10 system  (p1=10, p2=10) — 83 total, 80 unique
LANDMARK_LABELS_1010 = [
    # reference (5)
    "Nz", "Iz", "Lpa", "Rpa", "Cz",
    # cm: coronal medial [10:10:90] (9)
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    # sm: sagittal medial [10:10:90] (9)
    "Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz",
    # aal: anterior axial left, step=20 (4)
    "FT7", "F7", "AF7", "Fp1",
    # apl: posterior axial left, step=20 (4)
    "TP7", "P7", "PO7", "O1",
    # aar: anterior axial right, step=20 (4)
    "FT8", "F8", "AF8", "Fp2",
    # apr: posterior axial right, step=20 (4)
    "TP8", "P8", "PO8", "O2",
    # cal_1: anterior coronal left 1, step=25 (3)
    "FC1", "FC3", "FC5",
    # car_1: anterior coronal right 1, step=25 (3)
    "FC2", "FC4", "FC6",
    # cal_2: anterior coronal left 2, step=25 (3)
    "F1", "F3", "F5",
    # car_2: anterior coronal right 2, step=25 (3)
    "F2", "F4", "F6",
    # cal_3: anterior coronal left 3, step=50 (1)
    "AF3",
    # car_3: anterior coronal right 3, step=50 (1)
    "AF4",
    # cpl_1: posterior coronal left 1, step=25 (3)
    "CP1", "CP3", "CP5",
    # cpr_1: posterior coronal right 1, step=25 (3)
    "CP2", "CP4", "CP6",
    # cpl_2: posterior coronal left 2, step=25 (3)
    "P1", "P3", "P5",
    # cpr_2: posterior coronal right 2, step=25 (3)
    "P2", "P4", "P6",
    # cpl_3: posterior coronal left 3, step=50 (1)
    "PO3",
    # cpr_3: posterior coronal right 3, step=50 (1)
    "PO4",
    # paal: principal axial anterior left, step=20 (4)
    "FT9", "F9", "", "",
    # papl: principal axial posterior left, step=20 (4)
    "TP9", "P9", "PO9", "O9",
    # paar: principal axial anterior right, step=20 (4)
    "FT10", "F10", "", "",
    # papr: principal axial posterior right, step=20 (4)
    "TP10", "P10", "PO10", "O10",
]

# 10-5 system  (p1=10, p2=5) — 291 total
# Standard 10-5 names for cm/sm; group-positional for coronal/axial cuts.
LANDMARK_LABELS_105 = [
    # reference (5)
    "Nz", "Iz", "Lpa", "Rpa", "Cz",
    # cm: coronal medial [10:5:90] (17)
    "T7", "T7h", "C5", "C5h", "C3", "C3h", "C1", "C1h",
    "Cz",
    "C2h", "C2", "C4h", "C4", "C6h", "C6", "T8h", "T8",
    # sm: sagittal medial [10:5:90] (17)
    "Fpz", "AFpz", "AFz", "AFFz", "Fz", "FFCz", "FCz", "FCCz",
    "Cz",
    "CCPz", "CPz", "CPPz", "Pz", "PPOz", "POz", "POOz", "Oz",
    # aal: anterior axial left, step=10 (9)
    "aal_0", "aal_1", "aal_2", "aal_3", "aal_4",
    "aal_5", "aal_6", "aal_7", "aal_8",
    # apl: posterior axial left, step=10 (9)
    "apl_0", "apl_1", "apl_2", "apl_3", "apl_4",
    "apl_5", "apl_6", "apl_7", "apl_8",
    # aar: anterior axial right, step=10 (9)
    "aar_0", "aar_1", "aar_2", "aar_3", "aar_4",
    "aar_5", "aar_6", "aar_7", "aar_8",
    # apr: posterior axial right, step=10 (9)
    "apr_0", "apr_1", "apr_2", "apr_3", "apr_4",
    "apr_5", "apr_6", "apr_7", "apr_8",
    # cal_1: anterior coronal left 1, step=12.5 (7)
    "cal_1_0", "cal_1_1", "cal_1_2", "cal_1_3",
    "cal_1_4", "cal_1_5", "cal_1_6",
    # car_1: anterior coronal right 1, step=12.5 (7)
    "car_1_0", "car_1_1", "car_1_2", "car_1_3",
    "car_1_4", "car_1_5", "car_1_6",
    # cal_2 (7)
    "cal_2_0", "cal_2_1", "cal_2_2", "cal_2_3",
    "cal_2_4", "cal_2_5", "cal_2_6",
    # car_2 (7)
    "car_2_0", "car_2_1", "car_2_2", "car_2_3",
    "car_2_4", "car_2_5", "car_2_6",
    # cal_3 (7)
    "cal_3_0", "cal_3_1", "cal_3_2", "cal_3_3",
    "cal_3_4", "cal_3_5", "cal_3_6",
    # car_3 (7)
    "car_3_0", "car_3_1", "car_3_2", "car_3_3",
    "car_3_4", "car_3_5", "car_3_6",
    # cal_4 (7)
    "cal_4_0", "cal_4_1", "cal_4_2", "cal_4_3",
    "cal_4_4", "cal_4_5", "cal_4_6",
    # car_4 (7)
    "car_4_0", "car_4_1", "car_4_2", "car_4_3",
    "car_4_4", "car_4_5", "car_4_6",
    # cal_5 (7)
    "cal_5_0", "cal_5_1", "cal_5_2", "cal_5_3",
    "cal_5_4", "cal_5_5", "cal_5_6",
    # car_5 (7)
    "car_5_0", "car_5_1", "car_5_2", "car_5_3",
    "car_5_4", "car_5_5", "car_5_6",
    # cal_6 (7)
    "cal_6_0", "cal_6_1", "cal_6_2", "cal_6_3",
    "cal_6_4", "cal_6_5", "cal_6_6",
    # car_6 (7)
    "car_6_0", "car_6_1", "car_6_2", "car_6_3",
    "car_6_4", "car_6_5", "car_6_6",
    # cal_7: anterior coronal left 7, step=25 (3)
    "cal_7_0", "cal_7_1", "cal_7_2",
    # car_7: anterior coronal right 7, step=25 (3)
    "car_7_0", "car_7_1", "car_7_2",
    # cpl_1: posterior coronal left 1, step=12.5 (7)
    "cpl_1_0", "cpl_1_1", "cpl_1_2", "cpl_1_3",
    "cpl_1_4", "cpl_1_5", "cpl_1_6",
    # cpr_1 (7)
    "cpr_1_0", "cpr_1_1", "cpr_1_2", "cpr_1_3",
    "cpr_1_4", "cpr_1_5", "cpr_1_6",
    # cpl_2 (7)
    "cpl_2_0", "cpl_2_1", "cpl_2_2", "cpl_2_3",
    "cpl_2_4", "cpl_2_5", "cpl_2_6",
    # cpr_2 (7)
    "cpr_2_0", "cpr_2_1", "cpr_2_2", "cpr_2_3",
    "cpr_2_4", "cpr_2_5", "cpr_2_6",
    # cpl_3 (7)
    "cpl_3_0", "cpl_3_1", "cpl_3_2", "cpl_3_3",
    "cpl_3_4", "cpl_3_5", "cpl_3_6",
    # cpr_3 (7)
    "cpr_3_0", "cpr_3_1", "cpr_3_2", "cpr_3_3",
    "cpr_3_4", "cpr_3_5", "cpr_3_6",
    # cpl_4 (7)
    "cpl_4_0", "cpl_4_1", "cpl_4_2", "cpl_4_3",
    "cpl_4_4", "cpl_4_5", "cpl_4_6",
    # cpr_4 (7)
    "cpr_4_0", "cpr_4_1", "cpr_4_2", "cpr_4_3",
    "cpr_4_4", "cpr_4_5", "cpr_4_6",
    # cpl_5 (7)
    "cpl_5_0", "cpl_5_1", "cpl_5_2", "cpl_5_3",
    "cpl_5_4", "cpl_5_5", "cpl_5_6",
    # cpr_5 (7)
    "cpr_5_0", "cpr_5_1", "cpr_5_2", "cpr_5_3",
    "cpr_5_4", "cpr_5_5", "cpr_5_6",
    # cpl_6 (7)
    "cpl_6_0", "cpl_6_1", "cpl_6_2", "cpl_6_3",
    "cpl_6_4", "cpl_6_5", "cpl_6_6",
    # cpr_6 (7)
    "cpr_6_0", "cpr_6_1", "cpr_6_2", "cpr_6_3",
    "cpr_6_4", "cpr_6_5", "cpr_6_6",
    # cpl_7: posterior coronal left 7, step=25 (3)
    "cpl_7_0", "cpl_7_1", "cpl_7_2",
    # cpr_7: posterior coronal right 7, step=25 (3)
    "cpr_7_0", "cpr_7_1", "cpr_7_2",
    # paal: principal axial anterior left, step=10 (9)
    "paal_0", "paal_1", "paal_2", "paal_3", "paal_4",
    "paal_5", "paal_6", "paal_7", "paal_8",
    # papl: principal axial posterior left, step=10 (9)
    "papl_0", "papl_1", "papl_2", "papl_3", "papl_4",
    "papl_5", "papl_6", "papl_7", "papl_8",
    # paar: principal axial anterior right, step=10 (9)
    "paar_0", "paar_1", "paar_2", "paar_3", "paar_4",
    "paar_5", "paar_6", "paar_7", "paar_8",
    # papr: principal axial posterior right, step=10 (9)
    "papr_0", "papr_1", "papr_2", "papr_3", "papr_4",
    "papr_5", "papr_6", "papr_7", "papr_8",
]


# ═══════════════════════════════════════════════════════════════════════
#  brain1020 group key → standard names mapping
#
#  Used by brain1020mesh.py to assign proper per-vertex labels at
#  generation time.  Keyed by (group_name, group_size).
# ═══════════════════════════════════════════════════════════════════════

GROUP_TO_STANDARD = {
    # Reference (single point each)
    "nz":  {1: ["Nz"]},
    "iz":  {1: ["Iz"]},
    "lpa": {1: ["Lpa"]},
    "rpa": {1: ["Rpa"]},
    "cz":  {1: ["Cz"]},
    # Coronal medial — 10-20 / 10-10 / 10-5
    "cm": {
        5:  ["T3", "C3", "Cz", "C4", "T4"],
        9:  ["T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8"],
        17: ["T7", "T7h", "C5", "C5h", "C3", "C3h", "C1", "C1h", "Cz",
             "C2h", "C2", "C4h", "C4", "C6h", "C6", "T8h", "T8"],
    },
    # Sagittal medial
    "sm": {
        5:  ["Fpz", "Fz", "Cz", "Pz", "Oz"],
        9:  ["Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz"],
        17: ["Fpz", "AFpz", "AFz", "AFFz", "Fz", "FFCz", "FCz", "FCCz", "Cz",
             "CCPz", "CPz", "CPPz", "Pz", "PPOz", "POz", "POOz", "Oz"],
    },
    # Axial arcs — 10-20 (2 pts) / 10-10 (4 pts)
    "aal": {2: ["F7", "Fp1"],  4: ["FT7", "F7", "AF7", "Fp1"]},
    "apl": {2: ["T5", "O1"],   4: ["TP7", "P7", "PO7", "O1"]},
    "aar": {2: ["F8", "Fp2"],  4: ["FT8", "F8", "AF8", "Fp2"]},
    "apr": {2: ["T6", "O2"],   4: ["TP8", "P8", "PO8", "O2"]},
    # Anterior coronal cuts
    "cal_1": {1: ["F3"],  3: ["FC1", "FC3", "FC5"]},
    "car_1": {1: ["F4"],  3: ["FC2", "FC4", "FC6"]},
    "cal_2": {3: ["F1", "F3", "F5"]},
    "car_2": {3: ["F2", "F4", "F6"]},
    "cal_3": {1: ["AF3"], 3: ["", "AF3", ""]},
    "car_3": {1: ["AF4"], 3: ["", "AF4", ""]},
    # Posterior coronal cuts
    "cpl_1": {1: ["P3"],  3: ["CP1", "CP3", "CP5"]},
    "cpr_1": {1: ["P4"],  3: ["CP2", "CP4", "CP6"]},
    "cpl_2": {3: ["P1", "P3", "P5"]},
    "cpr_2": {3: ["P2", "P4", "P6"]},
    "cpl_3": {1: ["PO3"], 3: ["", "PO3", ""]},
    "cpr_3": {1: ["PO4"], 3: ["", "PO4", ""]},
    # Baseplane
    "paal": {4: ["FT9", "F9", "", ""]},
    "papl": {4: ["TP9", "P9", "PO9", "O9"]},
    "paar": {4: ["FT10", "F10", "", ""]},
    "papr": {4: ["TP10", "P10", "PO10", "O10"]},
}


def get_landmark_labels(landmark_mesh_obj):
    """
    Get landmark labels for a LandmarkMesh object.

    Priority:
      1. Stored labels on the mesh (if count matches vertex count)
      2. Hardcoded fallback by vertex count (10-5, 10-10, 10-20)

    Returns list of label strings (empty string for unlabeled vertices).
    Always returns exactly num_verts entries.
    """
    num_verts = len(landmark_mesh_obj.data.vertices)
    stored = landmark_mesh_obj.get("landmark_labels", None)

    if stored is not None:
        stored_list = list(stored)
        if len(stored_list) == num_verts:
            return stored_list

    # Exact match only — applying wrong labels causes barycentric
    # reconstruction to place optodes at completely wrong positions
    exact_matches = {
        len(LANDMARK_LABELS_1020): (LANDMARK_LABELS_1020, "10-20"),
        67: (LANDMARK_LABELS_1010[:67], "10-10 (no baseplane)"),
        len(LANDMARK_LABELS_1010): (LANDMARK_LABELS_1010, "10-10"),
        len(LANDMARK_LABELS_105): (LANDMARK_LABELS_105, "10-5"),
    }

    if num_verts in exact_matches:
        labels_ref, system_name = exact_matches[num_verts]
        return list(labels_ref)

    return [f"v{i}" for i in range(num_verts)]


def label_material():
    """material for landmark labels"""
    mat_name = "Landmark_Label_Material"
    label_color = (0.1, 0.3, 0.9, 1.0)

    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    mat.diffuse_color = label_color

    nodes = mat.node_tree.nodes
    nodes.clear()

    node_emission = nodes.new(type='ShaderNodeEmission')
    node_emission.inputs[0].default_value = label_color
    node_emission.inputs[1].default_value = 2.0

    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(node_emission.outputs[0], node_output.inputs[0])

    return mat


class NEUROCAPTAIN_OT_display_landmark_labels(bpy.types.Operator):
    """Display 10-10/10-20 landmark labels in 3D viewport"""
    bl_idname = "neurocaptain.display_landmark_labels"
    bl_label = "Display Landmark Labels"
    bl_options = {'REGISTER', 'UNDO'}

    label_size: bpy.props.FloatProperty(
        name="Label Size",
        default=3.0,
        min=0.5,
        max=20.0
    )

    offset_distance: bpy.props.FloatProperty(
        name="Offset Distance",
        default=1.0,
        min=-20.0,
        max=20.0
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        if "LandmarkMesh" not in bpy.data.objects:
            self.report({'ERROR'}, "LandmarkMesh not found")
            return {'CANCELLED'}

        headmesh_obj = bpy.data.objects.get("headmesh")
        if not headmesh_obj:
            self.report({'ERROR'}, "headmesh not found")
            return {'CANCELLED'}

        landmark_mesh_obj = bpy.data.objects["LandmarkMesh"]

        # Use validated label lookup
        labels = get_landmark_labels(landmark_mesh_obj)

        if "Landmark_Labels" in bpy.data.collections:
            label_collection = bpy.data.collections["Landmark_Labels"]
            for obj in list(label_collection.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
        else:
            label_collection = bpy.data.collections.new("Landmark_Labels")
            context.scene.collection.children.link(label_collection)

        label_mat = label_material()
        bvh = BVHTree.FromObject(headmesh_obj, context.evaluated_depsgraph_get())

        created_labels = 0
        for idx, vertex in enumerate(landmark_mesh_obj.data.vertices):
            if idx >= len(labels) or not labels[idx]:
                continue

            vertex_world_pos = landmark_mesh_obj.matrix_world @ vertex.co
            location, normal, _, _ = bvh.find_nearest(vertex_world_pos)

            surface_normal = normal if location else (landmark_mesh_obj.matrix_world.to_3x3() @ vertex.normal)
            surface_normal.normalize()
            label_position = vertex_world_pos + surface_normal * self.offset_distance

            text_data = bpy.data.curves.new(name=f"Label_{labels[idx]}_{idx}", type='FONT')
            text_data.body = labels[idx]
            text_data.size = self.label_size
            text_data.align_x = 'CENTER'
            text_data.align_y = 'CENTER'
            text_data.extrude = 0.1
            text_data.bevel_depth = 0.02

            text_obj = bpy.data.objects.new(name=f"Label_{labels[idx]}_{idx}", object_data=text_data)
            text_obj.location = label_position
            text_obj.rotation_mode = 'QUATERNION'
            text_obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(surface_normal)

            text_obj.data.materials.append(label_mat)
            label_collection.objects.link(text_obj)

            text_obj["is_landmark_label"] = True
            text_obj["landmark_index"] = idx
            created_labels += 1

        context.scene.neurocaptain.show_landmark_labels = True

        self.report({'INFO'}, f"Created {created_labels} landmark labels")
        return {'FINISHED'}


class NEUROCAPTAIN_OT_toggle_landmark_labels(bpy.types.Operator):
    """Toggle visibility of landmark labels"""
    bl_idname = "neurocaptain.toggle_landmark_labels"
    bl_label = "Toggle Landmark Labels"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.neurocaptain

        if "Landmark_Labels" not in bpy.data.collections:
            self.report({'WARNING'}, "No labels found. Generate them first.")
            props.show_landmark_labels = False
            return {'CANCELLED'}

        label_collection = bpy.data.collections["Landmark_Labels"]

        for obj in label_collection.objects:
            obj.hide_viewport = not props.show_landmark_labels
            obj.hide_render = not props.show_landmark_labels

        status = "visible" if props.show_landmark_labels else "hidden"
        self.report({'INFO'}, f"Labels {status}")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(NEUROCAPTAIN_OT_display_landmark_labels)
    bpy.utils.register_class(NEUROCAPTAIN_OT_toggle_landmark_labels)


def unregister():
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_toggle_landmark_labels)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_display_landmark_labels)
