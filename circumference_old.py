import bpy
from bpy.types import Operator
from bpy.props import EnumProperty, StringProperty, CollectionProperty
import bmesh


class circumference_calc(Operator):
    bl_label = "Calculate circumference of cap"
    bl_idname = "neurocaptain.circumference"
    bl_description = "Approximate the circumference of the cap model"

    def execute(self, context):
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass
        cap = bpy.context.scene.objects["headmesh"]
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = cap
        cap.select_set(True)
        bpy.ops.object.duplicate_move(
            OBJECT_OT_duplicate={"linked": False, "mode": "TRANSLATION"},
            TRANSFORM_OT_translate={
                "value": (0.212906, 0.0140968, 0.0237914),
                "orient_axis_ortho": "X",
                "orient_type": "GLOBAL",
                "orient_matrix": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                "orient_matrix_type": "GLOBAL",
                "constraint_axis": (False, False, False),
                "mirror": False,
                "use_proportional_edit": False,
                "proportional_edit_falloff": "SMOOTH",
                "proportional_size": 1,
                "use_proportional_connected": False,
                "use_proportional_projected": False,
                "snap": False,
                "snap_elements": {"INCREMENT"},
                "use_snap_project": False,
                "snap_target": "CLOSEST",
                "use_snap_self": True,
                "use_snap_edit": True,
                "use_snap_nonedit": True,
                "use_snap_selectable": False,
                "snap_point": (0, 0, 0),
                "snap_align": False,
                "snap_normal": (0, 0, 0),
                "gpencil_strokes": False,
                "cursor_transform": False,
                "texture_space": False,
                "remove_on_cancel": False,
                "view2d_edge_pan": False,
                "release_confirm": False,
                "use_accurate": False,
                "use_automerge_and_split": False,
            },
        )
        cap_dup = bpy.context.object
        cap_dup.name = "cap_copy"
        self.reference_point(context=context)
        self.place_cube(context=context)
        return {"FINISHED"}

    @staticmethod
    def get_lengths():
        lengths = []
        ruler_data = bpy.data.grease_pencils["Annotations"].layers["RulerData3D"]
        frame = ruler_data.frames[0]
        for stroke in frame.strokes:
            p1, p2 = stroke.points[0], stroke.points[-1]
            length = (p1.co - p2.co).length
            lengths.append(length)
        return lengths

    @staticmethod
    def reference_point(context):
        for ob in bpy.context.scene.objects:
            if ob.type == "MESH":
                print(ob.name)

        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass
        # mode = bpy.context.active_object.mode
        # we need to switch from Edit mode to Object mode so the selection gets updated
        bpy.ops.object.select_all(action="DESELECT")
        obj = bpy.data.objects["headmesh.001"]
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        # breakpoint()
        # try:
        selectedVerts = [v for v in obj.data.vertices if v.select]
        global vselect
        vselect = []
        for n in range(len(selectedVerts)):
            vert = selectedVerts[n].co
            v_global = obj.matrix_world @ vert

            vselect = v_global
            print("reference is ", v_global)

        # back to the previous mode
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass

        print("selected vertex is:", vselect)

        return vselect

    @staticmethod
    def place_cube(context):
        global co0_t, co1_t, l
        head = bpy.data.objects["headmesh.001"]

        bpy.ops.mesh.primitive_cube_add()
        obj2 = bpy.context.selected_objects[0]
        obj2.name = "capcut"

        # moving the bottom cube
        obj2.scale = (head.dimensions[0], head.dimensions[1], head.dimensions[0] / 2)
        obj2.location = (
            vselect[0],
            vselect[1] - (head.dimensions[1] / 3),
            vselect[2] - (head.dimensions[2] / 3.5),
        )
        zhead = head.dimensions[0]
        print("the zheaddimension is:", zhead)

        capcopy = bpy.data.objects["cap_copy"]
        bpy.ops.object.mode_set(mode="OBJECT")
        bool_one = capcopy.modifiers.new(type="BOOLEAN", name="bool 1")
        bool_one.object = obj2
        bool_one.operation = "DIFFERENCE"
        bool_one.solver = "FAST"
        # face.hide_set(True)
        bpy.context.view_layer.objects.active = capcopy
        bpy.ops.object.modifier_apply(modifier="bool 1")
        print("face boolean complete")

        bpy.data.objects["capcut"].select_set(True)
        bpy.ops.object.delete()

        bpy.context.view_layer.objects.active = capcopy

        capcopy = bpy.context.scene.objects["cap_copy"]
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = capcopy
        capcopy.select_set(True)

        try:
            bpy.ops.object.mode_set(mode="EDIT")
        except:
            pass

        bpy.ops.mesh.edge_face_add()
        bpy.ops.mesh.duplicate_move(
            MESH_OT_duplicate={"mode": 1},
            TRANSFORM_OT_translate={
                "value": (116.693, -12.1011, 83.9973),
                "orient_axis_ortho": "X",
                "orient_type": "GLOBAL",
                "orient_matrix": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                "orient_matrix_type": "GLOBAL",
                "constraint_axis": (False, False, False),
                "mirror": False,
                "use_proportional_edit": False,
                "proportional_edit_falloff": "SMOOTH",
                "proportional_size": 1,
                "use_proportional_connected": False,
                "use_proportional_projected": False,
                "snap": False,
                "snap_elements": {"INCREMENT"},
                "use_snap_project": False,
                "snap_target": "CLOSEST",
                "use_snap_self": True,
                "use_snap_edit": True,
                "use_snap_nonedit": True,
                "use_snap_selectable": False,
                "snap_point": (0, 0, 0),
                "snap_align": False,
                "snap_normal": (0, 0, 0),
                "gpencil_strokes": False,
                "cursor_transform": False,
                "texture_space": False,
                "remove_on_cancel": False,
                "view2d_edge_pan": False,
                "release_confirm": False,
                "use_accurate": False,
                "use_automerge_and_split": False,
            },
        )
        bpy.ops.mesh.separate(type="SELECTED")

        bpy.ops.object.mode_set(mode="OBJECT")
        capcopy2 = bpy.data.objects["cap_copy.001"]
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = capcopy2
        capcopy2.select_set(True)

        try:
            bpy.ops.object.mode_set(mode="EDIT")
        except:
            pass
        bpy.ops.mesh.select_all(action="SELECT")

        merge_distance = zhead / 9
        bpy.ops.mesh.remove_doubles(threshold=merge_distance)

        capcopy2.select_set(True)

        me = bpy.context.object.data
        bm = bmesh.from_edit_mesh(me)
        object = bpy.context.active_object
        mesh = object.data
        # list to populate with lengths
        edge_lengths = []
        # if edge is selected, get its length
        for e in bm.edges:
            edge_lengths.append(e.calc_length())
        circ = sum(edge_lengths) / 1000
        circ = round(circ, 3)
        print(
            "The Circumfrence is: ", circ, "meters"
        )  # default unit in blender is meters

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

        bpy.data.objects["cap_copy"].select_set(True)

        # bpy.ops.object.delete()

        capcopy2.name = "CIR_face"

        pass
