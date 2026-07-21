import bpy
from bpy.types import Operator
import bmesh


class circumference_calc(Operator):
    bl_label = "Calculate Cap Circumference"
    bl_idname = "neurocaptain.circumference"
    bl_description = (
        "Estimate the circumference of the cap model - first select the "
        "topmost vertex on headmesh.001 in Edit Mode (unhide it in the "
        "outliner if needed)"
    )

    def execute(self, context):
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass
        bpy.ops.object.select_all(action="DESELECT")
        ob = bpy.data.objects["headmesh.001"]
        context.view_layer.objects.active = ob
        ob.hide_set(False)
        ob.select_set(True)

        bpy.ops.object.duplicate()
        for obj in bpy.context.selected_objects:
            obj.name = "headcopy"
            obj.data.name = "headcopy"
        # head_dup.name = "headcopy"
        vselect = self.reference_point(context=context)
        if len(vselect) == 0:
            self.report(
                {"ERROR"},
                "No vertex selected on headmesh.001 - unhide it in the outliner, "
                "select its topmost vertex in Edit Mode, then run this again",
            )
            return {"CANCELLED"}
        self.place_cube(context=context)
        self.boolean_cut(context=context)
        self.measure(context=context)
        return {"FINISHED"}

    @staticmethod
    def reference_point(context):
        # switch from Edit mode to Object mode so the selection gets updated
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass
        obj = bpy.data.objects["headmesh.001"]
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        selectedVerts = [v for v in obj.data.vertices if v.select]
        global vselect
        vselect = []
        for n in range(len(selectedVerts)):
            vert = selectedVerts[n].co
            v_global = obj.matrix_world @ vert

            vselect = v_global

        # back to theprevious mode
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass

        return vselect

    @staticmethod
    def place_cube(context):
        bpy.ops.mesh.primitive_cube_add()
        obj2 = bpy.context.selected_objects[0]
        obj2.name = "cube_meas"
        # scale the cube relative to the size of the head mesh
        head = bpy.data.objects["headmesh.001"]
        head_dimension = head.dimensions
        obj2.scale = (head.dimensions[0], head.dimensions[1], head.dimensions[0] / 3)
        # moving the bottom cube
        obj2.location = (
            vselect[0],
            vselect[1],
            vselect[2],
        )

    @staticmethod
    def boolean_cut(context):
        head = bpy.data.objects["headcopy"]
        cube = bpy.data.objects["cube_meas"]

        # make the bottom cut, switch to edit mode and delete faces created by the operation
        bpy.ops.object.mode_set(mode="OBJECT")
        bool_two = head.modifiers.new(type="BOOLEAN", name="bool 2")
        bool_two.object = cube
        bool_two.operation = "DIFFERENCE"
        bool_two.solver = "FAST" if bpy.app.version < (4, 0, 0) else "EXACT"
        cube.hide_set(True)
        bpy.context.view_layer.objects.active = head
        bpy.ops.object.modifier_apply(modifier="bool 2")
        # make the face cut, switch to edit mode and delete faces created by the operation

    @staticmethod
    def measure(context):
        bpy.ops.object.mode_set(mode="EDIT")
        me = bpy.context.active_object.data
        bm = bmesh.from_edit_mesh(me)
        bm.verts.ensure_lookup_table()
        bpy.ops.mesh.select_all(action="DESELECT")
        top_verts = [bm.verts[0]]
        for v in bm.verts:
            if v.co.z < top_verts[0].co.z:
                top_verts = [v]

        for v in top_verts:
            for f in v.link_faces:
                f.select_set(True)
        translate_args = {
            "value": (86.631, 151.812, 21.379),
            "orient_type": "GLOBAL",
            "orient_matrix": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        }
        # This used to add orient_axis_ortho="X" for bpy.app.version < (4,0,0),
        # on the assumption TRANSFORM_OT_translate needed it pre-4.0. Confirmed
        # via CI on Blender 3.6 that's wrong: TRANSFORM_OT_translate doesn't
        # recognize that keyword at all, on any currently-supported version
        # (3.6/4.2/5.0) - it raised "TypeError: ... keyword
        # 'orient_axis_ortho' unrecognized". 4.2/5.0 never hit this (the
        # version gate already skipped adding it there); dropping it
        # entirely fixes 3.6 without changing 4.2/5.0 behavior at all.
        bpy.ops.mesh.duplicate_move(
            MESH_OT_duplicate={"mode": 1},
            TRANSFORM_OT_translate=translate_args,
        )

        bm = bmesh.from_edit_mesh(me)
        edge_lengths = []
        # if edge is selected, get its length
        for e in bm.edges:
            if e.select:
                edge_lengths.append(e.calc_length())
        # print(edge_lengths)
        circ = sum(edge_lengths)
        circ = round(circ, 3)
        # Stashed so callers (and tests) can read the computed circumference
        # instead of parsing it back out of the console print below.
        context.scene["last_circumference_mm"] = circ
        unit = bpy.context.scene.unit_settings.length_unit
        print("The estimated circumfrence is: ", circ, unit)  # default unit in blender is meter
        print("NOTE: the units are defined in blender scene properties")
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass
        head = bpy.data.objects["headcopy"]
        cube = bpy.data.objects["cube_meas"]
        cube.select_set(True)
        bpy.ops.object.delete()
        head.select_set(True)
        bpy.ops.object.delete()

        return {"FINISHED"}
