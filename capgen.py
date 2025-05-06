import bpy
from bpy.types import Operator
from bpy.props import EnumProperty, FloatProperty, BoolProperty
from .utils import *

enum_action = [
    ("REFERENCE_POINT", "reference_point", "select Nz vertice, then press okay"),
    (
        "PLACE_CUTOUTS",
        "place_cutouts",
        "place the cutouts in generic locations (can be altered by user)",
    ),
    (
        "BOOLEAN_CUT",
        "boolean_cut",
        "performs the boolean cut, wireframe and remesh of the NeuroCap",
    ),
]


class cap_generation(Operator):
    bl_label = "Generate NeuroCap"
    bl_idname = "braincapgen.cap_generation"
    bl_description = "Generate the Generic NeuroCap"

    action: EnumProperty(items=enum_action)

    # Properties for PLACE_CUTOUTS
    add_cylinder: BoolProperty(name="Include Ear Cutout", default=True)

    # Properties for BOOLEAN_CUT
    thick: FloatProperty(name="Thickness", default=2)
    voxel: FloatProperty(name="Voxel Size", default=0.5)

    @classmethod
    def description(cls, context, properties):
        hints = {item[0]: item[2] for item in enum_action}
        return hints.get(properties.action, "")

    def execute(self, context):
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass

        bpy.ops.object.select_all(action="DESELECT")
        head = bpy.data.objects.get("headmesh")
        if head:
            head.select_set(True)
            bpy.context.view_layer.objects.active = head
        else:
            self.report({"ERROR"}, "headmesh not found!")
            return {"CANCELLED"}

        if self.action == "REFERENCE_POINT":
            self.reference_point(context)

        elif self.action == "PLACE_CUTOUTS":
            self.place_cutouts(context)

        elif self.action == "BOOLEAN_CUT":
            global thickness
            global voxelsize
            thickness = self.thick
            voxelsize = self.voxel
            self.boolean_cut(context)

        return {"FINISHED"}

    def invoke(self, context, event):
        if self.action == "PLACE_CUTOUTS":
            return context.window_manager.invoke_props_dialog(self, width=300)
        elif self.action == "BOOLEAN_CUT":
            return context.window_manager.invoke_props_dialog(self, width=300)
        else:
            return self.execute(context)

    def draw(self, context):
        layout = self.layout
        if self.action == "PLACE_CUTOUTS":
            layout.prop(self, "add_cylinder", text="Include Ear Cutout")
        elif self.action == "BOOLEAN_CUT":
            layout.prop(self, "thick", text="Wireframe Thickness")
            layout.prop(self, "voxel", text="Remesh Voxel Size")

    @staticmethod
    def reference_point(context):
        global vselect

        saved_nz = context.scene.get("saved_nz")

        mode = bpy.context.active_object.mode
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        selected_verts = [v for v in obj.data.vertices if v.select]

        if len(selected_verts) == 1:
            vert = selected_verts[0].co
            v_global = obj.matrix_world @ vert
            vselect = list(v_global)
            context.scene["vselect"] = vselect
            context.scene["nz_assigned"] = True
            context.scene["saved_nz"] = vselect
            print("Reference Nz (manual selection):", vselect)

        elif saved_nz is not None:
            vselect = saved_nz
            print("Reference Nz (saved):", vselect)

        else:
            ShowMessageBox(
                "Select the vertex that corresponds to Nz and select OK", "Error", "ERROR"
            )
            context.scene["nz_assigned"] = False
            bpy.ops.object.mode_set(mode=mode)
            return {"CANCELLED"}

        bpy.ops.object.mode_set(mode=mode)
        return vselect

    def place_cutouts(self, context):
        global vselect

        saved_nz = context.scene.get("saved_nz")

        mode = bpy.context.active_object.mode
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        selected_verts = [v for v in obj.data.vertices if v.select]

        if len(selected_verts) == 1:
            vert = selected_verts[0].co
            v_global = obj.matrix_world @ vert
            vselect = list(v_global)
            context.scene["vselect"] = vselect
            context.scene["nz_assigned"] = True
            context.scene["saved_nz"] = vselect
            print("Reference Nz (manual selection in cutouts):", vselect)

        elif saved_nz is not None:
            vselect = saved_nz
            print("Reference Nz (saved in cutouts):", vselect)

        else:
            ShowMessageBox(
                "Select the vertex that corresponds to Nz and select OK", "Error", "ERROR"
            )
            context.scene["nz_assigned"] = False
            bpy.ops.object.mode_set(mode=mode)
            return {"CANCELLED"}

        bpy.ops.object.mode_set(mode=mode)

        # Place cutouts
        head = bpy.data.objects["headmesh"]

        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.selected_objects[0]
        bev = obj.modifiers.new(name="bevel", type="BEVEL")
        bev.width = 0.6
        bev.segments = 30
        bpy.ops.object.modifier_apply(modifier="bevel")
        obj.name = "face_cutout"

        obj.scale = (
            (head.dimensions[0] / 3) + ((head.dimensions[0] / 10) / 2),
            head.dimensions[0] / 3,
            head.dimensions[0] / 3,
        )
        obj.location = (vselect[0], vselect[1], vselect[2] - (head.dimensions[2] / 10))

        bpy.ops.mesh.primitive_cube_add()
        obj2 = bpy.context.selected_objects[0]
        obj2.name = "bottom_cutout"
        obj2.scale = (head.dimensions[0], head.dimensions[1], head.dimensions[0] / 3)
        obj2.location = (
            vselect[0],
            vselect[1] - (head.dimensions[1] / 2),
            vselect[2] - (head.dimensions[2] / 3.5),
        )
        bpy.context.object.rotation_euler[0] = 0.0523599

        if self.add_cylinder:
            bpy.ops.mesh.primitive_cylinder_add()
            obj3 = bpy.context.selected_objects[0]
            bpy.context.object.rotation_euler[1] = 1.5708
            obj3.name = "ear_cutout"
            obj3.scale = (head.dimensions[0] / 9, head.dimensions[1] / 7, head.dimensions[2] * 1.2)

            # align cylinder so that top of bottom_cutout = center of ear_cutout
            bottom_cutout_top_z = obj2.location.z + obj2.scale[2]
            obj3.location = (vselect[0], vselect[1] - (head.dimensions[1] / 2), bottom_cutout_top_z)

        return {"FINISHED"}

    @staticmethod
    def boolean_cut(context):
        head = bpy.data.objects["headmesh"]
        face = bpy.data.objects["face_cutout"]
        bottom = bpy.data.objects["bottom_cutout"]
        ear = bpy.data.objects["ear_cutout"]

        bpy.ops.object.mode_set(mode="OBJECT")
        bool_three = head.modifiers.new(type="BOOLEAN", name="bool 3")
        bool_three.object = ear
        bool_three.operation = "DIFFERENCE"
        bool_three.solver = "FAST"
        ear.hide_set(True)
        bpy.context.view_layer.objects.active = head
        bpy.ops.object.modifier_apply(modifier="bool 3")
        print("ear boolean complete")

        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass
        # decrease number of faces
        bpy.ops.object.select_all(action="DESELECT")
        head = bpy.data.objects["headmesh"]
        # bpy.ops.object.select_all(action="DESELECT")
        head.select_set(True)
        bpy.context.view_layer.objects.active = head
        bpy.ops.object.mode_set(mode="OBJECT")
        bool_two = head.modifiers.new(type="BOOLEAN", name="bool 2")
        bool_two.object = bottom
        bool_two.operation = "DIFFERENCE"
        bool_two.solver = "FAST"
        bottom.hide_set(True)
        bpy.context.view_layer.objects.active = head
        bpy.ops.object.modifier_apply(modifier="bool 2")
        print("bottom boolean complete")

        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.delete(type="FACE")

        bpy.ops.object.mode_set(mode="OBJECT")
        bool_one = head.modifiers.new(type="BOOLEAN", name="bool 1")
        bool_one.object = face
        bool_one.operation = "DIFFERENCE"
        bool_one.solver = "FAST"
        face.hide_set(True)
        bpy.context.view_layer.objects.active = head
        bpy.ops.object.modifier_apply(modifier="bool 1")
        print("face boolean complete")

        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass
        # decrease number of faces
        bpy.ops.object.select_all(action="DESELECT")
        head = bpy.data.objects["headmesh"]
        # bpy.ops.object.select_all(action="DESELECT")
        head.select_set(True)
        bpy.context.view_layer.objects.active = head

        wire = head.modifiers.new(type="WIREFRAME", name="wireframe")
        wire.thickness = thickness
        wire.use_even_offset = False
        wire.use_boundary = True
        wire.use_crease = False
        bpy.ops.object.modifier_apply(modifier="wireframe")
        print("wireframe complete")

        remesh = head.modifiers.new(type="REMESH", name="remesh")
        remesh.voxel_size = voxelsize
        bpy.ops.object.modifier_apply(modifier="remesh")
        bpy.context.view_layer.objects.active = head
        print("remesh complete")

        return {"FINISHED"}
