import bpy
from bpy.types import Operator
from bpy.props import EnumProperty, StringProperty, CollectionProperty

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
        "performs the boolean cut, wirefram and remesh of the NeuroCap",
    ),
]


class cap_generation(Operator):
    bl_label = "Generate NeuroCap"
    bl_idname = "braincapgen.cap_generation"
    bl_description = "Generate the Generic NeuroCap "
    action: EnumProperty(
        items=[
            ("REFERENCE_POINT", "reference_point", "reference_point"),
            ("PLACE_CUTOUTS", "place_cutouts", "place_cutouts"),
            ("BOOLEAN_CUT", "boolean_cut", "boolean_cut"),
        ]
    )
    # user input- wireframe thickness and voxel size of remesh
    thick: bpy.props.FloatProperty(name="thickness", default=2)
    voxel: bpy.props.FloatProperty(name="voxel size", default=0.5)

    @classmethod
    def description(cls, context, properties):
        hints = {}
        for item in enum_action:
            hints[item[0]] = item[2]
        return hints[properties.action]

    def execute(self, context):
        # obtain reference point to guide boolean difference objects
        if self.action == "REFERENCE_POINT":
            self.reference_point(context=context)

        # allow placement of cubes for boolean cut (can be adjusted by the user)
        elif self.action == "PLACE_CUTOUTS":
            self.place_cutouts(context=context)

        # performs the boolean cut, wireframe and remesh
        elif self.action == "BOOLEAN_CUT":
            global thickness
            global voxelsize
            thickness = self.thick
            voxelsize = self.voxel
            self.boolean_cut(context=context)

        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    @staticmethod
    def reference_point(context):
        mode = bpy.context.active_object.mode
        # switch from Edit mode to Object mode so the selection gets updated
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        selectedVerts = [v for v in bpy.context.active_object.data.vertices if v.select]
        global vselect
        vselect = []
        for n in range(len(selectedVerts)):
            vert = selectedVerts[n].co
            v_global = obj.matrix_world @ vert

            vselect = v_global
            print("reference is ", v_global)

        # back to theprevious mode
        bpy.ops.object.mode_set(mode="OBJECT")

        return vselect

    @staticmethod
    def place_cutouts(context):
        # adding the face cube
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.selected_objects[0]
        # bevel cube to create rounded shape for face cutout
        bev = obj.modifiers.new(name="bevel", type="BEVEL")
        bev.width = 0.6
        bev.segments = 30
        bpy.ops.object.modifier_apply(modifier="bevel")
        obj.name = "face_cutout"

        # scale cube relative to the size of the head mesh
        head = bpy.data.objects["headmesh"]
        head_dimension = head.dimensions
        obj.scale = (
            (head.dimensions[0] / 3) + ((head.dimensions[0] / 10) / 2),
            head.dimensions[0] / 3,
            head.dimensions[0] / 3,
        )
        # moving the face cube
        obj.location = (vselect[0], vselect[1], vselect[2] - (head.dimensions[2] / 10))

        # adding bottom cube
        bpy.ops.mesh.primitive_cube_add()
        obj2 = bpy.context.selected_objects[0]
        obj2.name = "bottom_cutout"
        # scale the cube relative to the size of the head mesh
        obj2.scale = (head.dimensions[0], head.dimensions[1], head.dimensions[0] / 3)
        # moving the bottom cube
        obj2.location = (
            vselect[0],
            vselect[1] - (head.dimensions[1] / 2),
            vselect[2] - (head.dimensions[2] / 3.5),
        )
        # rotate the bottom cut 2 degrees - preserve more landmarks
        bpy.context.object.rotation_euler[0] = 0.0523599

        pass

    @staticmethod
    def boolean_cut(context):
        head = bpy.data.objects["headmesh"]
        face = bpy.data.objects["face_cutout"]
        bottom = bpy.data.objects["bottom_cutout"]

        # make the bottom cut, switch to edit mode and delete faces created by the operation
        bpy.ops.object.mode_set(mode="OBJECT")
        bool_two = head.modifiers.new(type="BOOLEAN", name="bool 2")
        bool_two.object = bottom
        bool_two.operation = "DIFFERENCE"
        bool_two.solver = "FAST"
        bottom.hide_set(True)
        bpy.context.view_layer.objects.active = head
        bpy.ops.object.modifier_apply(modifier="bool 2")
        print("bottom boolean complete")
        # make the face cut, switch to edit mode and delete faces created by the operation
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
        bpy.context.view_layer.objects.active = head

        # generate a wireframe mesh from cap design
        head = bpy.data.objects["headmesh"]
        bpy.ops.object.mode_set(mode="OBJECT")
        wire = head.modifiers.new(type="WIREFRAME", name="wireframe")
        wire.thickness = thickness
        wire.use_even_offset = False
        wire.use_boundary = True
        wire.use_crease = False
        bpy.ops.object.modifier_apply(modifier="wireframe")
        print("wireframe complete")

        # remesh the wireframe mesh- create one contingent mesh
        remesh = head.modifiers.new(type="REMESH", name="remesh")
        remesh.voxel_size = voxelsize
        bpy.ops.object.modifier_apply(modifier="remesh")
        bpy.context.view_layer.objects.active = head
        print("remesh complete")

        return {"FINISHED"}
