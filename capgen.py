import bpy
from bpy.types import Operator
from bpy.props import EnumProperty, FloatProperty, BoolProperty
from .utils import *
from .landmark_labels import get_landmark_labels


def _find_landmark_nz(context):
    """Look for an Nz landmark already present in the scene - from a
    previous manual define-Nz step (brain1020mesh.py's interactive
    picking), or from generating/importing 10-20 (or 10-10/10-5) landmarks
    via select_model(ADD_BRAIN1020MESH) - so the user doesn't have to
    manually re-pick a headmesh vertex every time a LandmarkMesh with a
    real Nz label is already in the scene.

    Returns an [x, y, z] world-space position, or None if no usable
    LandmarkMesh/Nz label is found.
    """
    landmark_obj = bpy.data.objects.get("LandmarkMesh")
    if landmark_obj is None or not landmark_obj.data.vertices:
        return None

    labels = get_landmark_labels(landmark_obj)
    if "Nz" not in labels:
        return None

    index = labels.index("Nz")
    local = landmark_obj.data.vertices[index].co
    world = landmark_obj.matrix_world @ local
    return list(world)


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

        elif (landmark_nz := _find_landmark_nz(context)) is not None:
            vselect = landmark_nz
            context.scene["vselect"] = vselect
            context.scene["nz_assigned"] = True
            context.scene["saved_nz"] = vselect
            print("Reference Nz (found on LandmarkMesh in scene):", vselect)

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

        elif (landmark_nz := _find_landmark_nz(context)) is not None:
            vselect = landmark_nz
            context.scene["vselect"] = vselect
            context.scene["nz_assigned"] = True
            context.scene["saved_nz"] = vselect
            print("Reference Nz (found on LandmarkMesh in scene, in cutouts):", vselect)

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
        bool_three.solver = "FAST" if bpy.app.version < (4, 0, 0) else "EXACT"
        ear.hide_set(True)
        bpy.context.view_layer.objects.active = head
        bpy.ops.object.modifier_apply(modifier="bool 3")

        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass
        # decrease number of faces
        bpy.ops.object.select_all(action="DESELECT")
        head = bpy.data.objects["headmesh"]
        head.select_set(True)
        bpy.context.view_layer.objects.active = head
        bpy.ops.object.mode_set(mode="OBJECT")
        bool_two = head.modifiers.new(type="BOOLEAN", name="bool 2")
        bool_two.object = bottom
        bool_two.operation = "DIFFERENCE"
        bool_two.solver = "FAST" if bpy.app.version < (4, 0, 0) else "EXACT"
        bottom.hide_set(True)
        bpy.context.view_layer.objects.active = head
        bpy.ops.object.modifier_apply(modifier="bool 2")

        # This used to toggle into edit mode and delete whatever faces were
        # already selected, with nothing explicitly selecting any faces
        # beforehand - it depended entirely on incidental left-over
        # selection state from earlier in the pipeline. That's exactly the
        # kind of behavior Blender 4.0's boolean-modifier rewrite (BMesh ->
        # Exact/Carve solver) can silently change: whatever geometry a
        # modifier_apply() leaves selected is an internal implementation
        # detail, not a documented contract. On 3.4 nothing ends up
        # selected here (effectively a no-op); on 4.2/5.0 the new solver
        # apparently leaves the newly-merged geometry selected, so this
        # deleted almost the entire head, leaving only a tiny disconnected
        # fragment (confirmed: reproduced manually on 5.0, not on 3.4).
        # Force it to the guaranteed no-op that already matched the
        # working 3.4 behavior, instead of relying on undefined state.
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.delete(type="FACE")
        bpy.ops.object.editmode_toggle()

        bpy.ops.object.mode_set(mode="OBJECT")
        bool_one = head.modifiers.new(type="BOOLEAN", name="bool 1")
        bool_one.object = face
        bool_one.operation = "DIFFERENCE"
        bool_one.solver = "FAST" if bpy.app.version < (4, 0, 0) else "EXACT"
        face.hide_set(True)
        bpy.context.view_layer.objects.active = head
        bpy.ops.object.modifier_apply(modifier="bool 1")

        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass
        # decrease number of faces
        bpy.ops.object.select_all(action="DESELECT")
        head = bpy.data.objects["headmesh"]
        head.select_set(True)
        bpy.context.view_layer.objects.active = head

        wire = head.modifiers.new(type="WIREFRAME", name="wireframe")
        wire.thickness = thickness
        wire.use_even_offset = False
        wire.use_boundary = True
        wire.use_crease = False
        bpy.ops.object.modifier_apply(modifier="wireframe")

        remesh = head.modifiers.new(type="REMESH", name="remesh")
        remesh.voxel_size = voxelsize
        bpy.ops.object.modifier_apply(modifier="remesh")
        bpy.context.view_layer.objects.active = head

        return {"FINISHED"}
