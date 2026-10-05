import bpy
from bpy.types import Operator


class decimate_mesh(bpy.types.Operator):
    bl_idname = "braincapgen.decimate_mesh"
    bl_label = "Decimate Headmesh"
    bl_description = "Reduce headmesh's face count before cutting out landmark shapes"
    number: bpy.props.FloatProperty(
        name="Decimate Ratio",
        description="Fraction of faces to keep, e.g. 0.05 keeps 5%",
        default=1,
    )

    def execute(self, context):
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass
        bpy.ops.object.select_all(action="DESELECT")
        head = bpy.data.objects["headmesh"]
        head.select_set(True)
        bpy.context.view_layer.objects.active = head
        obj = bpy.context.object
        mod = obj.modifiers.new(name="decimate", type="DECIMATE")
        mod.decimate_type = "COLLAPSE"
        # user defined decimate ratio of faces to keep
        mod.ratio = self.number

        head = bpy.data.objects["headmesh"]
        head.select_set(True)
        bpy.ops.object.modifier_apply(modifier="decimate")

        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
