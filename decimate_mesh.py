import bpy
from bpy.types import Operator


class decimate_mesh(bpy.types.Operator):
    bl_idname = "braincapgen.decimate_mesh"
    bl_label = "Modify Mesh Density"
    bl_description = "modify density of mesh after choosing Nz,Lpa,Rpa)"
    number: bpy.props.FloatProperty(name="Decimate Ratio", default=1)

    def execute(self, context):
        # decrease number of faces
        obj = bpy.context.object
        mod = obj.modifiers.new(name="decimate", type="DECIMATE")
        mod.decimate_type = "COLLAPSE"
        # user defined decimate ratio of faces to keep
        mod.ratio = self.number
        bpy.ops.object.modifier_apply(modifier="mod")
        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
