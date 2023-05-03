import bpy
from bpy.types import Operator

class decimate_mesh(bpy.types.Operator):
    bl_label = "Modify the Density of Head Mesh"
    bl_idname = 'braincapgen.decimate_mesh'

    number : bpy.props.FloatProperty(name = "Decimate Ratio", default = 1)

    def execute(self, context):
        
        obj = bpy.context.object
        mod = obj.modifiers.new(name='decimate', type='DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = self.number
       
        #bpy.context.object.modifiers["Decimate"].ratio = 1  ###

        return{'FINISHED'}
    
    def invoke(self, context, event):

        return context.window_manager.invoke_props_dialog(self)




def register():
    bpy.utils.register_class(decimate_mesh)


def unregister():
    bpy.utils.unregister_class(decimate_mesh)

if __name__ == '__main__':
    register()


#bpy.ops.braincapgen.decimate_mesh('INVOKE_DEFAULT')
