import bpy
from bpy.props import EnumProperty
from bpy.types import Operator
from bpy.utils import register_class, unregister_class
 

class insert_shape(Operator):
    bl_idname = 'braincapgen.insert_shape'
    bl_label = 'insertshape'
    bl_description = 'inserts a shape'
    bl_options = {'REGISTER', 'UNDO'}
 
    action: EnumProperty(
        items=[
            ('ADD_CYLINDER', 'add cylinder', 'add cylinder'),
            ('ADD_CUBE', 'add cube', 'add cube'),
            ('ADD_TRIANGLE', 'add triangle', 'add triangle'),
            ('ADD_CUSTOM', 'add custom', 'add custom')
        ]
    )

    def execute(self, context):
        if self.action == 'ADD_CYLINDER':
            self.add_cylinder(context=context)
        elif self.action == 'ADD_CUBE':
            self.add_cube(context=context)
        elif self.action == 'ADD_TRIANGLE':
            self.add_triangle(context=context)
        elif self.action == 'ADD_CUSTOM':
            self.add_custom(context=context)

            
        return {'FINISHED'}
 
    @staticmethod
    def add_cylinder(context):
        bpy.ops.mesh.primitive_cylinder_add()
        obj = bpy.context.selected_objects[0]
        obj.name = "cutout"
 
    @staticmethod
    def add_cube(context):
        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.selected_objects[0]
        obj.name = "cutout"
 
 
    @staticmethod
    def add_triangle(context):
        bpy.ops.mesh.primitive_cylinder_add(vertices=3)
        obj = bpy.context.selected_objects[0]
        obj.name = "cutout"

    @staticmethod
    def add_custom(context):
        obj = bpy.context.selected_objects[0]
        obj.name = "cutout"

 
 
def register():
    register_class(insert_shape)
 
 
def unregister():
    unregister_class(insert_shape)

if __name__ == '__main__':
    register()
