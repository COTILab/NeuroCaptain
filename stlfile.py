import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator, PropertyGroup
from bpy.props import StringProperty, CollectionProperty
import os

class stl_file_import(Operator,ImportHelper):
    """Import .stl or .off files"""
    bl_idname = "stlfile.invoke_import"
    bl_label = "Import File"
    bl_description = "Import headmesh file (if mesh for brain 10-20 points also exists upload at SAME time)"
    bl_options = {'PRESET', 'UNDO'}

    

    # ImportHelper mixin class uses this
    filename_ext = ".stl, .off"

    filter_glob: StringProperty(
            default="*.stl;*.off",
            options={'HIDDEN'},
            )

    # Selected files
    files: CollectionProperty(type=PropertyGroup)
   # def execute(self,context):
    @classmethod
    def func(self,context):
        bpy.ops.object.select_all(action = 'DESELECT')
        
        bpy.data.objects['Cube'].select_set(True)
        bpy.data.objects['Camera'].select_set(True)
        bpy.data.objects['Light'].select_set(True)
        bpy.ops.object.delete()

        return {"FINISHED"}

    def execute(self, context):
        self.func(context)
        # Get the folder
        folder = os.path.dirname(self.filepath)
        
        obs = []
        # Iterate through the selected files
        for i in self.files:

            # Generate full path to file
            path_to_file = (os.path.join(folder, i.name))
            bpy.ops.import_mesh.stl(filepath=path_to_file, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.stl")
            # Append Object(s) to the list
            obs.append(context.selected_objects[:])
            bpy.context.object.rotation_euler[0] = 4.71239 ## I needed this line idk if eveyone will
            obj = bpy.context.object
            obj.name = "headmesh"
            bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(0.212906, 0.0140968, 0.0237914), "orient_axis_ortho":'X', "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
            obj_dup = bpy.context.object
            obj_dup.hide_set(True)
            obj_dup.name = "headmesh.001"
            # Print the imported object reference
            print ("Imported object:", context.object)
            
            return {"FINISHED"}
        


