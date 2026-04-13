import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator, PropertyGroup
from bpy.props import StringProperty, CollectionProperty
import numpy as np
import jdata as jd
import os
from .utils import *
from .dependencies import safe_import, require_dependency, show_error_message


class file_import(Operator, ImportHelper):
    """Import .stl or .off files"""

    bl_idname = "stlfile.invoke_import"
    bl_label = "Import File"
    bl_description = "Import headmesh file"
    bl_options = {"PRESET", "UNDO"}

    filename_ext = ".json,.jmsh,.bmsh,.stl, .off,.obj"

    filter_glob: StringProperty(
        default="*.json;*.jmsh;*.bmsh;*.stl;*.off;*.obj",
        options={"HIDDEN"},
    )

    files: CollectionProperty(type=PropertyGroup)

    @classmethod
    def func(self, context):
        bpy.ops.object.select_all(action="SELECT")
        for ob in bpy.context.selected_objects:
            print(ob.type)
            if (
                ob.type == "CAMERA"
                or ob.type == "LIGHT"
                or ob.type == "EMPTY"
                or ob.type == "LAMP"
                or ob.type == "SPEAKER"
                or ob.type == "CUBE"
            ):
                ob.select_set(True)
            else:
                ob.select_set(False)
        bpy.ops.object.delete()

        for o in bpy.context.scene.objects:
            if o.name == "Cube":
                o.select_set(True)
                bpy.ops.object.delete()
            else:
                pass
        return {"FINISHED"}

    def execute(self, context):
        self.func(context)
        for i in self.files:
            folder = os.path.dirname(self.filepath)
            path_to_file = os.path.join(folder, i.name)
        print("folder is", folder)
        print("pathis  is", path_to_file)
        root_ext = os.path.splitext(path_to_file)
        file_ex = root_ext[1]
        print("file ext is ", file_ex)
        obs = []
        if file_ex == ".stl":
            path_to_file = os.path.join(folder, i.name)
            bpy.ops.import_mesh.stl(
                filepath=path_to_file,
                axis_forward="-Z",
                axis_up="Y",
                filter_glob="*.obj;*.stl",
            )
            obs.append(context.selected_objects[:])
            bpy.context.object.rotation_euler[0] = 4.71239

            obj = bpy.context.object
            obj.name = "importedmodel"

        elif file_ex == ".obj":
            path_to_file = os.path.join(folder, i.name)
            if bpy.app.version >= (4, 0, 0):
                bpy.ops.wm.obj_import(
                    filepath=path_to_file,
                    filter_glob="*.obj;*.stl",
                )
            else:
                bpy.ops.import_scene.obj(
                    filepath=path_to_file,
                    axis_forward="-Z",
                    axis_up="Y",
                    filter_glob="*.obj;*.stl",
                )

            imported_objects = context.selected_objects[:]
            if imported_objects:
                imported_objects[0].name = "importedmodel"
        else:
            try:
                surfdata = jd.load(self.filepath)
                print("Loaded mesh data:", surfdata.keys() if hasattr(surfdata, 'keys') else type(surfdata))
                
                if "MeshVertex3" in surfdata and "MeshTri3" in surfdata:
                    AddMeshFromNodeFace(
                        surfdata["MeshVertex3"],
                        (np.array(surfdata["MeshTri3"]) - 1).astype(np.int32).tolist(),
                        "importedmodel",
                    )
                elif "node" in surfdata and "face" in surfdata:
                    AddMeshFromNodeFace(
                        surfdata["node"],
                        (np.array(surfdata["face"]) - 1).astype(np.int32).tolist(),
                        "importedmodel",
                    )
                else:
                    show_error_message(f"Unsupported mesh format in file. Available keys: {surfdata.keys()}")
                    return {'CANCELLED'}
                    
            except Exception as e:
                print(f"Error loading mesh: {e}")
                show_error_message(f"Failed to load mesh file: {str(e)}")
                return {'CANCELLED'}

        mod = bpy.data.objects["importedmodel"]
        bpy.ops.object.select_all(action="DESELECT")
        mod.select_set(True)
        bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN", center="MEDIAN")
        bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)

        print(
            "please rename object in blender to indicate either 'headmesh' or 'LandmarkMesh, or use the 'label custom' button to assign landmark geometry"
        )
        return {"FINISHED"}