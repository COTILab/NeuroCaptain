import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator, PropertyGroup
from bpy.props import StringProperty, CollectionProperty
import numpy as np
import jdata as jd
import os
from .utils import *


class file_import(Operator, ImportHelper):
    """Import .stl or .off files"""

    bl_idname = "stlfile.invoke_import"
    bl_label = "Import File"
    bl_description = "Import headmesh file"
    bl_options = {"PRESET", "UNDO"}

    # ImportHelperclass uses this
    filename_ext = ".json,.jmsh,.bmsh,.stl, .off"

    filter_glob: StringProperty(
        default="*.json;*.jmsh;*.bmsh;*.stl;*.off",
        options={"HIDDEN"},
    )

    # Selected files
    files: CollectionProperty(type=PropertyGroup)

    @classmethod
    def func(self, context):
        bpy.ops.object.select_all(action="SELECT")
        # delete unnecessary default objects
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
        # print(self.files)

        self.func(context)
        # Get the folder
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
            # Iterate through the selected files

            # Generate full path to file
            path_to_file = os.path.join(folder, i.name)
            bpy.ops.import_mesh.stl(
                filepath=path_to_file,
                axis_forward="-Z",
                axis_up="Y",
                filter_glob="*.obj;*.stl",
            )
            # Append Object(s) to the list
            obs.append(context.selected_objects[:])
            bpy.context.object.rotation_euler[
                0
            ] = 4.71239  ## I needed this line idk if eveyone will
            # bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')
            obj = bpy.context.object
            obj.name = "importedmodel"

        else:
            try:
                if bpy.context.scene.blender_photonics.backend == "octave":
                    import oct2py as op

                    oc = op.Oct2Py()
                else:
                    import matlab.engine as op

                    oc = op.start_matlab()
            except ImportError:
                raise ImportError(
                    "To run this feature, you must install the oct2py or matlab.engine Python modulem first, based on your choice of the backend"
                )

            oc.addpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "script"))
            # load json
            try:
                surfdata = oc.feval("loadjson", self.filepath)
                AddMeshFromNodeFace(
                    surfdata["MeshVertex3"],
                    (np.array(surfdata["MeshTri3"]) - 1).astype(np.int32).tolist(),
                    "importedmodel",
                )
            # load bmsh
            except:
                surfdata = oc.feval("surf2jmesh", self.filepath)
                print("data is", surfdata)
                AddMeshFromNodeFace(
                    surfdata["MeshVertex3"],
                    (np.array(surfdata["MeshTri3"]) - 1).astype(np.int16).tolist(),
                    "importedmodel",
                )

            print(
                "please rename object in blender to indicate either 'headmesh' or 'LandmarkMesh"
            )
        return {"FINISHED"}
