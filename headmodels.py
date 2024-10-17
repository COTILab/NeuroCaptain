import bpy
import subprocess
import sys
from bpy.props import EnumProperty
import addon_utils
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator, PropertyGroup
from bpy.props import StringProperty, CollectionProperty
import os
from .utils import *
import numpy as np
import jdata as jd

enum_action = [
    ("ADD_HEADMESH", "add headmesh", "Access a folder called: HeadModels"),
    ("ADD_BRAIN1020MESH", "add brain1020mesh", "Access a folder called: BrainLandmarks"),
]


class select_model(Operator, ImportHelper):
    bl_label = "Select a head model"
    bl_description = "Access folders with head models and brain-landmark meshes"
    bl_idname = "braincapgen.select_model"

    action: EnumProperty(
        items=[
            ("ADD_HEADMESH", "add headmesh", "add headmesh"),
            ("ADD_BRAIN1020MESH", "add brain1020mesh", "add brain1020mesh"),
        ]
    )

    filename_ext = ".json,.jmsh,.bmsh,.stl, .off"
    filter_glob: StringProperty(
        default="*.json;*.jmsh;*.bmsh;*.stl;*.off",
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

    @classmethod
    def description(cls, context, properties):
        hints = {}
        for item in enum_action:
            hints[item[0]] = item[2]
        return hints[properties.action]

    def execute(self, context):
        self.func(context)
        for i in self.files:
            folder = os.path.dirname(self.filepath)
            path_to_file = os.path.join(folder, i.name)
            print("folder is", folder)
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
            print(
                "the path is:", os.path.join(os.path.dirname(os.path.abspath(__file__)), "script")
            )
            oc.addpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "script"))

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

        if self.action == "ADD_HEADMESH":
            self.add_headmesh(context=context)

        elif self.action == "ADD_BRAIN1020MESH":
            self.add_brain1020mesh(context=context)

        return {"FINISHED"}

    def invoke(self, context, event):
        for mod in addon_utils.modules():
            if mod.bl_info["name"] == "NeuroCaptain":
                path = os.path.dirname(mod.__file__)
                path = os.path.join(path, "Models")
                print("path is", path)
                obs = []
                self.filepath = path
                wm = context.window_manager.fileselect_add(self)
                return {"RUNNING_MODAL"}

    @staticmethod
    def add_headmesh(context):
        obj = bpy.context.object

        head = bpy.data.objects["importedmodel"]
        bpy.ops.object.select_all(action="DESELECT")
        head.select_set(True)
        bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN", center="MEDIAN")

        head.name = "headmesh"
        head.select_set(True)

        bpy.ops.object.duplicate_move(
            OBJECT_OT_duplicate={"linked": False, "mode": "TRANSLATION"},
            TRANSFORM_OT_translate={
                "value": (0.212906, 0.0140968, 0.0237914),
                "orient_axis_ortho": "X",
                "orient_type": "GLOBAL",
                "orient_matrix": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                "orient_matrix_type": "GLOBAL",
                "constraint_axis": (False, False, False),
                "mirror": False,
                "use_proportional_edit": False,
                "proportional_edit_falloff": "SMOOTH",
                "proportional_size": 1,
                "use_proportional_connected": False,
                "use_proportional_projected": False,
                "snap": False,
                "snap_elements": {"INCREMENT"},
                "use_snap_project": False,
                "snap_target": "CLOSEST",
                "use_snap_self": True,
                "use_snap_edit": True,
                "use_snap_nonedit": True,
                "use_snap_selectable": False,
                "snap_point": (0, 0, 0),
                "snap_align": False,
                "snap_normal": (0, 0, 0),
                "gpencil_strokes": False,
                "cursor_transform": False,
                "texture_space": False,
                "remove_on_cancel": False,
                "view2d_edge_pan": False,
                "release_confirm": False,
                "use_accurate": False,
                "use_automerge_and_split": False,
            },
        )
        ob = bpy.context.scene.objects["headmesh.001"]
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = ob  # Make the cube the active object
        ob.select_set(True)
        ob.hide_set(True)

        return {"FINISHED"}

    @staticmethod
    def add_brain1020mesh(context):
        # obj = bpy.context.object
        brain = bpy.data.objects["importedmodel"]
        bpy.ops.object.select_all(action="DESELECT")
        brain.select_set(True)
        brain.name = "LandmarkMesh"
        return {"FINISHED"}
