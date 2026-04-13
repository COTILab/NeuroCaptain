import bpy
from bpy import context
import subprocess
import sys
from .utils import *
#import oct2py
import numpy as np
import jdata as jd
import pathlib
from bpy.types import Operator, PropertyGroup
import os
from bpy.props import EnumProperty, StringProperty, CollectionProperty

enum_action = [
    (
        "CUSTOM_GENERATE",
        "custom_generate",
        "Select vertices on head mesh corresponding to custom landmark geometry, then press button",
    ),
    (
        "CUSTOM_MESH",
        "custom_mesh",
        "select the mesh corresponding to custom landmark geometry then press button",
    ),
]


class customLandmarks(Operator):
    bl_label = "Select vertices to calculate 10-20 points"
    bl_description = "Click this button to generate mesh from brain landmarks "
    bl_idname = "braincapgen.customlandmark"
    action: EnumProperty(
        items=[
            ("CUSTOM_GENERATE", "custom_generate", "select the vertices of desired landmarks"),
            ("CUSTOM_MESH", "custom_mesh", "custom_mesh"),
        ]
    )

    @classmethod
    def description(cls, context, properties):
        hints = {}
        for item in enum_action:
            hints[item[0]] = item[2]
        return hints[properties.action]

    def execute(self, context):
        outputdir = GetBPWorkFolder()
        print("output directory is:", outputdir)
        if not os.path.isdir(outputdir):
            os.makedirs(outputdir)
        obj = bpy.context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="OBJECT")

        if self.action == "CUSTOM_GENERATE":
            self.custom_generate(context=context)

        elif self.action == "CUSTOM_MESH":
            self.custom_mesh(context=context)

        return {"FINISHED"}

    @staticmethod
    def custom_generate(context):
        # select vertice cloest to Nz, saves coordinate
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="OBJECT")
        selectedverts_customLayout = [
            v for v in bpy.context.active_object.data.vertices if v.select
        ]

        vselect_custom = []
        # formats the global coordinates [x,y,z]
        for n in range(len(selectedverts_customLayout)):
            vert_custom = selectedverts_customLayout[n].co
            v_global_custom = obj.matrix_world @ vert_custom
            vselect_custom.append(v_global_custom)

        try:
            bpy.ops.object.mode_set(mode="EDIT")
        except:
            pass
        bpy.ops.mesh.duplicate()  # Duplicate selected vertices
        bpy.ops.mesh.separate(type="SELECTED")  # Separate duplicated vertices into a new object

        bpy.ops.object.mode_set(mode="OBJECT")

        # Find the newly created object
        new_obj = [
            obj
            for obj in bpy.context.scene.objects
            if obj != bpy.context.object and obj.type == "MESH"
        ][-1]
        new_obj.name = "LandmarkMesh"
        new_obj.data.name = "LandmarkMesh"
        pass

    @staticmethod
    def custom_mesh(context):
        if bpy.context.selected_objects:
            obj = bpy.context.selected_objects[0]
            obj.name = "LandmarkMesh"
        else:
            print("Please select a mesh object corresponding to the desired landmarks.")
        pass


def register():
    bpy.utils.register_class(init_points)


def unregister():
    bpy.utils.unregister_class(init_points)


if __name__ == "__main__":
    register()

    bpy.ops.braincapgen.brain1020mesh("INVOKE_DEFAULT")
