import bpy
from bpy import context
import subprocess
import sys
from .utils import *
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
    (
        "OPTODE_LANDMARKS",
        "optode_landmarks",
        "Create LandmarkMesh from Source and Detector optode scalp-surface positions",
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
            ("OPTODE_LANDMARKS", "optode_landmarks", "Create LandmarkMesh from optode positions"),
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
        if not os.path.isdir(outputdir):
            os.makedirs(outputdir)
        obj = bpy.context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="OBJECT")

        if self.action == "CUSTOM_GENERATE":
            self.custom_generate(context=context)

        elif self.action == "CUSTOM_MESH":
            self.custom_mesh(context=context)

        elif self.action == "OPTODE_LANDMARKS":
            return self.optode_landmarks(context=context)

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

    @staticmethod
    def custom_mesh(context):
        if bpy.context.selected_objects:
            obj = bpy.context.selected_objects[0]
            obj.name = "LandmarkMesh"
        else:
            print("Please select a mesh object corresponding to the desired landmarks.")

    def optode_landmarks(self, context):
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree

        depsgraph = context.evaluated_depsgraph_get()

        optodes = []
        for obj in bpy.data.objects:
            parts = obj.name.split("_")
            if len(parts) >= 2 and parts[0] in ("Source", "Detector") and obj.type == "MESH":
                try:
                    int(parts[1])
                except ValueError:
                    continue
                optodes.append(obj)

        if not optodes:
            self.report({"WARNING"}, "No Source_* or Detector_* objects found")
            return {"CANCELLED"}

        optodes.sort(key=lambda o: (0 if o.name.startswith("Source") else 1,
                                    int(o.name.split("_")[1])))

        headmesh = bpy.data.objects.get("headmesh")
        bvh = None
        head_mat_inv = None
        head_mat = None
        if headmesh and headmesh.type == "MESH":
            bvh = BVHTree.FromObject(headmesh, depsgraph)
            head_mat = headmesh.matrix_world.copy()
            head_mat_inv = head_mat.inverted()

        positions = []
        names = []
        for obj in optodes:
            eval_obj = obj.evaluated_get(depsgraph)
            world_pos = eval_obj.matrix_world.translation.copy()

            if bvh is not None:
                local_pos = head_mat_inv @ world_pos
                loc, _, _, _ = bvh.find_nearest(local_pos)
                if loc is not None:
                    world_pos = head_mat @ loc

            positions.append(world_pos)
            names.append(obj.name)

        old_lm = bpy.data.objects.get("LandmarkMesh")
        if old_lm:
            bpy.data.objects.remove(old_lm, do_unlink=True)

        mesh_data = bpy.data.meshes.new("LandmarkMesh")
        mesh_data.from_pydata([list(p) for p in positions], [], [])
        mesh_data.update()

        lm_obj = bpy.data.objects.new("LandmarkMesh", mesh_data)
        context.collection.objects.link(lm_obj)

        lm_obj["landmark_labels"] = names

        self.report({"INFO"}, f"Created LandmarkMesh from {len(positions)} optodes")
        return {"FINISHED"}
