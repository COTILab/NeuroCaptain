import bpy
from bpy import context
import subprocess
import sys
from .utils import *
import oct2py
import numpy as np
import jdata as jd
import pathlib
from bpy.types import Operator, PropertyGroup
import os
from bpy.props import EnumProperty, StringProperty, CollectionProperty

enum_action = [
    ("NZ_SELECT", "nz_select", "select the vertice closest to Nz, then press okay"),
    ("LPA_SELECT", "lpa_select", "select the vertice closest to Lpa, then press okay"),
    ("RPA_SELECT", "RPA_select", "select the vertice closest to Rpa, then press okay"),
    ("IZ_SELECT", "iz_select", "select the vertice closest to Iz, then press okay"),
    ("CZ_SELECT", "cz_select", "select the vertice closest to Cz, then press okay"),
    (
        "BRAIN1020_MESH",
        "brain1020_mesh",
        "enter p1 and p2 corresponding to 10(p1)-20(p2) points, then press okay",
    ),
]


class brain1020mesh(Operator):
    bl_label = "Select vertices to calculate 10-20 points"
    bl_description = "Click this button to generate mesh from brain landmarks "
    bl_idname = "braincapgen.init_points"
    action: EnumProperty(
        items=[
            ("NZ_SELECT", "nz_select", "nz_select"),
            ("LPA_SELECT", "lpa_select", "lpa_select"),
            ("RPA_SELECT", "RPA_select", "rpa_select"),
            ("IZ_SELECT", "iz_select", "iz_select"),
            ("CZ_SELECT", "cz_select", "cz_select"),
            ("BRAIN1020_MESH", "brain1020_mesh", "brain1020_mesh"),
        ]
    )
    point1: bpy.props.FloatProperty(name="p1", default=10)
    point2: bpy.props.FloatProperty(name="p2", default=10)

    @classmethod
    def description(cls, context, properties):
        hints = {}
        for item in enum_action:
            hints[item[0]] = item[2]
        return hints[properties.action]

    def execute(self, context):
        outputdir = GetBPWorkFolder()
        print("output directory is:", outputdir)
        obj = bpy.context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="OBJECT")

        if self.action == "NZ_SELECT":
            self.nz_select(context=context)

        elif self.action == "LPA_SELECT":
            self.lpa_select(context=context)

        elif self.action == "RPA_SELECT":
            self.rpa_select(context=context)

        elif self.action == "IZ_SELECT":
            self.iz_select(context=context)

        elif self.action == "CZ_SELECT":
            self.cz_select(context=context)

        elif self.action == "BRAIN1020_MESH":
            global p1, p2
            p1 = self.point1
            p2 = self.point2
            self.brain1020_mesh(context=context)

        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    @staticmethod
    def nz_select(context):
        # select vertice cloest to Nz, saves coordinate
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="OBJECT")
        selectedverts_nz = [v for v in bpy.context.active_object.data.vertices if v.select]

        vselect_nz = []
        # formats the global coordinates [x,y,z]
        for n in range(len(selectedverts_nz)):
            vert_nz = selectedverts_nz[n].co
            v_global_nz = obj.matrix_world @ vert_nz
            vselect_nz.append(v_global_nz)
        global nz
        nz = np.array(vselect_nz)
        print("nz is", nz)
        return nz

    @staticmethod
    def lpa_select(context):
        # select vertice cloest to Lpa, saves coordinate
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="OBJECT")
        selectedverts_lpa = [v for v in bpy.context.active_object.data.vertices if v.select]

        vselect_lpa = []
        # formats the global coordinates [x,y,z]
        for n in range(len(selectedverts_lpa)):
            vert_lpa = selectedverts_lpa[n].co
            v_global_lpa = obj.matrix_world @ vert_lpa
            vselect_lpa.append(v_global_lpa)
        global lpa
        lpa = np.array(vselect_lpa)
        print("lpa is", lpa)
        return lpa

    @staticmethod
    def rpa_select(context):
        # select vertice cloest to Rpa, saves coordinate
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="OBJECT")
        selectedverts_rpa = [v for v in bpy.context.active_object.data.vertices if v.select]

        vselect_rpa = []
        # formats the global coordinates [x,y,z]
        for n in range(len(selectedverts_rpa)):
            vert_rpa = selectedverts_rpa[n].co
            v_global_rpa = obj.matrix_world @ vert_rpa
            vselect_rpa.append(v_global_rpa)
        global rpa
        rpa = np.array(vselect_rpa)
        print("rpa is", rpa)
        return rpa

    @staticmethod
    def iz_select(context):
        # select vertice cloest to Iz, saves coordinate
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="OBJECT")
        selectedverts_iz = [v for v in bpy.context.active_object.data.vertices if v.select]

        vselect_iz = []
        # formats the global coordinates [x,y,z]
        for n in range(len(selectedverts_iz)):
            vert_iz = selectedverts_iz[n].co
            v_global_iz = obj.matrix_world @ vert_iz
            vselect_iz.append(v_global_iz)
        global iz
        iz = np.array(vselect_iz)
        print("iz is", iz)
        return iz

    @staticmethod
    def cz_select(context):
        # select vertice cloest to Cz, saves coordinate
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="OBJECT")
        selectedverts_cz = [v for v in bpy.context.active_object.data.vertices if v.select]

        vselect_cz = []
        # formats the global coordinates [x,y,z]
        for n in range(len(selectedverts_cz)):
            vert_cz = selectedverts_cz[n].co
            v_global_cz = obj.matrix_world @ vert_cz
            vselect_cz.append(v_global_cz)
        global cz
        cz = np.array(vselect_cz)
        print("cz is", cz)
        return cz

    @staticmethod
    def brain1020_mesh(context):
        print("entering brain1020 mesh")
        outputdir = GetBPWorkFolder()
        obj = bpy.context.view_layer.objects.active
        # vs-the intial points Brain1020 uses
        vs = np.vstack((nz, iz, lpa, rpa, cz))
        verts = []
        # saving the head mesh from the scene
        for n in range(len(obj.data.vertices)):
            vert = obj.data.vertices[n].co
            v_global = obj.matrix_world @ vert
            verts.append(v_global)
        faces = [(np.array(face.vertices[:]) + 1).tolist() for face in obj.data.polygons]
        v = np.array(verts)
        f = np.array(faces)
        # meshdata- saves the input requirements for brian1020mesh.m
        meshdata = {
            "_DataInfo_": {
                "JMeshVersion": "0.5",
                "Comment": "Created by BlenderPhotonics (http:\/\/mcx.space\/BlenderPhotonics)",
            },
            "MeshVertex3": v,
            "MeshTri3": f,
            "param": {"initpoints": vs, "p1": p1, "p2": p2},
        }

        jd.save(meshdata, os.path.join(outputdir, "brain1020input.jmsh"))
        # determine which backend the suer has selected: matlab or octave
        try:
            if bpy.context.scene.blender_photonics.backend == "octave":
                import oct2py as op

                oc = op.Oct2Py()
            else:
                import matlab.engine as op

                oc = op.start_matlab()
        except ImportError:
            raise ImportError(
                "To run this feature, you must install the `oct2py` or `matlab.engine` Python module first, based on your choice of the backend"
            )
        print(os.path.join(os.path.dirname(os.path.abspath(__file__)), "script"))
        oc.addpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "script"))
        # runs brain1020mesh.m in backend
        oc.feval("brain1020mesh", os.path.join(outputdir, "brain1020input.jmsh"))

        # load the resulting file with the brain1020 mesh into Blender space
        outputmesh = jd.load(os.path.join(outputdir, "brain1020output.jmsh"))
        AddMeshFromNodeFace(
            outputmesh["MeshVertex3"],
            (np.array(outputmesh["MeshTri3"]) - 1).tolist(),
            "LandmarkMesh",
        )
        bpy.context.view_layer.objects.active = bpy.data.objects["LandmarkMesh"]

        ShowMessageBox("Generating 10-20 points is completed", "BrainCapGen")

        pass


def register():
    bpy.utils.register_class(init_points)


def unregister():
    bpy.utils.unregister_class(init_points)


if __name__ == "__main__":
    register()

    bpy.ops.braincapgen.brain1020mesh("INVOKE_DEFAULT")
