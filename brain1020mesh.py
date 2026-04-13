import bpy
from bpy import context
import subprocess
import sys
from .utils import *
from .dependencies import safe_import, require_dependency, show_error_message
import numpy as np
import jdata as jd
import pathlib
from bpy.types import Operator, PropertyGroup
import os
from bpy.props import EnumProperty, StringProperty, CollectionProperty

bpy.types.Scene.neurocaptain_selected_action = bpy.props.StringProperty(
    name="NeuroCaptain Selected Action", default=""
)
bpy.types.Scene.nz_assigned = bpy.props.BoolProperty(name="NZ Assigned", default=False)
bpy.types.Scene.lpa_assigned = bpy.props.BoolProperty(name="LPA Assigned", default=False)
bpy.types.Scene.rpa_assigned = bpy.props.BoolProperty(name="RPA Assigned", default=False)
bpy.types.Scene.iz_assigned = bpy.props.BoolProperty(name="IZ Assigned", default=False)
bpy.types.Scene.cz_assigned = bpy.props.BoolProperty(name="CZ Assigned", default=False)


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
    bl_idname = "braincapgen.brain1020mesh"
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
        if not os.path.isdir(outputdir):
            os.makedirs(outputdir)
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

        context.scene.neurocaptain_selected_action = self.action
        return {"FINISHED"}

    def invoke(self, context, event):
        if self.action == "BRAIN1020_MESH":
            return context.window_manager.invoke_props_dialog(self)
        else:
            return self.execute(context)

    @staticmethod
    def nz_select(context):
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="OBJECT")
        selectedverts_nz = [v for v in bpy.context.active_object.data.vertices if v.select]
        if len(selectedverts_nz) == 0:
            ShowMessageBox(
                "Select the vertice that corresponds to NZ and select OK", "Error", "ERROR"
            )
            print("Select the vertice that corresponds to NZ and select OK")
            context.scene.nz_assigned = False

            return {"CANCELLED"}

        elif len(selectedverts_nz) > 1:
            ShowMessageBox("Please select only 1 vertex", "Error", "ERROR")
            print("Please select only 1 vertex")
            context.scene.nz_assigned = False
            return {"CANCELLED"}

        else:
            vert_nz = selectedverts_nz[0].co
            v_global_nz = obj.matrix_world @ vert_nz
            global nz
            nz = np.array([v_global_nz])
            context.scene.nz_assigned = True
            context.scene["saved_nz"] = nz
            print("nz is", nz)
            return nz

    @staticmethod
    def lpa_select(context):
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        selectedverts_lpa = [v for v in bpy.context.active_object.data.vertices if v.select]

        if len(selectedverts_lpa) == 0:
            ShowMessageBox(
                "Select the vertex that corresponds to LPA and select OK", "Error", "ERROR"
            )
            print("Select the vertice that corresponds to LPA and select OK")
            context.scene.lpa_assigned = False
            return {"CANCELLED"}

        elif len(selectedverts_lpa) > 1:
            ShowMessageBox("Please select only 1 vertex", "Error", "ERROR")
            print("Please select only 1 vertex")
            context.scene.lpa_assigned = False
            return {"CANCELLED"}

        else:
            vert_lpa = selectedverts_lpa[0].co
            v_global_lpa = obj.matrix_world @ vert_lpa
            global lpa
            lpa = np.array([v_global_lpa])
            context.scene.lpa_assigned = True
            print("lpa is", lpa)
            return lpa

    @staticmethod
    def rpa_select(context):
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        selectedverts_rpa = [v for v in bpy.context.active_object.data.vertices if v.select]

        if len(selectedverts_rpa) == 0:
            ShowMessageBox(
                "Select the vertex that corresponds to RPA and select OK", "Error", "ERROR"
            )
            print("Select the vertice that corresponds to RPA and select OK")
            context.scene.rpa_assigned = False
            return {"CANCELLED"}

        elif len(selectedverts_rpa) > 1:
            ShowMessageBox("Please select only 1 vertex", "Error", "ERROR")
            print("Please select only 1 vertex")
            context.scene.rpa_assigned = False
            return {"CANCELLED"}

        else:
            vert_rpa = selectedverts_rpa[0].co
            v_global_rpa = obj.matrix_world @ vert_rpa
            global rpa
            rpa = np.array([v_global_rpa])
            context.scene.rpa_assigned = True
            print("rpa is", rpa)
            return rpa

    @staticmethod
    def iz_select(context):
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        selectedverts_iz = [v for v in bpy.context.active_object.data.vertices if v.select]

        if len(selectedverts_iz) == 0:
            ShowMessageBox(
                "Select the vertex that corresponds to IZ and select OK", "Error", "ERROR"
            )
            print("Select the vertice that corresponds to IZ and select OK")
            context.scene.iz_assigned = False
            return {"CANCELLED"}

        elif len(selectedverts_iz) > 1:
            ShowMessageBox("Please select only 1 vertex", "Error", "ERROR")
            print("Please select only 1 vertex")
            context.scene.iz_assigned = False
            return {"CANCELLED"}

        else:
            vert_iz = selectedverts_iz[0].co
            v_global_iz = obj.matrix_world @ vert_iz
            global iz
            iz = np.array([v_global_iz])
            context.scene.iz_assigned = True
            print("iz is", iz)
            return iz

    @staticmethod
    def cz_select(context):
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        selectedverts_cz = [v for v in bpy.context.active_object.data.vertices if v.select]

        if len(selectedverts_cz) == 0:
            ShowMessageBox(
                "Select the vertex that corresponds to CZ and select OK", "Error", "ERROR"
            )
            print("Select the vertice that corresponds to CZ and select OK")
            context.scene.cz_assigned = False
            return {"CANCELLED"}

        elif len(selectedverts_cz) > 1:
            ShowMessageBox("Please select only 1 vertex", "Error", "ERROR")
            print("Please select only 1 vertex")
            context.scene.cz_assigned = False
            return {"CANCELLED"}

        else:
            vert_cz = selectedverts_cz[0].co
            v_global_cz = obj.matrix_world @ vert_cz
            global cz
            cz = np.array([v_global_cz])
            context.scene.cz_assigned = True
            print("cz is", cz)
            return cz

    @staticmethod
    def brain1020_mesh(context):
        print("entering brain1020 mesh")
        
        pmesh = safe_import('iso2mesh')
        if pmesh is None:
            show_error_message("iso2mesh is required for this feature")
            return {'CANCELLED'}

        scipy_spatial = safe_import('scipy.spatial')
        if scipy_spatial is None:
            show_error_message("scipy is required for this feature")
            return {'CANCELLED'}
        
        outputdir = GetBPWorkFolder()
        if not os.path.isdir(outputdir):
            os.makedirs(outputdir)
        
        obj = bpy.context.view_layer.objects.active
        
        vs = np.vstack((nz, iz, lpa, rpa, cz))
        
        verts = []
        for n in range(len(obj.data.vertices)):
            vert = obj.data.vertices[n].co
            v_global = obj.matrix_world @ vert
            verts.append(v_global)
        
        faces = [(np.array(face.vertices[:]) + 1).tolist() for face in obj.data.polygons]
        
        v = np.array(verts)
        f = np.array(faces)
        
        meshdata = {
            "_DataInfo_": {
                "JMeshVersion": "0.5",
                "Comment": "Created by BlenderPhotonics (http://mcx.space/BlenderPhotonics)",
            },
            "MeshVertex3": v,
            "MeshTri3": f,
            "param": {"initpoints": vs, "p1": p1, "p2": p2},
        }
        
        jd.save(meshdata, os.path.join(outputdir, "brain1020input.jmsh"))
        
        try:
            landmarks = pmesh.brain1020(
                meshdata["MeshVertex3"],
                meshdata["MeshTri3"],
                meshdata["param"]["initpoints"],
                meshdata["param"]["p1"],
                meshdata["param"]["p2"],
                cztol=1e-8,
                display=0
            )
            
            # points1 rows are in the same order as landmarks[0].keys()
            # preserve this ordering explicitly
            label_list = list(landmarks[0].keys())
            points1 = np.vstack([landmarks[0][k] for k in label_list])
            
            tet = scipy_spatial.Delaunay(points1).simplices + 1
            
            face = pmesh.volface(tet)[0]
            face = np.array(face)
            
            outputmesh = {
                "_DataInfo_": {
                    "JMeshVersion": "0.5",
                    "Comment": "Created by NeuroCaptain",
                },
                "MeshVertex3": points1,
                "MeshTri3": face
            }
            
            jd.save(outputmesh, os.path.join(outputdir, "brain1020output.jmsh"))
            
        except Exception as e:
            print(f"Error in brain1020_mesh: {e}")
            import traceback
            traceback.print_exc()
            show_error_message(f"Error generating 10-20 mesh: {str(e)}")
            return {'CANCELLED'}
        
        # Remove stale LandmarkMesh before creating new one
        # This prevents stale landmark_labels from persisting
        if "LandmarkMesh" in bpy.data.objects:
            old_mesh = bpy.data.objects["LandmarkMesh"]
            old_mesh_data = old_mesh.data
            bpy.data.objects.remove(old_mesh, do_unlink=True)
            bpy.data.meshes.remove(old_mesh_data)
        
        AddMeshFromNodeFace(
            outputmesh["MeshVertex3"],
            (np.array(outputmesh["MeshTri3"]) - 1).tolist(),
            "LandmarkMesh",
        )
        
        mesh_obj = bpy.data.objects["LandmarkMesh"]
        num_verts = len(mesh_obj.data.vertices)
        num_labels = len(label_list)
        
        print(f"brain1020_mesh: {num_labels} labels, {num_verts} vertices")
        
        if num_labels == num_verts:
            mesh_obj["landmark_labels"] = label_list
            print(f"Stored {num_labels} landmark labels: {label_list[:5]}...")
        else:
            # Mismatch — clear any stale property and report
            if "landmark_labels" in mesh_obj:
                del mesh_obj["landmark_labels"]
            print(f"WARNING: label count ({num_labels}) != vertex count ({num_verts})")
            print(f"Labels: {label_list}")
            show_error_message(
                f"Landmark label mismatch: {num_labels} labels vs {num_verts} vertices. "
                f"Labels not stored — import/export will use hardcoded fallback."
            )
        
        bpy.context.view_layer.objects.active = mesh_obj
        ShowMessageBox("Generating 10-20 points is completed", "BrainCapGen")

def register():
    bpy.utils.register_class(brain1020mesh)
    bpy.types.Scene.neurocaptain_selected_action = bpy.props.StringProperty(
        name="NeuroCaptain Selected Action", default=""
    )
    bpy.types.Scene.nz_assigned = bpy.props.BoolProperty(name="NZ Assigned", default=False)
    bpy.types.Scene.lpa_assigned = bpy.props.BoolProperty(name="LPA Assigned", default=False)
    bpy.types.Scene.rpa_assigned = bpy.props.BoolProperty(name="RPA Assigned", default=False)
    bpy.types.Scene.iz_assigned = bpy.props.BoolProperty(name="IZ Assigned", default=False)
    bpy.types.Scene.cz_assigned = bpy.props.BoolProperty(name="CZ Assigned", default=False)


def unregister():
    bpy.utils.unregister_class(brain1020mesh)
    del bpy.types.Scene.nz_selected
    del bpy.types.Scene.neurocaptain_selected_action
    del bpy.types.Scene.nz_assigned
    del bpy.types.Scene.lpa_assigned
    del bpy.types.Scene.rpa_assigned
    del bpy.types.Scene.cz_assigned
    del bpy.types.Scene.iz_assigned