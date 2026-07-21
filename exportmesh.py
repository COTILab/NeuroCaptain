import bpy
from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty
from .utils import *
import numpy as np
import jdata as jd


class exportmesh(bpy.types.Operator, ExportHelper):
    bl_idname = "braincapgen.export_mesh"
    bl_label = "Export Mesh"
    bl_description = "Export the active mesh as a .bmsh/.jmsh file"

    filename_ext = ".jmsh"
    filter_glob: StringProperty(
        default="*.jmsh;*.bmsh",
        options={"HIDDEN"},
    )

    def execute(self, context):
        obj = bpy.context.view_layer.objects.active
        if obj is None:
            self.report({"ERROR"}, "No active object to export")
            return {"CANCELLED"}

        bpy.ops.object.modifier_add(type="TRIANGULATE")
        bpy.ops.object.modifier_apply(modifier="Triangulate")

        verts = []
        # saving the head mesh from the scene
        for n in range(len(obj.data.vertices)):
            vert = obj.data.vertices[n].co
            v_global = obj.matrix_world @ vert
            verts.append(v_global)

        faces = [[v + 1 for v in face.vertices[:]] for face in obj.data.polygons]

        v = np.array(verts)

        f = np.array(faces)

        meshdata = {
            "_DataInfo_": {
                "JMeshVersion": "0.5",
                "Comment": "Created by BlenderPhotonics (http:\/\/mcx.space\/BlenderPhotonics)",
            },
            "MeshVertex3": v,
            "MeshTri3": f,
        }
        jd.save(meshdata, self.filepath)
        self.report({"INFO"}, f"Exported mesh to: {self.filepath}")

        return {"FINISHED"}
