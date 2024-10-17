import bpy
from .utils import *
import numpy as np
import jdata as jd
import os

# from bpy_extras.io_utils import ImportHelper
# from bpy.utils import register_class, unregister_class

g_action = "export"


class MatlabFunction:
    pass


class exportmesh(bpy.types.Operator):
    bl_idname = "braincapgen.export_mesh"
    bl_label = "Export Mesh to bmsh/jmsh"
    bl_description = "Export mesh as bmsh/jmsh"

    filename: bpy.props.StringProperty(name="File Name", default=" ")

    def execute(self, context):
        outputdir = GetBPWorkFolder()
        print("the saved mesh directory is:", outputdir)
        obj = bpy.context.view_layer.objects.active

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

        print(len(faces))
        f = np.array(faces)

        print([v.dtype, v.shape, f.dtype, f.shape])
        meshdata = {
            "_DataInfo_": {
                "JMeshVersion": "0.5",
                "Comment": "Created by BlenderPhotonics (http:\/\/mcx.space\/BlenderPhotonics)",
            },
            "MeshVertex3": v,
            "MeshTri3": f,
        }
        jd.save(meshdata, os.path.join(outputdir, self.filename))

        print([v.dtype, v.shape, f.dtype, f.shape])

        return {"FINISHED"}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)
