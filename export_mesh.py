import bpy
from bpy_extras.io_utils import ImportHelper
import numpy as np
import jdata as jd
import os
from bpy.utils import register_class, unregister_class
from .utils import *


class export_mesh(bpy.types.Operator):
    bl_idname = "braincapgen.export_mesh"
    bl_label = "Export Mesh to bmsh/jmsh"
    bl_description = "Export mesh as bmsh/jmsh"
    # number: bpy.props.FloatProperty(name="Decimate Ratio", default=1)

    def execute(self, context):
        surfdata = {
            "_DataInfo_": {
                "JMeshVersion": "0.5",
                "Comment": "Object surface mesh created by BlenderPhotonics (http:\/\/mcx.space\/BlenderPhotonics)",
            }
        }
        surfdata["MeshGroup"] = []
        for ob in bpy.context.selected_objects:
            objsurf = GetNodeFacefromObject(ob, self.convtri)
            surfdata["MeshGroup"].append(objsurf)

        surfdata["param"] = {"action": self.action, "level": self.actionparam}

        jd.save(surfdata, os.path.join(outputdir, "blendersurf.jmsh"))

        return {"Finished"}
