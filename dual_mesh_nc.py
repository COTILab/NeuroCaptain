import bpy
from bpy.types import Operator
from bpy.props import (
    BoolProperty,
    EnumProperty,
)
import bmesh
from .utils import *
from mesh_tissue.dual_mesh import dual_mesh_tessellated


class dual_mesh_NC(Operator):
    bl_idname = "object.dual_mesh"
    bl_label = "Convert to Dual Mesh"
    bl_description = "Convert a generic mesh into a polygonal mesh. (Destructive)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            bpy.ops.object.dual_mesh_tessellated()
        except AttributeError:
            print("MESH:TISSUE ADDON NOT INSTALLED")
        head = bpy.data.objects["headmesh"]
        head.name = "old copy"
        dual = bpy.data.objects["DualMesh"]
        bpy.ops.object.select_all(action="DESELECT")
        head.select_set(True)
        bpy.ops.object.delete()
        dual.select_set(True)
        dual.name = "headmesh"

        return {"FINISHED"}
