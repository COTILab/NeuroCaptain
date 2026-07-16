import bmesh
import bpy
from bpy.types import Operator


class decimate_mesh(bpy.types.Operator):
    bl_idname = "braincapgen.decimate_mesh"
    bl_label = "Modify Mesh Density"
    bl_description = "modify density of mesh after choosing Nz,Lpa,Rpa)"
    number: bpy.props.FloatProperty(name="Decimate Ratio", default=1)

    def execute(self, context):
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass
        bpy.ops.object.select_all(action="DESELECT")
        head = bpy.data.objects["headmesh"]
        head.select_set(True)
        bpy.context.view_layer.objects.active = head
        obj = bpy.context.object
        mod = obj.modifiers.new(name="decimate", type="DECIMATE")
        mod.decimate_type = "COLLAPSE"
        # user defined decimate ratio of faces to keep
        mod.ratio = self.number
        # Keeps the collapsed result cleanly triangulated instead of merging
        # into potentially-degenerate ngons - at aggressive ratios (e.g.
        # 0.05) COLLAPSE decimation is otherwise prone to leaving zero-area
        # slivers and inconsistent face winding, which destabilizes the
        # downstream boolean-cut + wireframe + voxel-remesh cap-generation
        # steps (observed: a near-empty sliver mesh, or a boolean-operator
        # crash on Blender's pre-4.0 FAST solver).
        if hasattr(mod, "use_collapse_triangulate"):
            mod.use_collapse_triangulate = True

        head = bpy.data.objects["headmesh"]
        head.select_set(True)
        bpy.ops.object.modifier_apply(modifier="decimate")

        # Belt-and-suspenders cleanup after the modifier apply: remove any
        # zero-area faces the collapse still left behind and make sure face
        # winding is consistent, so downstream booleans get well-formed
        # input regardless of how aggressive the ratio is.
        bm = bmesh.new()
        bm.from_mesh(head.data)
        bmesh.ops.dissolve_degenerate(bm, dist=0.0001, edges=bm.edges)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(head.data)
        bm.free()
        head.data.update()

        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
