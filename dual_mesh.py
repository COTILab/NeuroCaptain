# SPDX-License-Identifier: GPL-2.0-or-later

# --------------------------------- DUAL MESH -------------------------------- #
# -------------------------------- version 0.3 ------------------------------- #
#                                                                              #
# Convert a generic mesh to its dual. With open meshes it can get some wired   #
# effect on the borders.                                                       #
#                                                                              #
#                        (c)   Alessandro Zomparelli                           #
#                                    (2017)                                    #
#                                                                              #
# http://www.co-de-it.com/                                                     #
#                                                                              #
# ############################################################################ #


import bpy
from bpy.types import Operator
from bpy.props import (
        BoolProperty,
        EnumProperty,
        )
import bmesh
from .utils import *

class dual_mesh(Operator):
    bl_idname = "object.dual_mesh"
    bl_label = "Convert to Dual Mesh"
    bl_description = ("Convert a generic mesh into a polygonal mesh. (Destructive)")
    bl_options = {'REGISTER', 'UNDO'}

    quad_method : EnumProperty(
            items=[('BEAUTY', 'Beauty',
                    'Split the quads in nice triangles, slower method'),
                    ('FIXED', 'Fixed',
                    'Split the quads on the 1st and 3rd vertices'),
                    ('FIXED_ALTERNATE', 'Fixed Alternate',
                    'Split the quads on the 2nd and 4th vertices'),
                    ('SHORTEST_DIAGONAL', 'Shortest Diagonal',
                    'Split the quads based on the distance between the vertices')
                    ],
            name="Quad Method",
            description="Method for splitting the quads into triangles",
            default="FIXED",
            options={'LIBRARY_EDITABLE'}
            )
    polygon_method : EnumProperty(
            items=[
                ('BEAUTY', 'Beauty', 'Arrange the new triangles evenly'),
                ('CLIP', 'Clip',
                 'Split the N-gon with an ear clipping algorithm')],
            name="N-gon Method",
            description="Method for splitting the N-gons into triangles",
            default="BEAUTY",
            options={'LIBRARY_EDITABLE'}
            )
    preserve_borders : BoolProperty(
            name="Preserve Borders",
            default=True,
            description="Preserve original borders"
            )
    apply_modifiers : BoolProperty(
            name="Apply Modifiers",
            default=True,
            description="Apply object's modifiers"
            )

    def execute(self, context):
        mode = context.mode
        if mode == 'EDIT_MESH':
            mode = 'EDIT'
        act = context.active_object
        if mode != 'OBJECT':
            sel = [act]
            bpy.ops.object.mode_set(mode='OBJECT')
        else:
            sel = context.selected_objects
        doneMeshes = []

        for ob0 in sel:
            if ob0.type != 'MESH':
                continue
            if ob0.data.name in doneMeshes:
                continue
            ob = ob0
            mesh_name = ob0.data.name

            # store linked objects
            clones = []
            n_users = ob0.data.users
            count = 0
            for o in bpy.data.objects:
                if o.type != 'MESH':
                    continue
                if o.data.name == mesh_name:
                    count += 1
                    clones.append(o)
                if count == n_users:
                    break

            if self.apply_modifiers:
                bpy.ops.object.convert(target='MESH')
            ob.data = ob.data.copy()
            bpy.ops.object.select_all(action='DESELECT')
            ob.select_set(True)
            context.view_layer.objects.active = ob0
            bpy.ops.object.mode_set(mode='EDIT')

            # prevent borders erosion
            bpy.ops.mesh.select_mode(
                    use_extend=False, use_expand=False, type='EDGE'
                    )
            bpy.ops.mesh.select_non_manifold(
                    extend=False, use_wire=False, use_boundary=True,
                    use_multi_face=False, use_non_contiguous=False,
                    use_verts=False
                    )
            bpy.ops.mesh.extrude_region_move(
                    MESH_OT_extrude_region={"mirror": False},
                    TRANSFORM_OT_translate={"value": (0, 0, 0)}
                    )

            bpy.ops.mesh.select_mode(
                    use_extend=False, use_expand=False, type='VERT',
                    action='TOGGLE'
                    )
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.quads_convert_to_tris(
                    quad_method=self.quad_method, ngon_method=self.polygon_method
                    )
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')
            subsurf_modifier = context.object.modifiers.new("dual_mesh_subsurf", 'SUBSURF')
            context.object.modifiers.move(len(context.object.modifiers)-1, 0)

            bpy.ops.object.modifier_apply(modifier=subsurf_modifier.name)

            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='DESELECT')

            verts = ob.data.vertices

            bpy.ops.object.mode_set(mode='OBJECT')
            verts[-1].select = True
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_more(use_face_step=False)

            bpy.ops.mesh.select_similar(
                type='EDGE', compare='EQUAL', threshold=0.01)
            bpy.ops.mesh.select_all(action='INVERT')

            bpy.ops.mesh.dissolve_verts()
            bpy.ops.mesh.select_all(action='DESELECT')

            bpy.ops.mesh.select_non_manifold(
                extend=False, use_wire=False, use_boundary=True,
                use_multi_face=False, use_non_contiguous=False, use_verts=False)
            bpy.ops.mesh.select_more()

            # find boundaries
            bpy.ops.object.mode_set(mode='OBJECT')
            bound_v = [v.index for v in ob.data.vertices if v.select]
            bound_e = [e.index for e in ob.data.edges if e.select]
            bound_p = [p.index for p in ob.data.polygons if p.select]
            bpy.ops.object.mode_set(mode='EDIT')

            # select quad faces
            context.tool_settings.mesh_select_mode = (False, False, True)
            bpy.ops.mesh.select_face_by_sides(number=4, extend=False)

            # deselect boundaries
            bpy.ops.object.mode_set(mode='OBJECT')
            for i in bound_v:
                context.active_object.data.vertices[i].select = False
            for i in bound_e:
                context.active_object.data.edges[i].select = False
            for i in bound_p:
                context.active_object.data.polygons[i].select = False

            bpy.ops.object.mode_set(mode='EDIT')

            context.tool_settings.mesh_select_mode = (False, False, True)
            bpy.ops.mesh.edge_face_add()
            context.tool_settings.mesh_select_mode = (True, False, False)
            bpy.ops.mesh.select_all(action='DESELECT')

            # delete boundaries
            bpy.ops.mesh.select_non_manifold(
                    extend=False, use_wire=True, use_boundary=True,
                    use_multi_face=False, use_non_contiguous=False, use_verts=True
                    )
            bpy.ops.mesh.delete(type='VERT')

            # remove middle vertices
            bm = bmesh.from_edit_mesh(ob.data)
            for v in bm.verts:
                if len(v.link_edges) == 2 and len(v.link_faces) < 3:
                    v.select = True

            # dissolve
            bpy.ops.mesh.dissolve_verts()
            bpy.ops.mesh.select_all(action='DESELECT')

            # remove border faces
            if not self.preserve_borders:
                bpy.ops.mesh.select_non_manifold(
                    extend=False, use_wire=False, use_boundary=True,
                    use_multi_face=False, use_non_contiguous=False, use_verts=False
                    )
                bpy.ops.mesh.select_more()
                bpy.ops.mesh.delete(type='FACE')

            # clean wires
            bpy.ops.mesh.select_non_manifold(
                    extend=False, use_wire=True, use_boundary=False,
                    use_multi_face=False, use_non_contiguous=False, use_verts=False
                    )
            bpy.ops.mesh.delete(type='EDGE')

            bpy.ops.object.mode_set(mode='OBJECT')
            ob0.data.name = 'hello'
            doneMeshes.append('hello')

            for o in clones:
                o.data = ob.data
            bm.free()

            for o in sel:
                o.select_set(True)
            
            return {'FINISHED'}
