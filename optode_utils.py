import bpy
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
import bpy_extras.view3d_utils as view3d_utils


class _OptodeMixin:
    """Shared helpers for Add Source / Add Detector modal operators."""

    _optode = None
    _bvh = None
    _head_mesh = None
    _mat_world = None
    _mat_inv = None
    _mat3_inv_T = None
    _optode_diameter = 0.0
    _optode_thickness = 0.0

    # ------------------------------------------------------------------
    # Modal placement
    # ------------------------------------------------------------------

    def _invoke_modal(self, context, event, optode_type, color, collection_name):
        head_mesh = self.find_head_mesh()
        if not head_mesh:
            self.report({'ERROR'}, "Could not find headmesh")
            return {'CANCELLED'}

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # If invoked outside a 3D view (e.g. scripting), fall back to execute
        if not context.area or context.area.type != 'VIEW_3D':
            return self.execute(context)

        self._head_mesh = head_mesh
        depsgraph = context.evaluated_depsgraph_get()
        self._bvh = BVHTree.FromObject(head_mesh, depsgraph)
        self._mat_world = head_mesh.matrix_world.copy()
        self._mat_inv = head_mesh.matrix_world.inverted()
        self._mat3_inv_T = head_mesh.matrix_world.to_3x3().inverted().transposed()

        diameter, thickness = self.get_head_scale(head_mesh)
        self._optode_diameter = diameter
        self._optode_thickness = thickness

        num = self.get_next_optode_number(optode_type)
        name = f"{optode_type}_{num}"

        pos, normal = self._raycast_mouse(context, event)
        if pos is None:
            pos, normal = self._fallback_position(head_mesh)

        collection = self.get_or_create_collection(collection_name)
        self._optode = self.create_optode_disc(pos, normal, name, color, diameter, thickness)
        self.add_to_collection(self._optode, collection)

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            pos, normal = self._raycast_mouse(context, event)
            if pos is not None:
                self._update_transform(pos, normal)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self.apply_shrinkwrap(self._optode, self._head_mesh)
            self.report({'INFO'}, f"Placed {self._optode.name}")
            return {'FINISHED'}

        if event.type in ('RIGHTMOUSE', 'ESC'):
            bpy.data.objects.remove(self._optode, do_unlink=True)
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def _raycast_mouse(self, context, event):
        """Cast ray from mouse into 3D view; return (world_pos, world_normal) or (None, None)."""
        if not context.area or context.area.type != 'VIEW_3D':
            return None, None
        region = context.region
        rv3d = context.space_data.region_3d
        if region is None or rv3d is None:
            return None, None

        coord = (event.mouse_region_x, event.mouse_region_y)
        ray_origin_world = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        ray_dir_world = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

        # BVH operates in head-mesh local space
        ray_origin_local = self._mat_inv @ ray_origin_world
        ray_dir_local = (self._mat_inv.to_3x3() @ ray_dir_world).normalized()

        hit_local, normal_local, _, _ = self._bvh.ray_cast(ray_origin_local, ray_dir_local)
        if hit_local is None:
            return None, None

        hit_world = self._mat_world @ hit_local
        normal_world = (self._mat3_inv_T @ normal_local).normalized()
        return np.array(hit_world), np.array(normal_world)

    def _update_transform(self, position, normal):
        """Directly set optode location and orientation from a surface hit (no constraint overhead)."""
        normal_vec = Vector(normal).normalized()
        self._optode.rotation_euler = self._normal_to_euler(normal_vec)
        self._optode.location = Vector(position) + normal_vec * (self._optode_thickness * 0.5)

    def _normal_to_euler(self, normal_vec):
        """Stable Z-up → normal rotation using a consistent up-vector (avoids quaternion flips)."""
        z = Vector(normal_vec).normalized()
        up = Vector((0, 1, 0))
        if abs(z.dot(up)) > 0.98:
            up = Vector((1, 0, 0))
        x = up.cross(z).normalized()
        y = z.cross(x).normalized()
        return Matrix((x, y, z)).transposed().to_euler()

    def _fallback_position(self, head_mesh):
        """Return (world_pos, world_normal) at Cz or the highest head point."""
        cz_obj = bpy.data.objects.get("Cz")
        if cz_obj:
            pos = np.array(cz_obj.location)
        else:
            verts = np.array([v.co for v in head_mesh.data.vertices])
            mat = np.array(head_mesh.matrix_world)[:3, :]
            world_z = (mat[2, :3] @ verts.T) + mat[2, 3]
            pos = np.array([0, 0, float(world_z.max())])

        local_pos = self._mat_inv @ Vector(pos.tolist())
        hit_local, normal_local, _, _ = self._bvh.find_nearest(local_pos)
        if hit_local is None:
            return pos, np.array([0, 0, 1])
        hit_world = self._mat_world @ hit_local
        normal_world = (self._mat3_inv_T @ normal_local).normalized()
        return np.array(hit_world), np.array(normal_world)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def find_head_mesh(self):
        obj = bpy.data.objects.get("headmesh")
        if obj and obj.type == 'MESH':
            return obj
        for obj in bpy.data.objects:
            if "headmesh" in obj.name.lower() and obj.type == 'MESH':
                return obj
        return None

    def get_optode_positions(self, context, head_mesh):
        """Return world-space positions from selected vertices, or fall back to Cz / highest point."""
        positions = []
        if context.mode == 'EDIT_MESH':
            edit_object = next(
                (o for o in context.selected_objects if o.type == 'MESH' and o.mode == 'EDIT'),
                context.active_object if (context.active_object and
                    context.active_object.type == 'MESH' and
                    context.active_object.mode == 'EDIT') else None
            )
            if edit_object:
                import bmesh
                bm = bmesh.from_edit_mesh(edit_object.data)
                for vert in bm.verts:
                    if vert.select:
                        wp = edit_object.matrix_world @ vert.co.copy()
                        positions.append(np.array([wp.x, wp.y, wp.z]))
                bpy.ops.object.mode_set(mode='OBJECT')
                if positions:
                    return positions
        elif context.mode == 'OBJECT':
            if context.active_object and context.active_object.type == 'MESH':
                obj = context.active_object
                for vert in obj.data.vertices:
                    if vert.select:
                        wp = obj.matrix_world @ vert.co.copy()
                        positions.append(np.array([wp.x, wp.y, wp.z]))
                if positions:
                    return positions

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        cz_obj = bpy.data.objects.get("Cz")
        if cz_obj:
            loc = cz_obj.location
            return [np.array([loc.x, loc.y, loc.z])]

        if head_mesh:
            verts = np.array([v.co for v in head_mesh.data.vertices])
            mat = np.array(head_mesh.matrix_world)[:3, :]
            world_z = (mat[2, :3] @ verts.T) + mat[2, 3]
            return [np.array([0, 0, float(world_z.max())])]

        return [np.array([0, 0, 0.12])]

    def get_next_optode_number(self, prefix):
        numbers = []
        for obj in bpy.data.objects:
            if obj.name.startswith(f"{prefix}_"):
                try:
                    numbers.append(int(obj.name.split("_")[1]))
                except (IndexError, ValueError):
                    pass
        return max(numbers) + 1 if numbers else 1

    def get_head_scale(self, head_mesh):
        verts = np.array([v.co for v in head_mesh.data.vertices])
        bbox_min = head_mesh.matrix_world @ Vector(verts.min(axis=0).tolist())
        bbox_max = head_mesh.matrix_world @ Vector(verts.max(axis=0).tolist())
        head_size = max(abs((bbox_max - bbox_min).x),
                        abs((bbox_max - bbox_min).y),
                        abs((bbox_max - bbox_min).z))
        diameter = head_size * 0.02
        thickness = diameter * 0.15
        return diameter, thickness

    def snap_to_mesh_surface(self, position, head_mesh):
        """Snap a world-space position to the nearest point on the head mesh surface."""
        bvh = BVHTree.FromObject(head_mesh, bpy.context.evaluated_depsgraph_get())
        mat_inv = head_mesh.matrix_world.inverted()
        mat3_inv_T = head_mesh.matrix_world.to_3x3().inverted().transposed()
        hit_local, normal_local, _, _ = bvh.find_nearest(mat_inv @ Vector(position.tolist()))
        if hit_local is None:
            return position, np.array([0, 0, 1])
        hit_world = head_mesh.matrix_world @ hit_local
        normal_world = (mat3_inv_T @ normal_local).normalized()
        return np.array(hit_world), np.array(normal_world)

    def create_optode_disc(self, position, normal, name, color, diameter, thickness):
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=16,
            radius=diameter / 2.0,
            depth=thickness,
            enter_editmode=False,
            align='WORLD',
            location=(0, 0, 0)
        )
        optode = bpy.context.active_object
        optode.name = name

        normal_vec = Vector(normal).normalized()
        optode.rotation_euler = self._normal_to_euler(normal_vec)
        optode.location = Vector(position) + normal_vec * (thickness * 0.5)

        mat_name = f"{name}_Material"
        mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        mat.diffuse_color = color
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Metallic'].default_value = 0.3
            bsdf.inputs['Roughness'].default_value = 0.4
        if len(optode.data.materials) > 0:
            optode.data.materials[0] = mat
        else:
            optode.data.materials.append(mat)
        if bpy.app.version >= (4, 1, 0):
            bpy.ops.object.shade_smooth()
        else:
            for face in optode.data.polygons:
                face.use_smooth = True

        return optode

    def get_or_create_collection(self, collection_name):
        collection = bpy.data.collections.get(collection_name)
        if not collection:
            collection = bpy.data.collections.new(collection_name)
            bpy.context.scene.collection.children.link(collection)
        return collection

    def add_to_collection(self, obj, collection):
        for col in obj.users_collection:
            col.objects.unlink(obj)
        collection.objects.link(obj)

    def apply_shrinkwrap(self, obj, head_mesh):
        for con in [c for c in obj.constraints if c.type == 'SHRINKWRAP']:
            obj.constraints.remove(con)
        constraint = obj.constraints.new(type='SHRINKWRAP')
        constraint.target = head_mesh
        constraint.shrinkwrap_type = 'NEAREST_SURFACE'
        constraint.wrap_mode = 'ON_SURFACE'
        constraint.use_track_normal = True
        constraint.track_axis = 'TRACK_Z'


# ----------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------

class NEUROCAPTAIN_OT_add_source(_OptodeMixin, bpy.types.Operator):
    """Add a Source Optode — move mouse to position, left-click to confirm, ESC/right-click to cancel"""
    bl_idname = "neurocaptain.add_source"
    bl_label = "Add Source"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        return self._invoke_modal(context, event, "Source", (1.0, 0.0, 0.0, 1.0), "Sources")

    def execute(self, context):
        head_mesh = self.find_head_mesh()
        if not head_mesh:
            self.report({'ERROR'}, "Could not find headmesh")
            return {'CANCELLED'}
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        head_mesh.select_set(True)
        bpy.context.view_layer.objects.active = head_mesh

        positions = self.get_optode_positions(context, head_mesh)
        diameter, thickness = self.get_head_scale(head_mesh)
        collection = self.get_or_create_collection("Sources")
        count = 0
        for position in positions:
            name = f"Source_{self.get_next_optode_number('Source')}"
            pos, normal = self.snap_to_mesh_surface(position, head_mesh)
            optode = self.create_optode_disc(pos, normal, name, (1.0, 0.0, 0.0, 1.0), diameter, thickness)
            self.add_to_collection(optode, collection)
            self.apply_shrinkwrap(optode, head_mesh)
            count += 1
        self.report({'INFO'}, f"Added {count} source(s)")
        return {'FINISHED'}


class NEUROCAPTAIN_OT_add_detector(_OptodeMixin, bpy.types.Operator):
    """Add a Detector Optode — move mouse to position, left-click to confirm, ESC/right-click to cancel"""
    bl_idname = "neurocaptain.add_detector"
    bl_label = "Add Detector"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        return self._invoke_modal(context, event, "Detector", (0.0, 0.0, 0.0, 1.0), "Detectors")

    def execute(self, context):
        head_mesh = self.find_head_mesh()
        if not head_mesh:
            self.report({'ERROR'}, "Could not find headmesh")
            return {'CANCELLED'}
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        head_mesh.select_set(True)
        bpy.context.view_layer.objects.active = head_mesh

        positions = self.get_optode_positions(context, head_mesh)
        diameter, thickness = self.get_head_scale(head_mesh)
        collection = self.get_or_create_collection("Detectors")
        count = 0
        for position in positions:
            name = f"Detector_{self.get_next_optode_number('Detector')}"
            pos, normal = self.snap_to_mesh_surface(position, head_mesh)
            optode = self.create_optode_disc(pos, normal, name, (0.0, 0.0, 0.0, 1.0), diameter, thickness)
            self.add_to_collection(optode, collection)
            self.apply_shrinkwrap(optode, head_mesh)
            count += 1
        self.report({'INFO'}, f"Added {count} detector(s)")
        return {'FINISHED'}


class NEUROCAPTAIN_OT_move_optode(_OptodeMixin, bpy.types.Operator):
    """Reposition selected Source/Detector optodes — move mouse to new position, left-click to confirm, ESC to cancel"""
    bl_idname = "neurocaptain.move_optode"
    bl_label = "Move Selected Optode"
    bl_options = {'REGISTER', 'UNDO'}

    _optodes = []
    _saved_states = []
    _anchor_pos = None
    _offsets = []

    def invoke(self, context, event):
        selected = [
            obj for obj in context.selected_objects
            if obj.name.startswith("Source_") or obj.name.startswith("Detector_")
        ]
        if not selected:
            self.report({'WARNING'}, "Select one or more Source or Detector optodes first")
            return {'CANCELLED'}

        head_mesh = self.find_head_mesh()
        if not head_mesh:
            self.report({'ERROR'}, "Could not find headmesh")
            return {'CANCELLED'}

        if not context.area or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "Must be used from the 3D Viewport")
            return {'CANCELLED'}

        self._optodes = selected
        self._saved_states = [(obj.location.copy(), obj.rotation_euler.copy()) for obj in selected]

        self._head_mesh = head_mesh
        depsgraph = context.evaluated_depsgraph_get()
        self._bvh = BVHTree.FromObject(head_mesh, depsgraph)
        self._mat_world = head_mesh.matrix_world.copy()
        self._mat_inv = head_mesh.matrix_world.inverted()
        self._mat3_inv_T = head_mesh.matrix_world.to_3x3().inverted().transposed()

        _, thickness = self.get_head_scale(head_mesh)
        self._optode_thickness = thickness

        # Anchor = where the mouse currently hits the surface; offsets keep relative spacing
        anchor, _ = self._raycast_mouse(context, event)
        if anchor is None:
            anchor = np.array(selected[0].location)
        self._anchor_pos = anchor
        self._offsets = [np.array(obj.location) - anchor for obj in selected]

        for obj in selected:
            for con in [c for c in obj.constraints if c.type == 'SHRINKWRAP']:
                obj.constraints.remove(con)

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            pos, _ = self._raycast_mouse(context, event)
            if pos is not None:
                self._move_group(pos)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            for obj in self._optodes:
                self.apply_shrinkwrap(obj, self._head_mesh)
            names = ", ".join(obj.name for obj in self._optodes)
            self.report({'INFO'}, f"Moved {names}")
            return {'FINISHED'}

        if event.type in ('RIGHTMOUSE', 'ESC'):
            for obj, (loc, rot) in zip(self._optodes, self._saved_states):
                obj.location = loc
                obj.rotation_euler = rot
                self.apply_shrinkwrap(obj, self._head_mesh)
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def _move_group(self, anchor_pos):
        """Move all optodes by snapping each to the surface at its offset from the anchor."""
        for obj, offset in zip(self._optodes, self._offsets):
            target = Vector(anchor_pos + offset)
            local_pos = self._mat_inv @ target
            hit_local, normal_local, _, _ = self._bvh.find_nearest(local_pos)
            if hit_local is not None:
                hit_world = self._mat_world @ hit_local
                normal_world = (self._mat3_inv_T @ normal_local).normalized()
                obj.location = hit_world + normal_world * (self._optode_thickness * 0.5)
                obj.rotation_euler = self._normal_to_euler(normal_world)
            else:
                obj.location = target


class NEUROCAPTAIN_OT_ensure_optode_constraints(bpy.types.Operator):
    """Ensure all optodes have proper shrinkwrap constraints"""
    bl_idname = "neurocaptain.ensure_optode_constraints"
    bl_label = "Ensure Optode Constraints"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        optodes = [o for o in bpy.data.objects
                   if o.name.startswith("Source_") or o.name.startswith("Detector_")]
        if not optodes:
            self.report({'WARNING'}, "No optodes found (Source_# or Detector_#)")
            return {'CANCELLED'}

        head_mesh = bpy.data.objects.get("headmesh")
        if not head_mesh:
            for obj in bpy.data.objects:
                if "headmesh" in obj.name.lower() and obj.type == 'MESH':
                    head_mesh = obj
                    break
        if not head_mesh:
            self.report({'ERROR'}, "Could not find headmesh for shrinkwrap target")
            return {'CANCELLED'}

        for optode in optodes:
            for con in [c for c in optode.constraints if c.type == 'SHRINKWRAP']:
                optode.constraints.remove(con)
            constraint = optode.constraints.new(type='SHRINKWRAP')
            constraint.target = head_mesh
            constraint.shrinkwrap_type = 'NEAREST_SURFACE'
            constraint.wrap_mode = 'ON_SURFACE'
            constraint.use_track_normal = True
            constraint.track_axis = 'TRACK_Z'

        self.report({'INFO'}, f"Fixed {len(optodes)} optodes → targeting '{head_mesh.name}'")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(NEUROCAPTAIN_OT_add_source)
    bpy.utils.register_class(NEUROCAPTAIN_OT_add_detector)
    bpy.utils.register_class(NEUROCAPTAIN_OT_move_optode)
    bpy.utils.register_class(NEUROCAPTAIN_OT_ensure_optode_constraints)


def unregister():
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_ensure_optode_constraints)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_move_optode)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_add_detector)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_add_source)
