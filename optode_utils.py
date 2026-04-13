import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


class NEUROCAPTAIN_OT_add_source(bpy.types.Operator):
    """add a Source Optode"""
    bl_idname = "neurocaptain.add_source"
    bl_label = "Add Source"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        head_mesh = self.find_head_mesh()
        if not head_mesh:
            self.report({'ERROR'}, "Could not find headmesh")
            return {'CANCELLED'}
        #ensure object mode
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        head_mesh.select_set(True)
        bpy.context.view_layer.objects.active = head_mesh
        #positions from selected vertices or default to Cz 
        positions = self.get_optode_positions(context, head_mesh)
        #optode size relative ot headmesh scale 
        optode_diameter, optode_thickness = self.get_head_scale(head_mesh)
        #create Sources collection
        sources_collection = self.get_or_create_collection("Sources")
        
        # create optode for each position
        created_count = 0
        for position in positions:
            source_num = self.get_next_optode_number("Source")
            optode_name = f"Source_{source_num}"
            
            # snap to surface and get normal
            snapped_pos, normal = self.snap_to_mesh_surface(position, head_mesh)
            
            # create optode disc for visualization
            optode = self.create_optode_disc(
                snapped_pos,
                normal,
                optode_name,
                (1.0, 0.0, 0.0, 1.0),
                optode_diameter,
                optode_thickness
            )
            
            self.add_to_collection(optode, sources_collection)
            
            # apply shrinkwrap constraint
            self.apply_shrinkwrap(optode, head_mesh)
            
            created_count += 1
        
        self.report({'INFO'}, f"Added {created_count} source(s)")
        return {'FINISHED'}
    
    def find_head_mesh(self):
        """Find the head mesh in the scene.

        Priority:
          1. Exact name 'headmesh'
          2. Any object whose name contains 'headmesh' (case-insensitive)
        We intentionally do NOT match bare 'head' to avoid picking up
        5-layer visualisation objects like 'Head_Surface_5L'.
        """
        # Exact match first (fastest, most reliable)
        obj = bpy.data.objects.get("headmesh")
        if obj and obj.type == 'MESH':
            return obj
        # Substring match for "headmesh" only — not just "head"
        for obj in bpy.data.objects:
            if "headmesh" in obj.name.lower() and obj.type == 'MESH':
                return obj
        return None

    def get_optode_positions(self, context, head_mesh):
        """get positions from selected vertices or default to Cz"""
        positions = []
        
        # check if in edit mode
        edit_object = None
        if context.mode == 'EDIT_MESH':
            for obj in context.selected_objects:
                if obj.type == 'MESH' and obj.mode == 'EDIT':
                    edit_object = obj
                    break
            
            # fallback to active object if nothing selected
            if not edit_object and context.active_object:
                if context.active_object.type == 'MESH' and context.active_object.mode == 'EDIT':
                    edit_object = context.active_object
        
            # get selected vertex positions in edit mode
            if edit_object:
                import bmesh
                bm = bmesh.from_edit_mesh(edit_object.data)
                
                for vert in bm.verts:
                    if vert.select:
                        local_pos = vert.co.copy()
                        world_pos = edit_object.matrix_world @ local_pos
                        positions.append(np.array([world_pos.x, world_pos.y, world_pos.z]))
                
                bpy.ops.object.mode_set(mode='OBJECT')
                
                # return any selected positions
                if positions:
                    return positions
        
        # Check if we're in object mode with active mesh 
        elif context.mode == 'OBJECT':
            if context.active_object and context.active_object.type == 'MESH':
                obj = context.active_object
                mesh = obj.data
                for vert in mesh.vertices:
                    if vert.select:
                        local_pos = vert.co.copy()
                        world_pos = obj.matrix_world @ local_pos
                        positions.append(np.array([world_pos.x, world_pos.y, world_pos.z]))
                
                # return any selected vertices 
                if positions:
                    return positions
        
        # ensure object mode (pass if already in object mode)
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # find Cz landmark (top of head)
        cz_obj = bpy.data.objects.get("Cz")
        if cz_obj:
            loc = cz_obj.location
            positions.append(np.array([loc.x, loc.y, loc.z]))
            return positions
        
        # If no Cz, use the highest point on the head mesh (numpy — fast on high-poly)
        if head_mesh:
            verts = np.array([v.co for v in head_mesh.data.vertices])
            mat = np.array(head_mesh.matrix_world)[:3, :]
            world_z = (mat[2, :3] @ verts.T) + mat[2, 3]
            highest_z = float(world_z.max())
            positions.append(np.array([0, 0, highest_z]))
        else:
            positions.append(np.array([0, 0, 0.12]))

        return positions

    def get_next_optode_number(self, prefix):
        """Optode namming sequenc"""
        existing = [obj.name for obj in bpy.data.objects if obj.name.startswith(f"{prefix}_")]
        numbers = []
        for name in existing:
            try:
                num = int(name.split("_")[1])
                numbers.append(num)
            except (IndexError, ValueError):
                continue
        
        return max(numbers) + 1 if numbers else 1
    
    def get_head_scale(self, head_mesh):
        """Optode size based on head dimensions (numpy — fast on high-poly meshes)."""
        verts = np.array([v.co for v in head_mesh.data.vertices])
        bbox_min = Vector(verts.min(axis=0).tolist())
        bbox_max = Vector(verts.max(axis=0).tolist())
        bbox_min_world = head_mesh.matrix_world @ bbox_min
        bbox_max_world = head_mesh.matrix_world @ bbox_max
        dimensions = bbox_max_world - bbox_min_world
        head_size = max(dimensions)
        optode_diameter = head_size * 0.02
        optode_thickness = optode_diameter * 0.15
        return optode_diameter, optode_thickness

    def snap_to_mesh_surface(self, position, head_mesh):
        """Snap to nearest surface location"""
        bvh = BVHTree.FromObject(head_mesh, bpy.context.evaluated_depsgraph_get())
        location, normal, index, distance = bvh.find_nearest(Vector(position))
        
        if location is None:
            return position, np.array([0, 0, 1])
        
        return np.array(location), np.array(normal)
    
    def create_optode_disc(self, position, normal, name, color, diameter, thickness):
        """Create optode disc at position"""
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=diameter / 2.0,
            depth=thickness,
            enter_editmode=False,
            align='WORLD',
            location=(0, 0, 0)
        )
        
        optode = bpy.context.active_object
        optode.name = name
        
        # align to surface normal
        z_axis = Vector((0, 0, 1))
        normal_vec = Vector(normal)
        rotation_quat = z_axis.rotation_difference(normal_vec)
        optode.rotation_euler = rotation_quat.to_euler()
        
        # center of cylinder on surface of head 
        offset_position = Vector(position) + (normal_vec.normalized() * (1))
        optode.location = offset_position 
        
        # Create unique material for this optode
        mat_name = f"{name}_Material"
        mat = bpy.data.materials.get(mat_name)
        if not mat:
            mat = bpy.data.materials.new(name=mat_name)
        
        mat.use_nodes = True
        mat.diffuse_color = color
        # bsdf combines multiple layers into one layer used for modeling materials
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Metallic'].default_value = 0.3
            bsdf.inputs['Roughness'].default_value = 0.4
        
        # assign material 
        if len(optode.data.materials) > 0:
            optode.data.materials[0] = mat
        else:
            optode.data.materials.append(mat)
        
        # smooh shading (just looks nicer)
        for face in optode.data.polygons:
            face.use_smooth = True
        
        return optode
    
    def get_or_create_collection(self, collection_name):
        """Get existing collection or create new one"""
        collection = bpy.data.collections.get(collection_name)
        if not collection:
            collection = bpy.data.collections.new(collection_name)
            bpy.context.scene.collection.children.link(collection)
        return collection
    
    def add_to_collection(self, obj, collection):
        """Add object to collection"""
        # Unlink from all current collections
        for col in obj.users_collection:
            col.objects.unlink(obj)
        
        # Link to target collection
        collection.objects.link(obj)
    
    def apply_shrinkwrap(self, obj, head_mesh):
        """Clear any existing shrinkwrap constraints then add one clean one."""
        for con in [c for c in obj.constraints if c.type == 'SHRINKWRAP']:
            obj.constraints.remove(con)
        constraint = obj.constraints.new(type='SHRINKWRAP')
        constraint.target = head_mesh
        constraint.shrinkwrap_type = 'NEAREST_SURFACE'
        constraint.wrap_mode = 'ON_SURFACE'
        constraint.use_track_normal = True
        constraint.track_axis = 'TRACK_Z'


class NEUROCAPTAIN_OT_add_detector(bpy.types.Operator):
    """Add a Detector Optode"""
    bl_idname = "neurocaptain.add_detector"
    bl_label = "Add Detector"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # headmesh is selected and active
        head_mesh = self.find_head_mesh()
        if not head_mesh:
            self.report({'ERROR'}, "Could not find headmesh")
            return {'CANCELLED'}
                #ensure object mode
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        head_mesh.select_set(True)
        bpy.context.view_layer.objects.active = head_mesh
        
         # Get positions from selected vertices or default to Cz 
        positions = self.get_optode_positions(context, head_mesh)
        # optode size relative to headmesh scale
        optode_diameter, optode_thickness = self.get_head_scale(head_mesh)
        
        # create Detectors collection
        detectors_collection = self.get_or_create_collection("Detectors")
        
        # create optode for each position
        created_count = 0
        for position in positions:
            detector_num = self.get_next_optode_number("Detector")
            optode_name = f"Detector_{detector_num}"
            
            # snap to surface and get normal
            snapped_pos, normal = self.snap_to_mesh_surface(position, head_mesh)
            
            #  optode disc
            optode = self.create_optode_disc(
                snapped_pos,
                normal,
                optode_name,
                (0.0, 0.0, 0.0, 1.0),
                optode_diameter,
                optode_thickness
            )
            
            self.add_to_collection(optode, detectors_collection)
            self.apply_shrinkwrap(optode, head_mesh)
            
            created_count += 1
        
        self.report({'INFO'}, f"Added {created_count} detector(s)")
        return {'FINISHED'}
    
    def find_head_mesh(self):
        """Find the head mesh in the scene.

        Priority:
          1. Exact name 'headmesh'
          2. Any object whose name contains 'headmesh' (case-insensitive)
        We intentionally do NOT match bare 'head' to avoid picking up
        5-layer visualisation objects like 'Head_Surface_5L'.
        """
        obj = bpy.data.objects.get("headmesh")
        if obj and obj.type == 'MESH':
            return obj
        for obj in bpy.data.objects:
            if "headmesh" in obj.name.lower() and obj.type == 'MESH':
                return obj
        return None

    def get_optode_positions(self, context, head_mesh):
        """Get positions from selected vertices or default to Cz"""
        positions = []
                # Check if we're in edit mode with selected vertices
        edit_object = None
        if context.mode == 'EDIT_MESH':
            for obj in context.selected_objects:
                if obj.type == 'MESH' and obj.mode == 'EDIT':
                    edit_object = obj
                    break
            
            # active object if nothing selected
            if not edit_object and context.active_object:
                if context.active_object.type == 'MESH' and context.active_object.mode == 'EDIT':
                    edit_object = context.active_object
        
            # get selected vertices in edit mofe 
            if edit_object:
                import bmesh
                bm = bmesh.from_edit_mesh(edit_object.data)
                
                for vert in bm.verts:
                    if vert.select:
                        local_pos = vert.co.copy()
                        world_pos = edit_object.matrix_world @ local_pos
                        positions.append(np.array([world_pos.x, world_pos.y, world_pos.z]))
                
                # switch to object mode
                bpy.ops.object.mode_set(mode='OBJECT')
                
                # return any seleted vertices (if any)
                if positions:
                    return positions
        
        # ensure object mode with active mesh
        elif context.mode == 'OBJECT':
            if context.active_object and context.active_object.type == 'MESH':
                obj = context.active_object
                mesh = obj.data
                for vert in mesh.vertices:
                    if vert.select:
                        local_pos = vert.co.copy()
                        world_pos = obj.matrix_world @ local_pos
                        positions.append(np.array([world_pos.x, world_pos.y, world_pos.z]))
                
                # return any selected vertices
                if positions:
                    return positions
        
        # ensure object mode (pass if already in object mode)
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # try to find Cz landmark
        cz_obj = bpy.data.objects.get("Cz")
        if cz_obj:
            loc = cz_obj.location
            positions.append(np.array([loc.x, loc.y, loc.z]))
            return positions
        
        # if no Cz, find the highest point on the head mesh (numpy — fast on high-poly)
        if head_mesh:
            verts = np.array([v.co for v in head_mesh.data.vertices])
            mat = np.array(head_mesh.matrix_world)[:3, :]
            world_z = (mat[2, :3] @ verts.T) + mat[2, 3]
            highest_z = float(world_z.max())
            positions.append(np.array([0, 0, highest_z]))
        else:
            # Last resort fallback
            positions.append(np.array([0, 0, 0.12]))
        
        return positions
    
    def get_next_optode_number(self, prefix):
        """Find next available number for optode naming"""
        existing = [obj.name for obj in bpy.data.objects if obj.name.startswith(f"{prefix}_")]
        numbers = []
        for name in existing:
            try:
                num = int(name.split("_")[1])
                numbers.append(num)
            except (IndexError, ValueError):
                continue
        
        return max(numbers) + 1 if numbers else 1
    
    def get_head_scale(self, head_mesh):
        """Optode size based on head dimensions (numpy — fast on high-poly meshes)."""
        verts = np.array([v.co for v in head_mesh.data.vertices])
        bbox_min = Vector(verts.min(axis=0).tolist())
        bbox_max = Vector(verts.max(axis=0).tolist())
        bbox_min_world = head_mesh.matrix_world @ bbox_min
        bbox_max_world = head_mesh.matrix_world @ bbox_max
        dimensions = bbox_max_world - bbox_min_world
        head_size = max(dimensions)
        optode_diameter = head_size * 0.02
        optode_thickness = optode_diameter * 0.15
        return optode_diameter, optode_thickness

    def snap_to_mesh_surface(self, position, head_mesh):
        """Snap point to nearest surface location"""
        bvh = BVHTree.FromObject(head_mesh, bpy.context.evaluated_depsgraph_get())
        location, normal, index, distance = bvh.find_nearest(Vector(position))
        
        if location is None:
            return position, np.array([0, 0, 1])
        
        return np.array(location), np.array(normal)
    
    def create_optode_disc(self, position, normal, name, color, diameter, thickness):
        """Create optode disc at position (like SD import)"""
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=diameter / 2.0,
            depth=thickness,
            enter_editmode=False,
            align='WORLD',
            location=(0, 0, 0)
        )
        
        optode = bpy.context.active_object
        optode.name = name
        
        # align to surface normal
        z_axis = Vector((0, 0, 1))
        normal_vec = Vector(normal)
        rotation_quat = z_axis.rotation_difference(normal_vec)
        optode.rotation_euler = rotation_quat.to_euler()
        
        #  we want the center of the cylinder on the surface of the head
        offset_position = Vector(position) + (normal_vec.normalized() * (1))
        optode.location = offset_position
        
        # create material for this optode
        mat_name = f"{name}_Material"
        mat = bpy.data.materials.get(mat_name)
        if not mat:
            mat = bpy.data.materials.new(name=mat_name)
        
        mat.use_nodes = True
        mat.diffuse_color = color
        
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Metallic'].default_value = 0.3
            bsdf.inputs['Roughness'].default_value = 0.4
        
        # assign material 
        if len(optode.data.materials) > 0:
            optode.data.materials[0] = mat
        else:
            optode.data.materials.append(mat)
        
        # smooth shading (just visually nicer)
        for face in optode.data.polygons:
            face.use_smooth = True
        
        return optode
    
    def get_or_create_collection(self, collection_name):
        """Get existing collection or create new one"""
        collection = bpy.data.collections.get(collection_name)
        if not collection:
            collection = bpy.data.collections.new(collection_name)
            bpy.context.scene.collection.children.link(collection)
        return collection
    
    def add_to_collection(self, obj, collection):
        """Add object to collection and remove from others"""
        # Uunlink from all current collections
        for col in obj.users_collection:
            col.objects.unlink(obj)
        
        #lLink to target collection
        collection.objects.link(obj)
    
    def apply_shrinkwrap(self, obj, head_mesh):
        """Clear any existing shrinkwrap constraints then add one clean one."""
        for con in [c for c in obj.constraints if c.type == 'SHRINKWRAP']:
            obj.constraints.remove(con)
        constraint = obj.constraints.new(type='SHRINKWRAP')
        constraint.target = head_mesh
        constraint.shrinkwrap_type = 'NEAREST_SURFACE'
        constraint.wrap_mode = 'ON_SURFACE'
        constraint.use_track_normal = True
        constraint.track_axis = 'TRACK_Z'


class NEUROCAPTAIN_OT_ensure_optode_constraints(bpy.types.Operator):
    """Ensure all optodes have proper shrinkwrap constraints"""
    bl_idname = "neurocaptain.ensure_optode_constraints"
    bl_label = "Ensure Optode Constraints"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # ensure object mode 
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # all sources and detectors 
        optodes = []
        for obj in bpy.data.objects:
            if obj.name.startswith("Source_") or obj.name.startswith("Detector_"):
                optodes.append(obj)
        
        if not optodes:
            self.report({'WARNING'}, "No optodes found (Source_# or Detector_#)")
            return {'CANCELLED'}
        
        # find headmesh — prefer exact name, then "headmesh" substring only
        # (avoids matching high-poly 5-layer objects like Head_Surface_5L)
        head_mesh = bpy.data.objects.get("headmesh")
        if not head_mesh:
            for obj_candidate in bpy.data.objects:
                if "headmesh" in obj_candidate.name.lower() and obj_candidate.type == 'MESH':
                    head_mesh = obj_candidate
                    break

        if not head_mesh:
            self.report({'ERROR'}, "Could not find headmesh for shrinkwrap target")
            return {'CANCELLED'}

        fixed_count = 0

        for optode in optodes:
            # Remove ALL existing shrinkwrap constraints (clears stacked/bad ones)
            for con in [c for c in optode.constraints if c.type == 'SHRINKWRAP']:
                optode.constraints.remove(con)
            # Add one clean constraint targeting the correct headmesh
            constraint = optode.constraints.new(type='SHRINKWRAP')
            constraint.target = head_mesh
            constraint.shrinkwrap_type = 'NEAREST_SURFACE'
            constraint.wrap_mode = 'ON_SURFACE'
            constraint.use_track_normal = True
            constraint.track_axis = 'TRACK_Z'
            fixed_count += 1

        self.report({'INFO'}, f"Fixed {fixed_count} optodes → targeting '{head_mesh.name}'")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(NEUROCAPTAIN_OT_add_source)
    bpy.utils.register_class(NEUROCAPTAIN_OT_add_detector)
    bpy.utils.register_class(NEUROCAPTAIN_OT_ensure_optode_constraints)


def unregister():
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_ensure_optode_constraints)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_add_detector)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_add_source)