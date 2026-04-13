import bpy
import bmesh
from mathutils import Vector , Matrix, Quaternion
import math
import numpy as np
from scipy.spatial import Delaunay
from mathutils.bvhtree import BVHTree


# HELPER FUNCTIONS
def create_connection_material(obj, mat_name, color):
    """create material for connection objects"""
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        mat.diffuse_color = color
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = color
    
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def setup_optode_hooks(conn_obj, optodes):
    """hook modifiers for optode connections"""
    # create vertex groups of optodes_connections mesh (generated in previous NC step)
    for idx, optode in enumerate(optodes):
        vgroup = conn_obj.vertex_groups.new(name=f"VG_{optode.name}")
        vgroup.add([idx], 1.0, 'REPLACE')
    
    bpy.ops.object.select_all(action='DESELECT')
    conn_obj.select_set(True)
    bpy.context.view_layer.objects.active = conn_obj
    
    # create hooks
    for idx, optode in enumerate(optodes):
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        for v in conn_obj.data.vertices:
            v.select = False
        
        conn_obj.data.vertices[idx].select = True
        conn_obj.data.update()
        bpy.ops.object.mode_set(mode='EDIT')
        #hook modififer applied to connected object hooking to optodes 
        hook = conn_obj.modifiers.new(name=f"Hook_{optode.name}", type='HOOK')
        hook.object = optode
        hook.vertex_group = f"VG_{optode.name}"
        
        bpy.ops.object.hook_assign(modifier=hook.name)
        bpy.ops.object.hook_reset(modifier=hook.name)
    
    bpy.ops.object.mode_set(mode='OBJECT')


def create_anchor_ring_material():
    """create visual color distinction for anchor optodes"""
    mat_name = "Anchor_Ring_Material"
    
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]
    
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    mat.diffuse_color = (1.0, 0.85, 0.0, 1.0) #yellow
    
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    node_emission = nodes.new(type='ShaderNodeEmission')
    node_emission.inputs[0].default_value = (1.0, 0.85, 0.0, 1.0)#yellow
    node_emission.inputs[1].default_value = 5.0
    
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(node_emission.outputs[0], node_output.inputs[0])
    
    return mat

def create_torus_mesh(name, major_radius, minor_radius, major_segments=32, minor_segments=12):
    """create physical torus mesh for anchor optodes"""
    bpy.ops.mesh.primitive_torus_add(
    align='WORLD',
    major_radius=major_radius,
    minor_radius=minor_radius,
    major_segments= major_segments,
    minor_segments=minor_segments,
)
    mesh = bpy.context.active_object
    mesh.data.name = name
    return mesh

def get_or_create_collection(collection_name):
    """Get existing collection or create new one"""
    collection = bpy.data.collections.get(collection_name)
    if not collection:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    return collection

def add_to_collection(obj, collection):
    """Add object to collection"""
    # Unlink from all current collections
    for col in obj.users_collection:
        col.objects.unlink(obj)
    
    # Link to target collection
    collection.objects.link(obj)

## OPERATORS ##
class NEUROCAPTAIN_OT_create_optode_connections(bpy.types.Operator):
    """create edges connecting nearby optodes with soft body springs"""
    bl_idname = "neurocaptain.create_optode_connections"
    bl_label = "Create Optode Connections"
    bl_options = {'REGISTER', 'UNDO'}
    
    distance_threshold: bpy.props.FloatProperty(
        name="Distance Threshold",
        description="Maximum distance between optodes to create connection",
        default=30.0,
        min=0.1,
        max=1000.0,
        unit='LENGTH'
    )
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        optodes = [obj for obj in bpy.data.objects 
                   if obj.name.startswith("Source_") or obj.name.startswith("Detector_")]
        optodes.sort(key=lambda x: x.name)
        
        if len(optodes) < 2:
            self.report({'ERROR'}, "Need at least 2 optodes")
            return {'CANCELLED'}
        #remove any old connections
        if "Optode_Connections" in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects["Optode_Connections"], do_unlink=True) 
        
        mesh = bpy.data.meshes.new("Optode_Connections_Mesh")
        conn_obj = bpy.data.objects.new("Optode_Connections", mesh)
        context.collection.objects.link(conn_obj)
        
        bm = bmesh.new()
        #access  evaluated IDs (such as object, material, .. state from an original ID- Blender
        depsgraph = context.evaluated_depsgraph_get()
        
        optode_to_vert = {}
        optode_to_index = {}
        for idx, optode in enumerate(optodes):
            optode_eval = optode.evaluated_get(depsgraph)
            final_position = optode_eval.matrix_world.translation
            vert = bm.verts.new(final_position)
            optode_to_vert[optode.name] = vert
            optode_to_index[optode.name] = idx
            vert.index = idx
        
        bm.verts.ensure_lookup_table()
        
        created_edges = 0
        edge_info = []
        
        for i, optode1 in enumerate(optodes):
            for j, optode2 in enumerate(optodes):
                if i >= j:
                    continue
                
                distance = (optode1.location - optode2.location).length
                if distance <= self.distance_threshold:
                    vert1 = optode_to_vert[optode1.name] #generate new vertices based on optode positions
                    vert2 = optode_to_vert[optode2.name]
                    bm.edges.new([vert1, vert2]) #generate new edge between two vertices 
                    created_edges += 1
                    
                    edge_info.append({
                        'optode1': optode1.name,
                        'optode2': optode2.name,
                        'length': distance,
                        'v1_idx': optode_to_index[optode1.name],
                        'v2_idx': optode_to_index[optode2.name]
                    })
        
        if created_edges == 0:
            self.report({'ERROR'}, "No connections created. Try increasing distance threshold.")
            bm.free()
            bpy.data.objects.remove(conn_obj, do_unlink=True)
            return {'CANCELLED'}
        
        bm.to_mesh(mesh)
        bm.free()
        
        conn_obj["optode_names"] = [opt.name for opt in optodes]
        conn_obj["edge_info"] = edge_info
        
        spring_states = {}
        for edge in edge_info:
            optode_pair = sorted([edge['optode1'], edge['optode2']])
            edge_key = f"{optode_pair[0]}_{optode_pair[1]}"
            #set up struct for json export 
            spring_states[edge_key] = {
                "optode1": edge['optode1'],
                "optode2": edge['optode2'],
                "rest_length": edge['length'],
                "pull": 0.9, #default is stiff 
                "push": 0.9, #default is stiff 
                "is_flexible": False
            }
        
        conn_obj["spring_states"] = spring_states
        #setting soft body physics in Blender 
        soft_body_mod = conn_obj.modifiers.new(name="Softbody", type='SOFT_BODY')
        sb_settings = conn_obj.soft_body
        sb_settings.use_edges = True
        sb_settings.use_goal = True
        sb_settings.goal_default = 1.0
        sb_settings.pull = 0.9
        sb_settings.push = 0.9
        sb_settings.spring_length = 0
        sb_settings.use_stiff_quads = False
        sb_settings.use_edge_collision = False
        sb_settings.use_face_collision = False
        sb_settings.use_self_collision = False
        
        conn_obj["is_optode_connections"] = True
        
        create_connection_material(conn_obj, "Optode_Connection_Stiff", (0.2, 0.5, 1.0, 1.0))
        setup_optode_hooks(conn_obj, optodes)
        
        self.report({'INFO'}, f"Created {created_edges} connections between {len(optodes)} optodes")
        return {'FINISHED'}


class NEUROCAPTAIN_OT_create_optode_connections_delaunay(bpy.types.Operator):
    """create edges connecting optodes using Delaunay triangulation"""
    bl_idname = "neurocaptain.create_optode_connections_delaunay"
    bl_label = "Create Optode Connections (Delaunay)"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        optodes = [obj for obj in bpy.data.objects 
                   if obj.name.startswith("Source_") or obj.name.startswith("Detector_")]
        optodes.sort(key=lambda x: x.name)
        
        if len(optodes) < 3:
            self.report({'ERROR'}, "Need at least 3 optodes for Delaunay triangulation, consider connecting by distance threshold")
            return {'CANCELLED'}
        
        if "Optode_Connections" in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects["Optode_Connections"], do_unlink=True)
        
        # project optode positions to 2D using PCA
        positions = np.array([list(opt.location) for opt in optodes])
        centroid = positions.mean(axis=0)
        centered = positions - centroid
        
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        points_2d = centered @ eigenvectors[:, :2]
        
        try:
            tri = Delaunay(points_2d)
        except Exception as e:
            self.report({'ERROR'}, f"Delaunay triangulation failed: {str(e)}")
            return {'CANCELLED'}
        
        # Create mesh
        mesh = bpy.data.meshes.new("Optode_Connections_Mesh")
        conn_obj = bpy.data.objects.new("Optode_Connections", mesh)
        context.collection.objects.link(conn_obj)
        
        bm = bmesh.new()
        depsgraph = context.evaluated_depsgraph_get()
        
        optode_to_vert = {}
        optode_to_index = {}
        for idx, optode in enumerate(optodes):
            optode_eval = optode.evaluated_get(depsgraph)
            final_position = optode_eval.matrix_world.translation
            vert = bm.verts.new(final_position)
            optode_to_vert[optode.name] = vert
            optode_to_index[optode.name] = idx
            vert.index = idx
        
        bm.verts.ensure_lookup_table()
        
        # Create edges from triangulation
        edges_set = set()
        for simplex in tri.simplices:
            for i in range(3):
                v1 = int(simplex[i])
                v2 = int(simplex[(i + 1) % 3])
                edge = tuple(sorted([v1, v2]))
                edges_set.add(edge)
        
        created_edges = 0
        edge_info = []
        
        for v1_idx, v2_idx in edges_set:
            optode1 = optodes[v1_idx]
            optode2 = optodes[v2_idx]
            
            vert1 = optode_to_vert[optode1.name]
            vert2 = optode_to_vert[optode2.name]
            bm.edges.new([vert1, vert2])
            created_edges += 1
            
            distance = (optode1.location - optode2.location).length
            edge_info.append({
                'optode1': optode1.name,
                'optode2': optode2.name,
                'length': float(distance),
                'v1_idx': v1_idx,
                'v2_idx': v2_idx
            })
        
        bm.to_mesh(mesh)
        bm.free()
        
        conn_obj["optode_names"] = [opt.name for opt in optodes]
        conn_obj["edge_info"] = edge_info
        
        spring_states = {}
        for edge in edge_info:
            optode_pair = sorted([edge['optode1'], edge['optode2']])
            edge_key = f"{optode_pair[0]}_{optode_pair[1]}"
            spring_states[edge_key] = {
                "optode1": edge['optode1'],
                "optode2": edge['optode2'],
                "rest_length": edge['length'],
                "pull": 0.9,
                "push": 0.9,
                "is_flexible": False
            }
        
        conn_obj["spring_states"] = spring_states
        
        soft_body_mod = conn_obj.modifiers.new(name="Softbody", type='SOFT_BODY')
        sb_settings = conn_obj.soft_body
        sb_settings.use_edges = True
        sb_settings.use_goal = True
        sb_settings.goal_default = 1.0
        sb_settings.pull = 0.9
        sb_settings.push = 0.9
        sb_settings.spring_length = 0
        sb_settings.use_stiff_quads = False
        sb_settings.use_edge_collision = False
        sb_settings.use_face_collision = False
        sb_settings.use_self_collision = False
        
        conn_obj["is_optode_connections"] = True
        
        create_connection_material(conn_obj, "Optode_Connection_Stiff", (0.2, 0.5, 1.0, 1.0))
        setup_optode_hooks(conn_obj, optodes)
        
        self.report({'INFO'}, f"Created {created_edges} connections using Delaunay triangulation")
        return {'FINISHED'}


class NEUROCAPTAIN_OT_modify_spring_properties(bpy.types.Operator):
    """Modify spring properties to make flexible"""
    bl_idname = "neurocaptain.modify_spring_properties"
    bl_label = "Set Flexible Spring Properties"
    bl_options = {'REGISTER', 'UNDO'}
    
    spring_pull: bpy.props.FloatProperty(
        name="Spring Pull (Stretch Resistance)",
        description="Resistance to stretching (0=no resistance, 1=maximum)",
        default=0.3,
        min=0.0,
        max=1.0
    )
    
    spring_push: bpy.props.FloatProperty(
        name="Spring Push (Compression Resistance)",
        description="Resistance to compression (0=no resistance, 1=maximum)",
        default=0.3,
        min=0.0,
        max=1.0
    )
    
    def invoke(self, context, event):
        obj = context.active_object
        # ensure user has proper selections to run this feature 
        if not obj or not obj.get("is_optode_connections"):
            self.report({'ERROR'}, "Please select the Optode_Connections object")
            return {'CANCELLED'}
        
        if context.mode != 'EDIT_MESH':
            self.report({'ERROR'}, "Must be in Edit Mode with edges selected")
            return {'CANCELLED'}
        
        bm = bmesh.from_edit_mesh(obj.data)
        if not any(e.select for e in bm.edges):
            self.report({'ERROR'}, "No edges selected")
            return {'CANCELLED'}
        
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or not obj.get("is_optode_connections"):
            self.report({'ERROR'}, "Please select the Optode_Connections object")
            return {'CANCELLED'}
        
        optode_names = obj.get("optode_names", [])
        if not optode_names:
            self.report({'ERROR'}, "No optode reference data found")
            return {'CANCELLED'}
        
        bm = bmesh.from_edit_mesh(obj.data)
        bevel_layer = bm.edges.layers.bevel_weight.verify()
        crease_layer = bm.edges.layers.crease.verify()
        
        spring_states = obj.get("spring_states", {})
        
        num_modified = 0
        for edge in bm.edges:
            if edge.select:
                v1_idx = edge.verts[0].index
                v2_idx = edge.verts[1].index
                
                if v1_idx >= len(optode_names) or v2_idx >= len(optode_names):
                    continue
                    
                optode1_name = optode_names[v1_idx]
                optode2_name = optode_names[v2_idx]
                
                optode_pair = sorted([optode1_name, optode2_name])
                edge_key = f"{optode_pair[0]}_{optode_pair[1]}"
                
                rest_length = (edge.verts[0].co - edge.verts[1].co).length
                # update to spring states for json export
                spring_states[edge_key] = {
                    "optode1": optode1_name,
                    "optode2": optode2_name,
                    "rest_length": rest_length,
                    "pull": self.spring_pull,
                    "push": self.spring_push,
                    "is_flexible": True
                }
                
                edge[bevel_layer] = 1.0
                edge[crease_layer] = 1.0
                
                num_modified += 1
        
        if num_modified == 0:
            self.report({'ERROR'}, "No edges selected")
            return {'CANCELLED'}
        
        obj["spring_states"] = spring_states
        bmesh.update_edit_mesh(obj.data)
        
        mat_name = "Optode_Connection_Flexible"
        if mat_name not in bpy.data.materials:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            nodes.clear()
            
            node_emission = nodes.new(type='ShaderNodeEmission')
            node_emission.inputs[0].default_value = (1.0, 0.6, 0.0, 1.0)
            node_emission.inputs[1].default_value = 2.0
            
            node_output = nodes.new(type='ShaderNodeOutputMaterial')
            mat.node_tree.links.new(node_emission.outputs[0], node_output.inputs[0])
        
        if "Bevel_Springs" not in obj.modifiers:
            bevel_mod = obj.modifiers.new(name="Bevel_Springs", type='BEVEL')
            bevel_mod.width = 0.5
            bevel_mod.segments = 2
            bevel_mod.limit_method = 'WEIGHT'
            bevel_mod.affect = 'EDGES'
        
        self.report({'INFO'}, 
                   f"Set flexible spring properties for {num_modified} edges "
                   f"(Pull: {self.spring_pull:.2f}, Push: {self.spring_push:.2f})")
        
        return {'FINISHED'}


class NEUROCAPTAIN_OT_make_spring_stiff(bpy.types.Operator):
    """Make selected edges stiff springs with optional distance setting"""
    bl_idname = "neurocaptain.make_spring_stiff"
    bl_label = "Make Springs Stiff"
    bl_options = {'REGISTER', 'UNDO'}
    
    set_distance: bpy.props.BoolProperty(
        name="Set Custom Distance",
        description="Set a custom distance between optodes",
        default=False
    )
    
    target_distance: bpy.props.FloatProperty(
        name="Target Distance (mm)",
        description="Target distance between optodes",
        default=30.0,
        min=1.0,
        max=200.0,
        unit='LENGTH'
    )
    
    distance_type: bpy.props.EnumProperty(
        name="Distance Type",
        description="Type of distance calculation",
        items=[
            ('EUCLIDEAN', "Euclidean", "Straight-line distance through space"),
            ('GEODESIC', "Geodesic", "Surface distance along the mesh")
        ],
        default='EUCLIDEAN'
    )
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "set_distance")
        
        if self.set_distance:
            layout.prop(self, "target_distance")
            layout.prop(self, "distance_type")
    
    def calculate_geodesic_distance(self, mesh, v1_co, v2_co):
        """calculate approximate geodesic distance along headmesh surface"""
        bvh = BVHTree.FromBMesh(mesh)
        
        num_samples = 20
        total_distance = 0.0
        prev_point = v1_co.copy()
        
        for i in range(1, num_samples + 1):
            t = i / num_samples
            sample_point = v1_co.lerp(v2_co, t)
            
            location, normal, index, distance = bvh.find_nearest(sample_point)
            
            if location:
                total_distance += (location - prev_point).length
                prev_point = location
            else:
                return (v2_co - v1_co).length
        
        return total_distance
    
    def move_optode_to_distance(self, context, optode1_obj, optode2_obj, target_dist, mesh):
        """move optode2 to target distance from optode1 along surface"""
        depsgraph = context.evaluated_depsgraph_get()
        optode1_eval = optode1_obj.evaluated_get(depsgraph)
        optode2_eval = optode2_obj.evaluated_get(depsgraph)
        
        v1_co = optode1_eval.location
        v2_co = optode2_eval.location
        
        direction = (v2_co - v1_co).normalized()
        new_position = v1_co + direction * target_dist
        
        if self.distance_type == 'GEODESIC':
            bvh = BVHTree.FromBMesh(mesh)
            location, normal, index, distance = bvh.find_nearest(new_position)
            if location:
                new_position = location
        
        optode2_obj.location = new_position
    
    def execute(self, context):
        obj = context.active_object
        
        if not obj or not obj.get("is_optode_connections"):
            self.report({'ERROR'}, "Please select the Optode_Connections object")
            return {'CANCELLED'}
        
        if context.mode != 'EDIT_MESH':
            self.report({'ERROR'}, "Must be in Edit Mode with edges selected")
            return {'CANCELLED'}
        
        optode_names = obj.get("optode_names", [])
        if not optode_names:
            self.report({'ERROR'}, "No optode reference data found")
            return {'CANCELLED'}
        
        bm = bmesh.from_edit_mesh(obj.data)
        bevel_layer = bm.edges.layers.bevel_weight.verify()
        crease_layer = bm.edges.layers.crease.verify()
        
        spring_states = obj.get("spring_states", {})
        
        head_mesh = None
        if self.set_distance and self.distance_type == 'GEODESIC':
            head_obj = context.scene.objects.get("headmesh")
            if head_obj and head_obj.type == 'MESH':
                head_bm = bmesh.new()
                head_bm.from_mesh(head_obj.data)
                head_mesh = head_bm
        
        num_processed = 0
        edges_to_update = []
        
        for edge in bm.edges:
            if edge.select:
                v1_idx = edge.verts[0].index
                v2_idx = edge.verts[1].index
                
                if v1_idx >= len(optode_names) or v2_idx >= len(optode_names):
                    continue
                    
                optode1_name = optode_names[v1_idx]
                optode2_name = optode_names[v2_idx]
                
                optode_pair = sorted([optode1_name, optode2_name])
                edge_key = f"{optode_pair[0]}_{optode_pair[1]}"
                
                if self.set_distance:
                    rest_length = self.target_distance
                    edges_to_update.append((optode1_name, optode2_name, edge))
                else:
                    rest_length = (edge.verts[0].co - edge.verts[1].co).length
                
                if edge_key in spring_states:
                    spring_states[edge_key]["rest_length"] = rest_length
                    spring_states[edge_key]["pull"] = 0.9
                    spring_states[edge_key]["push"] = 0.9
                    spring_states[edge_key]["is_flexible"] = False
                else:
                    spring_states[edge_key] = {
                        "optode1": optode1_name,
                        "optode2": optode2_name,
                        "rest_length": rest_length,
                        "pull": 0.9,
                        "push": 0.9,
                        "is_flexible": False
                    }
                
                edge[bevel_layer] = 0.0
                edge[crease_layer] = 0.0
                
                num_processed += 1
        
        if num_processed == 0:
            self.report({'ERROR'}, "No edges selected")
            return {'CANCELLED'}
        
        obj["spring_states"] = spring_states
        bmesh.update_edit_mesh(obj.data)
        
        if self.set_distance and edges_to_update:
            bpy.ops.object.mode_set(mode='OBJECT')
            
            for optode1_name, optode2_name, edge in edges_to_update:
                optode1_obj = context.scene.objects.get(optode1_name)
                optode2_obj = context.scene.objects.get(optode2_name)
                
                if optode1_obj and optode2_obj:
                    self.move_optode_to_distance(context, optode1_obj, optode2_obj, 
                                                 self.target_distance, head_mesh)
            
            bpy.ops.object.mode_set(mode='EDIT')
            bm = bmesh.from_edit_mesh(obj.data)
            
            for edge_info in edges_to_update:
                optode1_name, optode2_name = edge_info[0], edge_info[1]
                optode1_obj = context.scene.objects.get(optode1_name)
                optode2_obj = context.scene.objects.get(optode2_name)
                
                if optode1_obj and optode2_obj:
                    v1_idx = optode_names.index(optode1_name)
                    v2_idx = optode_names.index(optode2_name)
                    
                    bm.verts.ensure_lookup_table()
                    bm.verts[v1_idx].co = optode1_obj.location
                    bm.verts[v2_idx].co = optode2_obj.location
            
            bmesh.update_edit_mesh(obj.data)
            
            if head_mesh:
                head_mesh.free()
        
        msg = f"Made {num_processed} springs stiff"
        if self.set_distance:
            msg += f" with {self.distance_type.lower()} distance {self.target_distance:.1f}mm"
        
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class NEUROCAPTAIN_OT_update_optode_connections(bpy.types.Operator):
    """Update connection vertex positions to match optode locations (force sync)"""
    bl_idname = "neurocaptain.update_optode_connections"
    bl_label = "Update Connection Positions"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Check for Optode_Connections
        if "Optode_Connections" not in bpy.data.objects:
            self.report({'ERROR'}, "No Optode_Connections object found")
            return {'CANCELLED'}
        
        conn_obj = bpy.data.objects["Optode_Connections"]
        
        # Get list of optode names in order
        optode_names = conn_obj.get("optode_names", [])
        if not optode_names:
            self.report({'ERROR'}, "No optode_names data found on Optode_Connections")
            return {'CANCELLED'}
        
        print("\n=== Starting position update ===")
        
        # STEP 1: Get evaluated positions BEFORE disabling anything
        depsgraph = context.evaluated_depsgraph_get()
        
        # Store target positions (where hooks have pulled vertices)
        target_positions = {}
        for i, optode_name in enumerate(optode_names):
            optode = bpy.data.objects.get(optode_name)
            if optode and i < len(conn_obj.data.vertices):
                optode_eval = optode.evaluated_get(depsgraph)
                target_positions[i] = optode_eval.matrix_world.translation.copy()
        
        print(f"Collected {len(target_positions)} target positions")
        
        # STEP 2: Switch to Edit Mode to modify base mesh
        bpy.ops.object.select_all(action='DESELECT')
        conn_obj.select_set(True)
        context.view_layer.objects.active = conn_obj
        
        # Must be in object mode first
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Get bmesh for editing
        import bmesh
        bm = bmesh.from_edit_mesh(conn_obj.data)
        bm.verts.ensure_lookup_table()
        
        updated_count = 0
        max_distance = 0.0
        
        print("=== Updating base mesh vertices ===")
        
        # STEP 3: Update base mesh vertices
        conn_world_matrix_inv = conn_obj.matrix_world.inverted()
        
        for i, target_world_pos in target_positions.items():
            if i >= len(bm.verts):
                continue
            
            vert = bm.verts[i]
            old_pos = vert.co.copy()
            
            # Convert to local space
            target_local_pos = conn_world_matrix_inv @ target_world_pos
            
            # Update vertex
            vert.co = target_local_pos
            
            distance = (vert.co - old_pos).length
            if distance > 0.001:
                optode_name = optode_names[i]
                print(f"  Vertex {i} ({optode_name}): moved {distance:.4f} units")
                updated_count += 1
                if distance > max_distance:
                    max_distance = distance
        
        # STEP 4: Apply changes and return to object mode
        bmesh.update_edit_mesh(conn_obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        print("=== Resetting hooks ===")
        
        # STEP 5: Reset each hook modifier
        # This makes hooks recalculate their reference positions
        for mod in conn_obj.modifiers:
            if mod.type == 'HOOK':
                # Find the vertex this hook controls
                if mod.vertex_group and mod.vertex_group in conn_obj.vertex_groups:
                    vgroup = conn_obj.vertex_groups[mod.vertex_group]
                    
                    # Select only this vertex
                    for v in conn_obj.data.vertices:
                        v.select = False
                    
                    for v in conn_obj.data.vertices:
                        try:
                            weight = vgroup.weight(v.index)
                            if weight > 0.5:
                                v.select = True
                                break
                        except:
                            pass
                    
                    # Enter edit mode to reset this hook
                    bpy.ops.object.mode_set(mode='EDIT')
                    bpy.ops.object.hook_reset(modifier=mod.name)
                    bpy.ops.object.mode_set(mode='OBJECT')
                    
                    print(f"  Reset {mod.name}")
        
        # Deselect all
        for v in conn_obj.data.vertices:
            v.select = False
        
        # STEP 6: Force complete scene update
        context.view_layer.update()
        depsgraph.update()
        
        print(f"=== Update complete: {updated_count} vertices moved ===\n")
        
        # Build report message
        msg = f"Snapped {updated_count} vertices to optodes"
        if max_distance > 0:
            msg += f" (max distance: {max_distance:.4f})"
        
        if updated_count == 0:
            self.report({'INFO'}, "All vertices already aligned with optodes")
        else:
            self.report({'INFO'}, msg)
        
        return {'FINISHED'}


class NEUROCAPTAIN_OT_toggle_connection_visibility(bpy.types.Operator):
    """toggle visibility of optode connections"""
    bl_idname = "neurocaptain.toggle_connection_visibility"
    bl_label = "Toggle Connections"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        if "Optode_Connections" in bpy.data.objects:
            obj = bpy.data.objects["Optode_Connections"]
            obj.hide_viewport = not obj.hide_viewport
            obj.hide_render = not obj.hide_render
            
            status = "hidden" if obj.hide_viewport else "visible"
            self.report({'INFO'}, f"Optode connections are now {status}")
        else:
            self.report({'WARNING'}, "No Optode_Connections object found")
        
        return {'FINISHED'}



# ==============================================================================
#  EXPORT
# ==============================================================================

class NEUROCAPTAIN_OT_export_optode_json(bpy.types.Operator):
    """Export optode probe configuration to JSON file"""
    bl_idname = "neurocaptain.export_optode_json"
    bl_label = "Export Probe Configuration"
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filename: bpy.props.StringProperty(default="probe_config.json")

    def invoke(self, context, event):
        self.filepath = self.filename
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        import json
        from mathutils.bvhtree import BVHTree

        depsgraph = context.evaluated_depsgraph_get()

        # ── Landmark mesh setup ──────────────────────────────────────────
        landmark_mesh, landmark_labels = self._get_landmark_mesh_and_labels()
        landmark_bvh = None
        if landmark_mesh:
            landmark_bvh = BVHTree.FromObject(landmark_mesh, depsgraph)

        # ── Gather optodes ───────────────────────────────────────────────
        optodes = []
        for obj in bpy.data.objects:
            if not (obj.name.startswith("Source_") or obj.name.startswith("Detector_")):
                continue

            optode_eval = obj.evaluated_get(depsgraph)
            optode_pos = optode_eval.matrix_world.translation.copy()

            optode_data = {
                "name": obj.name,
                "position": [float(optode_pos.x), float(optode_pos.y), float(optode_pos.z)],
                "type": "source" if obj.name.startswith("Source_") else "detector",
                "is_anchor": bool(obj.get("is_anchor", False)),
            }

            # Compute barycentric for ALL optodes
            if landmark_mesh and landmark_bvh:
                bary_info = self._compute_barycentric(
                    optode_pos, landmark_mesh, landmark_bvh, landmark_labels
                )
                if bary_info:
                    optode_data["registration"] = bary_info

            # Custom properties
            custom_props = {}
            for key in obj.keys():
                if key not in ("_RNA_UI", "is_anchor") and not key.startswith("_"):
                    try:
                        json.dumps(obj[key])
                        custom_props[key] = obj[key]
                    except Exception:
                        pass
            if custom_props:
                optode_data["custom_properties"] = custom_props

            optodes.append(optode_data)

        if not optodes:
            self.report({'WARNING'}, "No optodes found to export")
            return {'CANCELLED'}

        # ── Gather spring connections ────────────────────────────────────
        springs = []
        if "Optode_Connections" in bpy.data.objects:
            conn_obj = bpy.data.objects["Optode_Connections"]
            spring_states = conn_obj.get("spring_states", {})
            for edge_key, props in spring_states.items():
                springs.append({
                    "optode1": props["optode1"],
                    "optode2": props["optode2"],
                    "rest_length": float(props["rest_length"]),
                    "pull": float(props["pull"]),
                    "push": float(props["push"]),
                    "is_flexible": bool(props["is_flexible"]),
                })

        if not springs:
            self.report({'WARNING'}, "No spring connections found – exporting optodes only")

        # ── Build export payload ─────────────────────────────────────────
        export_data = {
            "format": "NeuroCaptain Probe Configuration",
            "version": "2.0",
            "metadata": {
                "num_sources": len([o for o in optodes if o["type"] == "source"]),
                "num_detectors": len([o for o in optodes if o["type"] == "detector"]),
                "num_connections": len(springs),
                "num_anchors": len([o for o in optodes if o["is_anchor"]]),
                "num_landmarks": len(landmark_labels) if landmark_labels else 0,
            },
            "optodes": optodes,
            "connections": springs,
            "global_settings": {
                "default_pull": 0.9,
                "default_push": 0.9,
            },
        }

        # ── Write file ───────────────────────────────────────────────────
        try:
            with open(self.filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.report(
                {'INFO'},
                f"Exported {len(optodes)} optodes, {len(springs)} connections  (v2.0 full-barycentric)"
            )
        except Exception as e:
            self.report({'ERROR'}, f"Failed to export: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_landmark_mesh_and_labels():
        """Return (landmark_mesh_object, labels_list) or (None, [])."""
        if "LandmarkMesh" not in bpy.data.objects:
            return None, []
        lm = bpy.data.objects["LandmarkMesh"]
        n = len(lm.data.vertices)
        stored = lm.get("landmark_labels", None)
        if stored is not None:
            return lm, list(stored)
        if n >= 77:
            labels = [
                "Nz","Iz","Lpa","Rpa","Cz",
                "T7","C5","C3","C1","Cz","C2","C4","C6","T8",
                "Fpz","AFz","Fz","FCz","Cz","CPz","Pz","POz","Oz",
                "FT7","F7","AF7","Fp1","TP7","P7","PO7","O1",
                "FT8","F8","AF8","Fp2","TP8","P8","PO8","O2",
                "FC1","FC3","FC5","FC2","FC4","FC6",
                "F1","F3","F5","F2","F4","F6","AF3","AF4",
                "CP1","CP3","CP5","CP2","CP4","CP6",
                "P1","P3","P5","P2","P4","P6","PO3","PO4",
                "FT9","F9","","","TP9","P9","PO9","O9",
                "FT10","F10","","","TP10","P10","PO10","O10",
            ]
        elif n >= 27:
            labels = [
                "Nz","Iz","Lpa","Rpa","Cz",
                "T3","C3","Cz","C4","T4",
                "Fpz","Fz","Cz","Pz","Oz",
                "F7","Fp1","T5","O1","F8","Fp2","T6","O2",
                "F3","F4","P3","P4",
            ]
        else:
            labels = [f"vertex_{i}" for i in range(n)]
        return lm, labels

    def _compute_barycentric(self, point, landmark_mesh, landmark_bvh, landmark_labels):
        """Compute barycentric registration data for a single point."""
        location, normal, face_index, distance = landmark_bvh.find_nearest(point)
        if location is None or face_index is None:
            return None

        face = landmark_mesh.data.polygons[face_index]
        vert_indices = list(face.vertices)
        v0 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[vert_indices[0]].co
        v1 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[vert_indices[1]].co
        v2 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[vert_indices[2]].co

        bary = self._barycentric_coords(point, v0, v1, v2)
        if bary is None:
            return None  # degenerate triangle

        # Build labels — use index-qualified labels for uniqueness
        vlabels = []
        for vi in vert_indices:
            raw = landmark_labels[vi] if vi < len(landmark_labels) else ""
            vlabels.append(raw if raw else f"vertex_{vi}")

        # Nearest-landmark fallback (brute-force closest vertex)
        min_dist = float('inf')
        closest_idx = -1
        for i, vert in enumerate(landmark_mesh.data.vertices):
            d = (landmark_mesh.matrix_world @ vert.co - point).length
            if d < min_dist:
                min_dist = d
                closest_idx = i

        nearest_lm = None
        if 0 <= closest_idx < len(landmark_labels) and landmark_labels[closest_idx]:
            lm_world = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[closest_idx].co
            offset = point - lm_world
            nearest_lm = {
                "landmark": landmark_labels[closest_idx],
                "landmark_index": int(closest_idx),
                "offset": [float(offset.x), float(offset.y), float(offset.z)],
                "distance": float(min_dist),
            }

        return {
            "method": "barycentric",
            "face_index": int(face_index),
            "vertex_indices": [int(v) for v in vert_indices],
            "vertex_labels": vlabels,
            "barycentric_coords": [float(bary[0]), float(bary[1]), float(bary[2])],
            "distance_to_surface": float(distance),
            "nearest_landmark": nearest_lm,
        }

    @staticmethod
    def _barycentric_coords(p, v0, v1, v2):
        """Compute barycentric coords with degenerate-triangle guard."""
        p = Vector(p); v0 = Vector(v0); v1 = Vector(v1); v2 = Vector(v2)
        e1 = v1 - v0; e2 = v2 - v0; ep = p - v0
        d00 = e1.dot(e1); d01 = e1.dot(e2); d02 = e1.dot(ep)
        d11 = e2.dot(e2); d12 = e2.dot(ep)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            return None  # degenerate
        inv = 1.0 / denom
        u = (d11 * d02 - d01 * d12) * inv
        v = (d00 * d12 - d01 * d02) * inv
        return (1.0 - u - v, u, v)


# ==============================================================================
#  IMPORT
# ==============================================================================

class NEUROCAPTAIN_OT_import_optode_json(bpy.types.Operator):
    """Import optode probe configuration from JSON"""
    bl_idname = "neurocaptain.import_optode_json"
    bl_label = "Import Probe Configuration"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})

    relaxation_iterations: bpy.props.IntProperty(
        name="Relaxation Iterations",
        description="Light spring relaxation passes after barycentric placement (0 = disable)",
        default=40,
        min=0,
        max=300,
    )
    relaxation_step: bpy.props.FloatProperty(
        name="Max Step (mm)",
        description="Maximum displacement per relaxation iteration",
        default=0.5,
        min=0.01,
        max=5.0,
    )
    relaxation_damping: bpy.props.FloatProperty(
        name="Damping",
        description="Spring force damping factor (lower = gentler adjustment)",
        default=0.25,
        min=0.01,
        max=1.0,
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        import json
        from mathutils.bvhtree import BVHTree

        SRC_COLOR = (1.0, 0.0, 0.0, 1.0)
        DET_COLOR = (0.0, 0.0, 0.0, 1.0)

        if "headmesh" not in bpy.data.objects:
            self.report({'ERROR'}, "No 'headmesh' found in scene")
            return {'CANCELLED'}
        headmesh = bpy.data.objects["headmesh"]

        # ── Landmark mesh + labels ───────────────────────────────────────
        landmark_mesh, landmark_labels = NEUROCAPTAIN_OT_export_optode_json._get_landmark_mesh_and_labels()
        num_target_landmarks = len(landmark_labels)

        # Build label → [list of indices] (handles duplicates like "Cz")
        label_to_indices = defaultdict(list)
        for i, lbl in enumerate(landmark_labels):
            if lbl:
                label_to_indices[lbl].append(i)

        if not landmark_mesh:
            self.report({'WARNING'}, "No 'LandmarkMesh' found — using raw positions only")

        # ── Load JSON ────────────────────────────────────────────────────
        try:
            with open(self.filepath, 'r') as f:
                config = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load JSON: {e}")
            return {'CANCELLED'}

        if config.get("format") != "NeuroCaptain Probe Configuration":
            self.report({'ERROR'}, "Invalid JSON format")
            return {'CANCELLED'}

        version = config.get("version", "1.0")
        num_source_landmarks = config.get("metadata", {}).get("num_landmarks", 0)
        same_landmark_system = (num_source_landmarks == num_target_landmarks and num_target_landmarks > 0)

        print(f"\n{'='*60}")
        print(f"JSON IMPORT v{version}  |  source landmarks: {num_source_landmarks}  |  target landmarks: {num_target_landmarks}")
        print(f"Same landmark system: {same_landmark_system}")
        print(f"{'='*60}")

        # ── Head scale ───────────────────────────────────────────────────
        optode_diameter, optode_thickness = self._get_head_scale(headmesh)
        head_bvh = BVHTree.FromObject(headmesh, context.evaluated_depsgraph_get())

        bpy.ops.object.select_all(action='DESELECT')

        # ── Place optodes ────────────────────────────────────────────────
        optode_objects = {}
        placement_stats = {"barycentric_index": 0, "barycentric_label": 0, "nearest_lm": 0, "raw": 0}

        for optode_data in config.get("optodes", []):
            name = optode_data["name"]
            optode_type = optode_data["type"]
            is_anchor = optode_data.get("is_anchor", False)

            initial_pos, method = self._resolve_position(
                optode_data, landmark_mesh, landmark_labels,
                label_to_indices, same_landmark_system, num_target_landmarks,
            )
            placement_stats[method] += 1

            # Snap to headmesh surface
            snapped_pos, surface_normal = self._snap_to_surface(initial_pos, head_bvh)

            # Create the disc
            color = SRC_COLOR if optode_type == "source" else DET_COLOR
            collection_name = "Sources" if optode_type == "source" else "Detectors"
            coll = get_or_create_collection(collection_name)

            optode = self._create_optode_disc(snapped_pos, surface_normal, name,
                                              color, optode_diameter, optode_thickness)
            add_to_collection(optode, coll)
            optode["is_anchor"] = 1 if is_anchor else 0

            # Anchor ring
            if is_anchor:
                anchors_coll = get_or_create_collection("Anchor Indicators")
                ring_mat = create_anchor_ring_material()
                bpy.ops.mesh.primitive_torus_add(
                    location=(0, 0, 0), major_radius=2.5, minor_radius=0.3
                )
                indicator = context.active_object
                add_to_collection(indicator, anchors_coll)
                indicator.name = f"Anchor_{name}"
                indicator.parent = optode
                indicator.location = (0, 0, 0)
                indicator.hide_render = True
                indicator.data.materials.append(ring_mat)

            # Shrinkwrap
            cst = optode.constraints.new(type='SHRINKWRAP')
            cst.target = headmesh
            cst.shrinkwrap_type = 'NEAREST_SURFACE'
            cst.distance = 0.0
            cst.influence = 1.0

            optode_objects[name] = optode

        print(f"\nPlacement summary:")
        print(f"  Barycentric (index):  {placement_stats['barycentric_index']}")
        print(f"  Barycentric (label):  {placement_stats['barycentric_label']}")
        print(f"  Nearest landmark:     {placement_stats['nearest_lm']}")
        print(f"  Raw position:         {placement_stats['raw']}")

        # ── Light spring relaxation ──────────────────────────────────────
        connections = config.get("connections", [])
        if self.relaxation_iterations > 0 and connections:
            print(f"\nRunning light relaxation: {self.relaxation_iterations} iterations, "
                  f"step={self.relaxation_step}, damping={self.relaxation_damping}")
            self._run_light_relaxation(
                context, optode_objects, connections, head_bvh,
                n_iters=self.relaxation_iterations,
                max_step=self.relaxation_step,
                damping=self.relaxation_damping,
            )

        # ── Build connection mesh + hooks + soft body ────────────────────
        mesh = bpy.data.meshes.new("Optode_Connections")
        conn_obj = bpy.data.objects.new("Optode_Connections", mesh)
        context.collection.objects.link(conn_obj)

        names_ordered = list(optode_objects.keys())
        name_to_idx = {n: i for i, n in enumerate(names_ordered)}

        vertices = [optode_objects[n].location.copy() for n in names_ordered]
        edges = []
        spring_states = {}

        for cd in connections:
            n1, n2 = cd["optode1"], cd["optode2"]
            if n1 in name_to_idx and n2 in name_to_idx:
                edges.append((name_to_idx[n1], name_to_idx[n2]))
                ek = f"{min(n1, n2)}_{max(n1, n2)}"
                spring_states[ek] = {
                    "optode1": n1, "optode2": n2,
                    "rest_length": cd["rest_length"],
                    "pull": cd["pull"], "push": cd["push"],
                    "is_flexible": cd["is_flexible"],
                }

        mesh.from_pydata(vertices, edges, [])
        mesh.update()
        conn_obj["spring_states"] = spring_states

        mat = bpy.data.materials.get("Connection_Material")
        if not mat:
            mat = bpy.data.materials.new(name="Connection_Material")
            mat.use_nodes = True
            mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.8, 0.8, 1)
        if conn_obj.data.materials:
            conn_obj.data.materials[0] = mat
        else:
            conn_obj.data.materials.append(mat)

        # Vertex groups + hooks
        bpy.ops.object.select_all(action='DESELECT')
        conn_obj.select_set(True)
        context.view_layer.objects.active = conn_obj

        for i, oname in enumerate(names_ordered):
            vg = conn_obj.vertex_groups.new(name=f"VG_{oname}")
            vg.add([i], 1.0, 'REPLACE')

        print(f"\n=== Creating hooks for {len(names_ordered)} optodes ===")
        bpy.ops.object.select_all(action='DESELECT')
        conn_obj.select_set(True)
        context.view_layer.objects.active = conn_obj

        for i, oname in enumerate(names_ordered):
            optode = optode_objects[oname]
            bpy.ops.object.mode_set(mode='OBJECT')
            hm = conn_obj.modifiers.new(name=f"Hook_{oname}", type='HOOK')
            hm.object = optode
            hm.vertex_group = f"VG_{oname}"
            for v in conn_obj.data.vertices:
                v.select = False
            conn_obj.data.vertices[i].select = True
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.object.hook_assign(modifier=hm.name)
            bpy.ops.object.hook_reset(modifier=hm.name)
            bpy.ops.object.hook_recenter(modifier=hm.name)

        bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.update()
        print("=== Hook creation complete ===")

        # Soft body
        context.view_layer.objects.active = conn_obj
        sb_mod = conn_obj.modifiers.new(name="Softbody", type='SOFT_BODY')
        sb = sb_mod.settings

        goal_group = conn_obj.vertex_groups.new(name="Goals")
        for i, oname in enumerate(names_ordered):
            if optode_objects[oname].get("is_anchor", 0):
                goal_group.add([i], 1.0, 'REPLACE')

        sb.use_goal = True
        sb.goal_default = 0.0
        sb.vertex_group_goal = "Goals"
        sb.goal_spring = 0.99
        sb.goal_friction = 0.0
        sb.use_edges = True
        sb.pull = 0.9
        sb.push = 0.9
        sb.damping = 0.5
        sb.plastic = 0
        sb.bend = 0.5
        sb.use_edge_collision = False
        sb.use_face_collision = False
        sb.collision_type = 'MANUAL'
        sb.step_min = 10
        sb.step_max = 100
        sb.mass = 1.0
        sb.speed = 1.0

        self.report(
            {'INFO'},
            f"Imported {len(optode_objects)} optodes, {len(edges)} connections  "
            f"(bary: {placement_stats['barycentric_index']+placement_stats['barycentric_label']}, "
            f"fallback: {placement_stats['nearest_lm']+placement_stats['raw']})"
        )
        return {'FINISHED'}

    # ──────────────────────────────────────────────────────────────────────
    #  Position resolution — the core accuracy logic
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_position(self, optode_data, landmark_mesh, landmark_labels,
                          label_to_indices, same_landmark_system, num_target_landmarks):
        """
        Return (Vector position, method_string).

        Resolution priority:
          1. Barycentric via vertex INDICES  (exact when same landmark system)
          2. Barycentric via vertex LABELS   (cross-system fallback)
          3. Nearest-landmark + offset       (legacy / partial data)
          4. Raw stored position             (last resort)
        """
        name = optode_data["name"]

        # Support both v2.0 "registration" key and v1.x "anchor_info" key
        reg = optode_data.get("registration") or optode_data.get("anchor_info")

        if reg and landmark_mesh and reg.get("method") == "barycentric":
            bary = reg.get("barycentric_coords")
            stored_indices = reg.get("vertex_indices")
            vlabels = reg.get("vertex_labels", [])

            if bary and len(bary) == 3:

                # ── Strategy 1: vertex indices (same landmark system) ────
                if same_landmark_system and stored_indices and len(stored_indices) == 3:
                    all_valid = all(0 <= idx < num_target_landmarks for idx in stored_indices)
                    if all_valid:
                        pos = self._reconstruct_from_indices(
                            landmark_mesh, stored_indices, bary
                        )
                        if pos is not None:
                            print(f"  {name}: BARY-INDEX  tri=[{stored_indices}]  w={[f'{b:.3f}' for b in bary]}")
                            return pos, "barycentric_index"

                # ── Strategy 2: vertex labels (cross-system) ─────────────
                if vlabels and len(vlabels) == 3:
                    resolved = self._resolve_labels_to_indices(
                        vlabels, stored_indices, label_to_indices, num_target_landmarks
                    )
                    if resolved is not None:
                        pos = self._reconstruct_from_indices(
                            landmark_mesh, resolved, bary
                        )
                        if pos is not None:
                            print(f"  {name}: BARY-LABEL  labels={vlabels} -> idx={resolved}  w={[f'{b:.3f}' for b in bary]}")
                            return pos, "barycentric_label"

                self.report({'WARNING'}, f"{name}: barycentric labels/indices failed, trying fallback")

        # ── Strategy 3: nearest landmark + offset ────────────────────────
        if reg and landmark_mesh:
            nl = reg.get("nearest_landmark")
            if nl:
                lm_name = nl.get("landmark", "")
                offset = nl.get("offset")
                candidates = label_to_indices.get(lm_name, [])
                if candidates and offset:
                    # Pick the first candidate — for duplicates like Cz they share position
                    lm_idx = candidates[0]
                    lm_world = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[lm_idx].co
                    pos = lm_world + Vector(offset)
                    print(f"  {name}: NEAREST-LM  landmark={lm_name}  idx={lm_idx}")
                    return pos, "nearest_lm"

        # ── Strategy 4: raw position ─────────────────────────────────────
        pos = Vector(optode_data["position"])
        is_anchor = optode_data.get("is_anchor", False)
        if is_anchor:
            self.report({'WARNING'}, f"Anchor {name}: using RAW position — cross-atlas will be wrong!")
        print(f"  {name}: RAW POSITION FALLBACK")
        return pos, "raw"

    @staticmethod
    def _reconstruct_from_indices(landmark_mesh, indices, bary):
        """Reconstruct world position from vertex indices + barycentric weights."""
        try:
            v0 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[indices[0]].co
            v1 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[indices[1]].co
            v2 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[indices[2]].co
            return bary[0] * v0 + bary[1] * v1 + bary[2] * v2
        except (IndexError, KeyError):
            return None

    @staticmethod
    def _resolve_labels_to_indices(vlabels, stored_indices, label_to_indices, num_verts):
        """
        Resolve three vertex labels to indices on the target LandmarkMesh.

        Handles duplicate labels (e.g. "Cz" at indices 4, 9, 18) by:
          - If label has exactly one candidate → use it.
          - If label has multiple candidates → prefer the stored_index if it matches
            one of the candidates; otherwise use the first candidate (all duplicates
            like "Cz" share the same physical position, so any is correct).
        """
        resolved = []
        for i, lbl in enumerate(vlabels):
            candidates = label_to_indices.get(lbl, [])
            if not candidates:
                return None  # label not found at all

            if len(candidates) == 1:
                resolved.append(candidates[0])
            else:
                # Multiple candidates — prefer stored index if valid
                hint = stored_indices[i] if (stored_indices and i < len(stored_indices)) else None
                if hint is not None and hint in candidates and hint < num_verts:
                    resolved.append(hint)
                else:
                    resolved.append(candidates[0])

        # Verify all indices are in range
        if all(0 <= idx < num_verts for idx in resolved):
            return resolved
        return None

    # ──────────────────────────────────────────────────────────────────────
    #  Light spring relaxation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _run_light_relaxation(context, optode_objects, connections, head_bvh,
                              n_iters=40, max_step=0.5, damping=0.25):
        """
        Gently adjust non-anchor optode positions to honour rest lengths
        while keeping them on the head surface.  Works in numpy for speed
        and applies final positions to Blender objects at the end.
        """
        names = list(optode_objects.keys())
        n = len(names)
        name_to_i = {nm: i for i, nm in enumerate(names)}

        # Current positions
        pos = np.zeros((n, 3), dtype=np.float64)
        for i, nm in enumerate(names):
            loc = optode_objects[nm].location
            pos[i] = (loc.x, loc.y, loc.z)

        # Edges: (i, j, rest_length)
        edge_list = []
        for cd in connections:
            n1, n2 = cd["optode1"], cd["optode2"]
            if n1 in name_to_i and n2 in name_to_i:
                edge_list.append((name_to_i[n1], name_to_i[n2], float(cd["rest_length"])))

        if not edge_list:
            return

        anchor_mask = np.array(
            [bool(optode_objects[nm].get("is_anchor", 0)) for nm in names],
            dtype=bool,
        )

        # Store initial barycentric positions so we can penalise drift
        init_pos = pos.copy()

        for iteration in range(n_iters):
            forces = np.zeros_like(pos)

            # Spring forces
            for i1, i2, rest_len in edge_list:
                diff = pos[i2] - pos[i1]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    continue
                displacement = dist - rest_len
                f = (diff / dist) * displacement * damping
                forces[i1] += f
                forces[i2] -= f

            # Light anchor-return force for non-anchors to prevent drift
            # from barycentric placement (weak — 10% of spring damping)
            drift = init_pos - pos
            forces += drift * (damping * 0.10)

            # Zero out forces on anchors
            forces[anchor_mask] = 0.0

            # Clamp per-optode displacement
            mags = np.linalg.norm(forces, axis=1, keepdims=True)
            mags_safe = np.maximum(mags, 1e-10)
            scale = np.minimum(1.0, max_step / mags_safe)
            forces *= scale

            pos += forces

            # Snap non-anchors back to head surface
            for i in range(n):
                if anchor_mask[i]:
                    continue
                loc, normal, idx, dist = head_bvh.find_nearest(Vector(pos[i].tolist()))
                if loc is not None:
                    pos[i] = (loc.x, loc.y, loc.z)

        # Compute and report displacement from initial placement
        displacements = np.linalg.norm(pos - init_pos, axis=1)
        moved = displacements[~anchor_mask]
        if len(moved) > 0:
            print(f"  Relaxation complete: mean shift = {np.mean(moved):.3f} mm, "
                  f"max = {np.max(moved):.3f} mm  (anchors unchanged)")

        # Apply back to Blender
        for i, nm in enumerate(names):
            optode_objects[nm].location = Vector(pos[i].tolist())

        context.view_layer.update()

    # ──────────────────────────────────────────────────────────────────────
    #  Geometry helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_head_scale(head_mesh):
        bbox_min = Vector([min(v.co[i] for v in head_mesh.data.vertices) for i in range(3)])
        bbox_max = Vector([max(v.co[i] for v in head_mesh.data.vertices) for i in range(3)])
        dims = (head_mesh.matrix_world @ bbox_max) - (head_mesh.matrix_world @ bbox_min)
        hs = max(dims)
        return hs * 0.02, hs * 0.02 * 0.15

    @staticmethod
    def _snap_to_surface(position, bvh):
        location, normal, index, distance = bvh.find_nearest(Vector(position))
        if location is None:
            return Vector(position), Vector((0, 0, 1))
        return Vector(location), Vector(normal)

    @staticmethod
    def _create_optode_disc(position, normal, name, color, diameter, thickness):
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32, radius=diameter / 2.0, depth=thickness,
            enter_editmode=False, align='WORLD', location=(0, 0, 0),
        )
        optode = bpy.context.active_object
        optode.name = name
        rot_q = Vector((0, 0, 1)).rotation_difference(Vector(normal))
        optode.rotation_euler = rot_q.to_euler()
        optode.location = Vector(position)

        mat = bpy.data.materials.new(name=f"{name}_mat")
        mat.use_nodes = True
        mat.diffuse_color = color
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Metallic'].default_value = 0.3
            bsdf.inputs['Roughness'].default_value = 0.4
        optode.data.materials.append(mat)
        for face in optode.data.polygons:
            face.use_smooth = True
        return optode


# ==============================================================================
#  ANCHOR DEFINITION (unchanged interface)
# ==============================================================================

class NEUROCAPTAIN_OT_define_anchor_optode(bpy.types.Operator):
    """Define selected optode(s) as anchor points"""
    bl_idname = "neurocaptain.define_anchor_optode"
    bl_label = "Define as Anchor"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        selected = [
            obj for obj in context.selected_objects
            if obj.name.startswith("Source_") or obj.name.startswith("Detector_")
        ]
        if not selected:
            self.report({'ERROR'}, "No optodes selected")
            return {'CANCELLED'}

        ring_mat = create_anchor_ring_material()

        for optode in selected:
            if optode.get("is_anchor", False):
                continue
            optode["is_anchor"] = True

            bbox = [Vector(c) for c in optode.bound_box]
            radius = (max(v.x for v in bbox) - min(v.x for v in bbox)) / 2

            ring_mesh = create_torus_mesh(
                name=f"Anchor_{optode.name}",
                major_radius=radius * 1.4,
                minor_radius=0.4,
                major_segments=24,
                minor_segments=8,
            )
            anchors_coll = get_or_create_collection("Anchor Indicators")
            ring_obj = bpy.context.active_object
            ring_obj.name = f"Anchor_{optode.name}"
            ring_obj.data.materials.append(ring_mat)
            add_to_collection(ring_obj, anchors_coll)

            ring_obj.parent = optode
            ring_obj.location = (0, 0, 0)
            ring_obj.rotation_euler = (0, 0, 0)
            ring_obj["is_anchor_ring"] = True
            ring_obj["parent_optode"] = optode.name
            ring_obj.hide_select = True

        self.report({'INFO'}, f"Defined {len(selected)} optode(s) as anchors")
        return {'FINISHED'}


class NEUROCAPTAIN_OT_undefine_anchor_optode(bpy.types.Operator):
    """Remove anchor status from selected optode(s)"""
    bl_idname = "neurocaptain.undefine_anchor_optode"
    bl_label = "Remove Anchor Status"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        selected_optodes = [obj for obj in context.selected_objects 
                           if (obj.name.startswith("Source_") or obj.name.startswith("Detector_"))
                           and obj.get("is_anchor", False)]
        
        if not selected_optodes:
            self.report({'ERROR'}, "No anchor optodes selected")
            return {'CANCELLED'}
        
        for optode in selected_optodes:
            if "is_anchor" in optode:
                del optode["is_anchor"]
            
            rings_to_remove = []
            
            for child in optode.children:
                if child.get("is_anchor_ring"):
                    rings_to_remove.append(child)
            
            for obj in bpy.data.objects:
                if obj.get("is_anchor_ring") and obj.get("parent_optode") == optode.name:
                    if obj not in rings_to_remove:
                        rings_to_remove.append(obj)
            
            for ring in rings_to_remove:
                mesh = ring.data
                bpy.data.objects.remove(ring, do_unlink=True)
                if mesh and mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
        
        self.report({'INFO'}, f"Removed anchor status from {len(selected_optodes)} optode(s)")
        return {'FINISHED'}

import bpy
import bmesh
from mathutils import Vector, Matrix, Quaternion
from mathutils.bvhtree import BVHTree
import math

class NEUROCAPTAIN_OT_rigid_rotate_optodes(bpy.types.Operator):
    """Rigidly rotate all optodes while maintaining surface constraints"""
    bl_idname = "neurocaptain.rigid_rotate_optodes"
    bl_label = "Rigid Rotate Optodes"
    bl_options = {'REGISTER', 'UNDO', 'GRAB_CURSOR', 'BLOCKING'}
    
    # State variables
    initial_positions = {}
    initial_rotations = {}
    rotation_center = None
    optodes = []
    bvh_tree = None
    original_distances = {}
    rotation_axis = Vector((0, 0, 1))
    _initial_mouse_x = 0
    _current_angle = 0.0
    distance_tolerance = 0.15

    def modal(self, context, event):
        context.area.tag_redraw()

        if event.type == 'MOUSEMOVE':
            self._current_angle = (event.mouse_x - self._initial_mouse_x) * 0.01

            if self.apply_rotation(context):
                angle_deg = math.degrees(self._current_angle)
                context.area.header_text_set(f"Rotate: {angle_deg:.1f}°")
            else:
                context.area.header_text_set("⚠ Rotation limited by constraints")
            return {'RUNNING_MODAL'}

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self.update_connections(context)
            context.area.header_text_set(None)
            return {'FINISHED'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            for opt in self.optodes:
                opt.location = self.initial_positions[opt.name]
                opt.rotation_euler = self.initial_rotations[opt.name]
            context.view_layer.update()
            context.area.header_text_set(None)
            return {'CANCELLED'}
        
        elif event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'MIDDLEMOUSE'}:
            return {'PASS_THROUGH'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        # Auto-select all optodes and connections
        bpy.ops.object.select_all(action='DESELECT')
        self.optodes = [obj for obj in bpy.data.objects 
                       if obj.name.startswith(('Source_', 'Detector_'))]
        
        for opt in self.optodes:
            opt.select_set(True)
        
        if "Optode_Connections" in bpy.data.objects:
            bpy.data.objects["Optode_Connections"].select_set(True)
        
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        if len(self.optodes) < 2:
            self.report({'ERROR'}, "Need at least 2 optodes")
            return {'CANCELLED'}
        
        # Find head mesh
        head_mesh = bpy.data.objects.get("headmesh")
        if not head_mesh:
            for opt in self.optodes:
                for con in opt.constraints:
                    if con.type == 'SHRINKWRAP' and con.target:
                        head_mesh = con.target
                        break
                if head_mesh:
                    break
        
        if not head_mesh:
            self.report({'ERROR'}, "No head mesh found")
            return {'CANCELLED'}
        
        # Build BVH tree
        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = head_mesh.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.transform(head_mesh.matrix_world)
        self.bvh_tree = BVHTree.FromBMesh(bm)
        bm.free()
        eval_obj.to_mesh_clear()
        
        # Store initial state
        self.rotation_center = Vector((0, 0, 0))
        for opt in self.optodes:
            self.initial_positions[opt.name] = opt.location.copy()
            self.initial_rotations[opt.name] = opt.rotation_euler.copy()
            self.rotation_center += opt.location
        self.rotation_center /= len(self.optodes)
        
        # Store distances
        for i, opt1 in enumerate(self.optodes):
            for opt2 in self.optodes[i+1:]:
                key = tuple(sorted([opt1.name, opt2.name]))
                self.original_distances[key] = (opt1.location - opt2.location).length
        
        # Get rotation axis from view
        if context.region_data:
            view_matrix = context.region_data.view_matrix.inverted()
            self.rotation_axis = view_matrix.col[2].xyz.normalized()
        
        self._initial_mouse_x = event.mouse_x
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def apply_rotation(self, context):
        """Apply rotation with constraints"""
        rot_quat = Quaternion(self.rotation_axis, self._current_angle)
        rot_matrix = rot_quat.to_matrix().to_4x4()

        new_positions = {}
        new_rotations = {}

        # Calculate new positions
        for opt in self.optodes:
            rel_pos = self.initial_positions[opt.name] - self.rotation_center
            rotated = rot_matrix @ rel_pos
            new_pos = self.rotation_center + rotated

            # Project to surface
            location, normal, _, _ = self.bvh_tree.find_nearest(new_pos)
            # Only reject nearly-straight-down normals (chin underside).
            # -0.8 ≈ 144° from up — much less restrictive than the old -0.3.
            if not location or normal.z < -0.8:
                return False
            
            new_positions[opt.name] = location
            new_rotations[opt.name] = Vector((0, 0, 1)).rotation_difference(normal).to_euler()
        
        # Validate distances
        for i, opt1 in enumerate(self.optodes):
            for opt2 in self.optodes[i+1:]:
                key = tuple(sorted([opt1.name, opt2.name]))
                orig_dist = self.original_distances[key]
                new_dist = (new_positions[opt1.name] - new_positions[opt2.name]).length
                ratio = new_dist / orig_dist
                if ratio < (1 - self.distance_tolerance) or ratio > (1 + self.distance_tolerance):
                    return False
        
        # Apply
        for opt in self.optodes:
            opt.location = new_positions[opt.name]
            opt.rotation_euler = new_rotations[opt.name]
        
        context.view_layer.update()
        return True
    
    def update_connections(self, context):
        """Update connection mesh"""
        conn_obj = bpy.data.objects.get('Optode_Connections')
        if not conn_obj:
            return
        
        optode_names = conn_obj.get("optode_names", [])
        if not optode_names:
            return
        
        # Update in edit mode
        bpy.ops.object.select_all(action='DESELECT')
        conn_obj.select_set(True)
        context.view_layer.objects.active = conn_obj
        
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(conn_obj.data)
        bm.verts.ensure_lookup_table()
        
        conn_inv = conn_obj.matrix_world.inverted()
        for i, name in enumerate(optode_names):
            if i < len(bm.verts):
                opt = bpy.data.objects.get(name)
                if opt and opt in self.optodes:
                    bm.verts[i].co = conn_inv @ opt.location
        
        bmesh.update_edit_mesh(conn_obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Reset hooks
        for mod in conn_obj.modifiers:
            if mod.type == 'HOOK' and mod.object in self.optodes:
                if mod.vertex_group in conn_obj.vertex_groups:
                    vg = conn_obj.vertex_groups[mod.vertex_group]
                    for v in conn_obj.data.vertices:
                        v.select = False
                    for v in conn_obj.data.vertices:
                        try:
                            if vg.weight(v.index) > 0.5:
                                v.select = True
                                break
                        except:
                            pass
                    bpy.ops.object.mode_set(mode='EDIT')
                    bpy.ops.object.hook_reset(modifier=mod.name)
                    bpy.ops.object.mode_set(mode='OBJECT')
        
        context.view_layer.update()






# REGISTRATION
def register():
    bpy.utils.register_class(NEUROCAPTAIN_OT_create_optode_connections)
    bpy.utils.register_class(NEUROCAPTAIN_OT_create_optode_connections_delaunay)
    bpy.utils.register_class(NEUROCAPTAIN_OT_modify_spring_properties)
    bpy.utils.register_class(NEUROCAPTAIN_OT_make_spring_stiff)
    bpy.utils.register_class(NEUROCAPTAIN_OT_update_optode_connections)
    bpy.utils.register_class(NEUROCAPTAIN_OT_toggle_connection_visibility)
    bpy.utils.register_class(NEUROCAPTAIN_OT_export_optode_json)
    bpy.utils.register_class(NEUROCAPTAIN_OT_import_optode_json)
    bpy.utils.register_class(NEUROCAPTAIN_OT_define_anchor_optode)
    bpy.utils.register_class(NEUROCAPTAIN_OT_undefine_anchor_optode)
    bpy.utils.register_class(NEUROCAPTAIN_OT_rigid_rotate_optodes)


def unregister():
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_undefine_anchor_optode)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_define_anchor_optode)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_import_optode_json)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_export_optode_json)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_toggle_connection_visibility)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_update_optode_connections)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_make_spring_stiff)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_modify_spring_properties)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_create_optode_connections_delaunay)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_create_optode_connections)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_rigid_rotate_optodes)

if __name__ == "__main__":
    register()