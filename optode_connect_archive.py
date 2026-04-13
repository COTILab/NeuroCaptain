import bpy
import bmesh
from mathutils import Vector
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
        if "Optode_Connections" not in bpy.data.objects:
            self.report({'ERROR'}, "No Optode_Connections object found")
            return {'CANCELLED'}
        
        conn_obj = bpy.data.objects["Optode_Connections"]
        
        updated_count = 0
        max_distance = 0.0
        
        print("\n=== Updating Optode_Connections vertices ===")
        
        # Force update each vertex to match its optode position
        for modifier in conn_obj.modifiers:
            if modifier.type == 'HOOK' and modifier.object:
                optode = modifier.object
                
                # Find which vertex this hook controls
                if modifier.vertex_group and modifier.vertex_group in conn_obj.vertex_groups:
                    vgroup = conn_obj.vertex_groups[modifier.vertex_group]
                    
                    for v in conn_obj.data.vertices:
                        try:
                            weight = vgroup.weight(v.index)
                            if weight > 0.5:
                                # This is the vertex controlled by this hook
                                old_pos = v.co.copy()
                                
                                # FORCE vertex to optode position (ignore current position)
                                v.co = optode.location.copy()
                                
                                distance = (v.co - old_pos).length
                                if distance > max_distance:
                                    max_distance = distance
                                
                                if distance > 0.001:
                                    print(f"  Vertex {v.index} ({optode.name}): moved {distance:.4f} units")
                                    updated_count += 1
                                
                                break  # Found the vertex, move to next modifier
                        except:
                            pass
        
        # CRITICAL: Update mesh geometry so edges redraw
        conn_obj.data.update()
        
        # Force viewport refresh
        context.view_layer.update()
        
        # Also force depsgraph update
        context.evaluated_depsgraph_get().update()
        
        print(f"=== Update complete: {updated_count} vertices moved ===\n")
        
        if updated_count == 0:
            self.report({'INFO'}, "All vertices already match optode positions")
        else:
            self.report({'INFO'}, f"Synced {updated_count} vertices (max movement: {max_distance:.4f} units)")
        
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
        
        landmark_mesh = None
        landmark_labels = []
        # used to reference anchors to nearest landmark
        if "LandmarkMesh" in bpy.data.objects:
            landmark_mesh = bpy.data.objects["LandmarkMesh"]
            num_landmarks = len(landmark_mesh.data.vertices)
            
            if num_landmarks >= 77:
                landmark_labels = [
                    "Nz", "Iz", "Lpa", "Rpa", "Cz",
                    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
                    "Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz",
                    "FT7", "F7", "AF7", "Fp1", "TP7", "P7", "PO7", "O1",
                    "FT8", "F8", "AF8", "Fp2", "TP8", "P8", "PO8", "O2",
                    "FC1", "FC3", "FC5", "FC2", "FC4", "FC6",
                    "F1", "F3", "F5", "F2", "F4", "F6", "AF3", "AF4",
                    "CP1", "CP3", "CP5", "CP2", "CP4", "CP6",
                    "P1", "P3", "P5", "P2", "P4", "P6", "PO3", "PO4",
                    "FT9", "F9", "", "", "TP9", "P9", "PO9", "O9",
                    "FT10", "F10", "", "", "TP10", "P10", "PO10", "O10"
                ]
            elif num_landmarks >= 27:
                landmark_labels = [
                    "Nz", "Iz", "Lpa", "Rpa", "Cz",
                    "T3", "C3", "Cz", "C4", "T4",
                    "Fpz", "Fz", "Cz", "Pz", "Oz",
                    "F7", "Fp1", "T5", "O1", "F8", "Fp2", "T6", "O2",
                    "F3", "F4", "P3", "P4"
                ]
        
        landmark_bvh = None
        if landmark_mesh:
            depsgraph = context.evaluated_depsgraph_get() #accessing indexable data 
            landmark_bvh = BVHTree.FromObject(landmark_mesh, depsgraph)
        
        optodes = []
        for obj in bpy.data.objects:
            if obj.name.startswith("Source_") or obj.name.startswith("Detector_"):
                optode_data = {
                    "name": obj.name,
                    "position": list(obj.location),
                    "type": "source" if obj.name.startswith("Source_") else "detector",
                    "is_anchor": obj.get("is_anchor", False),
                }
                
                if optode_data["is_anchor"] and landmark_mesh and landmark_bvh:
                    optode_pos = obj.location
                    location, normal, face_index, distance = landmark_bvh.find_nearest(optode_pos)
                    
                    if location is not None and face_index is not None:
                        face = landmark_mesh.data.polygons[face_index]
                        vert_indices = list(face.vertices)
                        v0_co = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[vert_indices[0]].co
                        v1_co = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[vert_indices[1]].co
                        v2_co = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[vert_indices[2]].co
                        
                        bary_coords = self.calculate_barycentric(optode_pos, v0_co, v1_co, v2_co)
                        
                        vertex_labels = []
                        for v_idx in vert_indices:
                            if v_idx < len(landmark_labels):
                                label = landmark_labels[v_idx]
                                vertex_labels.append(label if label else f"vertex_{v_idx}")
                            else:
                                vertex_labels.append(f"vertex_{v_idx}")
                        
                        optode_data["anchor_info"] = {
                            "method": "barycentric",
                            "face_index": int(face_index),
                            "vertex_indices": [int(v) for v in vert_indices],
                            "vertex_labels": vertex_labels,
                            "barycentric_coords": [float(bary_coords[0]), 
                                                   float(bary_coords[1]), 
                                                   float(bary_coords[2])],
                            "distance_to_surface": float(distance)
                        }
                        
                        min_dist = float('inf')
                        closest_idx = -1
                        
                        for i, vert in enumerate(landmark_mesh.data.vertices):
                            vert_world = landmark_mesh.matrix_world @ vert.co
                            dist = (vert_world - optode_pos).length
                            if dist < min_dist:
                                min_dist = dist
                                closest_idx = i
                        
                        if closest_idx >= 0 and closest_idx < len(landmark_labels):
                            landmark_name = landmark_labels[closest_idx]
                            if landmark_name:
                                closest_vert = landmark_mesh.data.vertices[closest_idx]
                                landmark_world = landmark_mesh.matrix_world @ closest_vert.co
                                offset = optode_pos - landmark_world
                                
                                optode_data["anchor_info"]["nearest_landmark"] = {
                                    "landmark": landmark_name,
                                    "landmark_index": int(closest_idx),
                                    "offset": [float(offset.x), float(offset.y), float(offset.z)],
                                    "distance": float(min_dist)
                                }
                
                custom_props = {}
                for key in obj.keys():
                    if key not in ["_RNA_UI", "is_anchor"] and not key.startswith("_"):
                        try:
                            json.dumps(obj[key])
                            custom_props[key] = obj[key]
                        except:
                            pass
                if custom_props:
                    optode_data["custom_properties"] = custom_props
                
                optodes.append(optode_data)
        
        if not optodes:
            self.report({'WARNING'}, "No optodes found to export")
            return {'CANCELLED'}
        
        num_anchors = len([o for o in optodes if o["is_anchor"]])
        if num_anchors > 0 and not landmark_mesh:
            self.report({'WARNING'}, 
                f"Found {num_anchors} anchor optodes but no LandmarkMesh")
        
        springs = []
        if "Optode_Connections" in bpy.data.objects:
            conn_obj = bpy.data.objects["Optode_Connections"]
            spring_states = conn_obj.get("spring_states", {})
            
            for edge_key, props in spring_states.items():
                springs.append({
                    "optode1": props["optode1"],
                    "optode2": props["optode2"],
                    "rest_length": props["rest_length"],
                    "pull": props["pull"],
                    "push": props["push"],
                    "is_flexible": props["is_flexible"]
                })
        
        if not springs:
            self.report({'WARNING'}, "No spring connections found")
        
        export_data = {
            "format": "NeuroCaptain Probe Configuration",
            "version": "1.1",
            "metadata": {
                "export_date": str(bpy.context.scene.frame_current),
                "num_sources": len([o for o in optodes if o["type"] == "source"]),
                "num_detectors": len([o for o in optodes if o["type"] == "detector"]),
                "num_connections": len(springs),
                "num_anchors": len([o for o in optodes if o["is_anchor"]])
            },
            "optodes": optodes,
            "connections": springs,
            "global_settings": {
                "default_pull": 0.9,
                "default_push": 0.9
            }
        }
        
        try:
            with open(self.filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.report({'INFO'}, 
                f"Exported {len(optodes)} optodes with {len(springs)} connections")
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to export: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}
    
    def calculate_barycentric(self, point, v0, v1, v2):
        """Calculate barycentric coordinates of point relative to triangle"""
        p = Vector(point)
        v0 = Vector(v0)
        v1 = Vector(v1)
        v2 = Vector(v2)
        
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = p - v0
        
        dot00 = v0v1.dot(v0v1)
        dot01 = v0v1.dot(v0v2)
        dot02 = v0v1.dot(v0p)
        dot11 = v0v2.dot(v0v2)
        dot12 = v0v2.dot(v0p)
        
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        w = 1.0 - u - v
        
        return (w, u, v)


class NEUROCAPTAIN_OT_import_optode_json(bpy.types.Operator):
    """Import optode probe configuration from JSON"""
    bl_idname = "neurocaptain.import_optode_json"
    bl_label = "Import Probe Configuration"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        import json
        from mathutils.bvhtree import BVHTree
        
        source_color = (1.0, 0.0, 0.0, 1.0)
        detector_color = (0.0, 0.0, 0.0, 1.0)
        
        if "headmesh" not in bpy.data.objects:
            self.report({'ERROR'}, "No 'headmesh' found")
            return {'CANCELLED'}
        
        landmark_mesh = None
        landmark_labels = []
        
        if "LandmarkMesh" in bpy.data.objects:
            landmark_mesh = bpy.data.objects["LandmarkMesh"]
            num_landmarks = len(landmark_mesh.data.vertices)
            
            if num_landmarks >= 77:
                landmark_labels = [
                    "Nz", "Iz", "Lpa", "Rpa", "Cz",
                    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
                    "Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz",
                    "FT7", "F7", "AF7", "Fp1", "TP7", "P7", "PO7", "O1",
                    "FT8", "F8", "AF8", "Fp2", "TP8", "P8", "PO8", "O2",
                    "FC1", "FC3", "FC5", "FC2", "FC4", "FC6",
                    "F1", "F3", "F5", "F2", "F4", "F6", "AF3", "AF4",
                    "CP1", "CP3", "CP5", "CP2", "CP4", "CP6",
                    "P1", "P3", "P5", "P2", "P4", "P6", "PO3", "PO4",
                    "FT9", "F9", "", "", "TP9", "P9", "PO9", "O9",
                    "FT10", "F10", "", "", "TP10", "P10", "PO10", "O10"
                ]
            elif num_landmarks >= 27:
                landmark_labels = [
                    "Nz", "Iz", "Lpa", "Rpa", "Cz",
                    "T3", "C3", "Cz", "C4", "T4",
                    "Fpz", "Fz", "Cz", "Pz", "Oz",
                    "F7", "Fp1", "T5", "O1", "F8", "Fp2", "T6", "O2",
                    "F3", "F4", "P3", "P4"
                ]
        else:
            self.report({'WARNING'}, "No 'LandmarkMesh' found")
        
        try:
            with open(self.filepath, 'r') as f:
                config = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load JSON: {str(e)}")
            return {'CANCELLED'}
        
        if config.get("format") != "NeuroCaptain Probe Configuration":
            self.report({'ERROR'}, "Invalid JSON format")
            return {'CANCELLED'}
        
        headmesh = bpy.data.objects["headmesh"]
        optode_objects = {}
        
        optode_diameter, optode_thickness = self.get_head_scale(headmesh)
        
        bpy.ops.object.select_all(action='DESELECT')
        
        for optode_data in config.get("optodes", []):
            name = optode_data["name"]
            optode_type = optode_data["type"]
            is_anchor = optode_data.get("is_anchor", False)
            
            if is_anchor and "anchor_info" in optode_data and landmark_mesh:
                anchor_info = optode_data["anchor_info"]
                                
                if anchor_info.get("method") == "barycentric" and "barycentric_coords" in anchor_info:
                    vertex_indices = anchor_info["vertex_indices"]
                    bary_coords = anchor_info["barycentric_coords"]
                    
                    try:
                        v0 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[vertex_indices[0]].co
                        v1 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[vertex_indices[1]].co
                        v2 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[vertex_indices[2]].co
                        
                        initial_pos = (bary_coords[0] * v0 + 
                                      bary_coords[1] * v1 + 
                                      bary_coords[2] * v2)
                    except (IndexError, KeyError) as e:
                        self.report({'WARNING'}, f"Failed barycentric for {name}: {e}")
                        initial_pos = Vector(optode_data["position"])
                
                elif "nearest_landmark" in anchor_info:
                    landmark_data = anchor_info["nearest_landmark"]
                    landmark_name = landmark_data["landmark"]
                    offset = Vector(landmark_data["offset"])
                    
                    try:
                        landmark_idx = landmark_labels.index(landmark_name)
                        landmark_vert = landmark_mesh.data.vertices[landmark_idx]
                        landmark_world = landmark_mesh.matrix_world @ landmark_vert.co
                        initial_pos = landmark_world + offset
                    except (ValueError, IndexError):
                        self.report({'WARNING'}, f"Landmark {landmark_name} not found for {name}")
                        initial_pos = Vector(optode_data["position"])
                else:
                    initial_pos = Vector(optode_data["position"])
            else:
                initial_pos = Vector(optode_data["position"])
            
            snapped_pos, surface_normal = self.snap_to_mesh_surface(initial_pos, headmesh)
            
            if optode_type == "source":
                sources_collection = get_or_create_collection("Sources")

                optode = self.create_optode_disc(snapped_pos, surface_normal, name, 
                                                 source_color, optode_diameter, optode_thickness)
                add_to_collection(optode, sources_collection)
            else:
                detectors_collection = get_or_create_collection("Detectors")
                optode = self.create_optode_disc(snapped_pos, surface_normal, name, 
                                                 detector_color, optode_diameter, optode_thickness)
                add_to_collection(optode, detectors_collection)    
            
            optode["is_anchor"] = 1 if is_anchor else 0
            
            if is_anchor:
                
                Anchors_collection = get_or_create_collection("Anchor Indicators")
                ring_mat = create_anchor_ring_material()
                bpy.ops.mesh.primitive_torus_add(location=(0, 0, 0), major_radius=2.5, minor_radius=0.3)
                indicator = context.active_object
                add_to_collection(indicator, Anchors_collection)   
                indicator.name = f"Anchor_{name}"
                indicator.parent = optode
                indicator.location = (0, 0, 0)
                indicator.hide_render = True
                indicator.data.materials.append(ring_mat)
                            
            constraint = optode.constraints.new(type='SHRINKWRAP')
            constraint.target = headmesh
            constraint.shrinkwrap_type = 'NEAREST_SURFACE'
            constraint.distance = 0.0
            constraint.influence = 1.0
            
            optode_objects[name] = optode
        
        mesh = bpy.data.meshes.new("Optode_Connections")
        conn_obj = bpy.data.objects.new("Optode_Connections", mesh)
        context.collection.objects.link(conn_obj)
        
        vertices = []
        edges = []
        spring_states = {}
        
        optode_names_ordered = list(optode_objects.keys())
        
        for optode_name in optode_names_ordered:
            optode = optode_objects[optode_name]
            vertices.append(optode.location.copy())
        
        optode_name_to_idx = {name: i for i, name in enumerate(optode_names_ordered)}
        
        for conn_data in config.get("connections", []):
            opt1_name = conn_data["optode1"]
            opt2_name = conn_data["optode2"]
            
            if opt1_name in optode_name_to_idx and opt2_name in optode_name_to_idx:
                idx1 = optode_name_to_idx[opt1_name]
                idx2 = optode_name_to_idx[opt2_name]
                edges.append((idx1, idx2))
                
                edge_key = f"{min(idx1, idx2)}_{max(idx1, idx2)}"
                spring_states[edge_key] = {
                    "optode1": opt1_name,
                    "optode2": opt2_name,
                    "rest_length": conn_data["rest_length"],
                    "pull": conn_data["pull"],
                    "push": conn_data["push"],
                    "is_flexible": conn_data["is_flexible"]
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
        
        bpy.ops.object.select_all(action='DESELECT')
        conn_obj.select_set(True)
        bpy.context.view_layer.objects.active = conn_obj
        
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        for i, optode_name in enumerate(optode_names_ordered):
            vgroup = conn_obj.vertex_groups.new(name=f"VG_{optode_name}")
            vgroup.add([i], 1.0, 'REPLACE')
        

        # ===== FIXED HOOK CREATION =====
        # First, create vertex groups for each vertex
        for i, optode_name in enumerate(optode_names_ordered):
            vgroup = conn_obj.vertex_groups.new(name=f"VG_{optode_name}")
            vgroup.add([i], 1.0, 'REPLACE')

        print(f"\n=== Creating hooks for {len(optode_names_ordered)} optodes ===")

        # Select conn_obj and enter edit mode
        bpy.ops.object.select_all(action='DESELECT')
        conn_obj.select_set(True)
        bpy.context.view_layer.objects.active = conn_obj

        for i, optode_name in enumerate(optode_names_ordered):
            optode = optode_objects[optode_name]
            
            # Create the hook modifier first (in object mode)
            bpy.ops.object.mode_set(mode='OBJECT')
            hook_mod = conn_obj.modifiers.new(name=f"Hook_{optode_name}", type='HOOK')
            hook_mod.object = optode
            hook_mod.vertex_group = f"VG_{optode_name}"
            
            # Now we need to properly initialize the hook
            # Go to edit mode and select the vertex
            for v in conn_obj.data.vertices:
                v.select = False
            conn_obj.data.vertices[i].select = True
            
            bpy.ops.object.mode_set(mode='EDIT')
            
            # Assign the selected vertex to this hook modifier
            bpy.ops.object.hook_assign(modifier=hook_mod.name)
            
            # Reset and recenter the hook to bind it properly
            bpy.ops.object.hook_reset(modifier=hook_mod.name)
            bpy.ops.object.hook_recenter(modifier=hook_mod.name)
            
            print(f"  Created and bound {hook_mod.name}")

        # Return to object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.update()

        print(f"=== Hook creation complete ===")
        # ===== END HOOK FIX =====
        
        bpy.context.view_layer.objects.active = conn_obj

        soft_body_mod = conn_obj.modifiers.new(name="Softbody", type='SOFT_BODY')
        soft_body = soft_body_mod.settings

        # Only use goals for ANCHORS, not flexible optodes
        goal_group = conn_obj.vertex_groups.new(name="Goals")

        for i, optode_name in enumerate(optode_names_ordered):
            optode = optode_objects[optode_name]
            if optode.get("is_anchor", 0):
                # Anchors only: fully pinned
                goal_group.add([i], 1.0, 'REPLACE')
            # Don't add flexible optodes to goals - let them be controlled purely by hooks and springs

        # Configure soft body
        soft_body.use_goal = True
        soft_body.goal_default = 0.0  # Default no goal
        soft_body.vertex_group_goal = "Goals"  # Only anchors have goals
        soft_body.goal_spring = 0.99
        soft_body.goal_friction = 0.0

        soft_body.use_edges = True
        soft_body.pull = 0.9
        soft_body.push = 0.9
        soft_body.damping = 0.5
        soft_body.plastic = 0
        soft_body.bend = 0.5

        soft_body.use_edge_collision = False
        soft_body.use_face_collision = False
        soft_body.collision_type = 'MANUAL'

        soft_body.step_min = 10
        soft_body.step_max = 100
        soft_body.mass = 1.0
        soft_body.speed = 1.0
        
        self.report({'INFO'}, 
            f"Imported {len(optode_objects)} optodes with {len(edges)} connections")
        
        return {'FINISHED'}
    
    def get_head_scale(self, head_mesh):
        """Calculate optode size based on head dimensions"""
        bbox_min = Vector([min(v.co[i] for v in head_mesh.data.vertices) for i in range(3)])
        bbox_max = Vector([max(v.co[i] for v in head_mesh.data.vertices) for i in range(3)])
        
        dimensions = (head_mesh.matrix_world @ bbox_max) - (head_mesh.matrix_world @ bbox_min)
        head_size = max(dimensions)
        
        return head_size * 0.02, head_size * 0.02 * 0.15
    
    def snap_to_mesh_surface(self, position, head_mesh):
        """Snap point to nearest surface location"""
        bvh = BVHTree.FromObject(head_mesh, bpy.context.evaluated_depsgraph_get())
        location, normal, index, distance = bvh.find_nearest(Vector(position))
        
        if location is None:
            return position, Vector((0, 0, 1))
        
        return Vector(location), Vector(normal)
    
    def create_optode_disc(self, position, normal, name, color, diameter, thickness):
        """Create optode disc at position"""
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32, radius=diameter/2.0, depth=thickness,
            enter_editmode=False, align='WORLD', location=(0, 0, 0)
        )
        
        optode = bpy.context.active_object
        optode.name = name
        
        z_axis = Vector((0, 0, 1))
        rotation_quat = z_axis.rotation_difference(Vector(normal))
        optode.rotation_euler = rotation_quat.to_euler()
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
    
class NEUROCAPTAIN_OT_import_optode_json_goals(bpy.types.Operator):
    """Import optode probe configuration from JSON with goal-based spring system"""
    bl_idname = "neurocaptain.import_optode_json_goals"
    bl_label = "Import Probe Configuration (Goals)"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        import json
        from mathutils.bvhtree import BVHTree
        
        # Turn off gravity
        bpy.context.scene.use_gravity = False
        
        source_color = (1.0, 0.0, 0.0, 1.0)
        detector_color = (0.0, 0.0, 0.0, 1.0)
        
        if "headmesh" not in bpy.data.objects:
            self.report({'ERROR'}, "No 'headmesh' found")
            return {'CANCELLED'}
        
        landmark_mesh = None
        landmark_labels = []
        
        if "LandmarkMesh" in bpy.data.objects:
            landmark_mesh = bpy.data.objects["LandmarkMesh"]
            num_landmarks = len(landmark_mesh.data.vertices)
            
            if num_landmarks >= 77:
                landmark_labels = [
                    "Nz", "Iz", "Lpa", "Rpa", "Cz",
                    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
                    "Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz",
                    "FT7", "F7", "AF7", "Fp1", "TP7", "P7", "PO7", "O1",
                    "FT8", "F8", "AF8", "Fp2", "TP8", "P8", "PO8", "O2",
                    "FC1", "FC3", "FC5", "FC2", "FC4", "FC6",
                    "F1", "F3", "F5", "F2", "F4", "F6", "AF3", "AF4",
                    "CP1", "CP3", "CP5", "CP2", "CP4", "CP6",
                    "P1", "P3", "P5", "P2", "P4", "P6", "PO3", "PO4",
                    "FT9", "F9", "", "", "TP9", "P9", "PO9", "O9",
                    "FT10", "F10", "", "", "TP10", "P10", "PO10", "O10"
                ]
            elif num_landmarks >= 27:
                landmark_labels = [
                    "Nz", "Iz", "Lpa", "Rpa", "Cz",
                    "T3", "C3", "Cz", "C4", "T4",
                    "Fpz", "Fz", "Cz", "Pz", "Oz",
                    "F7", "Fp1", "T5", "O1", "F8", "Fp2", "T6", "O2",
                    "F3", "F4", "P3", "P4"
                ]
        else:
            self.report({'WARNING'}, "No 'LandmarkMesh' found")
        
        try:
            with open(self.filepath, 'r') as f:
                config = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load JSON: {str(e)}")
            return {'CANCELLED'}
        
        if config.get("format") != "NeuroCaptain Probe Configuration":
            self.report({'ERROR'}, "Invalid JSON format")
            return {'CANCELLED'}
        
        headmesh = bpy.data.objects["headmesh"]
        optode_objects = {}
        
        optode_diameter, optode_thickness = self.get_head_scale(headmesh)
        
        bpy.ops.object.select_all(action='DESELECT')
        
        # Create optodes (same as original)
        for optode_data in config.get("optodes", []):
            name = optode_data["name"]
            optode_type = optode_data["type"]
            is_anchor = optode_data.get("is_anchor", False)
            
            if is_anchor and "anchor_info" in optode_data and landmark_mesh:
                anchor_info = optode_data["anchor_info"]
                                
                if anchor_info.get("method") == "barycentric" and "barycentric_coords" in anchor_info:
                    vertex_indices = anchor_info["vertex_indices"]
                    bary_coords = anchor_info["barycentric_coords"]
                    
                    try:
                        v0 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[vertex_indices[0]].co
                        v1 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[vertex_indices[1]].co
                        v2 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[vertex_indices[2]].co
                        
                        initial_pos = (bary_coords[0] * v0 + 
                                      bary_coords[1] * v1 + 
                                      bary_coords[2] * v2)
                    except (IndexError, KeyError) as e:
                        self.report({'WARNING'}, f"Failed barycentric for {name}: {e}")
                        initial_pos = Vector(optode_data["position"])
                
                elif "nearest_landmark" in anchor_info:
                    landmark_data = anchor_info["nearest_landmark"]
                    landmark_name = landmark_data["landmark"]
                    offset = Vector(landmark_data["offset"])
                    
                    try:
                        landmark_idx = landmark_labels.index(landmark_name)
                        landmark_vert = landmark_mesh.data.vertices[landmark_idx]
                        landmark_world = landmark_mesh.matrix_world @ landmark_vert.co
                        initial_pos = landmark_world + offset
                    except (ValueError, IndexError):
                        self.report({'WARNING'}, f"Landmark {landmark_name} not found for {name}")
                        initial_pos = Vector(optode_data["position"])
                else:
                    initial_pos = Vector(optode_data["position"])
            else:
                initial_pos = Vector(optode_data["position"])
            
            snapped_pos, surface_normal = self.snap_to_mesh_surface(initial_pos, headmesh)
            
            if optode_type == "source":
                sources_collection = get_or_create_collection("Sources")
                optode = self.create_optode_disc(snapped_pos, surface_normal, name, 
                                                 source_color, optode_diameter, optode_thickness)
                add_to_collection(optode, sources_collection)
            else:
                detectors_collection = get_or_create_collection("Detectors")
                optode = self.create_optode_disc(snapped_pos, surface_normal, name, 
                                                 detector_color, optode_diameter, optode_thickness)
                add_to_collection(optode, detectors_collection)    
            
            optode["is_anchor"] = 1 if is_anchor else 0
            
            if is_anchor:
                Anchors_collection = get_or_create_collection("Anchor Indicators")
                ring_mat = create_anchor_ring_material()
                bpy.ops.mesh.primitive_torus_add(location=(0, 0, 0), major_radius=2.5, minor_radius=0.3)
                indicator = context.active_object
                add_to_collection(indicator, Anchors_collection)   
                indicator.name = f"Anchor_{name}"
                indicator.parent = optode
                indicator.location = (0, 0, 0)
                indicator.hide_render = True
                indicator.data.materials.append(ring_mat)
                            
            constraint = optode.constraints.new(type='SHRINKWRAP')
            constraint.target = headmesh
            constraint.shrinkwrap_type = 'NEAREST_SURFACE'
            constraint.distance = 0.0
            constraint.influence = 1.0
            
            optode_objects[name] = optode
        
        # Create connection mesh
        mesh = bpy.data.meshes.new("Optode_Connections")
        conn_obj = bpy.data.objects.new("Optode_Connections", mesh)
        context.collection.objects.link(conn_obj)
        
        vertices = []
        edges = []
        spring_states = {}
        
        optode_names_ordered = list(optode_objects.keys())
        
        for optode_name in optode_names_ordered:
            optode = optode_objects[optode_name]
            vertices.append(optode.location.copy())
        
        optode_name_to_idx = {name: i for i, name in enumerate(optode_names_ordered)}
        
        for conn_data in config.get("connections", []):
            opt1_name = conn_data["optode1"]
            opt2_name = conn_data["optode2"]
            
            if opt1_name in optode_name_to_idx and opt2_name in optode_name_to_idx:
                idx1 = optode_name_to_idx[opt1_name]
                idx2 = optode_name_to_idx[opt2_name]
                edges.append((idx1, idx2))
                
                edge_key = f"{min(idx1, idx2)}_{max(idx1, idx2)}"
                spring_states[edge_key] = {
                    "optode1": opt1_name,
                    "optode2": opt2_name,
                    "rest_length": conn_data["rest_length"],
                    "pull": conn_data["pull"],
                    "push": conn_data["push"],
                    "is_flexible": conn_data["is_flexible"]
                }
        
        mesh.from_pydata(vertices, edges, [])
        mesh.update()
        
        conn_obj["spring_states"] = spring_states
        
        # Material
        mat = bpy.data.materials.get("Connection_Material")
        if not mat:
            mat = bpy.data.materials.new(name="Connection_Material")
            mat.use_nodes = True
            mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.8, 0.8, 1)
        if conn_obj.data.materials:
            conn_obj.data.materials[0] = mat
        else:
            conn_obj.data.materials.append(mat)
        
        bpy.ops.object.select_all(action='DESELECT')
        conn_obj.select_set(True)
        bpy.context.view_layer.objects.active = conn_obj
        
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # Create vertex groups for hooks
        for i, optode_name in enumerate(optode_names_ordered):
            vgroup = conn_obj.vertex_groups.new(name=f"VG_{optode_name}")
            vgroup.add([i], 1.0, 'REPLACE')
        
        # ===== HOOK CREATION =====
        print(f"\n=== Creating hooks for {len(optode_names_ordered)} optodes ===")
        
        bpy.ops.object.select_all(action='DESELECT')
        conn_obj.select_set(True)
        bpy.context.view_layer.objects.active = conn_obj
        
        for i, optode_name in enumerate(optode_names_ordered):
            optode = optode_objects[optode_name]
            
            bpy.ops.object.mode_set(mode='OBJECT')
            hook_mod = conn_obj.modifiers.new(name=f"Hook_{optode_name}", type='HOOK')
            hook_mod.object = optode
            hook_mod.vertex_group = f"VG_{optode_name}"
            
            for v in conn_obj.data.vertices:
                v.select = False
            conn_obj.data.vertices[i].select = True
            
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.object.hook_assign(modifier=hook_mod.name)
            bpy.ops.object.hook_reset(modifier=hook_mod.name)
            bpy.ops.object.hook_recenter(modifier=hook_mod.name)
            
            print(f"  Created and bound {hook_mod.name}")
        
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.update()
        print(f"=== Hook creation complete ===")
        
        # ===== GOAL-BASED SOFT BODY SETUP =====
        bpy.context.view_layer.objects.active = conn_obj
        
        soft_body_mod = conn_obj.modifiers.new(name="Softbody", type='SOFT_BODY')
        soft_body = soft_body_mod.settings
        
        goal_group = conn_obj.vertex_groups.new(name="Goals")
        
        # Track spring types per optode
        optode_spring_types = {}
        for optode_name in optode_names_ordered:
            optode_spring_types[optode_name] = {'stiff_count': 0, 'flex_count': 0}
        
        # Count stiff vs flexible connections for each optode
        for edge_key, spring_data in spring_states.items():
            opt1 = spring_data["optode1"]
            opt2 = spring_data["optode2"]
            
            is_flexible = spring_data.get("is_flexible", 1)
            
            if is_flexible:
                optode_spring_types[opt1]['flex_count'] += 1
                optode_spring_types[opt2]['flex_count'] += 1
            else:
                optode_spring_types[opt1]['stiff_count'] += 1
                optode_spring_types[opt2]['stiff_count'] += 1
        
        # Assign goal weights based on optode type and spring connections
        print(f"\n=== Assigning goal weights ===")
        for i, optode_name in enumerate(optode_names_ordered):
            optode = optode_objects[optode_name]
            
            if optode.get("is_anchor", 0):
                # Anchors: completely fixed (highest goal weight)
                goal_weight = 1.0
                optode["goal_type"] = "anchor"
            else:
                stiff_count = optode_spring_types[optode_name]['stiff_count']
                flex_count = optode_spring_types[optode_name]['flex_count']
                total_count = stiff_count + flex_count
                
                if total_count == 0:
                    goal_weight = 0.5
                    optode["goal_type"] = "isolated"
                elif stiff_count > 0 and flex_count == 0:
                    # Only stiff connections: high goal weight
                    goal_weight = 0.9
                    optode["goal_type"] = "stiff"
                elif flex_count > 0 and stiff_count == 0:
                    # Only flexible connections: low goal weight
                    goal_weight = 0.1
                    optode["goal_type"] = "flexible"
                else:
                    # Mixed: weighted based on ratio
                    stiff_ratio = stiff_count / total_count
                    goal_weight = 0.2 + (stiff_ratio * 0.6)
                    optode["goal_type"] = "mixed"
            
            goal_group.add([i], goal_weight, 'REPLACE')
            print(f"  {optode_name}: goal={goal_weight:.2f} ({optode['goal_type']}) - stiff:{optode_spring_types[optode_name]['stiff_count']}, flex:{optode_spring_types[optode_name]['flex_count']}")
        
        # Configure soft body goals
        soft_body.use_goal = True
        soft_body.goal_default = 0.0
        soft_body.vertex_group_goal = "Goals"
        soft_body.goal_spring = 0.95
        soft_body.goal_friction = 0.0
        
        # ===== IMPROVED SPRING STIFFNESS CALCULATION =====
        print(f"\n=== Calculating spring stiffness ===")
        
        # Calculate spring stiffness based on connection types
        stiff_pull = 0.0
        stiff_push = 0.0
        stiff_count = 0
        
        flex_pull = 0.0
        flex_push = 0.0
        flex_count = 0
        
        for edge_key, spring_data in spring_states.items():
            pull_val = max(0.0, min(0.99, spring_data.get("pull", 0.9)))
            push_val = max(0.0, min(0.99, spring_data.get("push", 0.9)))
            
            is_flexible = spring_data.get("is_flexible", 1)
            
            if is_flexible:
                flex_pull += pull_val
                flex_push += push_val
                flex_count += 1
            else:
                stiff_pull += pull_val
                stiff_push += push_val
                stiff_count += 1
        
        # Determine overall spring stiffness
        if stiff_count > 0 and flex_count > 0:
            # Mixed: use weighted average but ensure stiff springs dominate
            avg_stiff = stiff_pull / stiff_count
            avg_flex = flex_pull / flex_count
            
            # Boost stiff springs to be near-rigid
            avg_stiff = max(avg_stiff, 0.95)
            
            # Weight toward stiff
            total = stiff_count + flex_count
            soft_body.pull = (avg_stiff * stiff_count + avg_flex * flex_count) / total
            soft_body.push = soft_body.pull
            
            print(f"  Mixed springs: {stiff_count} stiff (avg={avg_stiff:.2f}), {flex_count} flexible (avg={avg_flex:.2f})")
            print(f"  Using weighted average: pull={soft_body.pull:.2f}")
        
        elif stiff_count > 0:
            # Only stiff springs: use high stiffness
            avg_stiff = stiff_pull / stiff_count
            soft_body.pull = max(avg_stiff, 0.95)  # Ensure high stiffness
            soft_body.push = soft_body.pull
            print(f"  All stiff springs: pull={soft_body.pull:.2f}")
        
        elif flex_count > 0:
            # Only flexible springs: use their values but boost if too low
            avg_flex = flex_pull / flex_count
            # For flexible-only systems, springs still need moderate stiffness
            soft_body.pull = max(avg_flex, 0.7)
            soft_body.push = soft_body.pull
            print(f"  All flexible springs: boosted from {avg_flex:.2f} to {soft_body.pull:.2f}")
        
        else:
            # Fallback
            soft_body.pull = 0.9
            soft_body.push = 0.9
            print(f"  No spring data, using default: pull=0.9")
        
        print(f"  Final spring stiffness: pull={soft_body.pull:.2f}, push={soft_body.push:.2f}")
        
        soft_body.use_edges = True
        soft_body.damping = 0.3
        soft_body.plastic = 0
        soft_body.bend = 0.5
        
        soft_body.use_edge_collision = False
        soft_body.use_face_collision = False
        soft_body.collision_type = 'MANUAL'
        
        soft_body.step_min = 10
        soft_body.step_max = 100
        soft_body.mass = 1.0
        soft_body.speed = 1.0
        
        # Summary
        anchor_count = sum(1 for o in optode_objects.values() if o.get("is_anchor", 0))
        stiff_count = sum(1 for o in optode_objects.values() if o.get("goal_type") == "stiff")
        flex_count = sum(1 for o in optode_objects.values() if o.get("goal_type") == "flexible")
        mixed_count = sum(1 for o in optode_objects.values() if o.get("goal_type") == "mixed")
        
        self.report({'INFO'}, 
            f"Imported {len(optode_objects)} optodes with goal-based springs: {anchor_count} anchors, {stiff_count} stiff, {flex_count} flexible, {mixed_count} mixed")
        
        return {'FINISHED'}
    
    def get_head_scale(self, head_mesh):
        """Calculate optode size based on head dimensions"""
        bbox_min = Vector([min(v.co[i] for v in head_mesh.data.vertices) for i in range(3)])
        bbox_max = Vector([max(v.co[i] for v in head_mesh.data.vertices) for i in range(3)])
        
        dimensions = (head_mesh.matrix_world @ bbox_max) - (head_mesh.matrix_world @ bbox_min)
        head_size = max(dimensions)
        
        return head_size * 0.02, head_size * 0.02 * 0.15
    
    def snap_to_mesh_surface(self, position, head_mesh):
        """Snap point to nearest surface location"""
        bvh = BVHTree.FromObject(head_mesh, bpy.context.evaluated_depsgraph_get())
        location, normal, index, distance = bvh.find_nearest(Vector(position))
        
        if location is None:
            return position, Vector((0, 0, 1))
        
        return Vector(location), Vector(normal)
    
    def create_optode_disc(self, position, normal, name, color, diameter, thickness):
        """Create optode disc at position"""
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32, radius=diameter/2.0, depth=thickness,
            enter_editmode=False, align='WORLD', location=(0, 0, 0)
        )
        
        optode = bpy.context.active_object
        optode.name = name
        
        z_axis = Vector((0, 0, 1))
        rotation_quat = z_axis.rotation_difference(Vector(normal))
        optode.rotation_euler = rotation_quat.to_euler()
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
    
    def get_head_scale(self, head_mesh):
        """Calculate optode size based on head dimensions"""
        bbox_min = Vector([min(v.co[i] for v in head_mesh.data.vertices) for i in range(3)])
        bbox_max = Vector([max(v.co[i] for v in head_mesh.data.vertices) for i in range(3)])
        
        dimensions = (head_mesh.matrix_world @ bbox_max) - (head_mesh.matrix_world @ bbox_min)
        head_size = max(dimensions)
        
        return head_size * 0.02, head_size * 0.02 * 0.15
    
    def snap_to_mesh_surface(self, position, head_mesh):
        """Snap point to nearest surface location"""
        bvh = BVHTree.FromObject(head_mesh, bpy.context.evaluated_depsgraph_get())
        location, normal, index, distance = bvh.find_nearest(Vector(position))
        
        if location is None:
            return position, Vector((0, 0, 1))
        
        return Vector(location), Vector(normal)
    
    def create_optode_disc(self, position, normal, name, color, diameter, thickness):
        """Create optode disc at position"""
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32, radius=diameter/2.0, depth=thickness,
            enter_editmode=False, align='WORLD', location=(0, 0, 0)
        )
        
        optode = bpy.context.active_object
        optode.name = name
        
        z_axis = Vector((0, 0, 1))
        rotation_quat = z_axis.rotation_difference(Vector(normal))
        optode.rotation_euler = rotation_quat.to_euler()
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

class NEUROCAPTAIN_OT_import_optode_json_goals_dialog(bpy.types.Operator):
    """Import optode probe with goal weight dialog"""
    bl_idname = "neurocaptain.import_optode_json_goals_dialog"
    bl_label = "Import Probe (Goals with Dialog)"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})
    
    flexible_goal_weight: bpy.props.FloatProperty(
        name="Flexible Optode Goal Weight",
        description="How strongly non-anchor optodes resist movement (0.0=free, 1.0=fixed)",
        default=0.0,
        min=0.0,
        max=1.0,
        step=1,
        precision=2
    )
    
    def invoke(self, context, event):
        if not self.filepath:
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}
        return context.window_manager.invoke_props_dialog(self, width=400)
    
    def draw(self, context):
        layout = self.layout
        layout.label(text="Set Goal Weight for Flexible Optodes:", icon='SETTINGS')
        layout.separator()
        
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Goal Weight Controls:")
        col.label(text="• 0.0 = Free to move (cloth-like, full propagation)")
        col.label(text="• 0.3 = Light resistance (some propagation)")
        col.label(text="• 0.5 = Moderate resistance (balanced)")
        col.label(text="• 0.8 = High resistance (minimal movement)")
        col.label(text="• 1.0 = Fixed (no movement)")
        
        layout.separator()
        layout.prop(self, "flexible_goal_weight", slider=True)
        layout.separator()
        layout.label(text="Note: Anchors always have goal weight 1.0 (fixed)", icon='INFO')
    
    def execute(self, context):
        import json
        from mathutils.bvhtree import BVHTree
        
        # Store the dialog value
        goal_weight = self.flexible_goal_weight
        
        # Turn off gravity
        bpy.context.scene.use_gravity = False
        
        source_color = (1.0, 0.0, 0.0, 1.0)
        detector_color = (0.0, 0.0, 0.0, 1.0)
        
        if "headmesh" not in bpy.data.objects:
            self.report({'ERROR'}, "No 'headmesh' found")
            return {'CANCELLED'}
        
        landmark_mesh = None
        landmark_labels = []
        
        if "LandmarkMesh" in bpy.data.objects:
            landmark_mesh = bpy.data.objects["LandmarkMesh"]
            num_landmarks = len(landmark_mesh.data.vertices)
            
            if num_landmarks >= 77:
                landmark_labels = [
                    "Nz", "Iz", "Lpa", "Rpa", "Cz",
                    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
                    "Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz",
                    "FT7", "F7", "AF7", "Fp1", "TP7", "P7", "PO7", "O1",
                    "FT8", "F8", "AF8", "Fp2", "TP8", "P8", "PO8", "O2",
                    "FC1", "FC3", "FC5", "FC2", "FC4", "FC6",
                    "F1", "F3", "F5", "F2", "F4", "F6", "AF3", "AF4",
                    "CP1", "CP3", "CP5", "CP2", "CP4", "CP6",
                    "P1", "P3", "P5", "P2", "P4", "P6", "PO3", "PO4",
                    "FT9", "F9", "", "", "TP9", "P9", "PO9", "O9",
                    "FT10", "F10", "", "", "TP10", "P10", "PO10", "O10"
                ]
            elif num_landmarks >= 27:
                landmark_labels = [
                    "Nz", "Iz", "Lpa", "Rpa", "Cz",
                    "T3", "C3", "Cz", "C4", "T4",
                    "Fpz", "Fz", "Cz", "Pz", "Oz",
                    "F7", "Fp1", "T5", "O1", "F8", "Fp2", "T6", "O2",
                    "F3", "F4", "P3", "P4"
                ]
        else:
            self.report({'WARNING'}, "No 'LandmarkMesh' found")
        
        try:
            with open(self.filepath, 'r') as f:
                config = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load JSON: {str(e)}")
            return {'CANCELLED'}
        
        if config.get("format") != "NeuroCaptain Probe Configuration":
            self.report({'ERROR'}, "Invalid JSON format")
            return {'CANCELLED'}
        
        headmesh = bpy.data.objects["headmesh"]
        optode_objects = {}
        
        optode_diameter, optode_thickness = self.get_head_scale(headmesh)
        
        bpy.ops.object.select_all(action='DESELECT')
        
        # Create optodes
        for optode_data in config.get("optodes", []):
            name = optode_data["name"]
            optode_type = optode_data["type"]
            is_anchor = optode_data.get("is_anchor", False)
            
            if is_anchor and "anchor_info" in optode_data and landmark_mesh:
                anchor_info = optode_data["anchor_info"]
                                
                if anchor_info.get("method") == "barycentric" and "barycentric_coords" in anchor_info:
                    vertex_indices = anchor_info["vertex_indices"]
                    bary_coords = anchor_info["barycentric_coords"]
                    
                    try:
                        v0 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[vertex_indices[0]].co
                        v1 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[vertex_indices[1]].co
                        v2 = landmark_mesh.matrix_world @ landmark_mesh.data.vertices[vertex_indices[2]].co
                        
                        initial_pos = (bary_coords[0] * v0 + 
                                      bary_coords[1] * v1 + 
                                      bary_coords[2] * v2)
                    except (IndexError, KeyError) as e:
                        self.report({'WARNING'}, f"Failed barycentric for {name}: {e}")
                        initial_pos = Vector(optode_data["position"])
                
                elif "nearest_landmark" in anchor_info:
                    landmark_data = anchor_info["nearest_landmark"]
                    landmark_name = landmark_data["landmark"]
                    offset = Vector(landmark_data["offset"])
                    
                    try:
                        landmark_idx = landmark_labels.index(landmark_name)
                        landmark_vert = landmark_mesh.data.vertices[landmark_idx]
                        landmark_world = landmark_mesh.matrix_world @ landmark_vert.co
                        initial_pos = landmark_world + offset
                    except (ValueError, IndexError):
                        self.report({'WARNING'}, f"Landmark {landmark_name} not found for {name}")
                        initial_pos = Vector(optode_data["position"])
                else:
                    initial_pos = Vector(optode_data["position"])
            else:
                initial_pos = Vector(optode_data["position"])
            
            snapped_pos, surface_normal = self.snap_to_mesh_surface(initial_pos, headmesh)
            
            if optode_type == "source":
                sources_collection = get_or_create_collection("Sources")
                optode = self.create_optode_disc(snapped_pos, surface_normal, name, 
                                                 source_color, optode_diameter, optode_thickness)
                add_to_collection(optode, sources_collection)
            else:
                detectors_collection = get_or_create_collection("Detectors")
                optode = self.create_optode_disc(snapped_pos, surface_normal, name, 
                                                 detector_color, optode_diameter, optode_thickness)
                add_to_collection(optode, detectors_collection)    
            
            optode["is_anchor"] = 1 if is_anchor else 0
            
            if is_anchor:
                Anchors_collection = get_or_create_collection("Anchor Indicators")
                ring_mat = create_anchor_ring_material()
                bpy.ops.mesh.primitive_torus_add(location=(0, 0, 0), major_radius=2.5, minor_radius=0.3)
                indicator = context.active_object
                add_to_collection(indicator, Anchors_collection)   
                indicator.name = f"Anchor_{name}"
                indicator.parent = optode
                indicator.location = (0, 0, 0)
                indicator.hide_render = True
                indicator.data.materials.append(ring_mat)
                            
            constraint = optode.constraints.new(type='SHRINKWRAP')
            constraint.target = headmesh
            constraint.shrinkwrap_type = 'NEAREST_SURFACE'
            constraint.distance = 0.0
            constraint.influence = 1.0
            
            optode_objects[name] = optode
        
        # Create connection mesh
        mesh = bpy.data.meshes.new("Optode_Connections")
        conn_obj = bpy.data.objects.new("Optode_Connections", mesh)
        context.collection.objects.link(conn_obj)
        
        vertices = []
        edges = []
        spring_states = {}
        
        optode_names_ordered = list(optode_objects.keys())
        
        for optode_name in optode_names_ordered:
            optode = optode_objects[optode_name]
            vertices.append(optode.location.copy())
        
        optode_name_to_idx = {name: i for i, name in enumerate(optode_names_ordered)}
        
        for conn_data in config.get("connections", []):
            opt1_name = conn_data["optode1"]
            opt2_name = conn_data["optode2"]
            
            if opt1_name in optode_name_to_idx and opt2_name in optode_name_to_idx:
                idx1 = optode_name_to_idx[opt1_name]
                idx2 = optode_name_to_idx[opt2_name]
                edges.append((idx1, idx2))
                
                edge_key = f"{min(idx1, idx2)}_{max(idx1, idx2)}"
                spring_states[edge_key] = {
                    "optode1": opt1_name,
                    "optode2": opt2_name,
                    "rest_length": conn_data["rest_length"],
                    "pull": conn_data["pull"],
                    "push": conn_data["push"],
                    "is_flexible": conn_data["is_flexible"]
                }
        
        mesh.from_pydata(vertices, edges, [])
        mesh.update()
        
        conn_obj["spring_states"] = spring_states
        
        # Material
        mat = bpy.data.materials.get("Connection_Material")
        if not mat:
            mat = bpy.data.materials.new(name="Connection_Material")
            mat.use_nodes = True
            mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.8, 0.8, 1)
        if conn_obj.data.materials:
            conn_obj.data.materials[0] = mat
        else:
            conn_obj.data.materials.append(mat)
        
        bpy.ops.object.select_all(action='DESELECT')
        conn_obj.select_set(True)
        bpy.context.view_layer.objects.active = conn_obj
        
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # Create vertex groups for hooks
        for i, optode_name in enumerate(optode_names_ordered):
            vgroup = conn_obj.vertex_groups.new(name=f"VG_{optode_name}")
            vgroup.add([i], 1.0, 'REPLACE')
        
        # Hook creation
        print(f"\n=== Creating hooks for {len(optode_names_ordered)} optodes ===")
        
        bpy.ops.object.select_all(action='DESELECT')
        conn_obj.select_set(True)
        bpy.context.view_layer.objects.active = conn_obj
        
        for i, optode_name in enumerate(optode_names_ordered):
            optode = optode_objects[optode_name]
            
            bpy.ops.object.mode_set(mode='OBJECT')
            hook_mod = conn_obj.modifiers.new(name=f"Hook_{optode_name}", type='HOOK')
            hook_mod.object = optode
            hook_mod.vertex_group = f"VG_{optode_name}"
            
            for v in conn_obj.data.vertices:
                v.select = False
            conn_obj.data.vertices[i].select = True
            
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.object.hook_assign(modifier=hook_mod.name)
            bpy.ops.object.hook_reset(modifier=hook_mod.name)
            bpy.ops.object.hook_recenter(modifier=hook_mod.name)
            
            print(f"  Created and bound {hook_mod.name}")
        
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.update()
        print(f"=== Hook creation complete ===")
        
        # Soft body setup
        bpy.context.view_layer.objects.active = conn_obj
        
        soft_body_mod = conn_obj.modifiers.new(name="Softbody", type='SOFT_BODY')
        soft_body = soft_body_mod.settings
        
        goal_group = conn_obj.vertex_groups.new(name="Goals")
        
        # Track spring types per optode
        optode_spring_types = {}
        for optode_name in optode_names_ordered:
            optode_spring_types[optode_name] = {'stiff_count': 0, 'flex_count': 0}
        
        # Count stiff vs flexible connections for each optode
        for edge_key, spring_data in spring_states.items():
            opt1 = spring_data["optode1"]
            opt2 = spring_data["optode2"]
            
            is_flexible = spring_data.get("is_flexible", 1)
            
            if is_flexible:
                optode_spring_types[opt1]['flex_count'] += 1
                optode_spring_types[opt2]['flex_count'] += 1
            else:
                optode_spring_types[opt1]['stiff_count'] += 1
                optode_spring_types[opt2]['stiff_count'] += 1
        
        # Assign goal weights - USE DIALOG VALUE
        print(f"\n=== Assigning goal weights (flexible={goal_weight:.2f}) ===")
        for i, optode_name in enumerate(optode_names_ordered):
            optode = optode_objects[optode_name]
            
            if optode.get("is_anchor", 0):
                # Anchors: completely fixed
                goal_group.add([i], 1.0, 'REPLACE')
                optode["goal_type"] = "anchor"
                print(f"  {optode_name}: ANCHOR (goal=1.0)")
            else:
                # Flexible: USE THE DIALOG VALUE
                goal_group.add([i], goal_weight, 'REPLACE')
                optode["goal_type"] = "flexible"
                print(f"  {optode_name}: FLEXIBLE (goal={goal_weight:.2f})")
        
        # Configure soft body goals
        soft_body.use_goal = True
        soft_body.goal_default = 0.0
        soft_body.vertex_group_goal = "Goals"
        soft_body.goal_spring = 0.95
        soft_body.goal_friction = 0.0
        
        # Calculate spring stiffness
        print(f"\n=== Calculating spring stiffness ===")
        
        stiff_pull = 0.0
        stiff_push = 0.0
        stiff_count = 0
        
        flex_pull = 0.0
        flex_push = 0.0
        flex_count = 0
        
        for edge_key, spring_data in spring_states.items():
            pull_val = max(0.0, min(0.99, spring_data.get("pull", 0.9)))
            push_val = max(0.0, min(0.99, spring_data.get("push", 0.9)))
            
            is_flexible = spring_data.get("is_flexible", 1)
            
            if is_flexible:
                flex_pull += pull_val
                flex_push += push_val
                flex_count += 1
            else:
                stiff_pull += pull_val
                stiff_push += push_val
                stiff_count += 1
        
        # Determine overall spring stiffness
        if stiff_count > 0 and flex_count > 0:
            avg_stiff = stiff_pull / stiff_count
            avg_flex = flex_pull / flex_count
            avg_stiff = max(avg_stiff, 0.95)
            total = stiff_count + flex_count
            soft_body.pull = (avg_stiff * stiff_count + avg_flex * flex_count) / total
            soft_body.push = soft_body.pull
            print(f"  Mixed springs: {stiff_count} stiff (avg={avg_stiff:.2f}), {flex_count} flexible (avg={avg_flex:.2f})")
            print(f"  Using weighted average: pull={soft_body.pull:.2f}")
        elif stiff_count > 0:
            avg_stiff = stiff_pull / stiff_count
            soft_body.pull = max(avg_stiff, 0.95)
            soft_body.push = soft_body.pull
            print(f"  All stiff springs: pull={soft_body.pull:.2f}")
        elif flex_count > 0:
            avg_flex = flex_pull / flex_count
            soft_body.pull = max(avg_flex, 0.7)
            soft_body.push = soft_body.pull
            print(f"  All flexible springs: boosted from {avg_flex:.2f} to {soft_body.pull:.2f}")
        else:
            soft_body.pull = 0.9
            soft_body.push = 0.9
            print(f"  No spring data, using default: pull=0.9")
        
        print(f"  Final spring stiffness: pull={soft_body.pull:.2f}, push={soft_body.push:.2f}")
        
        soft_body.use_edges = True
        soft_body.damping = 0.3
        soft_body.plastic = 0
        soft_body.bend = 0.5
        
        soft_body.use_edge_collision = False
        soft_body.use_face_collision = False
        soft_body.collision_type = 'MANUAL'
        
        soft_body.step_min = 10
        soft_body.step_max = 100
        soft_body.mass = 1.0
        soft_body.speed = 1.0
        
        # Summary
        anchor_count = sum(1 for o in optode_objects.values() if o.get("is_anchor", 0))
        stiff_count = sum(1 for o in optode_objects.values() if o.get("goal_type") == "stiff")
        flex_count = sum(1 for o in optode_objects.values() if o.get("goal_type") == "flexible")
        mixed_count = sum(1 for o in optode_objects.values() if o.get("goal_type") == "mixed")
        
        self.report({'INFO'}, 
            f"Imported with goal weight {goal_weight:.2f}: {anchor_count} anchors, {flex_count} flexible")
        
        return {'FINISHED'}
    
    def get_head_scale(self, head_mesh):
        """Calculate optode size based on head dimensions"""
        bbox_min = Vector([min(v.co[i] for v in head_mesh.data.vertices) for i in range(3)])
        bbox_max = Vector([max(v.co[i] for v in head_mesh.data.vertices) for i in range(3)])
        dimensions = (head_mesh.matrix_world @ bbox_max) - (head_mesh.matrix_world @ bbox_min)
        head_size = max(dimensions)
        return head_size * 0.02, head_size * 0.02 * 0.15
    
    def snap_to_mesh_surface(self, position, head_mesh):
        """Snap point to nearest surface location"""
        bvh = BVHTree.FromObject(head_mesh, bpy.context.evaluated_depsgraph_get())
        location, normal, index, distance = bvh.find_nearest(Vector(position))
        if location is None:
            return position, Vector((0, 0, 1))
        return Vector(location), Vector(normal)
    
    def create_optode_disc(self, position, normal, name, color, diameter, thickness):
        """Create optode disc at position"""
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32, radius=diameter/2.0, depth=thickness,
            enter_editmode=False, align='WORLD', location=(0, 0, 0)
        )
        optode = bpy.context.active_object
        optode.name = name
        z_axis = Vector((0, 0, 1))
        rotation_quat = z_axis.rotation_difference(Vector(normal))
        optode.rotation_euler = rotation_quat.to_euler()
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

class NEUROCAPTAIN_OT_set_all_flexible_goals(bpy.types.Operator):
    """Set goal weight for all non-anchor optodes"""
    bl_idname = "neurocaptain.set_all_flexible_goals"
    bl_label = "Set All Flexible Goal Weights"
    bl_options = {'REGISTER', 'UNDO'}
    
    goal_weight: bpy.props.FloatProperty(
        name="Goal Weight",
        description="Goal weight for all non-anchor optodes",
        default=0.0,
        min=0.0,
        max=1.0,
        step=1,
        precision=2
    )
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        layout.label(text="Set goal weight for all flexible optodes:")
        layout.prop(self, "goal_weight", slider=True)
    
    def execute(self, context):
        if "Optode_Connections" not in bpy.data.objects:
            self.report({'ERROR'}, "No Optode_Connections found")
            return {'CANCELLED'}
        
        conn_obj = bpy.data.objects["Optode_Connections"]
        
        if "Goals" not in conn_obj.vertex_groups:
            self.report({'ERROR'}, "No Goals vertex group found")
            return {'CANCELLED'}
        
        goals_vg = conn_obj.vertex_groups["Goals"]
        updated_count = 0
        
        # Find all flexible optodes and update their goal weights
        for mod in conn_obj.modifiers:
            if mod.type == 'HOOK' and mod.name.startswith("Hook_"):
                optode_name = mod.name.replace("Hook_", "")
                
                if optode_name in bpy.data.objects:
                    optode = bpy.data.objects[optode_name]
                    
                    # Skip anchors
                    if optode.get("is_anchor", 0):
                        continue
                    
                    # Update goal weight for this optode's vertex
                    if mod.vertex_group in conn_obj.vertex_groups:
                        vgroup = conn_obj.vertex_groups[mod.vertex_group]
                        for v in conn_obj.data.vertices:
                            try:
                                weight = vgroup.weight(v.index)
                                if weight > 0.5:
                                    goals_vg.add([v.index], self.goal_weight, 'REPLACE')
                                    updated_count += 1
                                    break
                            except:
                                pass
        
        context.view_layer.update()
        
        self.report({'INFO'}, f"Set goal weight to {self.goal_weight:.2f} for {updated_count} flexible optodes")
        return {'FINISHED'}


class NEUROCAPTAIN_OT_set_selected_goal_weights(bpy.types.Operator):
    """Set goal weight for selected optodes"""
    bl_idname = "neurocaptain.set_selected_goal_weights"
    bl_label = "Set Selected Goal Weights"
    bl_options = {'REGISTER', 'UNDO'}
    
    goal_weight: bpy.props.FloatProperty(
        name="Goal Weight",
        description="Goal weight for selected optodes",
        default=0.5,
        min=0.0,
        max=1.0,
        step=1,
        precision=2
    )
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        layout.label(text="Set goal weight for selected optodes:")
        layout.prop(self, "goal_weight", slider=True)
    
    def execute(self, context):
        if "Optode_Connections" not in bpy.data.objects:
            self.report({'ERROR'}, "No Optode_Connections found")
            return {'CANCELLED'}
        
        conn_obj = bpy.data.objects["Optode_Connections"]
        
        if "Goals" not in conn_obj.vertex_groups:
            self.report({'ERROR'}, "No Goals vertex group found")
            return {'CANCELLED'}
        
        # Get selected objects that are optodes
        selected_optodes = [obj for obj in context.selected_objects 
                           if obj.name.startswith("Source_") or obj.name.startswith("Detector_")]
        
        if not selected_optodes:
            self.report({'WARNING'}, "No optodes selected")
            return {'CANCELLED'}
        
        goals_vg = conn_obj.vertex_groups["Goals"]
        updated_count = 0
        
        # Update goal weight for each selected optode
        for optode in selected_optodes:
            # Find the hook for this optode
            hook_name = f"Hook_{optode.name}"
            
            if hook_name in conn_obj.modifiers:
                hook_mod = conn_obj.modifiers[hook_name]
                
                if hook_mod.vertex_group in conn_obj.vertex_groups:
                    vgroup = conn_obj.vertex_groups[hook_mod.vertex_group]
                    
                    for v in conn_obj.data.vertices:
                        try:
                            weight = vgroup.weight(v.index)
                            if weight > 0.5:
                                goals_vg.add([v.index], self.goal_weight, 'REPLACE')
                                updated_count += 1
                                print(f"Set {optode.name} goal weight to {self.goal_weight:.2f}")
                                break
                        except:
                            pass
        
        context.view_layer.update()
        
        self.report({'INFO'}, f"Set goal weight to {self.goal_weight:.2f} for {updated_count} optodes")
        return {'FINISHED'}

class NEUROCAPTAIN_OT_run_goal_spring_simulation(bpy.types.Operator):
    """Run soft body simulation using goal-based spring system (keeps hooks enabled)"""
    bl_idname = "neurocaptain.run_goal_spring_simulation"
    bl_label = "Run Goal Spring Simulation"
    bl_options = {'REGISTER', 'UNDO'}
    
    num_frames: bpy.props.IntProperty(
        name="Simulation Frames",
        description="Number of frames to simulate",
        default=100,
        min=10,
        max=500
    )
    
    def execute(self, context):
        from mathutils.bvhtree import BVHTree
        
        # Ensure gravity is off
        bpy.context.scene.use_gravity = False
        
        if "Optode_Connections" not in bpy.data.objects:
            self.report({'ERROR'}, "No Optode_Connections mesh found")
            return {'CANCELLED'}
        
        conn_obj = bpy.data.objects["Optode_Connections"]
        
        optode_objects = {}
        for obj in bpy.data.objects:
            if obj.name.startswith("Source_") or obj.name.startswith("Detector_"):
                optode_objects[obj.name] = obj
        
        if not optode_objects:
            self.report({'ERROR'}, "No optode objects found")
            return {'CANCELLED'}
        
        original_frame = context.scene.frame_current
        
        # NOTE: We keep hooks ENABLED for goal-based system
        self.report({'INFO'}, "Running goal-based simulation (hooks enabled)...")
        
        context.scene.frame_set(1)
        bpy.context.view_layer.objects.active = conn_obj
        conn_obj.select_set(True)
        
        soft_body_mod = None
        for mod in conn_obj.modifiers:
            if mod.type == 'SOFT_BODY':
                soft_body_mod = mod
                break
        
        if not soft_body_mod:
            self.report({'ERROR'}, "No soft body modifier found")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Running simulation for {self.num_frames} frames...")
        context.scene.frame_end = self.num_frames
        
        bpy.ops.ptcache.free_bake_all()
        bpy.ops.ptcache.bake_all(bake=True)
        
        context.scene.frame_set(self.num_frames)
        
        self.report({'INFO'}, "Applying relaxed positions to optodes...")

        depsgraph = context.evaluated_depsgraph_get()
        conn_obj_eval = conn_obj.evaluated_get(depsgraph)
        mesh_eval = conn_obj_eval.data

        # Map vertices to optodes and store ORIGINAL positions
        vertex_to_optode = {}
        original_positions = {}

        for mod in conn_obj.modifiers:
            if mod.type == 'HOOK' and mod.name.startswith("Hook_"):
                optode_name = mod.name.replace("Hook_", "")
                if mod.vertex_group and mod.vertex_group in conn_obj.vertex_groups:
                    vgroup = conn_obj.vertex_groups[mod.vertex_group]
                    for v in conn_obj.data.vertices:
                        try:
                            weight = vgroup.weight(v.index)
                            if weight > 0.5:
                                vertex_to_optode[v.index] = optode_name
                                if optode_name in optode_objects:
                                    original_positions[optode_name] = optode_objects[optode_name].location.copy()
                                break
                        except:
                            pass

        # Move optodes to match vertex positions and track changes
        moved_count = 0
        total_distance = 0.0
        max_distance = 0.0
        max_moved_optode = None

        for v_idx, optode_name in vertex_to_optode.items():
            if optode_name in optode_objects:
                if v_idx < len(mesh_eval.vertices):
                    optode = optode_objects[optode_name]
                    original_pos = original_positions.get(optode_name)
                    final_pos = mesh_eval.vertices[v_idx].co.copy()
                    
                    if original_pos:
                        distance = (final_pos - original_pos).length
                        total_distance += distance
                        
                        if distance > max_distance:
                            max_distance = distance
                            max_moved_optode = optode_name
                        
                        # Only count as "moved" if distance is significant
                        if distance > 0.001:  # 0.001 units threshold
                            print(f"  {optode_name}: moved {distance:.4f} units (goal_type: {optode.get('goal_type', 'unknown')})")
                            moved_count += 1
                    
                    optode.location = final_pos

        if moved_count > 0:
            avg_distance = total_distance / moved_count
            self.report({'INFO'}, 
                f"Moved {moved_count} optodes | Avg: {avg_distance:.4f} units | Max: {max_distance:.4f} units ({max_moved_optode})")
        else:
            self.report({'WARNING'}, "No significant optode movement detected - simulation may not be working")

        # Reset simulation
        context.scene.frame_set(1)
        bpy.ops.ptcache.free_bake_all()

        # Update base mesh to match optode positions
        for v_idx, optode_name in vertex_to_optode.items():
            if optode_name in optode_objects:
                conn_obj.data.vertices[v_idx].co = optode_objects[optode_name].location.copy()

        conn_obj.data.update()

        # Snap to head surface
        self.report({'INFO'}, "Snapping optodes to head surface...")

        headmesh = bpy.data.objects.get("headmesh")
        if headmesh:
            bvh = BVHTree.FromObject(headmesh, context.evaluated_depsgraph_get())
            
            for optode_name, optode in optode_objects.items():
                location, normal, index, distance = bvh.find_nearest(optode.location)
                if location:
                    optode.location = location
                    
                    z_axis = Vector((0, 0, 1))
                    rotation_quat = z_axis.rotation_difference(normal)
                    optode.rotation_euler = rotation_quat.to_euler()

        context.view_layer.update()
        context.scene.frame_set(original_frame)

        self.report({'INFO'}, f"Goal-based simulation complete!")
        return {'FINISHED'}

class NEUROCAPTAIN_OT_debug_goal_system(bpy.types.Operator):
    """Debug goal weights and spring connections"""
    bl_idname = "neurocaptain.debug_goal_system"
    bl_label = "Debug Goal System"
    
    def execute(self, context):
        if "Optode_Connections" not in bpy.data.objects:
            self.report({'ERROR'}, "No Optode_Connections mesh found")
            return {'CANCELLED'}
        
        conn_obj = bpy.data.objects["Optode_Connections"]
        
        print("\n" + "="*60)
        print("GOAL SYSTEM DEBUG")
        print("="*60)
        
        # Check goal vertex group
        if "Goals" in conn_obj.vertex_groups:
            goals_vg = conn_obj.vertex_groups["Goals"]
            print(f"\nGoal weights:")
            
            for v in conn_obj.data.vertices:
                try:
                    weight = goals_vg.weight(v.index)
                    if weight > 0:
                        # Find optode name
                        optode_name = None
                        for mod in conn_obj.modifiers:
                            if mod.type == 'HOOK' and mod.name.startswith("Hook_"):
                                if mod.vertex_group in conn_obj.vertex_groups:
                                    vg = conn_obj.vertex_groups[mod.vertex_group]
                                    try:
                                        if vg.weight(v.index) > 0.5:
                                            optode_name = mod.name.replace("Hook_", "")
                                            break
                                    except:
                                        pass
                        
                        if optode_name:
                            optode = bpy.data.objects.get(optode_name)
                            goal_type = optode.get("goal_type", "unknown") if optode else "unknown"
                            print(f"  Vertex {v.index} ({optode_name}): weight={weight:.2f}, type={goal_type}")
                except:
                    pass
        else:
            print("\nWARNING: No 'Goals' vertex group found!")
        
        # Check soft body settings
        soft_body_mod = None
        for mod in conn_obj.modifiers:
            if mod.type == 'SOFT_BODY':
                soft_body_mod = mod
                break
        
        if soft_body_mod:
            sb = soft_body_mod.settings
            print(f"\nSoft Body Settings:")
            print(f"  use_goal: {sb.use_goal}")
            print(f"  goal_default: {sb.goal_default}")
            print(f"  goal_spring: {sb.goal_spring}")
            print(f"  vertex_group_goal: {sb.vertex_group_goal}")
            print(f"  use_edges: {sb.use_edges}")
            print(f"  pull: {sb.pull}")
            print(f"  push: {sb.push}")
            print(f"  damping: {sb.damping}")
        else:
            print("\nWARNING: No soft body modifier found!")
        
        # Check spring states
        spring_states = conn_obj.get("spring_states", {})
        if spring_states:
            stiff_count = sum(1 for s in spring_states.values() if not s.get("is_flexible", 1))
            flex_count = sum(1 for s in spring_states.values() if s.get("is_flexible", 1))
            print(f"\nSpring connections:")
            print(f"  Total: {len(spring_states)}")
            print(f"  Stiff: {stiff_count}")
            print(f"  Flexible: {flex_count}")
        else:
            print("\nWARNING: No spring_states found!")
        
        print("="*60 + "\n")
        
        self.report({'INFO'}, "Debug info printed to console (Window > Toggle System Console)")
        return {'FINISHED'}

class NEUROCAPTAIN_OT_run_spring_relaxation(bpy.types.Operator):
    """Run soft body simulation to relax spring network"""
    bl_idname = "neurocaptain.run_spring_relaxation"
    bl_label = "Run Spring Relaxation"
    bl_options = {'REGISTER', 'UNDO'}
    
    num_frames: bpy.props.IntProperty(
        name="Simulation Frames",
        description="Number of frames to simulate",
        default=100,
        min=10,
        max=500
    )
    
    def execute(self, context):
        from mathutils.bvhtree import BVHTree
        
        bpy.context.scene.use_gravity = False
        
        if "Optode_Connections" not in bpy.data.objects:
            self.report({'ERROR'}, "No Optode_Connections mesh found")
            return {'CANCELLED'}
        
        conn_obj = bpy.data.objects["Optode_Connections"]
        
        optode_objects = {}
        for obj in bpy.data.objects:
            if obj.name.startswith("Source_") or obj.name.startswith("Detector_"):
                optode_objects[obj.name] = obj
        
        if not optode_objects:
            self.report({'ERROR'}, "No optode objects found")
            return {'CANCELLED'}
        
        original_frame = context.scene.frame_current
        
        self.report({'INFO'}, "Disabling hooks...")
        disabled_hooks = []
        for mod in conn_obj.modifiers:
            if mod.type == 'HOOK' and mod.show_viewport:
                mod.show_viewport = False
                disabled_hooks.append(mod.name)
        
        context.scene.frame_set(1)
        bpy.context.view_layer.objects.active = conn_obj
        conn_obj.select_set(True)
        
        soft_body_mod = None
        for mod in conn_obj.modifiers:
            if mod.type == 'SOFT_BODY':
                soft_body_mod = mod
                break
        
        if not soft_body_mod:
            self.report({'ERROR'}, "No soft body modifier found")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Running simulation for {self.num_frames} frames...")
        context.scene.frame_end = self.num_frames
        
        bpy.ops.ptcache.free_bake_all()
        bpy.ops.ptcache.bake_all(bake=True)
        
        context.scene.frame_set(self.num_frames)
        
        self.report({'INFO'}, "Applying relaxed positions to optodes...")
        
        depsgraph = context.evaluated_depsgraph_get()
        conn_obj_eval = conn_obj.evaluated_get(depsgraph)
        mesh_eval = conn_obj_eval.data
        
        vertex_to_optode = {}
        for mod in conn_obj.modifiers:
            if mod.type == 'HOOK' and mod.name.startswith("Hook_"):
                optode_name = mod.name.replace("Hook_", "")
                if mod.vertex_group and mod.vertex_group in conn_obj.vertex_groups:
                    vgroup = conn_obj.vertex_groups[mod.vertex_group]
                    for v in conn_obj.data.vertices:
                        try:
                            weight = vgroup.weight(v.index)
                            if weight > 0.5:
                                vertex_to_optode[v.index] = optode_name
                                break
                        except:
                            pass
        
        moved_count = 0
        for v_idx, optode_name in vertex_to_optode.items():
            if optode_name in optode_objects:
                if v_idx < len(mesh_eval.vertices):
                    final_pos = mesh_eval.vertices[v_idx].co.copy()
                    optode_objects[optode_name].location = final_pos
                    moved_count += 1
        
        self.report({'INFO'}, f"Updated {moved_count} optode positions")
        
        self.report({'INFO'}, "Re-enabling hooks...")
        for mod_name in disabled_hooks:
            if mod_name in conn_obj.modifiers:
                conn_obj.modifiers[mod_name].show_viewport = True
        
        context.scene.frame_set(1)
        bpy.ops.ptcache.free_bake_all()
        
        for v_idx, optode_name in vertex_to_optode.items():
            if optode_name in optode_objects:
                conn_obj.data.vertices[v_idx].co = optode_objects[optode_name].location.copy()
        
        conn_obj.data.update()
        
        self.report({'INFO'}, "Snapping optodes to head surface...")
        
        headmesh = bpy.data.objects.get("headmesh")
        if headmesh:
            bvh = BVHTree.FromObject(headmesh, context.evaluated_depsgraph_get())
            
            for optode_name, optode in optode_objects.items():
                location, normal, index, distance = bvh.find_nearest(optode.location)
                if location:
                    optode.location = location
                    
                    z_axis = Vector((0, 0, 1))
                    rotation_quat = z_axis.rotation_difference(normal)
                    optode.rotation_euler = rotation_quat.to_euler()
        
        context.view_layer.update()
        context.scene.frame_set(original_frame)
        
        self.report({'INFO'}, f"Spring relaxation complete! {moved_count} optodes repositioned")
        
        return {'FINISHED'}

class NEUROCAPTAIN_OT_toggle_dynamic_spring_mode(bpy.types.Operator):
    """Toggle dynamic spring editing mode - move optodes and see real-time spring response"""
    bl_idname = "neurocaptain.toggle_dynamic_spring_mode"
    bl_label = "Toggle Dynamic Spring Mode"
    bl_options = {'REGISTER', 'UNDO'}
    
    _handler = None
    
    @staticmethod
    def update_vertices_from_optodes(scene):
        """Frame change handler - updates vertex goal positions to follow optodes"""
        if "Optode_Connections" not in bpy.data.objects:
            return
        
        conn_obj = bpy.data.objects["Optode_Connections"]
        
        # Only run if dynamic mode is active
        if not conn_obj.get("dynamic_spring_mode", False):
            return
        
        # Update vertex positions to match optode positions (creates goal targets)
        for mod in conn_obj.modifiers:
            if mod.type == 'HOOK' and mod.name.startswith("Hook_"):
                optode_name = mod.name.replace("Hook_", "")
                
                if optode_name in bpy.data.objects and mod.vertex_group:
                    optode = bpy.data.objects[optode_name]
                    
                    # Find the vertex index for this optode
                    if mod.vertex_group in conn_obj.vertex_groups:
                        vgroup = conn_obj.vertex_groups[mod.vertex_group]
                        for v in conn_obj.data.vertices:
                            try:
                                weight = vgroup.weight(v.index)
                                if weight > 0.5:
                                    # Update base mesh vertex to optode position
                                    # This becomes the goal position for soft body
                                    conn_obj.data.vertices[v.index].co = optode.location.copy()
                                    break
                            except:
                                pass
        
        # Mark mesh as updated
        conn_obj.data.update()
    
    def execute(self, context):
        if "Optode_Connections" not in bpy.data.objects:
            self.report({'ERROR'}, "No Optode_Connections mesh found")
            return {'CANCELLED'}
        
        conn_obj = bpy.data.objects["Optode_Connections"]
        
        # Check current mode
        is_active = conn_obj.get("dynamic_spring_mode", False)
        
        if not is_active:
            # ===== ENTER DYNAMIC SPRING MODE =====
            self.report({'INFO'}, "Entering Dynamic Spring Mode...")
            
            # Step 1: Update vertex positions to match current optode positions (via hooks)
            context.view_layer.update()
            depsgraph = context.evaluated_depsgraph_get()
            conn_obj_eval = conn_obj.evaluated_get(depsgraph)
            
            # Copy evaluated (hooked) positions to base mesh
            for i, vert in enumerate(conn_obj.data.vertices):
                vert.co = conn_obj_eval.data.vertices[i].co.copy()
            
            conn_obj.data.update()
            
            # Step 2: Adjust soft body settings for dynamic mode
            for mod in conn_obj.modifiers:
                if mod.type == 'SOFT_BODY':
                    soft_body = mod.settings
                    
                    # Store original values
                    conn_obj["original_pull"] = soft_body.pull
                    conn_obj["original_push"] = soft_body.push
                    conn_obj["original_damping"] = soft_body.damping
                    conn_obj["original_goal_spring"] = soft_body.goal_spring
                    
                    # Adjust for better interactive response
                    # Springs should be moderate (not too stiff or too loose)
                    if soft_body.pull < 0.5:
                        soft_body.pull = 0.7
                        soft_body.push = 0.7
                    
                    # CRITICAL: Increase damping for stability
                    soft_body.damping = 0.8  # Much higher damping prevents jiggling
                    
                    # Goals should follow optodes but not too tightly
                    soft_body.goal_spring = 0.6  # Lower = smoother following
                    soft_body.goal_friction = 0.2  # Add friction to reduce oscillation
                    
                    print(f"Dynamic mode settings: pull={soft_body.pull:.2f}, damping={soft_body.damping:.2f}, goal_spring={soft_body.goal_spring:.2f}")
            
            # Step 3: Register frame change handler
            if NEUROCAPTAIN_OT_toggle_dynamic_spring_mode.update_vertices_from_optodes not in bpy.app.handlers.frame_change_post:
                bpy.app.handlers.frame_change_post.append(
                    NEUROCAPTAIN_OT_toggle_dynamic_spring_mode.update_vertices_from_optodes
                )
            
            # Step 4: Start animation playback
            if not context.screen.is_animation_playing:
                bpy.ops.screen.animation_play()
            
            # Mark as active
            conn_obj["dynamic_spring_mode"] = True
            
            self.report({'INFO'}, "Dynamic Spring Mode ON - Move optodes with G. Optodes stay connected to vertices.")
            
        else:
            # ===== EXIT DYNAMIC SPRING MODE =====
            self.report({'INFO'}, "Exiting Dynamic Spring Mode...")
            
            # Step 1: Stop animation FIRST
            if context.screen.is_animation_playing:
                bpy.ops.screen.animation_cancel()
            
            # Step 2: Remove frame change handler IMMEDIATELY
            # This prevents any more vertex updates
            if NEUROCAPTAIN_OT_toggle_dynamic_spring_mode.update_vertices_from_optodes in bpy.app.handlers.frame_change_post:
                bpy.app.handlers.frame_change_post.remove(
                    NEUROCAPTAIN_OT_toggle_dynamic_spring_mode.update_vertices_from_optodes
                )
            
            # Step 3: IMPORTANT - Don't move optodes to vertex positions
            # Instead, update base mesh vertices to match optode positions
            # This preserves where you manually moved the optodes
            
            context.view_layer.update()
            depsgraph = context.evaluated_depsgraph_get()
            conn_obj_eval = conn_obj.evaluated_get(depsgraph)
            
            # Update base mesh to match current evaluated (hooked) positions
            for i, vert in enumerate(conn_obj.data.vertices):
                if i < len(conn_obj_eval.data.vertices):
                    vert.co = conn_obj_eval.data.vertices[i].co.copy()
            
            conn_obj.data.update()
            
            # Step 4: Restore original soft body settings
            for mod in conn_obj.modifiers:
                if mod.type == 'SOFT_BODY':
                    soft_body = mod.settings
                    if "original_pull" in conn_obj:
                        soft_body.pull = conn_obj["original_pull"]
                        soft_body.push = conn_obj["original_push"]
                        soft_body.damping = conn_obj["original_damping"]
                        soft_body.goal_spring = conn_obj["original_goal_spring"]
                        soft_body.goal_friction = 0.0
                        
                        # Clean up stored values
                        del conn_obj["original_pull"]
                        del conn_obj["original_push"]
                        del conn_obj["original_damping"]
                        del conn_obj["original_goal_spring"]
            
            # Step 5: Restore goal weights based on optode types
            if "Goals" in conn_obj.vertex_groups:
                goals_vg = conn_obj.vertex_groups["Goals"]
                
                for mod in conn_obj.modifiers:
                    if mod.type == 'HOOK' and mod.name.startswith("Hook_"):
                        optode_name = mod.name.replace("Hook_", "")
                        if optode_name in bpy.data.objects:
                            optode = bpy.data.objects[optode_name]
                            if mod.vertex_group in conn_obj.vertex_groups:
                                vgroup = conn_obj.vertex_groups[mod.vertex_group]
                                for v in conn_obj.data.vertices:
                                    try:
                                        weight = vgroup.weight(v.index)
                                        if weight > 0.5:
                                            # Restore goal weight based on optode type
                                            if optode.get("is_anchor", 0):
                                                goals_vg.add([v.index], 1.0, 'REPLACE')
                                            elif optode.get("goal_type") == "stiff":
                                                goals_vg.add([v.index], 0.9, 'REPLACE')
                                            elif optode.get("goal_type") == "flexible":
                                                goals_vg.add([v.index], 0.1, 'REPLACE')
                                            elif optode.get("goal_type") == "mixed":
                                                # Calculate proper mixed weight
                                                goals_vg.add([v.index], 0.5, 'REPLACE')
                                            else:
                                                goals_vg.add([v.index], 0.4, 'REPLACE')
                                            break
                                    except:
                                        pass
            
            # Step 6: Snap optodes to head surface
            headmesh = bpy.data.objects.get("headmesh")
            if headmesh:
                from mathutils.bvhtree import BVHTree
                bvh = BVHTree.FromObject(headmesh, context.evaluated_depsgraph_get())
                
                snapped_count = 0
                for obj_name, obj in bpy.data.objects.items():
                    if obj_name.startswith("Source_") or obj_name.startswith("Detector_"):
                        original_pos = obj.location.copy()
                        location, normal, index, distance = bvh.find_nearest(obj.location)
                        
                        if location:
                            # Only snap if reasonably close (not jumping across head)
                            snap_distance = (location - original_pos).length
                            if snap_distance < 20.0:  # Adjust threshold as needed
                                obj.location = location
                                z_axis = Vector((0, 0, 1))
                                rotation_quat = z_axis.rotation_difference(normal)
                                obj.rotation_euler = rotation_quat.to_euler()
                                snapped_count += 1
                
                if snapped_count > 0:
                    self.report({'INFO'}, f"Snapped {snapped_count} optodes to head surface")
            
            # Step 7: Final update of base mesh to match optode positions after snapping
            context.view_layer.update()
            depsgraph = context.evaluated_depsgraph_get()
            conn_obj_eval = conn_obj.evaluated_get(depsgraph)
            
            for i, vert in enumerate(conn_obj.data.vertices):
                if i < len(conn_obj_eval.data.vertices):
                    vert.co = conn_obj_eval.data.vertices[i].co.copy()
            
            conn_obj.data.update()
            context.view_layer.update()
            
            # Mark as inactive
            conn_obj["dynamic_spring_mode"] = False
            
            self.report({'INFO'}, "Dynamic Spring Mode OFF - Optode positions preserved")
        
        return {'FINISHED'}
    
class NEUROCAPTAIN_OT_define_anchor_optode(bpy.types.Operator):
    """Define selected optode(s) as anchor points"""
    bl_idname = "neurocaptain.define_anchor_optode"
    bl_label = "Define as Anchor"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        selected_optodes = [obj for obj in context.selected_objects 
                           if obj.name.startswith("Source_") or obj.name.startswith("Detector_")]
        
        if not selected_optodes:
            self.report({'ERROR'}, "No optodes selected")
            return {'CANCELLED'}
        
        ring_mat = create_anchor_ring_material()
        
        for optode in selected_optodes:
            if optode.get("is_anchor", False):
                continue
            
            optode["is_anchor"] = True
            
            bbox = [Vector(corner) for corner in optode.bound_box]
            min_x = min(v.x for v in bbox)
            max_x = max(v.x for v in bbox)
            optode_radius = (max_x - min_x) / 2
            
            ring_mesh = create_torus_mesh(
                name=f"Anchor_{optode.name}",
                major_radius=optode_radius * 1.4,
                minor_radius=0.4,
                major_segments=24,
                minor_segments=8
            )
            Anchors_collection = get_or_create_collection("Anchor Indicators")
            ring_obj = bpy.context.active_object
            ring_obj.name = f"Anchor_{optode.name}"
            ring_obj.data.materials.append(ring_mat)
            add_to_collection(ring_obj, Anchors_collection)
            
            ring_obj.parent = optode
            ring_obj.location = (0, 0, 0)
            ring_obj.rotation_euler = (0, 0, 0)
            
            ring_obj["is_anchor_ring"] = True
            ring_obj["parent_optode"] = optode.name
            
            ring_obj.hide_select = True
        
        self.report({'INFO'}, f"Defined {len(selected_optodes)} optode(s) as anchors")
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
    bpy.utils.register_class(NEUROCAPTAIN_OT_run_spring_relaxation)
    bpy.utils.register_class(NEUROCAPTAIN_OT_toggle_dynamic_spring_mode)
    bpy.utils.register_class(NEUROCAPTAIN_OT_define_anchor_optode)
    bpy.utils.register_class(NEUROCAPTAIN_OT_undefine_anchor_optode)
    bpy.utils.register_class(NEUROCAPTAIN_OT_run_goal_spring_simulation)
    bpy.utils.register_class(NEUROCAPTAIN_OT_import_optode_json_goals)
    bpy.utils.register_class(NEUROCAPTAIN_OT_debug_goal_system)
    bpy.utils.register_class(NEUROCAPTAIN_OT_import_optode_json_goals_dialog)
    bpy.utils.register_class(NEUROCAPTAIN_OT_set_all_flexible_goals)
    bpy.utils.register_class(NEUROCAPTAIN_OT_set_selected_goal_weights)


def unregister():
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_undefine_anchor_optode)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_define_anchor_optode)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_run_spring_relaxation)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_toggle_dynamic_spring_mode)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_import_optode_json)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_export_optode_json)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_toggle_connection_visibility)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_update_optode_connections)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_make_spring_stiff)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_modify_spring_properties)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_create_optode_connections_delaunay)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_create_optode_connections)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_import_optode_json_goals)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_run_goal_spring_simulation)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_debug_goal_system)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_import_optode_json_goals_dialog)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_set_all_flexible_goals)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_set_selected_goal_weights)



if __name__ == "__main__":
    register()