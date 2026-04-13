import bpy
import numpy as np
from mathutils import Vector
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
    
def run_spring_relaxation(conn_obj, optode_objects, iterations=200, timestep=0.05, 
                          damping=0.85, stiff_strength=0.95, flexible_strength=0.3,
                          use_surface_projection=True, projection_strength=0.5,
                          report_callback=None):
    """
    Core spring relaxation function that can be called from multiple operators
    
    Returns: dictionary with displacement statistics
    """
    
    if "optode_names_ordered" not in conn_obj:
        if report_callback:
            report_callback({'ERROR'}, "Missing optode data")
        return None
    
    optode_names = list(conn_obj["optode_names_ordered"])
    spring_states = dict(conn_obj.get("spring_states", {}))
    
    print(f"\n{'='*70}")
    print(f"CUSTOM SPRING RELAXATION - Starting")
    print(f"{'='*70}")
    print(f"Optodes: {len(optode_names)}")
    print(f"Connections: {len(conn_obj.data.edges)}")
    
    # ===== INITIALIZE SIMULATION DATA =====
    n_verts = len(optode_names)
    
    positions = np.zeros((n_verts, 3), dtype=np.float64)
    velocities = np.zeros((n_verts, 3), dtype=np.float64)
    is_anchor = np.zeros(n_verts, dtype=bool)
    initial_positions = np.zeros((n_verts, 3), dtype=np.float64)  # For displacement tracking
    
    # Load initial data
    for i, name in enumerate(optode_names):
        if name not in optode_objects:
            if report_callback:
                report_callback({'ERROR'}, f"Optode '{name}' not found")
            return None
        
        optode = optode_objects[name]
        positions[i] = np.array(optode.location)
        initial_positions[i] = np.array(optode.location)  # Store initial position
        is_anchor[i] = (optode.get("is_anchor", 0) == 1)
    
    print(f"Anchored optodes: {is_anchor.sum()}")
    print(f"Free optodes: {(~is_anchor).sum()}")
    
    # ===== BUILD SPRING LIST =====
    springs = []
    stiff_count = 0
    flexible_count = 0
    
    print("\nBuilding spring list from edges...")
    
    for edge in conn_obj.data.edges:
        idx1, idx2 = edge.vertices[:]
        
        # Get optode names for these indices
        if idx1 >= len(optode_names) or idx2 >= len(optode_names):
            print(f"WARNING: Edge has invalid vertex indices: {idx1}, {idx2}")
            continue
        
        optode1_name = optode_names[idx1]
        optode2_name = optode_names[idx2]
        
        # Build key using optode names (matching how spring_states is stored)
        optode_pair = sorted([optode1_name, optode2_name])
        edge_key = f"{optode_pair[0]}_{optode_pair[1]}"
        
        if edge_key in spring_states:
            spring_data = spring_states[edge_key]
            
            rest_length = float(spring_data.get("rest_length", 1.0))
            is_flexible = spring_data.get("is_flexible", False)
            
            if is_flexible:
                stiffness = flexible_strength
                flexible_count += 1
            else:
                stiffness = stiff_strength
                stiff_count += 1
            
            springs.append({
                'i': idx1,
                'j': idx2,
                'rest_length': rest_length,
                'stiffness': stiffness,
                'is_flexible': is_flexible
            })
        else:
            print(f"WARNING: No spring_states entry for edge {optode1_name} - {optode2_name}")
            print(f"  Looking for key: '{edge_key}'")
            if len(spring_states) > 0:
                print(f"  Available keys sample: {list(spring_states.keys())[:3]}...")
    
    print(f"Springs - Stiff: {stiff_count}, Flexible: {flexible_count}")
    
    if len(springs) == 0:
        print("ERROR: No springs were created! Check spring_states keys.")
        if report_callback:
            report_callback({'ERROR'}, "No springs found - check spring_states data")
        return None
    
    # ===== SETUP SURFACE PROJECTION =====
    bvh = None
    headmesh = None
    if use_surface_projection and "headmesh" in bpy.data.objects:
        headmesh = bpy.data.objects["headmesh"]
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bvh = BVHTree.FromObject(headmesh, depsgraph)
        print(f"Surface projection: ENABLED (strength={projection_strength})")
    else:
        print(f"Surface projection: DISABLED")
    
    print(f"\n{'='*70}")
    print(f"SIMULATION PARAMETERS")
    print(f"{'='*70}")
    print(f"Iterations: {iterations}")
    print(f"Timestep: {timestep}")
    print(f"Damping: {damping}")
    print(f"Stiff spring strength: {stiff_strength}")
    print(f"Flexible spring strength: {flexible_strength}")
    print(f"\n{'='*70}")
    print(f"RUNNING SIMULATION")
    print(f"{'='*70}\n")
    
    # ===== MAIN SIMULATION LOOP =====
    for iteration in range(iterations):
        forces = np.zeros((n_verts, 3), dtype=np.float64)
        
        # Calculate spring forces
        for spring in springs:
            i, j = spring['i'], spring['j']
            
            delta = positions[j] - positions[i]
            current_length = np.linalg.norm(delta)
            
            if current_length < 1e-8:
                continue
            
            direction = delta / current_length
            extension = current_length - spring['rest_length']
            force_magnitude = spring['stiffness'] * extension
            force = force_magnitude * direction
            
            forces[i] += force
            forces[j] -= force
        
        # Update velocities
        velocities += forces * timestep
        velocities *= damping
        
        # Update positions (only non-anchors)
        positions[~is_anchor] += velocities[~is_anchor] * timestep
        
        # Surface projection
        if bvh and use_surface_projection:
            for i in range(n_verts):
                if not is_anchor[i]:
                    pos_vec = Vector(positions[i])
                    location, normal, index, distance = bvh.find_nearest(pos_vec)
                    
                    if location is not None:
                        surface_pos = np.array(location)
                        positions[i] = (positions[i] * (1.0 - projection_strength) + 
                                      surface_pos * projection_strength)
        
        # Progress reporting
        if iteration % 50 == 0 or iteration == iterations - 1:
            max_force = np.max(np.linalg.norm(forces, axis=1))
            max_velocity = np.max(np.linalg.norm(velocities, axis=1))
            print(f"  Iteration {iteration:4d}/{iterations}: "
                  f"max_force={max_force:.6f}, max_vel={max_velocity:.6f}")
    
    print(f"\n{'='*70}")
    print(f"CALCULATING DISPLACEMENT STATISTICS")
    print(f"{'='*70}")
    
    # ===== CALCULATE DISPLACEMENTS =====
    displacements = np.linalg.norm(positions - initial_positions, axis=1)
    
    # Categorize vertices by their edge types
    anchor_indices = []
    stiff_edge_indices = []
    flexible_edge_indices = []
    
    for i, name in enumerate(optode_names):
        optode = optode_objects[name]
        
        if is_anchor[i]:
            anchor_indices.append(i)
        else:
            # Check what types of edges this vertex has
            has_stiff = False
            has_flexible = False
            
            for spring_data in spring_states.values():
                if spring_data["optode1"] == name or spring_data["optode2"] == name:
                    if spring_data.get("is_flexible", False):
                        has_flexible = True
                    else:
                        has_stiff = True
            
            if has_stiff and not has_flexible:
                stiff_edge_indices.append(i)
            elif has_flexible:
                flexible_edge_indices.append(i)
    
    # Calculate statistics
    stats = {
        'anchor_displacement': 0.0,
        'stiff_displacement_mean': 0.0,
        'stiff_displacement_max': 0.0,
        'flexible_displacement_mean': 0.0,
        'flexible_displacement_max': 0.0,
        'anchor_count': len(anchor_indices),
        'stiff_count': len(stiff_edge_indices),
        'flexible_count': len(flexible_edge_indices)
    }
    
    if len(anchor_indices) > 0:
        anchor_displacements = displacements[anchor_indices]
        stats['anchor_displacement'] = np.max(anchor_displacements)
    
    if len(stiff_edge_indices) > 0:
        stiff_displacements = displacements[stiff_edge_indices]
        stats['stiff_displacement_mean'] = np.mean(stiff_displacements)
        stats['stiff_displacement_max'] = np.max(stiff_displacements)
    
    if len(flexible_edge_indices) > 0:
        flexible_displacements = displacements[flexible_edge_indices]
        stats['flexible_displacement_mean'] = np.mean(flexible_displacements)
        stats['flexible_displacement_max'] = np.max(flexible_displacements)
    
    # Print detailed statistics
    print(f"\nDISPLACEMENT ANALYSIS:")
    print(f"  Anchors ({stats['anchor_count']} optodes):")
    print(f"    Max displacement: {stats['anchor_displacement']:.6f} (should be ~0.0)")
    
    print(f"\n  Stiff-edge optodes ({stats['stiff_count']} optodes):")
    print(f"    Mean displacement: {stats['stiff_displacement_mean']:.6f} (should be small)")
    print(f"    Max displacement:  {stats['stiff_displacement_max']:.6f}")
    
    print(f"\n  Flexible-edge optodes ({stats['flexible_count']} optodes):")
    print(f"    Mean displacement: {stats['flexible_displacement_mean']:.6f} (can be larger)")
    print(f"    Max displacement:  {stats['flexible_displacement_max']:.6f}")
    
    # Detailed per-optode breakdown
    print(f"\n{'='*70}")
    print(f"PER-OPTODE DISPLACEMENT REPORT")
    print(f"{'='*70}")
    
    # Sort by displacement for easier reading
    sorted_indices = np.argsort(displacements)
    
    for idx in sorted_indices:
        name = optode_names[idx]
        disp = displacements[idx]
        
        # Determine category
        if is_anchor[idx]:
            category = "ANCHOR"
        elif idx in stiff_edge_indices:
            category = "STIFF"
        elif idx in flexible_edge_indices:
            category = "FLEXIBLE"
        else:
            category = "OTHER"
        
        print(f"  {name:12s} [{category:8s}]: {disp:8.6f}")
    
    print(f"\n{'='*70}")
    print(f"APPLYING RESULTS TO SCENE")
    print(f"{'='*70}")
    
    # ===== APPLY RESULTS =====
    for i, name in enumerate(optode_names):
        optode = optode_objects[name]
        new_pos = Vector(positions[i])
        optode.location = new_pos
    
    # Update connection mesh
    for i in range(len(conn_obj.data.vertices)):
        conn_obj.data.vertices[i].co = Vector(positions[i])
    
    conn_obj.data.update()
    bpy.context.view_layer.update()
    
    print(f"\n{'='*70}")
    print(f"RELAXATION COMPLETE")
    print(f"{'='*70}\n")
    
    return stats


class NEUROCAPTAIN_OT_import_optode_json_manualSpring(bpy.types.Operator):
    """Import optode probe configuration from JSON with automatic spring relaxation"""
    bl_idname = "neurocaptain.import_optode_json_manual_spring"
    bl_label = "Import Probe Configuration (Manual Spring)"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})
    
    auto_relax: bpy.props.BoolProperty(
        name="Auto Relax on Import",
        description="Automatically run spring relaxation after import",
        default=True
    )
    
    iterations: bpy.props.IntProperty(
        name="Relaxation Iterations",
        description="Number of simulation steps",
        default=200,
        min=10,
        max=2000
    )
    
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
                optode_pair = sorted([opt1_name, opt2_name])
                edge_key = f"{optode_pair[0]}_{optode_pair[1]}"
                #edge_key = f"{min(idx1, idx2)}_{max(idx1, idx2)}"
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
        conn_obj["optode_names_ordered"] = optode_names_ordered
        
        # Create material
        mat = bpy.data.materials.get("Connection_Material")
        if not mat:
            mat = bpy.data.materials.new(name="Connection_Material")
            mat.use_nodes = True
            mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.8, 0.8, 1)
        if conn_obj.data.materials:
            conn_obj.data.materials[0] = mat
        else:
            conn_obj.data.materials.append(mat)
        
        import_msg = f"Imported {len(optode_objects)} optodes with {len(edges)} connections"
        
        # ===== AUTO-RELAX IF ENABLED =====
        if self.auto_relax:
            print("\n" + "="*70)
            print("AUTO-RELAXATION ENABLED - Running spring simulation")
            print("="*70)
            
            stats = run_spring_relaxation(
                conn_obj=conn_obj,
                optode_objects=optode_objects,
                iterations=self.iterations,
                timestep=0.05,
                damping=0.85,
                stiff_strength=0.95,
                flexible_strength=0.3,
                use_surface_projection=True,
                projection_strength=0.5,
                report_callback=self.report
            )
            
            if stats:
                self.report({'INFO'}, 
                    f"{import_msg}. Relaxation: Anchor disp={stats['anchor_displacement']:.6f}, "
                    f"Stiff mean={stats['stiff_displacement_mean']:.6f}, "
                    f"Flex mean={stats['flexible_displacement_mean']:.6f}")
            else:
                self.report({'WARNING'}, f"{import_msg}. Relaxation failed!")
        else:
            self.report({'INFO'}, f"{import_msg}. Use 'Relax Probe' to run simulation.")
        
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


class NEUROCAPTAIN_OT_relax_probe_manual(bpy.types.Operator):
    """Manually run spring relaxation on imported probe or manually configured springs"""
    bl_idname = "neurocaptain.relax_probe_manual"
    bl_label = "Relax Probe (Manual)"
    bl_options = {'REGISTER', 'UNDO'}
    
    iterations: bpy.props.IntProperty(
        name="Iterations",
        description="Number of simulation steps",
        default=200,
        min=10,
        max=2000
    )
    
    timestep: bpy.props.FloatProperty(
        name="Time Step",
        description="Simulation time step (smaller = more stable)",
        default=0.05,
        min=0.001,
        max=0.5
    )
    
    damping: bpy.props.FloatProperty(
        name="Damping",
        description="Velocity damping (higher = settles faster)",
        default=0.85,
        min=0.0,
        max=0.99
    )
    
    stiff_spring_strength: bpy.props.FloatProperty(
        name="Stiff Spring Strength",
        description="Spring constant for stiff connections",
        default=0.95,
        min=0.1,
        max=1.0
    )
    
    flexible_spring_strength: bpy.props.FloatProperty(
        name="Flexible Spring Strength",
        description="Spring constant for flexible connections",
        default=0.3,
        min=0.01,
        max=1.0
    )
    
    use_surface_projection: bpy.props.BoolProperty(
        name="Project to Surface",
        description="Keep optodes on head surface during simulation",
        default=True
    )
    
    projection_strength: bpy.props.FloatProperty(
        name="Surface Projection Strength",
        description="How strongly to pull optodes to surface",
        default=0.5,
        min=0.0,
        max=1.0
    )
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=400)
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "iterations")
        layout.prop(self, "timestep")
        layout.prop(self, "damping")
        layout.separator()
        layout.label(text="Spring Strengths:")
        layout.prop(self, "stiff_spring_strength")
        layout.prop(self, "flexible_spring_strength")
        layout.separator()
        layout.prop(self, "use_surface_projection")
        if self.use_surface_projection:
            layout.prop(self, "projection_strength")
    
    def execute(self, context):
        print("\n" + "="*60)
        print("MANUAL SPRING RELAXATION - DIAGNOSTICS")
        print("="*60)
        
        if "Optode_Connections" not in bpy.data.objects:
            self.report({'ERROR'}, "No 'Optode_Connections' object found. Create connections first.")
            return {'CANCELLED'}
        
        conn_obj = bpy.data.objects["Optode_Connections"]
        
        # Try to get optode names from different possible sources
        optode_names = None
        
        # First try: optode_names_ordered (from JSON import)
        if "optode_names_ordered" in conn_obj:
            optode_names = conn_obj["optode_names_ordered"]
            print("Using optode_names_ordered from JSON import")
        
        # Second try: optode_names (from manual connection creation)
        elif "optode_names" in conn_obj:
            optode_names = conn_obj["optode_names"]
            print("Using optode_names from manual connection creation")
        
        # If no optode names found, error out
        if not optode_names:
            self.report({'ERROR'}, 
                "Optode_Connections missing required data. Re-import probe or create connections first.")
            return {'CANCELLED'}
        
        print(f"Found {len(optode_names)} optodes in optode_names list")
        
        # Get spring states (either from JSON import or manual spring definition)
        spring_states = conn_obj.get("spring_states", {})
        
        if not spring_states:
            self.report({'ERROR'}, 
                "No spring properties found. Re-import probe or define spring properties using "
                "'Set Flexible Spring Properties' or 'Make Springs Stiff'.")
            return {'CANCELLED'}
        
        print(f"Found {len(spring_states)} entries in spring_states")
        
        # Build edge list from spring_states
        edges = []
        for edge_key, spring_data in spring_states.items():
            try:
                edge = {
                    'optode1': spring_data['optode1'],
                    'optode2': spring_data['optode2'],
                    'rest_length': float(spring_data['rest_length']),
                    'is_flexible': spring_data.get('is_flexible', False),
                    'pull': float(spring_data.get('pull', 0.9)),
                    'push': float(spring_data.get('push', 0.9))
                }
                edges.append(edge)
            except Exception as e:
                print(f"WARNING: Failed to parse edge {edge_key}: {e}")
                print(f"  Data: {spring_data}")
        
        print(f"Successfully built {len(edges)} spring connections:")
        print(f"  Flexible: {sum(1 for e in edges if e['is_flexible'])}")
        print(f"  Stiff: {sum(1 for e in edges if not e['is_flexible'])}")
        
        # Get optode objects
        optode_objects = {}
        missing_optodes = []
        
        for name in optode_names:
            if name in bpy.data.objects:
                optode_objects[name] = bpy.data.objects[name]
            else:
                missing_optodes.append(name)
        
        if missing_optodes:
            print(f"ERROR: Missing optodes: {missing_optodes}")
            self.report({'ERROR'}, 
                f"Optodes not found: {', '.join(missing_optodes[:5])}"
                f"{' and more...' if len(missing_optodes) > 5 else ''}")
            return {'CANCELLED'}
        
        print(f"Found all {len(optode_objects)} optode objects")
        
        # Identify anchor optodes
        anchor_optodes = []
        for name in optode_names:
            obj = optode_objects.get(name)
            if obj and obj.get("is_anchor", False):
                anchor_optodes.append(name)
        
        print(f"Found {len(anchor_optodes)} anchor optodes: {anchor_optodes}")
        
        # CRITICAL: Build the spring_data structure that run_spring_relaxation expects
        # This needs to match the format from JSON import
        spring_data = {
            'optode_names': list(optode_names),
            'edges': edges,
            'anchors': anchor_optodes
        }
        
        # Store this on conn_obj - the relaxation function likely reads from here
        conn_obj["spring_data"] = spring_data
        
        # Also ensure optode_names_ordered exists (in case function checks for it)
        if "optode_names_ordered" not in conn_obj:
            conn_obj["optode_names_ordered"] = list(optode_names)
        
        print("\nStored spring_data on conn_obj:")
        print(f"  optode_names: {len(spring_data['optode_names'])} items")
        print(f"  edges: {len(spring_data['edges'])} items")
        print(f"  anchors: {len(spring_data['anchors'])} items")
        
        # Prepare parameters
        print("\nPreparing relaxation parameters:")
        print(f"  Iterations: {self.iterations}")
        print(f"  Timestep: {self.timestep}")
        print(f"  Damping: {self.damping}")
        print(f"  Stiff strength: {self.stiff_spring_strength}")
        print(f"  Flexible strength: {self.flexible_spring_strength}")
        print(f"  Use surface projection: {self.use_surface_projection}")
        print(f"  Projection strength: {self.projection_strength}")
        
        # Run relaxation with error catching
        print("\nRunning spring relaxation...")
        try:
            stats = run_spring_relaxation(
                conn_obj=conn_obj,
                optode_objects=optode_objects,
                iterations=self.iterations,
                timestep=self.timestep,
                damping=self.damping,
                stiff_strength=self.stiff_spring_strength,
                flexible_strength=self.flexible_spring_strength,
                use_surface_projection=self.use_surface_projection,
                projection_strength=self.projection_strength,
                report_callback=self.report
            )
            
            print(f"\nRelaxation returned: {stats}")
            
            if stats:
                print("SUCCESS! Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                
                self.report({'INFO'}, 
                    f"Relaxation complete! Anchor disp={stats.get('anchor_displacement', 0):.6f}, "
                    f"Stiff mean={stats.get('stiff_displacement_mean', 0):.6f}, "
                    f"Flex mean={stats.get('flexible_displacement_mean', 0):.6f}")
            else:
                print("ERROR: Relaxation returned None or empty result")
                self.report({'ERROR'}, "Relaxation failed - check console for details")
                return {'CANCELLED'}
                
        except Exception as e:
            print(f"\nEXCEPTION during relaxation: {e}")
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Relaxation error: {str(e)}")
            return {'CANCELLED'}
        
        print("="*60 + "\n")
        
        return {'FINISHED'}


# ===== REGISTER =====
def register():
    bpy.utils.register_class(NEUROCAPTAIN_OT_import_optode_json_manualSpring)
    bpy.utils.register_class(NEUROCAPTAIN_OT_relax_probe_manual)

def unregister():
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_import_optode_json_manualSpring)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_relax_probe_manual)

if __name__ == "__main__":
    register()
