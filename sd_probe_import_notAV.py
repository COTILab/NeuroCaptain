import bpy
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import Rbf
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy.types import Operator
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper

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

class NEUROCAPTAIN_OT_import_sd_probe(Operator, ImportHelper):
    """Import fNIRS probe from Homer/AtlasViewer SD file"""
    bl_idname = "neurocaptain.import_sd_probe"
    bl_label = "Import SD Probe"
    bl_description = "Import and register fNIRS probe from SD file to head mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    filename_ext = ".SD"
    filter_glob: StringProperty(default="*.SD;*.sd", options={'HIDDEN'})
    
    def execute(self, context):
        try:
            sd_file_path = self.filepath
            head_mesh_name = 'headmesh'
            landmark_mesh_name = 'LandmarkMesh'
            source_color = (1.0, 0.0, 0.0, 1.0)
            detector_color = (0.0, 0.0, 0.0, 1.0)
            dummy_color = (0.5, 0.5, 0.5, 1.0)  # Grey for dummies
            
            print("="*70)
            print("UPDATED CODE")
            print("fNIRS probe import from SD file")
            print("="*70)
            
            # load SD file
            sd_data = self.load_sd_file(sd_file_path)
            
            # get landmarks
            print("\n2. Loading 3D landmarks")
            if landmark_mesh_name not in bpy.data.objects:
                self.report({'ERROR'}, f"'{landmark_mesh_name}' not found. Run brain1020mesh first.")
                return {'CANCELLED'}            
            landmarks_3d = self.get_landmark_positions_with_labels(landmark_mesh_name)
            
            # match anchors to landmarks
            print("\n3. Pairing anchors to landmarks")
            anchors_3d, matched_labels = self.match_anchors_to_landmarks(
                sd_data['anchor_labels'], landmarks_3d
            )
            
            if len(anchors_3d) < 3:
                self.report({'ERROR'}, f"Only {len(anchors_3d)} anchors matched. Need at least 3 for unique placement.")
                return {'CANCELLED'}
            
            matched_indices = [i for i, label in enumerate(sd_data['anchor_labels']) 
                             if label in matched_labels]
            anchors_2d_matched = sd_data['anchors_2d'][matched_indices]
            matched_anchor_indices = [sd_data['anchor_indices'][i] for i in matched_indices]
            
            # headmesh 
            if head_mesh_name not in bpy.data.objects:
                self.report({'ERROR'}, f"Head mesh '{head_mesh_name}' not found in scene")
                return {'CANCELLED'}
            
            head_mesh = bpy.data.objects[head_mesh_name]
            
            # optode size based on headmesh scale 
            print("\n4. Calculating optode size")
            optode_diameter, optode_thickness = self.get_head_scale(head_mesh)
            
            # register probe with spring physics (relaxation)
            print("\n5. Registering probe with spring constraints")
            all_positions_3d = self.register_with_springs(
                sd_data['all_positions_2d'],
                anchors_2d_matched,
                anchors_3d,
                matched_anchor_indices,
                sd_data['spring_list'],
                head_mesh
            )
            
            # separate sources, detectors, and dummies
            n_srcs = sd_data['n_srcs']
            n_dets = sd_data['n_dets']
            src_pos_3d = all_positions_3d[:n_srcs]
            det_pos_3d = all_positions_3d[n_srcs:n_srcs+n_dets]
            dummy_pos_3d = all_positions_3d[n_srcs+n_dets:]
            
            # Get collections
            sources_collection = get_or_create_collection("Sources")
            detectors_collection = get_or_create_collection("Detectors")
            dummies_collection = get_or_create_collection("Dummies")
            anchors_collection = get_or_create_collection("Anchor Indicators")
            ring_mat = create_anchor_ring_material()
            
            # Debug output
            print("\n" + "="*70)
            print("ANCHOR DEBUG INFO")
            print("="*70)
            print(f"matched_anchor_indices: {matched_anchor_indices}")
            print(f"n_srcs: {n_srcs}, n_dets: {n_dets}, n_dummies: {len(dummy_pos_3d)}")
            print(f"Total optodes: {len(all_positions_3d)}")
            
            # Determine which anchors are sources, detectors, or dummies
            anchor_types = {}
            for anchor_idx in matched_anchor_indices:
                if anchor_idx < n_srcs:
                    anchor_types[anchor_idx] = f"Source_{anchor_idx+1}"
                elif anchor_idx < n_srcs + n_dets:
                    anchor_types[anchor_idx] = f"Detector_{anchor_idx - n_srcs + 1}"
                else:
                    anchor_types[anchor_idx] = f"Dummy_{anchor_idx - n_srcs - n_dets + 1}"
            
            print("\nAnchor optode types:")
            for idx, name in anchor_types.items():
                print(f"  Index {idx}: {name}")
            print("="*70 + "\n")
            
            # create source optodes
            print("\n6. Creating optodes...")
            print("  Creating sources...")
            for i, pos in enumerate(src_pos_3d):
                snapped_pos, normal = self.snap_to_mesh_surface(pos, head_mesh)
                optode = self.create_optode_disc(snapped_pos, normal, f"Source_{i+1}", 
                                       source_color, optode_diameter, optode_thickness)
                add_to_collection(optode, sources_collection)
                
                # Check if this optode is an anchor
                if i in matched_anchor_indices:
                    print(f"    ✓ Source_{i+1} (index {i}) is an ANCHOR")
                    optode["is_anchor"] = 1
                    
                    # Create anchor ring indicator
                    bpy.ops.mesh.primitive_torus_add(
                        location=(0, 0, 0), 
                        major_radius=2.5, 
                        minor_radius=0.3
                    )
                    indicator = context.active_object
                    add_to_collection(indicator, anchors_collection)
                    indicator.name = f"Anchor_Ring_Source_{i+1}"
                    indicator.parent = optode
                    indicator.location = (0, 0, 0)
                    indicator.hide_render = True
                    indicator.data.materials.append(ring_mat)
                else:
                    optode["is_anchor"] = 0
            
            # create detector optodes
            print("  Creating detectors...")
            for i, pos in enumerate(det_pos_3d):
                snapped_pos, normal = self.snap_to_mesh_surface(pos, head_mesh)
                optode = self.create_optode_disc(snapped_pos, normal, f"Detector_{i+1}", 
                                       detector_color, optode_diameter, optode_thickness)
                add_to_collection(optode, detectors_collection)
                
                # Check if this optode is an anchor
                anchor_idx = n_srcs + i
                if anchor_idx in matched_anchor_indices:
                    print(f"    ✓ Detector_{i+1} (index {anchor_idx}) is an ANCHOR")
                    optode["is_anchor"] = 1
                    
                    # Create anchor ring indicator
                    bpy.ops.mesh.primitive_torus_add(
                        location=(0, 0, 0), 
                        major_radius=2.5, 
                        minor_radius=0.3
                    )
                    indicator = context.active_object
                    add_to_collection(indicator, anchors_collection)
                    indicator.name = f"Anchor_Ring_Detector_{i+1}"
                    indicator.parent = optode
                    indicator.location = (0, 0, 0)
                    indicator.hide_render = True
                    indicator.data.materials.append(ring_mat)
                else:
                    optode["is_anchor"] = 0
            
            # create dummy optodes
            
            print("  Creating dummies...")
            print(f"    n_srcs={n_srcs}, n_dets={n_dets}")
            print(f"    matched_anchor_indices={matched_anchor_indices}")
            print(f"    Number of dummy positions: {len(dummy_pos_3d)}")

            for i, pos in enumerate(dummy_pos_3d):
                snapped_pos, normal = self.snap_to_mesh_surface(pos, head_mesh)
                optode = self.create_optode_disc(snapped_pos, normal, f"Dummy_{i+1}", 
                                    dummy_color, optode_diameter, optode_thickness)
                add_to_collection(optode, dummies_collection)
                
                # Check if this optode is an anchor
                anchor_idx = n_srcs + n_dets + i
                is_anchor = anchor_idx in matched_anchor_indices
                
                print(f"    Dummy_{i+1}: i={i}, anchor_idx={anchor_idx}, in list? {is_anchor}")
                
                if is_anchor:
                    print(f"    ✓✓✓ Dummy_{i+1} (index {anchor_idx}) IS AN ANCHOR - CREATING RING ✓✓✓")
                    optode["is_anchor"] = 1
                    
                    # Create anchor ring indicator
                    print(f"        About to create torus...")
                    bpy.ops.mesh.primitive_torus_add(
                        location=(0, 0, 0), 
                        major_radius=2.5, 
                        minor_radius=0.3
                    )
                    indicator = context.active_object
                    print(f"        Created torus: {indicator.name}")
                    
                    print(f"        Adding to collection: {anchors_collection.name}")
                    add_to_collection(indicator, anchors_collection)
                    
                    indicator.name = f"Anchor_Ring_Dummy_{i+1}"
                    print(f"        Renamed to: {indicator.name}")
                    
                    indicator.parent = optode
                    indicator.location = (0, 0, 0)
                    indicator.hide_render = True
                    
                    print(f"        Applying material...")
                    if indicator.data.materials:
                        indicator.data.materials[0] = ring_mat
                    else:
                        indicator.data.materials.append(ring_mat)
                    
                    print(f"        ✓ Complete: {indicator.name} parented to {optode.name}")
                else:
                    optode["is_anchor"] = 0
            
            print("\n" + "="*70)
            print("COMPLETE!")
            print("="*70)
            
            num_anchors = len(matched_anchor_indices)
            self.report({'INFO'}, 
                f"Created {len(src_pos_3d)} sources + {len(det_pos_3d)} detectors + {len(dummy_pos_3d)} dummies ({num_anchors} anchors)")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
    
    def load_sd_file(self, filepath):
        """Load Homer/AtlasViewer SD file"""
        mat_data = loadmat(filepath)
        SD = mat_data['SD']
        sd = SD[0, 0]
        
        src_pos = sd['SrcPos']
        det_pos = sd['DetPos']
        dummy_pos = sd['DummyPos']
        anchor_list = sd['AnchorList']
        spring_list = sd['SpringList'] if 'SpringList' in sd.dtype.names else None
        n_srcs = int(sd['nSrcs'][0, 0])
        n_dets = int(sd['nDets'][0, 0])
        
        print(f"   Sources: {n_srcs}, Detectors: {n_dets}, Dummies: {len(dummy_pos)}")
        if spring_list is not None:
            print(f"   Springs: {len(spring_list)}")
        
        all_positions_2d = np.vstack([src_pos, det_pos, dummy_pos])
        
        # anchors
        anchors_2d = []
        anchor_labels = []
        anchor_indices = []
        for anchor in anchor_list:
            idx = int(anchor[0][0, 0]) - 1
            label = str(anchor[1][0])
            
            if idx < len(all_positions_2d):
                pos = all_positions_2d[idx]
                anchors_2d.append([pos[0], pos[1]])
                anchor_labels.append(label)
                anchor_indices.append(idx)
        
        return {
            'all_positions_2d': all_positions_2d,
            'anchors_2d': np.array(anchors_2d),
            'anchor_labels': anchor_labels,
            'anchor_indices': anchor_indices,
            'spring_list': spring_list,
            'n_srcs': n_srcs,
            'n_dets': n_dets
        }
    
    def get_landmark_positions_with_labels(self, landmark_mesh_name):
        """Extract landmark positions with proper labels"""
        landmark_obj = bpy.data.objects[landmark_mesh_name]
        num_landmarks = len(landmark_obj.data.vertices)
        
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
            landmark_labels = [f"Landmark_{i+1}" for i in range(num_landmarks)]
        
        landmarks_3d = {}
        for i, vert in enumerate(landmark_obj.data.vertices):
            v_global = landmark_obj.matrix_world @ vert.co
            if i < len(landmark_labels) and landmark_labels[i]:
                landmarks_3d[landmark_labels[i]] = np.array([v_global.x, v_global.y, v_global.z])
        
        return landmarks_3d
    
    def match_anchors_to_landmarks(self, anchor_labels, landmarks_3d):
        """Match anchor labels to landmarks"""
        anchors_3d = []
        matched_labels = []
        
        for label in anchor_labels:
            if label in landmarks_3d:
                anchors_3d.append(landmarks_3d[label])
                matched_labels.append(label)
            else:
                self.report({'WARNING'}, f"Anchor '{label}' not found in LandmarkMesh")
        
        print(f"   Matched {len(anchors_3d)}/{len(anchor_labels)} anchors")
        return np.array(anchors_3d), matched_labels
    
    def register_with_springs(self, all_positions_2d, anchors_2d, anchors_3d, 
                             anchor_indices, spring_list, head_mesh):
        """Register probe using spring constraints"""
        n_points = len(all_positions_2d)
        
        # Thin-plate spline registration
        rbf_x = Rbf(anchors_2d[:, 0], anchors_2d[:, 1], anchors_3d[:, 0], function='thin_plate')
        rbf_y = Rbf(anchors_2d[:, 0], anchors_2d[:, 1], anchors_3d[:, 1], function='thin_plate')
        rbf_z = Rbf(anchors_2d[:, 0], anchors_2d[:, 1], anchors_3d[:, 2], function='thin_plate')
        
        positions_3d = np.column_stack([
            rbf_x(all_positions_2d[:, 0], all_positions_2d[:, 1]),
            rbf_y(all_positions_2d[:, 0], all_positions_2d[:, 1]),
            rbf_z(all_positions_2d[:, 0], all_positions_2d[:, 1])
        ])
        
        # project/snap to surface
        bvh = BVHTree.FromObject(head_mesh, bpy.context.evaluated_depsgraph_get())
        for i in range(n_points):
            location, normal, index, distance = bvh.find_nearest(Vector(positions_3d[i]))
            if location is not None:
                positions_3d[i] = np.array(location)
        
        # build stiff/flexible spring network
        spring_data = []
        if spring_list is not None:
            for spring in spring_list:
                idx1 = int(spring[0]) - 1
                idx2 = int(spring[1]) - 1
                if idx1 < n_points and idx2 < n_points:
                    if len(spring) > 2:
                        stored_length = float(spring[2])
                        is_stiff = stored_length > 0
                        rest_length = abs(stored_length)
                    else:
                        rest_length = np.linalg.norm(all_positions_2d[idx1] - all_positions_2d[idx2])
                        is_stiff = True
                    
                    spring_data.append((idx1, idx2, rest_length, is_stiff))
        
        # spring relaxation
        positions_refined = positions_3d.copy()
        n_iterations = 50
        stiff_spring_weight = 0.8
        flexible_spring_weight = 0.2
        surface_weight = 0.5
        max_displacement = 0.001
        
        for iteration in range(n_iterations):
            positions_new = positions_refined.copy()
            
            # apply spring forces
            for (idx1, idx2, rest_length, is_stiff) in spring_data:
                vec = positions_refined[idx2] - positions_refined[idx1]
                current_length = np.linalg.norm(vec)
                if current_length > 0:
                    scale_factor = rest_length / current_length
                    spring_weight = stiff_spring_weight if is_stiff else flexible_spring_weight
                    correction = vec * (1 - scale_factor) * spring_weight * 0.5
                    
                    correction_magnitude = np.linalg.norm(correction)
                    if correction_magnitude > max_displacement:
                        correction = correction * (max_displacement / correction_magnitude)
                    
                    if idx1 not in anchor_indices:
                        positions_new[idx1] += correction
                    if idx2 not in anchor_indices:
                        positions_new[idx2] -= correction
            
            # move towards head surface
            for i in range(n_points):
                if i not in anchor_indices:
                    location, normal, index, distance = bvh.find_nearest(Vector(positions_new[i]))
                    if location is not None:
                        surface_pos = np.array(location)
                        displacement = surface_pos - positions_new[i]
                        displacement_magnitude = np.linalg.norm(displacement)
                        if displacement_magnitude > max_displacement:
                            displacement = displacement * (max_displacement / displacement_magnitude)
                        positions_new[i] = positions_new[i] + displacement #* surface_weight
            
            # lock anchors
            for i, anchor_idx in enumerate(anchor_indices):
                positions_new[anchor_idx] = anchors_3d[i]
            
            positions_refined = positions_new
        
        return positions_refined
    
    def get_head_scale(self, head_mesh):
        """optode size based on head dimensions"""
        bbox_min = Vector([min(v.co[i] for v in head_mesh.data.vertices) for i in range(3)])
        bbox_max = Vector([max(v.co[i] for v in head_mesh.data.vertices) for i in range(3)])
        dimensions = (head_mesh.matrix_world @ bbox_max) - (head_mesh.matrix_world @ bbox_min)
        head_size = max(dimensions)
        return head_size * 0.02, head_size * 0.02 * 0.15
    
    def snap_to_mesh_surface(self, position, head_mesh):
        """snap point to nearest surface location"""
        bvh = BVHTree.FromObject(head_mesh, bpy.context.evaluated_depsgraph_get())
        location, normal, index, distance = bvh.find_nearest(Vector(position))
        return (np.array(location), np.array(normal)) if location else (position, np.array([0, 0, 1]))
    
    def create_optode_disc(self, position, normal, name, color, diameter, thickness):
        """create optode disc at position"""
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32, radius=diameter/2.0, depth=thickness,
            enter_editmode=False, align='WORLD', location=(0, 0, 0)
        )
        
        optode = bpy.context.active_object
        optode.name = name
        
        # Align to surface normal
        z_axis = Vector((0, 0, 1))
        rotation_quat = z_axis.rotation_difference(Vector(normal))
        optode.rotation_euler = rotation_quat.to_euler()
        optode.location = Vector(position)
        
        # create material
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


def register():
    bpy.utils.register_class(NEUROCAPTAIN_OT_import_sd_probe)

def unregister():
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_import_sd_probe)

if __name__ == "__main__":
    register()