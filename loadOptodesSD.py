import bpy
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import Rbf
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy.types import Operator
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper

class NEUROCAPTAIN_OT_import_sd_probe(Operator, ImportHelper):
    """Import fNIRS probe from Homer/AtlasViewer SD file"""
    bl_idname = "neurocaptain.import_sd_probe"
    bl_label = "Import SD Probe"
    bl_description = "Import and register fNIRS probe from SD file to head mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    filename_ext = ".SD"
    
    filter_glob: StringProperty(
        default="*.SD;*.sd",
        options={'HIDDEN'},
    )
    
    def execute(self, context):
        try:
            # Configuration
            sd_file_path = self.filepath
            head_mesh_name = 'headmesh'
            landmark_mesh_name = 'LandmarkMesh'
            source_color = (1.0, 0.0, 0.0, 1.0)
            detector_color = (0.0, 0.0, 0.0, 1.0)
            
            print("="*70)
            print("FNIRS PROBE REGISTRATION")
            print("="*70)
            
            # 1. Load SD file
            print(f"\n1. Loading SD file: {sd_file_path}")
            sd_data = self.load_sd_file(sd_file_path)
            
            # 2. Get landmarks
            print("\n2. Loading 3D landmarks...")
            if landmark_mesh_name not in bpy.data.objects:
                self.report({'ERROR'}, f"'{landmark_mesh_name}' not found. Run brain1020mesh first.")
                return {'CANCELLED'}
            
            landmarks_3d = self.get_landmark_positions_with_labels(landmark_mesh_name)
            
            # 3. Match anchors
            print("\n3. Matching anchors...")
            anchors_3d, matched_labels = self.match_anchors_to_landmarks(
                sd_data['anchor_labels'], 
                landmarks_3d
            )
            
            if len(anchors_3d) < 3:
                self.report({'ERROR'}, f"Only {len(anchors_3d)} anchors matched. Need at least 3.")
                return {'CANCELLED'}
            
            matched_indices = [i for i, label in enumerate(sd_data['anchor_labels']) 
                             if label in matched_labels]
            anchors_2d_matched = sd_data['anchors_2d'][matched_indices]
            matched_anchor_indices = [sd_data['anchor_indices'][i] for i in matched_indices]
            
            # 4. Get head mesh
            if head_mesh_name not in bpy.data.objects:
                self.report({'ERROR'}, f"Head mesh '{head_mesh_name}' not found in scene")
                return {'CANCELLED'}
            
            head_mesh = bpy.data.objects[head_mesh_name]
            
            # 5. Calculate optode size
            print("\n4. Calculating optode size...")
            optode_diameter, optode_thickness = self.get_head_scale(head_mesh)
            
            # 6. Register probe
            print("\n5. Registering probe with spring constraints...")
            all_positions_3d = self.register_with_springs(
                sd_data['all_positions_2d'],
                anchors_2d_matched,
                anchors_3d,
                matched_anchor_indices,
                sd_data['spring_list'],
                head_mesh
            )
            
            # Split into sources, detectors
            n_srcs = sd_data['n_srcs']
            n_dets = sd_data['n_dets']
            src_pos_3d = all_positions_3d[:n_srcs]
            det_pos_3d = all_positions_3d[n_srcs:n_srcs+n_dets]
            
            # 7. Create optodes
            print("\n6. Creating optodes...")
            for i, pos in enumerate(src_pos_3d):
                snapped_pos, normal = self.snap_to_mesh_surface(pos, head_mesh)
                self.create_optode_disc(snapped_pos, normal, f"Source_{i+1}", 
                                       source_color, optode_diameter, optode_thickness)
            
            for i, pos in enumerate(det_pos_3d):
                snapped_pos, normal = self.snap_to_mesh_surface(pos, head_mesh)
                self.create_optode_disc(snapped_pos, normal, f"Detector_{i+1}", 
                                       detector_color, optode_diameter, optode_thickness)
            
            print("\n" + "="*70)
            print("COMPLETE!")
            print("="*70)
            
            self.report({'INFO'}, f"Created {len(src_pos_3d)} sources + {len(det_pos_3d)} detectors")
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
            'src_pos': src_pos,
            'det_pos': det_pos,
            'dummy_pos': dummy_pos,
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
                "T7", "C5", "C3", "C1", "Cz",
                "C2", "C4", "C6", "T8",
                "Fpz", "AFz", "Fz", "FCz", "Cz",
                "CPz", "Pz", "POz", "Oz",
                "FT7", "F7", "AF7", "Fp1",
                "TP7", "P7", "PO7", "O1",
                "FT8", "F8", "AF8", "Fp2",
                "TP8", "P8", "PO8", "O2",
                "FC1", "FC3", "FC5",
                "FC2", "FC4", "FC6",
                "F1", "F3", "F5",
                "F2", "F4", "F6",
                "AF3", "AF4",
                "CP1", "CP3", "CP5",
                "CP2", "CP4", "CP6",
                "P1", "P3", "P5",
                "P2", "P4", "P6",
                "PO3", "PO4",
                "FT9", "F9", "", "",
                "TP9", "P9", "PO9", "O9",
                "FT10", "F10", "", "",
                "TP10", "P10", "PO10", "O10"
            ]
        elif num_landmarks >= 27:
            landmark_labels = [
                "Nz", "Iz", "Lpa", "Rpa", "Cz",
                "T3", "C3", "Cz", "C4", "T4",
                "Fpz", "Fz", "Cz", "Pz", "Oz",
                "F7", "Fp1", "T5", "O1",
                "F8", "Fp2", "T6", "O2",
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
        anchor_map = {
            'F7': ['F7'],
            'F8': ['F8'],
            'FC5': ['FC5', 'C5', 'FC3'],
            'FC6': ['FC6', 'C6', 'FC4'],
            'CP3': ['CP3', 'P3'],
            'CP4': ['CP4', 'P4'],
            'PO3': ['PO3'],
            'PO4': ['PO4'],
            'P3': ['P3', 'CP3'],
            'P4': ['P4', 'CP4']
        }
        
        anchors_3d = []
        matched_labels = []
        
        for label in anchor_labels:
            matched = False
            
            if label in landmarks_3d:
                anchors_3d.append(landmarks_3d[label])
                matched_labels.append(label)
                matched = True
            elif label in anchor_map:
                for alt_label in anchor_map[label]:
                    if alt_label in landmarks_3d:
                        anchors_3d.append(landmarks_3d[alt_label])
                        matched_labels.append(label)
                        matched = True
                        break
        
        print(f"   Matched {len(anchors_3d)}/{len(anchor_labels)} anchors")
        return np.array(anchors_3d), matched_labels
    
    def register_with_springs(self, all_positions_2d, anchors_2d, anchors_3d, 
                             anchor_indices, spring_list, head_mesh):
        """Register probe using spring constraints"""
        n_points = len(all_positions_2d)
        
        # Initial registration with TPS
        rbf_x = Rbf(anchors_2d[:, 0], anchors_2d[:, 1], anchors_3d[:, 0], function='thin_plate')
        rbf_y = Rbf(anchors_2d[:, 0], anchors_2d[:, 1], anchors_3d[:, 1], function='thin_plate')
        rbf_z = Rbf(anchors_2d[:, 0], anchors_2d[:, 1], anchors_3d[:, 2], function='thin_plate')
        
        positions_3d = np.column_stack([
            rbf_x(all_positions_2d[:, 0], all_positions_2d[:, 1]),
            rbf_y(all_positions_2d[:, 0], all_positions_2d[:, 1]),
            rbf_z(all_positions_2d[:, 0], all_positions_2d[:, 1])
        ])
        
        # Project to surface
        bvh = BVHTree.FromObject(head_mesh, bpy.context.evaluated_depsgraph_get())
        for i in range(n_points):
            location, normal, index, distance = bvh.find_nearest(Vector(positions_3d[i]))
            if location is not None:
                positions_3d[i] = np.array(location)
        
        # Calculate spring constraints
        spring_lengths = []
        spring_pairs = []
        if spring_list is not None:
            for spring in spring_list:
                idx1 = int(spring[0]) - 1
                idx2 = int(spring[1]) - 1
                if idx1 < n_points and idx2 < n_points:
                    original_length = np.linalg.norm(all_positions_2d[idx1] - all_positions_2d[idx2])
                    spring_lengths.append(original_length)
                    spring_pairs.append((idx1, idx2))
        
        # Spring relaxation
        positions_refined = positions_3d.copy()
        n_iterations = 50
        spring_weight = 0.5
        surface_weight = 0.5
        
        for iteration in range(n_iterations):
            positions_new = positions_refined.copy()
            
            # Apply spring forces
            for (idx1, idx2), target_length in zip(spring_pairs, spring_lengths):
                vec = positions_refined[idx2] - positions_refined[idx1]
                current_length = np.linalg.norm(vec)
                if current_length > 0:
                    scale_factor = target_length / current_length
                    correction = vec * (1 - scale_factor) * spring_weight * 0.5
                    
                    if idx1 not in anchor_indices:
                        positions_new[idx1] += correction
                    if idx2 not in anchor_indices:
                        positions_new[idx2] -= correction
            
            # Project to surface
            for i in range(n_points):
                if i not in anchor_indices:
                    location, normal, index, distance = bvh.find_nearest(Vector(positions_new[i]))
                    if location is not None:
                        positions_new[i] = surface_weight * np.array(location) + (1 - surface_weight) * positions_new[i]
            
            # Fix anchors
            for i, anchor_idx in enumerate(anchor_indices):
                positions_new[anchor_idx] = anchors_3d[i]
            
            positions_refined = positions_new
        
        return positions_refined
    
    def get_head_scale(self, head_mesh):
        """Calculate optode size based on head dimensions"""
        bbox_min = Vector([min(v.co[i] for v in head_mesh.data.vertices) for i in range(3)])
        bbox_max = Vector([max(v.co[i] for v in head_mesh.data.vertices) for i in range(3)])
        
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
        
        z_axis = Vector((0, 0, 1))
        normal_vec = Vector(normal)
        rotation_quat = z_axis.rotation_difference(normal_vec)
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
        
        optode.data.materials.clear()
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