import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# 10-10 landmark labels in brain1020 output order
LANDMARK_LABELS_77 = [
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

# 10-20 landmark labels in brain1020 output order
LANDMARK_LABELS_27 = [
    "Nz", "Iz", "Lpa", "Rpa", "Cz",
    "T3", "C3", "Cz", "C4", "T4",
    "Fpz", "Fz", "Cz", "Pz", "Oz",
    "F7", "Fp1", "T5", "O1",
    "F8", "Fp2", "T6", "O2",
    "F3", "F4", "P3", "P4"
]


def get_landmark_labels(landmark_mesh_obj):
    """
    Get landmark labels for a LandmarkMesh object.
    Prefers stored labels if they match vertex count,
    otherwise falls back to hardcoded lists by vertex count.
    Returns list of label strings (empty string for unlabeled vertices).
    """
    num_verts = len(landmark_mesh_obj.data.vertices)
    stored = landmark_mesh_obj.get("landmark_labels", None)

    if stored is not None:
        stored_list = list(stored)
        if len(stored_list) == num_verts:
            print(f"get_landmark_labels: using stored labels ({num_verts} labels)")
            return stored_list
        else:
            print(f"get_landmark_labels: stored label count ({len(stored_list)}) "
                  f"!= vertex count ({num_verts}), falling back to hardcoded list")

    if num_verts >= 77:
        print(f"get_landmark_labels: using hardcoded 10-10 list for {num_verts} vertices")
        return LANDMARK_LABELS_77
    elif num_verts >= 27:
        print(f"get_landmark_labels: using hardcoded 10-20 list for {num_verts} vertices")
        return LANDMARK_LABELS_27
    else:
        print(f"get_landmark_labels: unknown vertex count {num_verts}, returning empty labels")
        return [""] * num_verts


def label_material():
    """material for landmark labels"""
    mat_name = "Landmark_Label_Material"
    label_color = (0.1, 0.3, 0.9, 1.0)
    
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]
    
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    mat.diffuse_color = label_color
    
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    node_emission = nodes.new(type='ShaderNodeEmission')
    node_emission.inputs[0].default_value = label_color
    node_emission.inputs[1].default_value = 2.0
    
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(node_emission.outputs[0], node_output.inputs[0])
    
    return mat


class NEUROCAPTAIN_OT_display_landmark_labels(bpy.types.Operator):
    """Display 10-10/10-20 landmark labels in 3D viewport"""
    bl_idname = "neurocaptain.display_landmark_labels"
    bl_label = "Display Landmark Labels"
    bl_options = {'REGISTER', 'UNDO'}
    
    label_size: bpy.props.FloatProperty(
        name="Label Size",
        default=3.0,
        min=0.5,
        max=20.0
    )
    
    offset_distance: bpy.props.FloatProperty(
        name="Offset Distance",
        default=1.0,
        min=-20.0,
        max=20.0
    )
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        if "LandmarkMesh" not in bpy.data.objects:
            self.report({'ERROR'}, "LandmarkMesh not found")
            return {'CANCELLED'}
        
        headmesh_obj = bpy.data.objects.get("headmesh")
        if not headmesh_obj:
            self.report({'ERROR'}, "headmesh not found")
            return {'CANCELLED'}
        
        landmark_mesh_obj = bpy.data.objects["LandmarkMesh"]

        # Use validated label lookup
        labels = get_landmark_labels(landmark_mesh_obj)
        
        if "Landmark_Labels" in bpy.data.collections:
            label_collection = bpy.data.collections["Landmark_Labels"]
            for obj in list(label_collection.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
        else:
            label_collection = bpy.data.collections.new("Landmark_Labels")
            context.scene.collection.children.link(label_collection)
        
        label_mat = label_material()
        bvh = BVHTree.FromObject(headmesh_obj, context.evaluated_depsgraph_get())
        
        created_labels = 0
        for idx, vertex in enumerate(landmark_mesh_obj.data.vertices):
            if idx >= len(labels) or not labels[idx]:
                continue
            
            vertex_world_pos = landmark_mesh_obj.matrix_world @ vertex.co
            location, normal, _, _ = bvh.find_nearest(vertex_world_pos)
            
            surface_normal = normal if location else (landmark_mesh_obj.matrix_world.to_3x3() @ vertex.normal)
            surface_normal.normalize()
            label_position = vertex_world_pos + surface_normal * self.offset_distance
            
            text_data = bpy.data.curves.new(name=f"Label_{labels[idx]}_{idx}", type='FONT')
            text_data.body = labels[idx]
            text_data.size = self.label_size
            text_data.align_x = 'CENTER'
            text_data.align_y = 'CENTER'
            text_data.extrude = 0.1
            text_data.bevel_depth = 0.02
            
            text_obj = bpy.data.objects.new(name=f"Label_{labels[idx]}_{idx}", object_data=text_data)
            text_obj.location = label_position
            text_obj.rotation_mode = 'QUATERNION'
            text_obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(surface_normal)
            
            text_obj.data.materials.append(label_mat)
            label_collection.objects.link(text_obj)
            
            text_obj["is_landmark_label"] = True
            text_obj["landmark_index"] = idx
            created_labels += 1
        
        context.scene.neurocaptain.show_landmark_labels = True
        
        self.report({'INFO'}, f"Created {created_labels} landmark labels")
        return {'FINISHED'}


class NEUROCAPTAIN_OT_toggle_landmark_labels(bpy.types.Operator):
    """Toggle visibility of landmark labels"""
    bl_idname = "neurocaptain.toggle_landmark_labels"
    bl_label = "Toggle Landmark Labels"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.neurocaptain
        
        if "Landmark_Labels" not in bpy.data.collections:
            self.report({'WARNING'}, "No labels found. Generate them first.")
            props.show_landmark_labels = False
            return {'CANCELLED'}
        
        label_collection = bpy.data.collections["Landmark_Labels"]
        
        for obj in label_collection.objects:
            obj.hide_viewport = not props.show_landmark_labels
            obj.hide_render = not props.show_landmark_labels
        
        status = "visible" if props.show_landmark_labels else "hidden"
        self.report({'INFO'}, f"Labels {status}")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(NEUROCAPTAIN_OT_display_landmark_labels)
    bpy.utils.register_class(NEUROCAPTAIN_OT_toggle_landmark_labels)


def unregister():
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_toggle_landmark_labels)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_display_landmark_labels)