import random
import time
from bpy.types import Operator, PropertyGroup
from bpy.types import Node
import bpy


################################################################
# helper functions BEGIN
################################################################

class geo_nodes(Operator):

    """Implement the geometry nodes"""
    bl_idname = "braincapgen.geo_nodes"
    bl_label = "Apply geo nodes"
    bl_description = "Takes the LandmarkMesh and make cut outs at those locations on the ehad surface mesh"
    bl_options = {'PRESET', 'UNDO'}
    
    def active_object():
        """
        returns the currently active object
        """
        return bpy.context.active_object



    def set_fcurve_extrapolation_to_linear():
        for fc in bpy.context.active_object.animation_data.action.fcurves:
            fc.extrapolation = "LINEAR"


    def set_scene_props(fps, frame_count):
        """
        Set scene properties
        """
        scene = bpy.context.scene
        scene.frame_end = frame_count

        # set the world background to black
        world = bpy.data.worlds["World"]
        if "Background" in world.node_tree.nodes:
            world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

        scene.render.fps = fps

        scene.frame_current = 1
        scene.frame_start = 1





    ################################################################
    # helper functions END
    ################################################################
    @classmethod
    def link_nodes_by_mesh_socket(node_tree, from_node, to_node,type_from,type_to):
        node_tree.links.new(from_node.outputs[type_from], to_node.inputs[type_to])

    def create_node(node_tree, type_name, node_x_location, node_location_step_x=0):
        """Creates a node of a given type, and sets/updates the location of the node on the X axis.
        Returning the node object and the next location on the X axis for the next node.
        """
        node_obj = node_tree.nodes.new(type=type_name)
        node_obj.location.x = node_x_location
        node_x_location += node_location_step_x

        return node_obj, node_x_location



    def update_geo_node_tree(node_tree,self,context):
        """
        Adding a Cube Mesh, Subdiv, Triangulate, Edge Split, and Element scale geo node into the
        geo node tree
        Geo Node type names found here
        https://docs.blender.org/api/current/bpy.types.GeometryNode.html
        """
        out_node = node_tree.node["Group Output"]
        in_node  = node_tree.nodes["Group Input"]
        align_euler = node_tree.nodes.new(type="FunctionNodeAlignEulerToVector")
        node_x_location = 0
        node_location_step_x = 300
        
    #transform 
        transform_node, node_x_location = create_node(node_tree, "GeometryNodeTransform", node_x_location, node_location_step_x)
        transform_node.inputs["Rotation"].default_value = 0, 2, 0

    #Geomtry Proximity 
        geometry_proximity, node_x_location = create_node(node_tree, "GeometryNodeProximity", node_x_location, node_location_step_x)
        geometry_proximity.target_element = 'FACES'
        
    #set position 
        set_position, node_x_location = create_node(node_tree, "GeometryNodeSetPosition", node_x_location, node_location_step_x)
        set_position.inputs["Offset"].default_value = 0, 0, 0

    #instance on points
        instance_on_points, node_x_location = create_node(node_tree, "GeometryNodeInstanceOnPoints", node_x_location, node_location_step_x)
        instance_on_points.inputs["Scale"].default_value = 3, 3, 3

    #Mesh Boolean 
        mesh_boolean, node_x_location = create_node(node_tree, "GeometryNodeMeshBoolean", node_x_location, node_location_step_x)
        mesh_boolean.operation = 'DIFFERENCE'
        
    #Sample Nearest Surface
        sample_nearest_surface, node_x_location = create_node(node_tree, "GeometryNodeSampleNearestSurface", node_x_location, node_location_step_x)
        sample_nearest_surface.data_type = 'FLOAT_VECTOR'
        
    #Align Euler to Vector 
        align_euler_to_vector, node_x_location = create_node(node_tree, "GeometryNodeScaleElements", node_x_location, node_location_step_x)
        #align_euler_to_vector.axis = 'Z'
        
    #BrainMesh 
        object_info_brain, node_x_location = create_node(node_tree, "GeometryNodeObjectInfo", node_x_location, node_location_step_x)
        bpy.data.node_groups["Geometry Nodes"].nodes["Object Info"].inputs[0].default_value = bpy.data.objects["LandmarkMesh"]

    #cutout 

        object_info_cutout, node_x_location = create_node(node_tree, "GeometryNodeObjectInfo", node_x_location, node_location_step_x)
        bpy.data.node_groups["Geometry Nodes"].nodes["Object Info"].inputs[0].default_value = bpy.data.objects["cutout"]
        
    #normal 
        normal_node, node_x_location = create_node(node_tree, "GeometryNodeInputNormal", node_x_location, node_location_step_x)
        out_node.location.x = node_x_location

        link_nodes_by_mesh_socket(node_tree, from_node=in_node, to_node=geometry_proximity, type_from= 'Geometry',type_to='Target')
        link_nodes_by_mesh_socket(node_tree, from_node=geometry_proximity, to_node=set_position, type_from = 'Position', type_to = 'Position')
        link_nodes_by_mesh_socket(node_tree, from_node=set_position, to_node=instance_on_points, type_from= 'Geometry', type_to = 'Points')
        link_nodes_by_mesh_socket(node_tree, from_node=instance_on_points, to_node=mesh_boolean, type_from= 'Instances', type_to = 'Mesh 2')
        link_nodes_by_mesh_socket(node_tree, from_node=in_node, to_node=object_info_brain, type_from= 'Object', type_to = 'Object')
        link_nodes_by_mesh_socket(node_tree, from_node=object_info_brain, to_node= transform_node, type_from= 'Geometry', type_to = 'Geometry')
        link_nodes_by_mesh_socket(node_tree, from_node=transform_node, to_node = set_position, type_from= 'Geometry', type_to = 'Geometry')
        link_nodes_by_mesh_socket(node_tree, from_node=in_node, to_node = object_info_cutout, type_from= 'Object', type_to = 'Object')
        link_nodes_by_mesh_socket(node_tree, from_node=object_info_cutout, to_node = instance_on_points, type_from= 'Geometry', type_to = 'Instance')
        link_nodes_by_mesh_socket(node_tree, from_node=in_node, to_node = mesh_boolean, type_from= 'Geometry', type_to = 'Mesh 1')
        link_nodes_by_mesh_socket(node_tree, from_node=in_node, to_node = sample_nearest_surface, type_from= 'Geometry', type_to = 'Mesh')
        link_nodes_by_mesh_socket(node_tree, from_node=normal_node, to_node = sample_nearest_surface, type_from= 'Normal', type_to = 'Value')
        link_nodes_by_mesh_socket(node_tree, from_node=sample_nearest_surface, to_node = align_euler, type_from= 'Value', type_to = 'Vector')
        link_nodes_by_mesh_socket(node_tree, from_node=align_euler, to_node = instance_on_points, type_from= 'Rotation', type_to = 'Rotation')
        
        
        from_node = mesh_boolean
        to_node = out_node
        node_tree.links.new(from_node.outputs["Mesh"], to_node.inputs["Geometry"])

     
    def create_centerpiece(self,context):
        #selected_objects = bpy.context.selected_objects
        #active_object = bpy.context.active_object
        bpy.context.view_layer.objects.active = context.scene.objects.get('headmesh') 
        bpy.data.objects['headmesh'].select_set(True)
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.node.new_geometry_nodes_modifier()
        node_tree = bpy.data.node_groups["Geometry Nodes"]

        self.update_geo_node_tree(node_tree,context)
        #bpy.context.view_layer.objects.active
        

        bpy.context.active_object.modifiers["GeometryNodes"].is_active = True


    def execute(self, context):
        """
        Python code to generate an animated geo nodes node tree
        that consists of a subdivided & triangulated cube with animated faces
        """

        self.create_centerpiece(context)
        return {"FINISHED"}
       
