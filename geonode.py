import random
import time

import bpy


################################################################
# helper functions BEGIN
################################################################

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


def scene_setup():
    fps = 30
    loop_seconds = 12
    frame_count = fps * loop_seconds

    seed = 0
    if seed:
        random.seed(seed)
    else:
        time_seed()

    clean_scene()

    set_scene_props(fps, frame_count)


################################################################
# helper functions END
################################################################
def link_nodes_by_mesh_socket(node_tree, from_node, to_node):
    node_tree.links.new(from_node.outputs["Mesh"], to_node.inputs["Mesh"])

def create_node(node_tree, type_name, node_x_location, node_location_step_x=0):
    """Creates a node of a given type, and sets/updates the location of the node on the X axis.
    Returning the node object and the next location on the X axis for the next node.
    """
    node_obj = node_tree.nodes.new(type=type_name)
    node_obj.location.x = node_x_location
    node_x_location += node_location_step_x

    return node_obj, node_x_location

def update_geo_node_tree(node_tree):
    """
    Adding a Cube Mesh, Subdiv, Triangulate, Edge Split, and Element scale geo node into the
    geo node tree
    Geo Node type names found here
    https://docs.blender.org/api/current/bpy.types.GeometryNode.html
    """
    out_node = node_tree.nodes["Group Output"]

    node_x_location = 0
    node_location_step_x = 300
    
#transform 
    transform_node, node_x_location = create_node(node_tree, "GeometryNodeMeshTransform", node_x_location, node_location_step_x)
    transform_node.inputs["Rotation"].default_value = 0, 2, 0

#Geomtry Proximity 
    geometry_proximity, node_x_location = create_node(node_tree, "GeometryNodeProximity", node_x_location, node_location_step_x)
    geometry_proximity.target_element.default_value = 'faces'
    
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
    sample_nearest_surface.data_type = 'VECTOR'
    
#Align Euler to Vector 
    align_euler_to_vector, node_x_location = create_node(node_tree, "GeometryNodeScaleElements", node_x_location, node_location_step_x)
    align_euler_to_vector.axis = 'Z'
    
#BrainMesh 

#cutout 

    out_node.location.x = node_x_location

    link_nodes_by_mesh_socket(node_tree, from_node=mesh_cube_node, to_node=subdivide_mesh_node)
    link_nodes_by_mesh_socket(node_tree, from_node=subdivide_mesh_node, to_node=triangulate_node)
    link_nodes_by_mesh_socket(node_tree, from_node=triangulate_node, to_node=split_edges_node)

    from_node = split_edges_node
    to_node = scale_elements_node
    node_tree.links.new(from_node.outputs["Mesh"], to_node.inputs["Geometry"])

    from_node = scale_elements_node
    to_node = out_node
    node_tree.links.new(from_node.outputs["Geometry"], to_node.inputs["Geometry"])


def create_centerpiece():
    bpy.data.objects['headmesh'].select_set(True)

    bpy.ops.node.new_geometry_nodes_modifier()
    node_tree = bpy.data.node_groups["Geometry Nodes"]

    update_geo_node_tree(node_tree)

    


def main():
    """
    Python code to generate an animated geo nodes node tree
    that consists of a subdivided & triangulated cube with animated faces
    """
    scene_setup()
    create_centerpiece()


if __name__ == "__main__":
    main()
