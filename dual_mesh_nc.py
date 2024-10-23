import bpy
from bpy.types import Operator
from bpy.props import (
    BoolProperty,
    EnumProperty,
)
import bmesh
from .utils import *
from mesh_tissue.dual_mesh import dual_mesh_tessellated


class dual_mesh_NC(Operator):
    bl_idname = "object.dual_mesh"
    bl_label = "Convert to Dual Mesh"
    bl_description = "Convert a generic mesh into a polygonal mesh. (Destructive)"
    bl_options = {"REGISTER", "UNDO"}

    @staticmethod
    def link_nodes_by_mesh_socket(node_tree, from_node, to_node, type_from, type_to):
        node_tree.links.new(from_node.outputs[type_from], to_node.inputs[type_to])
        # create links between geometry nodes

    @staticmethod
    def create_node(node_tree, type_name, node_x_location, node_y_location, self):
        # Creates a node of a given type, and sets/updates the location of the node on the X axis.
        # Returning the node object and the next location on the X axis for the next node.

        node_obj = node_tree.nodes.new(type=type_name)
        node_obj.location.x = node_x_location

        node_obj.location.y = node_y_location

        return node_obj, node_x_location

    @staticmethod
    def update_geo_node_tree(node_tree, self):
        # add the specific geometry nodes and input parameters
        out_node = node_tree.nodes["Group Output"]
        global in_node
        in_node = node_tree.nodes["Group Input"]
        node_x_location = 0
        node_location_step_x = 300

        # ADDING NODES IN NODE EDITOR SPACE
        # transform
        dualmesh, node_x_location = self.create_node(
            node_tree, "GeometryNodeDualMesh", 350, 400, self
        )
        # transform_node.inputs["Geometry"].default_value = 0, 0, 0

        self.link_nodes_by_mesh_socket(
            node_tree,
            from_node=in_node,
            to_node=dualmesh,
            type_from="Geometry",
            type_to="Mesh",
        )

        self.link_nodes_by_mesh_socket(
            node_tree,
            from_node=dualmesh,
            to_node=out_node,
            type_from="Dual Mesh",
            type_to="Geometry",
        )

        # node_tree.links.new(dualmesh.outputs["DualMesh"], out_node.inputs["Geometry"])

    def execute(self, context):
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass
        # decrease number of faces
        bpy.ops.object.select_all(action="DESELECT")
        head = bpy.data.objects["headmesh"]
        # bpy.ops.object.select_all(action="DESELECT")
        head.select_set(True)
        bpy.context.view_layer.objects.active = head
        bpy.ops.node.new_geometry_nodes_modifier()
        global node_tree
        node_tree = bpy.data.node_groups["Geometry Nodes"]

        self.update_geo_node_tree(node_tree, self)
        bpy.ops.object.modifier_apply(modifier="GeometryNodes")
        return {"FINISHED"}
