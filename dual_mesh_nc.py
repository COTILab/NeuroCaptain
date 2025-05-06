import bpy
from bpy.types import Operator


class dual_mesh_NC(Operator):
    bl_idname = "object.dual_mesh"
    bl_label = "Convert to Dual Mesh"
    bl_description = "Convert a generic mesh into a polygonal mesh"
    bl_options = {"REGISTER", "UNDO"}

    def link_nodes_by_mesh_socket(self, node_tree, from_node, to_node, type_from, type_to):
        """Link one node's socket to another."""
        node_tree.links.new(from_node.outputs[type_from], to_node.inputs[type_to])

    def create_node(self, node_tree, type_name, node_x_location, node_y_location):
        """Create a new node and return it along with its position."""
        node = node_tree.nodes.new(type=type_name)
        node.location = (node_x_location, node_y_location)
        return node, node_x_location

    def ensure_geometry_sockets(self, node_tree):
        interface = node_tree.interface

        # see if geometry socket is missing
        if not any(
            socket.name == "Geometry" and socket.in_out == "INPUT"
            for socket in interface.items_tree
        ):
            new_input = interface.new_socket(
                name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
            )

        # make sure there is geometry socket
        if not any(
            socket.name == "Geometry" and socket.in_out == "OUTPUT"
            for socket in interface.items_tree
        ):
            new_output = interface.new_socket(
                name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
            )

        # input/output nodes
        in_node = node_tree.nodes.get("Group Input")
        out_node = node_tree.nodes.get("Group Output")
        if not in_node:
            in_node = node_tree.nodes.new("NodeGroupInput")
        if not out_node:
            out_node = node_tree.nodes.new("NodeGroupOutput")

        return in_node, out_node

    def update_geo_node_tree(self, node_tree):
        """Update the geometry node tree, adding and connecting necessary nodes."""
        # definie i/o nodes
        in_node, out_node = self.ensure_geometry_sockets(node_tree)

        # clear any other ndoes
        for node in node_tree.nodes:
            if node.name not in ["Group Input", "Group Output"]:
                node_tree.nodes.remove(node)

        # dual mesh node
        dualmesh, _ = self.create_node(node_tree, "GeometryNodeDualMesh", 0, 0)

        # link nodes
        self.link_nodes_by_mesh_socket(node_tree, in_node, dualmesh, "Geometry", "Mesh")
        self.link_nodes_by_mesh_socket(node_tree, dualmesh, out_node, "Dual Mesh", "Geometry")

    def execute(self, context):
        """Execute the operator: Set up nodes, apply modifier, and finalize the mesh."""
        # check Blender version
        if bpy.app.version < (3, 4, 0):
            self.report({"ERROR"}, "This add-on requires Blender 3.4 or newer.")
            return {"CANCELLED"}

        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass

        bpy.ops.object.select_all(action="DESELECT")
        head = bpy.data.objects.get("headmesh")
        if not head:
            self.report({"ERROR"}, 'Object "headmesh" not found.')
            return {"CANCELLED"}

        head.select_set(True)
        context.view_layer.objects.active = head

        # geometry nodes modifier
        mod = head.modifiers.new(name="GeometryNodes", type="NODES")

        if not mod.node_group:
            mod.node_group = bpy.data.node_groups.new("DualMeshNodeTree", "GeometryNodeTree")

        node_tree = mod.node_group

        # define i/o sockets
        major, minor, _ = bpy.app.version

        in_node = node_tree.nodes.get("Group Input")
        if not in_node:
            in_node = node_tree.nodes.new("NodeGroupInput")

        out_node = node_tree.nodes.get("Group Output")
        if not out_node:
            out_node = node_tree.nodes.new("NodeGroupOutput")

        if (major, minor) >= (3, 5):
            interface = node_tree.interface
            if not any(s.name == "Geometry" and s.in_out == "INPUT" for s in interface.items_tree):
                interface.new_socket(
                    name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
                )
            if not any(s.name == "Geometry" and s.in_out == "OUTPUT" for s in interface.items_tree):
                interface.new_socket(
                    name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
                )
        else:
            if "Geometry" not in node_tree.inputs:
                node_tree.inputs.new("NodeSocketGeometry", "Geometry")
            if "Geometry" not in node_tree.outputs:
                node_tree.outputs.new("NodeSocketGeometry", "Geometry")

        # ensure no previous nodes
        for node in node_tree.nodes:
            if node.name not in ["Group Input", "Group Output"]:
                node_tree.nodes.remove(node)

        # create dual mesh
        dualmesh = node_tree.nodes.new("GeometryNodeDualMesh")
        dualmesh.location = (0, 0)

        node_tree.links.new(in_node.outputs["Geometry"], dualmesh.inputs["Mesh"])
        node_tree.links.new(dualmesh.outputs["Dual Mesh"], out_node.inputs["Geometry"])

        # apply geonodes
        bpy.ops.object.modifier_apply(modifier=mod.name)

        return {"FINISHED"}
