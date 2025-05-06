import random
import time
from bpy.types import Operator, PropertyGroup, NodeLink
import bpy
from bpy.types import Operator, PropertyGroup, GeometryNode, FunctionNode, bpy_struct
from bpy import context
import bmesh


class geo_nodes(Operator):

    """Implement the geometry nodes"""

    bl_idname = "braincapgen.geo_nodes"
    bl_label = "Project 10-20 Landmarks"
    bl_description = (
        "Takes the LandmarkMesh and make cut outs at those locations on the head surface mesh"
    )
    bl_options = {"PRESET", "UNDO"}
    bl_space_type = "VIEW_3D"
    size_x: bpy.props.FloatProperty(name="cutout_x", default=3)
    size_y: bpy.props.FloatProperty(name="cutout_y", default=3)

    @staticmethod
    def link_nodes_by_mesh_socket(node_tree, from_node, to_node, type_from, type_to):
        node_tree.links.new(from_node.outputs[type_from], to_node.inputs[type_to])
        # create links between geometry nodes

    @staticmethod
    def create_node(node_tree, type_name, node_x_location, node_y_location, self):
        node_obj = node_tree.nodes.new(type=type_name)
        node_obj.location.x = node_x_location

        node_obj.location.y = node_y_location

        return node_obj, node_x_location

    @staticmethod
    def update_geo_node_tree(node_tree, self):
        # specify the i/o parameters
        out_node = node_tree.nodes["Group Output"]
        global in_node
        in_node = node_tree.nodes["Group Input"]
        node_x_location = 0
        node_location_step_x = 300

        # transform
        transform_node, node_x_location = self.create_node(
            node_tree, "GeometryNodeTransform", 350, 400, self
        )
        transform_node.inputs["Rotation"].default_value = 0, 0, 0

        # Geomtry Proximity
        geometry_proximity, node_x_location = self.create_node(
            node_tree, "GeometryNodeProximity", 600, 150, self
        )
        geometry_proximity.target_element = "FACES"

        # set position
        set_position, node_x_location = self.create_node(
            node_tree, "GeometryNodeSetPosition", 800, 200, self
        )
        set_position.inputs["Offset"].default_value = 0, 0, 0

        # instance on points
        instance_on_points, node_x_location = self.create_node(
            node_tree, "GeometryNodeInstanceOnPoints", 900, 0, self
        )
        global cutout_x, cutout_y
        cutout_x = self.size_x
        cutout_y = self.size_y
        instance_on_points.inputs["Scale"].default_value = cutout_x, cutout_y, 3

        # Mesh Boolean
        mesh_boolean, node_x_location = self.create_node(
            node_tree, "GeometryNodeMeshBoolean", 1500, 300, self
        )
        mesh_boolean.operation = "DIFFERENCE"

        # Sample Nearest Surface
        global sample_nearest_surface
        sample_nearest_surface, node_x_location = self.create_node(
            node_tree, "GeometryNodeSampleNearestSurface", 150, -300, self
        )
        sample_nearest_surface.data_type = "FLOAT_VECTOR"

        # Align Euler to Vector
        global align_euler_vector
        align_euler_vector, node_x_location = self.create_node(
            node_tree, "FunctionNodeAlignEulerToVector", 450, -300, self
        )
        align_euler_vector.axis = "Z"

        # BrainMesh
        object_info_brain, node_x_location = self.create_node(
            node_tree, "GeometryNodeObjectInfo", 150, 300, self
        )
        object_info_brain.inputs["Object"].default_value = bpy.data.objects["LandmarkMesh"]
        object_info_brain.transform_space = "RELATIVE"

        # cutout
        object_info_cutout, node_x_location = self.create_node(
            node_tree, "GeometryNodeObjectInfo", 500, 0, self
        )
        object_info_cutout.inputs["Object"].default_value = bpy.data.objects["cutout"]

        # normal
        global normal_node
        normal_node, node_x_location = self.create_node(
            node_tree, "GeometryNodeInputNormal", -50, -350, self
        )

        out_node.location.x = 1800
        out_node.location.y = 300

        ### linking nodes
        self.link_nodes_by_mesh_socket(
            node_tree,
            from_node=in_node,
            to_node=geometry_proximity,
            type_from="Geometry",
            type_to="Target",
        )
        self.link_nodes_by_mesh_socket(
            node_tree,
            from_node=geometry_proximity,
            to_node=set_position,
            type_from="Position",
            type_to="Position",
        )
        self.link_nodes_by_mesh_socket(
            node_tree,
            from_node=set_position,
            to_node=instance_on_points,
            type_from="Geometry",
            type_to="Points",
        )
        self.link_nodes_by_mesh_socket(
            node_tree,
            from_node=set_position,
            to_node=instance_on_points,
            type_from="Geometry",
            type_to="Points",
        )
        self.link_nodes_by_mesh_socket(
            node_tree,
            from_node=instance_on_points,
            to_node=mesh_boolean,
            type_from="Instances",
            type_to="Mesh 2",
        )
        self.link_nodes_by_mesh_socket(
            node_tree,
            from_node=transform_node,
            to_node=set_position,
            type_from="Geometry",
            type_to="Geometry",
        )
        self.link_nodes_by_mesh_socket(
            node_tree,
            from_node=object_info_cutout,
            to_node=instance_on_points,
            type_from="Geometry",
            type_to="Instance",
        )
        self.link_nodes_by_mesh_socket(
            node_tree,
            from_node=in_node,
            to_node=mesh_boolean,
            type_from="Geometry",
            type_to="Mesh 1",
        )
        self.link_nodes_by_mesh_socket(
            node_tree,
            from_node=in_node,
            to_node=sample_nearest_surface,
            type_from="Geometry",
            type_to="Mesh",
        )
        node_tree.links.new(
            object_info_brain.outputs["Geometry"], transform_node.inputs["Geometry"]
        )
        node_tree.links.new(in_node.outputs["Geometry"], sample_nearest_surface.inputs["Mesh"])
        node_tree.links.new(
            align_euler_vector.outputs["Rotation"],
            instance_on_points.inputs["Rotation"],
        )
        # final output
        node_tree.links.new(mesh_boolean.outputs["Mesh"], out_node.inputs["Geometry"])

    def execute(self, context):
        obj = bpy.context.object

        bpy.ops.object.mode_set(mode="EDIT")
        bm = bmesh.from_edit_mesh(obj.data)
        bpy.ops.mesh.select_all(action="DESELECT")

        # In order for the geometry nodes to treat the esh like a surfact mesh we need to delete a face
        # here, we are deleting the bottom most face since it will be removed anyways to make the cap-shape
        bottom_face = None
        min_z = float("inf")

        for face in bm.faces:
            avg_z = sum(v.co.z for v in face.verts) / len(face.verts)
            if avg_z < min_z:
                min_z = avg_z
                bottom_face = face

        if bottom_face:
            bottom_face.select_set(True)
            bmesh.ops.delete(bm, geom=[bottom_face], context="FACES")

        bmesh.update_edit_mesh(obj.data)

        # switch back to vertex mode (required for geometry nodes )
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)

        bpy.ops.object.mode_set(mode="OBJECT")

        # apply Geometry Nodes
        bpy.data.objects["headmesh"].select_set(True)
        # head = bpy.data.objects["headmesh"]

        head = bpy.context.scene.objects["headmesh"]
        bpy.context.view_layer.objects.active = head
        bpy.ops.object.mode_set(mode="EDIT")

        # make normals consistent and inward-facing
        bpy.ops.mesh.normals_make_consistent(inside=True)

        bpy.ops.object.mode_set(mode="OBJECT")

        bpy.context.view_layer.objects.active = head
        head.select_set(True)

        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = head
        head.select_set(True)
        bpy.ops.node.new_geometry_nodes_modifier()
        global node_tree
        node_tree = bpy.data.node_groups["Geometry Nodes"]

        self.update_geo_node_tree(node_tree, self)
        # Version control for Blender, Sample Nearest Surface

        major, minor, patch = bpy.app.version
        version_float = float(f"{major}.{minor}")

        if version_float >= 4.2:
            node_tree.links.new(normal_node.outputs["Normal"], sample_nearest_surface.inputs[1])
            node_tree.links.new(
                sample_nearest_surface.outputs[0], align_euler_vector.inputs["Vector"]
            )
        elif 3.4 <= version_float < 4.2:
            node_tree.links.new(normal_node.outputs["Normal"], sample_nearest_surface.inputs[3])
            node_tree.links.new(
                sample_nearest_surface.outputs[2], align_euler_vector.inputs["Vector"]
            )
        else:
            print(
                "This function requires Blender version 3.4 or newer. Please upgrade your Blender installation."
            )

        cuthide = bpy.data.objects["cutout"]
        cuthide.hide_set(True)
        brainhide = bpy.data.objects["LandmarkMesh"]
        brainhide.hide_set(True)
        # apply geoemtry nodes automaticcal- remove # from following line to modify
        bpy.ops.object.modifier_apply(modifier="GeometryNodes")
        try:
            bpy.ops.object.mode_set(mode="EDIT")
        except:
            pass
        # decrease number of faces
        bpy.ops.mesh.delete(type="FACE")
        bpy.ops.object.mode_set(mode="OBJECT")

        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
