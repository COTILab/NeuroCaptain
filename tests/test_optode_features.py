"""Headless integration test for the optode-placement pipeline.

Uses a synthetic "headmesh" (a UV sphere - the pipeline only needs *an*
object named/containing "headmesh", not a real head model) and drives the
real, scriptable execute() path of each operator:

    add_source / add_detector  -> place a Source_N/Detector_N disc with a
                                   SHRINKWRAP constraint onto the head mesh,
                                   using a pre-selected vertex position
                                   (NEUROCAPTAIN_OT_add_source.execute() ->
                                   _OptodeMixin.get_optode_positions()'s
                                   OBJECT-mode branch, which reads
                                   mesh.vertices[i].select flags - this is
                                   the real, scriptable placement path;
                                   move_optode/rigid_rotate_optodes are
                                   modal-only and have no such fallback, so
                                   they're out of scope here)
    ensure_optode_constraints  -> idempotent shrinkwrap-constraint repair
    create_optode_connections_delaunay -> Delaunay-triangulated edges
                                   between all placed optodes

Placing each optode at a distinct, pre-selected vertex (rather than calling
add_source/add_detector repeatedly with nothing selected, which would stack
every optode on the same "highest point" fallback) gives Delaunay
triangulation a non-degenerate point set to work with.
"""

import unittest

import bpy

from _addon_helpers import import_addon, register_addon

# Spread-out vertex indices on the default UV sphere (482 verts: 32 segments
# x 16 rings + 2 poles) so the resulting optode positions are non-coincident.
SOURCE_VERTEX_INDICES = [0, 120]
DETECTOR_VERTEX_INDICES = [240, 360]


class OptodeFeaturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()
        register_addon(cls.addon)

    @classmethod
    def tearDownClass(cls):
        cls.addon.unregister()

    def setUp(self):
        bpy.ops.mesh.primitive_uv_sphere_add()
        self.head = bpy.context.active_object
        self.head.name = "headmesh"

    def tearDown(self):
        for prefix in ("Source_", "Detector_"):
            for obj in [o for o in bpy.data.objects if o.name.startswith(prefix)]:
                mesh = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if mesh.users == 0:
                    bpy.data.meshes.remove(mesh)

        connections = bpy.data.objects.get("Optode_Connections")
        if connections is not None:
            mesh = connections.data
            bpy.data.objects.remove(connections, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)

        for collection_name in ("Sources", "Detectors"):
            collection = bpy.data.collections.get(collection_name)
            if collection is not None:
                bpy.data.collections.remove(collection)

        head = bpy.data.objects.get("headmesh")
        if head is not None:
            mesh = head.data
            bpy.data.objects.remove(head, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)

        for mat in [m for m in bpy.data.materials if m.name.endswith("_Material")]:
            bpy.data.materials.remove(mat)

    def _select_only_vertex(self, index):
        for vert in self.head.data.vertices:
            vert.select = False
        self.head.data.vertices[index].select = True

    def _add_optode(self, bl_idname, vertex_index):
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = self.head
        self.head.select_set(True)
        self._select_only_vertex(vertex_index)
        operator = getattr(bpy.ops.neurocaptain, bl_idname)
        result = operator()
        self.assertEqual(result, {"FINISHED"})

    def test_add_source_and_detector_creates_shrinkwrapped_optodes(self):
        for index in SOURCE_VERTEX_INDICES:
            self._add_optode("add_source", index)
        for index in DETECTOR_VERTEX_INDICES:
            self._add_optode("add_detector", index)

        sources = sorted(o.name for o in bpy.data.objects if o.name.startswith("Source_"))
        detectors = sorted(o.name for o in bpy.data.objects if o.name.startswith("Detector_"))
        self.assertEqual(len(sources), len(SOURCE_VERTEX_INDICES))
        self.assertEqual(len(detectors), len(DETECTOR_VERTEX_INDICES))

        for name in sources + detectors:
            obj = bpy.data.objects[name]
            shrinkwraps = [c for c in obj.constraints if c.type == "SHRINKWRAP"]
            self.assertEqual(len(shrinkwraps), 1, f"{name} should have exactly one shrinkwrap")
            self.assertEqual(shrinkwraps[0].target, self.head)

        # positions must actually be distinct, not all stacked at one fallback point
        positions = {tuple(round(c, 6) for c in bpy.data.objects[n].location) for n in sources + detectors}
        self.assertEqual(len(positions), len(sources) + len(detectors))

        result = bpy.ops.neurocaptain.ensure_optode_constraints()
        self.assertEqual(result, {"FINISHED"})
        for name in sources + detectors:
            obj = bpy.data.objects[name]
            shrinkwraps = [c for c in obj.constraints if c.type == "SHRINKWRAP"]
            self.assertEqual(len(shrinkwraps), 1, f"{name} should still have exactly one shrinkwrap")

        result = bpy.ops.neurocaptain.create_optode_connections_delaunay()
        self.assertEqual(result, {"FINISHED"})
        connections = bpy.data.objects.get("Optode_Connections")
        self.assertIsNotNone(connections, "Delaunay connection object was not created")
        self.assertGreater(len(connections.data.edges), 0)
