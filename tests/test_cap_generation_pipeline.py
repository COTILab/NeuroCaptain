"""Headless integration test for the cap-generation pipeline.

Drives the real operator chain a user follows to turn a head model into a
3D-printable cap, using the NDD_35-39Years sample data (a stand-in for the
30-34 age bracket, which isn't in the repo) and production-realistic
parameters (0.05 decimate ratio) instead of the lenient stand-in values a
smoke test would use:

    select_model(ADD_HEADMESH)      -> headmesh, headmesh.001
    select_model(ADD_BRAIN1020MESH) -> LandmarkMesh (precomputed landmarks)
    insert_shape(ADD_CYLINDER)      -> circular cutout
    geo_nodes()                     -> "project cutouts": cuts landmark
                                        holes into headmesh
    decimate_mesh(0.05)             -> "head density 0.05"
    select nearest headmesh vertex to Nz  -> scriptable equivalent of
                                        manually clicking the reference
                                        vertex in the viewport
    cap_generation(PLACE_CUTOUTS, BOOLEAN_CUT) -> wireframe cap shape
    dual_mesh()                     -> polygonal dual mesh
    export_mesh()                   -> writes a .jmsh file
    circumference()                 -> estimate cap circumference

This intentionally keeps the *order* proven to work by earlier iterations of
this test (geo_nodes/decimate before the boolean cut, dual_mesh last) since
that's what each operator's real object/state dependencies require, while
using the realistic decimate ratio and head model that exposed the cap-gen
regression this test suite failed to catch. It also checks more than "the
operator returned FINISHED" at each stage: face/vertex counts must actually
change where a step is supposed to change them, the final mesh must not be
riddled with non-manifold edges, and the exported mesh's physical size and
measured circumference must be plausible for a human head.

brain1020mesh.py's interactive 5-point (Nz/Iz/Lpa/Rpa/Cz) picking has no
scriptable equivalent and needs iso2mesh, so it's deliberately skipped by
loading the precomputed NDD_35-39_landmarks.jmsh landmark set instead - this
is the only realistic headless path through this pipeline.

select_model's ADD_HEADMESH path internally calls
bpy.ops.view3d.snap_selected_to_cursor(), which fails in plain
`blender --background` mode with "context is incorrect" unless wrapped in
a temp_override supplying a VIEW_3D area/region (verified empirically -
the default factory-startup screen does have one even headless).

add_headmesh() used to call bpy.ops.object.duplicate_move(...) (a macro
combining OBJECT_OT_duplicate + TRANSFORM_OT_translate) to create the
hidden headmesh.001 duplicate. That crashed the whole process with a native
EXCEPTION_ACCESS_VIOLATION/SIGSEGV inside Blender's transform/gizmo
machinery (ED_region_draw_cb_activate expects a real, GPU-initialized
viewport region that plain `--background` mode doesn't have) - reproduced
directly, isolated to this exact call, and confirmed it is NOT a Python
exception (temp_override can't fix a native crash). CI confirmed it on both
3.4 and 4.2. add_headmesh() now duplicates via bpy.data (obj.copy()) and
sets .location directly instead, which has no viewport dependency and no
longer crashes on any version, so this test no longer needs a version skip.
"""

import os
import unittest

import bmesh
import bpy

from _addon_helpers import (
    REPO_ROOT,
    get_view3d_area_and_region,
    import_addon,
    register_addon,
    select_nearest_vertex,
)

HEAD_MODEL_PATH = os.path.join(REPO_ROOT, "HeadModels", "NDD_35-39Years_scalp.bmsh")
LANDMARK_PATH = os.path.join(REPO_ROOT, "BrainLandmarks", "NDD_35-39_landmarks.jmsh")

# Actual production decimate ratio - the current suite used 0.5 (lenient),
# which never exercised the aggressive-decimation path most likely to break
# the boolean-cut/dual-mesh steps downstream.
DECIMATE_RATIO = 0.05

# Ground truth measured manually (Blender's own circumference feature) for
# this exact head model + landmark pair, in the mesh's native (mm-scale)
# blender units - the scene's unit label says "METERS" but that's just
# Blender's scene-property default string, not an actual unit conversion.
EXPECTED_CIRCUMFERENCE_MM = 560.244
CIRCUMFERENCE_TOLERANCE = 0.10  # +/- 10%

# A real human head circumference implies a roughly-head-sized bounding box;
# this catches "cap gen produced a degenerate sliver" style bugs that a bare
# non-empty-array check would miss.
MIN_PLAUSIBLE_HEAD_BBOX_DIAGONAL_MM = 100.0
MAX_PLAUSIBLE_HEAD_BBOX_DIAGONAL_MM = 350.0

# Aggressive decimation + boolean cuts + wireframe/remesh is exactly where a
# broken cap-gen pipeline produces non-manifold garbage; voxel remesh should
# normally hand back an (almost) fully closed/manifold shell.
MAX_NON_MANIFOLD_EDGE_RATIO = 0.05

CREATED_OBJECT_NAMES = [
    "headmesh",
    "headmesh.001",
    "headcopy",
    "cube_meas",
    "LandmarkMesh",
    "cutout",
    "face_cutout",
    "bottom_cutout",
    "ear_cutout",
    "importedmodel",
]
CREATED_NODE_GROUP_NAMES = ["Geometry Nodes", "DualMeshNodeTree"]


def _non_manifold_edge_ratio(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    total = len(bm.edges)
    non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    return (non_manifold / total) if total else 0.0


def _bbox_diagonal(vertices):
    mins = [min(v[axis] for v in vertices) for axis in range(3)]
    maxs = [max(v[axis] for v in vertices) for axis in range(3)]
    return sum((maxs[axis] - mins[axis]) ** 2 for axis in range(3)) ** 0.5


class CapGenerationPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()
        register_addon(cls.addon)
        cls.view3d_area, cls.view3d_region = get_view3d_area_and_region()

    @classmethod
    def tearDownClass(cls):
        cls.addon.unregister()

    def setUp(self):
        self.export_filename = "test_cap_export.jmsh"

    def tearDown(self):
        for name in CREATED_OBJECT_NAMES:
            obj = bpy.data.objects.get(name)
            if obj is None:
                continue
            mesh = obj.data if obj.type == "MESH" else None
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)

        for name in CREATED_NODE_GROUP_NAMES:
            node_group = bpy.data.node_groups.get(name)
            if node_group is not None:
                bpy.data.node_groups.remove(node_group)

        export_path = os.path.join(self._export_dir(), self.export_filename)
        if os.path.exists(export_path):
            os.remove(export_path)

        for scene_key in ("saved_nz", "vselect", "nz_assigned", "last_circumference_mm"):
            if scene_key in bpy.context.scene:
                del bpy.context.scene[scene_key]

    def _export_dir(self):
        return self.addon.utils.GetBPWorkFolder()

    @staticmethod
    def _landmark_world_position(landmark_obj, label):
        labels = list(landmark_obj.get("landmark_labels") or [])
        assert label in labels, f"landmark set should embed a {label} label"
        index = labels.index(label)
        local = landmark_obj.data.vertices[index].co
        return landmark_obj.matrix_world @ local

    def test_full_pipeline_produces_a_valid_cap_mesh(self):
        self.assertIsNotNone(
            self.view3d_area, "no VIEW_3D area found in factory-startup screen"
        )

        with bpy.context.temp_override(area=self.view3d_area, region=self.view3d_region):
            result = bpy.ops.braincapgen.select_model(
                action="ADD_HEADMESH",
                filepath=HEAD_MODEL_PATH,
                files=[{"name": os.path.basename(HEAD_MODEL_PATH)}],
            )
        self.assertEqual(result, {"FINISHED"})
        head = bpy.data.objects.get("headmesh")
        head_dup = bpy.data.objects.get("headmesh.001")
        self.assertIsNotNone(head)
        self.assertIsNotNone(head_dup)

        result = bpy.ops.braincapgen.select_model(
            action="ADD_BRAIN1020MESH",
            filepath=LANDMARK_PATH,
            files=[{"name": os.path.basename(LANDMARK_PATH)}],
        )
        self.assertEqual(result, {"FINISHED"})
        landmark_obj = bpy.data.objects.get("LandmarkMesh")
        self.assertIsNotNone(landmark_obj)

        nz_world = self._landmark_world_position(landmark_obj, "Nz")

        # --- circle cutout + project onto the head surface -----------------
        pre_cutout_vert_count = len(head.data.vertices)
        pre_cutout_face_count = len(head.data.polygons)

        result = bpy.ops.braincapgen.insert_shape(action="ADD_CYLINDER")
        self.assertEqual(result, {"FINISHED"})
        self.assertIsNotNone(bpy.data.objects.get("cutout"))

        result = bpy.ops.braincapgen.geo_nodes()
        self.assertEqual(result, {"FINISHED"})
        head = bpy.data.objects["headmesh"]
        self.assertGreater(len(head.data.vertices), 0)
        self.assertNotEqual(
            (len(head.data.vertices), len(head.data.polygons)),
            (pre_cutout_vert_count, pre_cutout_face_count),
            "geo_nodes() should have carved cutouts into headmesh, not left it unchanged",
        )

        # --- head density -----------------------------------------------
        pre_decimate_face_count = len(head.data.polygons)
        result = bpy.ops.braincapgen.decimate_mesh(number=DECIMATE_RATIO)
        self.assertEqual(result, {"FINISHED"})
        head = bpy.data.objects["headmesh"]
        post_decimate_face_count = len(head.data.polygons)
        self.assertGreater(
            post_decimate_face_count, 50,
            "decimate_mesh(0.05) should not collapse the head to near-nothing",
        )
        self.assertLess(
            post_decimate_face_count, pre_decimate_face_count,
            "decimate_mesh(0.05) should actually reduce the face count",
        )

        # --- choose the reference vertex closest to Nz --------------------
        # Scriptable equivalent of manually clicking the vertex nearest the
        # Nz landmark in the viewport, done on the now-decimated headmesh
        # (matches the real workflow: density is set before the reference
        # point is picked).
        select_nearest_vertex(head, nz_world)

        result = bpy.ops.braincapgen.cap_generation(action="PLACE_CUTOUTS", add_cylinder=True)
        self.assertEqual(result, {"FINISHED"})
        for name in ("face_cutout", "bottom_cutout", "ear_cutout"):
            self.assertIsNotNone(bpy.data.objects.get(name), f"{name} was not created")

        result = bpy.ops.braincapgen.cap_generation(action="BOOLEAN_CUT", thick=2, voxel=0.5)
        self.assertEqual(result, {"FINISHED"})
        head = bpy.data.objects["headmesh"]
        self.assertGreater(len(head.data.vertices), 0)
        self.assertGreater(len(head.data.polygons), 0)

        non_manifold_ratio = _non_manifold_edge_ratio(head)
        self.assertLess(
            non_manifold_ratio, MAX_NON_MANIFOLD_EDGE_RATIO,
            f"cap mesh has {non_manifold_ratio:.1%} non-manifold edges after "
            f"boolean cut + wireframe + remesh - likely broken topology",
        )

        result = bpy.ops.object.dual_mesh()
        self.assertEqual(result, {"FINISHED"})
        head = bpy.data.objects["headmesh"]
        self.assertGreater(len(head.data.vertices), 0)
        self.assertGreater(
            len(head.data.polygons), 20,
            "dual mesh conversion produced an implausibly low polygon count",
        )

        bpy.context.view_layer.objects.active = head
        result = bpy.ops.braincapgen.export_mesh(filename=self.export_filename)
        self.assertEqual(result, {"FINISHED"})

        export_path = os.path.join(self._export_dir(), self.export_filename)
        self.assertTrue(os.path.exists(export_path), f"export file not found at {export_path}")

        import jdata as jd

        exported = jd.load(export_path)
        self.assertIn("MeshVertex3", exported)
        self.assertIn("MeshTri3", exported)
        self.assertGreater(len(exported["MeshVertex3"]), 0)
        self.assertGreater(len(exported["MeshTri3"]), 0)

        bbox_diagonal = _bbox_diagonal(exported["MeshVertex3"])
        self.assertTrue(
            MIN_PLAUSIBLE_HEAD_BBOX_DIAGONAL_MM <= bbox_diagonal <= MAX_PLAUSIBLE_HEAD_BBOX_DIAGONAL_MM,
            f"exported cap bounding-box diagonal ({bbox_diagonal:.1f}mm) is not "
            f"plausible for a human head cap",
        )

        # --- circumference validity check ----------------------------------
        # headmesh.001 is the untouched, full-resolution duplicate created by
        # select_model(ADD_HEADMESH) - independently find its own vertex
        # nearest to Nz (its vertex indices don't correspond to the decimated
        # "headmesh" used above).
        head_dup = bpy.data.objects["headmesh.001"]
        select_nearest_vertex(head_dup, nz_world)
        result = bpy.ops.neurocaptain.circumference()
        self.assertEqual(result, {"FINISHED"})

        circumference_mm = bpy.context.scene.get("last_circumference_mm")
        self.assertIsNotNone(circumference_mm, "circumference operator did not report a result")

        lower_bound = EXPECTED_CIRCUMFERENCE_MM * (1 - CIRCUMFERENCE_TOLERANCE)
        upper_bound = EXPECTED_CIRCUMFERENCE_MM * (1 + CIRCUMFERENCE_TOLERANCE)
        self.assertTrue(
            lower_bound <= circumference_mm <= upper_bound,
            f"estimated circumference {circumference_mm}mm is outside +/-"
            f"{CIRCUMFERENCE_TOLERANCE:.0%} of the expected {EXPECTED_CIRCUMFERENCE_MM}mm "
            f"(bounds: [{lower_bound:.1f}, {upper_bound:.1f}])",
        )
