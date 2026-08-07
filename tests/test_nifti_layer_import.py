"""Tests for the NIfTI segmented-volume import path (see
test_flexible_layer_import.py for the sibling .mat/.jmsh importer).

Covers multi-file (one binary mask per file, role assigned per file) and
single-file (one multi-label volume, role assigned per label) entry points.
Most tests stub out iso2mesh.cgalv2m() with a canned 5-layer fixture to keep
focus on this add-on's own code; NiftiRealMeshingIntegrationTest runs the
real (unmocked) meshing call to confirm mesh conversion itself works too.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bpy
import numpy as np

from _addon_helpers import import_addon, register_addon
from test_flexible_layer_import import build_layered_sphere_mesh


def write_nifti_fixture(path, volume):
    # Uses the add-on's own vendored jdata (not a bare `import jdata`, which
    # may resolve to a differently-behaved pip-installed copy) so fixtures
    # are guaranteed compatible with what the add-on's own reader expects.
    vendored_jd = import_addon().layered_mesh_manager._vendored_jnifti()
    vendored_jd.savenifti(np.asarray(volume), str(path))


def _cleanup_layered_mesh_scene(addon):
    """Remove the LayeredModel_Visualization collection, headmesh, and any
    layer materials left behind by a layered-mesh-import test."""
    collection = bpy.data.collections.get("LayeredModel_Visualization")
    if collection is not None:
        for obj in list(collection.objects):
            mesh = obj.data if obj.type == "MESH" else None
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        bpy.data.collections.remove(collection)

    head = bpy.data.objects.get("headmesh")
    if head is not None:
        mesh = head.data
        bpy.data.objects.remove(head, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    for mat in [m for m in bpy.data.materials if m.name.endswith("_Mat")]:
        bpy.data.materials.remove(mat)

    addon.layered_mesh_manager.LAYERED_MESH.__init__()


def _canned_cgalv2m_mesh():
    """A 5-shell sphere with the pipeline's real fixed label convention
    (1=outermost/scalp .. 5=innermost/white matter), reusing the flexible
    importer test suite's own synthetic-mesh builder. Mimics cgalv2m()'s
    real (node, elem, face) return shape."""
    node, elem = build_layered_sphere_mesh(n_shells=5)
    return node, elem, np.array([])


class NiftiFilenameMatchingTest(unittest.TestCase):
    """Pure-logic tests for match_nifti_filename_to_tissue() - no bpy/files."""

    @classmethod
    def setUpClass(cls):
        cls.lmm = import_addon().layered_mesh_manager

    def test_matches_each_known_tissue(self):
        cases = {
            "scalp.nii": "scalp",
            "Skull_mask.nii.gz": "skull",
            "csf.nii": "csf",
            "gray_matter.nii": "gm",
            "greyMatter.nii.gz": "gm",
            "brain_wm.nii": "wm",
            "white_matter_seg.nii": "wm",
        }
        for filename, expected in cases.items():
            self.assertEqual(
                self.lmm.match_nifti_filename_to_tissue(filename), expected, filename
            )

    def test_unrecognized_filename_returns_none(self):
        self.assertIsNone(self.lmm.match_nifti_filename_to_tissue("scan001.nii"))


class NiftiRoleGuessTest(unittest.TestCase):
    """Pure-logic tests for guess_nifti_tissue_roles()."""

    @classmethod
    def setUpClass(cls):
        cls.lmm = import_addon().layered_mesh_manager

    def test_guesses_outer_to_inner_order(self):
        self.assertEqual(
            self.lmm.guess_nifti_tissue_roles([1, 2, 3]),
            {1: "scalp", 2: "skull", 3: "csf"},
        )

    def test_labels_beyond_five_default_to_other(self):
        guess = self.lmm.guess_nifti_tissue_roles([1, 2, 3, 4, 5, 6])
        self.assertEqual(guess[5], "white_matter")
        self.assertEqual(guess[6], "other")


class NiftiVoxelOrderRegressionTest(unittest.TestCase):
    """Guards against a repeat of the jnifti reshape axis-order bug, where
    reading a volume back scrambled its spatial layout."""

    @classmethod
    def setUpClass(cls):
        cls.lmm = import_addon().layered_mesh_manager

    def test_asymmetric_volume_round_trips_without_transposition(self):
        # Shape must be asymmetric and the marker off-center - a symmetric
        # cube looks the same after a C-order/F-order transposition.
        vol = np.zeros((3, 5, 7), dtype=np.uint8)
        vol[0, 1, 5] = 1
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "marker.nii"
            write_nifti_fixture(path, vol)
            read_back = self.lmm._read_nifti_volume(path)
        self.assertEqual(read_back.shape, (3, 5, 7))
        self.assertEqual(tuple(np.argwhere(read_back == 1)[0]), (0, 1, 5))


class NiftiSegDictBuilderTest(unittest.TestCase):
    """File-based tests for build_seg_dict_from_files_with_roles() and
    build_seg_dict_from_single_nifti() against real (tiny) .nii fixtures."""

    @classmethod
    def setUpClass(cls):
        cls.lmm = import_addon().layered_mesh_manager

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _path(self, name):
        return Path(self._tmpdir.name) / name

    def _mask(self, value):
        vol = np.zeros((4, 4, 4), dtype=np.uint8)
        vol[1:3, 1:3, 1:3] = value
        return vol

    def test_multi_file_builds_seg_dict_keyed_by_tissue(self):
        # Filenames are unrelated to tissue - roles come from file_roles only.
        roles = {"scalp": "a.nii", "skull": "b.nii", "csf": "c.nii",
                  "gray_matter": "d.nii", "white_matter": "e.nii"}
        file_roles = {}
        for role, name in roles.items():
            p = self._path(name)
            write_nifti_fixture(p, self._mask(1))
            file_roles[str(p)] = role

        seg = self.lmm.build_seg_dict_from_files_with_roles(list(file_roles.keys()), file_roles)
        self.assertEqual(set(seg.keys()), {"scalp", "skull", "csf", "gm", "wm"})
        for mask in seg.values():
            self.assertEqual(mask.dtype, bool)
            self.assertTrue(mask[1:3, 1:3, 1:3].all())

    def test_multi_file_missing_required_tissue_raises(self):
        paths = [self._path("a.nii"), self._path("b.nii")]
        for p in paths:
            write_nifti_fixture(p, self._mask(1))
        file_roles = {str(paths[0]): "scalp", str(paths[1]): "skull"}

        with self.assertRaises(ValueError) as ctx:
            self.lmm.build_seg_dict_from_files_with_roles([str(p) for p in paths], file_roles)
        self.assertIn("wm", str(ctx.exception))
        self.assertIn("gm", str(ctx.exception))

    def test_multi_file_same_role_assigned_twice_merges(self):
        # e.g. compact + spongy bone, both assigned "skull" - should OR together.
        paths = [self._path("a.nii"), self._path("b.nii"),
                 self._path("c.nii"), self._path("d.nii")]
        vol_a = self._mask(1)
        vol_b = np.zeros((4, 4, 4), dtype=np.uint8)
        vol_b[0, 0, 0] = 1  # disjoint from vol_a's region
        write_nifti_fixture(paths[0], vol_a)
        write_nifti_fixture(paths[1], vol_b)
        write_nifti_fixture(paths[2], self._mask(1))
        write_nifti_fixture(paths[3], self._mask(1))
        file_roles = {
            str(paths[0]): "skull", str(paths[1]): "skull",
            str(paths[2]): "gray_matter", str(paths[3]): "white_matter",
        }

        seg = self.lmm.build_seg_dict_from_files_with_roles([str(p) for p in paths], file_roles)
        self.assertTrue(seg["skull"][1:3, 1:3, 1:3].all())
        self.assertTrue(seg["skull"][0, 0, 0])

    def test_single_file_merges_labels_sharing_a_role(self):
        path = self._path("segmented.nii")
        vol = np.zeros((4, 4, 4), dtype=np.uint8)
        vol[1, 1, 1] = 4  # both mapped to gray_matter below
        vol[1, 1, 2] = 5
        vol[2, 2, 2] = 6  # mapped to white_matter
        write_nifti_fixture(path, vol)

        seg = self.lmm.build_seg_dict_from_single_nifti(
            path, {4: "gray_matter", 5: "gray_matter", 6: "white_matter"}
        )
        self.assertEqual(set(seg.keys()), {"gm", "wm"})
        self.assertTrue(seg["gm"][1, 1, 1] and seg["gm"][1, 1, 2])
        self.assertTrue(seg["wm"][2, 2, 2])

    def test_single_file_missing_required_role_raises(self):
        path = self._path("segmented.nii")
        vol = np.zeros((4, 4, 4), dtype=np.uint8)
        vol[1, 1, 1] = 1
        write_nifti_fixture(path, vol)

        with self.assertRaises(ValueError):
            self.lmm.build_seg_dict_from_single_nifti(path, {1: "scalp"})


class LoadNiftiSourceNeedsRolesTest(unittest.TestCase):
    """The single-file "ambiguous labels, ask before meshing" early return -
    real end-to-end (no cgalv2m stub needed: this path returns before ever
    calling it, by design). Still requires ISO2MESH_AVAILABLE, since that's
    checked before the early return."""

    @classmethod
    def setUpClass(cls):
        cls.lmm = import_addon().layered_mesh_manager

    def setUp(self):
        if not self.lmm.ISO2MESH_AVAILABLE:
            self.skipTest(
                "iso2mesh not available in this Python environment (known local-only "
                "quirk on some machines - see reference_neurocaptain_local_blender_testing)"
            )
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_single_file_without_roles_asks_before_meshing(self):
        path = Path(self._tmpdir.name) / "segmented.nii"
        vol = np.zeros((4, 4, 4), dtype=np.uint8)
        vol[1, 1, 1] = 1
        vol[2, 2, 2] = 3
        vol[3, 3, 3] = 2
        write_nifti_fixture(path, vol)

        with patch.object(self.lmm.i2m, "cgalv2m") as fake:
            result = self.lmm.load_layered_mesh_source_from_nifti([path])
            fake.assert_not_called()

        self.assertEqual(result, {"needs_roles": True, "unique_labels": [1, 2, 3]})


class NiftiToLayeredMeshBridgeTest(unittest.TestCase):
    """Full pipeline (real bpy scene + Blender object creation) from NIfTI
    file(s) through to imported layer objects, with cgalv2m()'s native
    meshing stubbed out - see module docstring for why."""

    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()
        register_addon(cls.addon)

    @classmethod
    def tearDownClass(cls):
        cls.addon.unregister()

    def setUp(self):
        lmm = self.addon.layered_mesh_manager
        if not lmm.ISO2MESH_AVAILABLE:
            self.skipTest(
                "iso2mesh not available in this Python environment (known local-only "
                "quirk on some machines - see reference_neurocaptain_local_blender_testing)"
            )
        self._tmpdir = tempfile.TemporaryDirectory()
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1)
        self.head = bpy.context.active_object
        self.head.name = "headmesh"
        self._cgalv2m_patch = patch.object(
            lmm.i2m, "cgalv2m",
            side_effect=lambda volume, opt, maxvol: _canned_cgalv2m_mesh(),
        )
        self.fake_cgalv2m = self._cgalv2m_patch.start()

    def tearDown(self):
        self._cgalv2m_patch.stop()
        self._tmpdir.cleanup()
        _cleanup_layered_mesh_scene(self.addon)

    def _path(self, name):
        return Path(self._tmpdir.name) / name

    def _mask(self, value=1):
        vol = np.zeros((4, 4, 4), dtype=np.uint8)
        vol[1:3, 1:3, 1:3] = value
        return vol

    def test_multi_file_needs_file_roles_then_completes_after_role_assignment(self):
        lmm = self.addon.layered_mesh_manager
        # Distinct, non-overlapping regions per file - identical regions
        # would let the last-applied (innermost) tissue silently overwrite
        # the others when combined.
        regions = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 0), (0, 3, 3)]
        paths = []
        for name, pos in zip(("a.nii", "b.nii", "c.nii", "d.nii", "e.nii"), regions):
            vol = np.zeros((4, 4, 4), dtype=np.uint8)
            vol[pos] = 1
            p = self._path(name)
            write_nifti_fixture(p, vol)
            paths.append(str(p))

        first = lmm.import_layered_head_model_from_nifti(paths, reference_obj_name="headmesh")
        self.assertTrue(first.get("needs_file_roles"))
        self.assertEqual(set(first["filenames"]), set(paths))
        self.fake_cgalv2m.assert_not_called()

        file_roles = {
            paths[0]: "scalp", paths[1]: "skull", paths[2]: "csf",
            paths[3]: "gray_matter", paths[4]: "white_matter",
        }
        result = lmm.import_layered_head_model_from_nifti(
            paths, file_roles=file_roles, reference_obj_name="headmesh")

        self.assertTrue(result["success"], result["message"])
        combined_volume = self.fake_cgalv2m.call_args[0][0]
        self.assertEqual(set(np.unique(combined_volume).tolist()) - {0}, {1, 2, 3, 4, 5})
        self.assertEqual(lmm.LAYERED_MESH.num_layers, 5)
        self.assertIsNotNone(bpy.data.objects.get("Layer1_Scalp"))
        self.assertIsNotNone(bpy.data.objects.get("Layer5_White_Matter"))

    def test_single_file_needs_roles_then_completes_after_role_assignment(self):
        lmm = self.addon.layered_mesh_manager
        path = self._path("segmented.nii")
        vol = np.zeros((4, 4, 4), dtype=np.uint8)
        vol[1, 1, 1] = 1
        vol[2, 2, 2] = 4
        vol[3, 3, 3] = 5
        write_nifti_fixture(path, vol)

        first = lmm.import_layered_head_model_from_nifti([str(path)], reference_obj_name="headmesh")
        self.assertTrue(first.get("needs_roles"))
        self.fake_cgalv2m.assert_not_called()

        second = lmm.import_layered_head_model_from_nifti(
            [str(path)], label_roles={1: "scalp", 4: "gray_matter", 5: "white_matter"},
            reference_obj_name="headmesh",
        )
        self.assertTrue(second["success"], second["message"])
        self.fake_cgalv2m.assert_called_once()
        self.assertEqual(lmm.LAYERED_MESH.num_layers, 5)

    def test_define_nifti_roles_operator_completes_import(self):
        lmm = self.addon.layered_mesh_manager
        path = self._path("segmented.nii")
        vol = np.zeros((4, 4, 4), dtype=np.uint8)
        vol[1, 1, 1] = 1
        vol[2, 2, 2] = 4
        vol[3, 3, 3] = 5
        write_nifti_fixture(path, vol)

        result = bpy.ops.neurocaptain.define_nifti_roles(
            filepath=str(path),
            reference_obj_name="headmesh",
            label_ids="1,4,5",
            role_0="scalp", role_1="gray_matter", role_2="white_matter",
        )

        self.assertEqual(result, {"FINISHED"})
        self.assertEqual(lmm.LAYERED_MESH.num_layers, 5)
        self.assertIsNotNone(bpy.data.objects.get("Layer1_Scalp"))

    def test_missing_required_tissue_reports_error_without_meshing(self):
        lmm = self.addon.layered_mesh_manager
        paths = []
        for name in ("a.nii", "b.nii"):
            p = self._path(name)
            write_nifti_fixture(p, self._mask())
            paths.append(str(p))
        file_roles = {paths[0]: "scalp", paths[1]: "skull"}  # gm/wm deliberately unassigned

        result = lmm.import_layered_head_model_from_nifti(
            paths, file_roles=file_roles, reference_obj_name="headmesh")

        self.assertFalse(result["success"])
        self.assertIn("wm", result["message"])
        self.fake_cgalv2m.assert_not_called()


def _build_nested_box_segmentation(shape=(20, 24, 28)):
    """5 concentric box shells, canonical outer-to-inner label order
    (1=scalp .. 5=white matter) - small enough for real cgalv2m() to mesh
    in a couple seconds."""
    vol = np.zeros(shape, dtype=np.uint8)
    for label, margin in enumerate([1, 3, 5, 7, 9], start=1):
        sl = tuple(slice(margin, s - margin) for s in shape)
        vol[sl] = label
    return vol


class NiftiRealMeshingIntegrationTest(unittest.TestCase):
    """Same pipeline as NiftiToLayeredMeshBridgeTest but with real cgalv2m()
    meshing (not mocked) - the actual "does mesh conversion work" check."""

    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()
        register_addon(cls.addon)

    @classmethod
    def tearDownClass(cls):
        cls.addon.unregister()

    def setUp(self):
        lmm = self.addon.layered_mesh_manager
        if not lmm.ISO2MESH_AVAILABLE:
            self.skipTest(
                "iso2mesh not available in this Python environment (known local-only "
                "quirk on some machines - see reference_neurocaptain_local_blender_testing)"
            )
        self._tmpdir = tempfile.TemporaryDirectory()
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1)
        self.head = bpy.context.active_object
        self.head.name = "headmesh"

    def tearDown(self):
        self._tmpdir.cleanup()
        _cleanup_layered_mesh_scene(self.addon)

    def test_real_meshing_produces_five_real_layers(self):
        lmm = self.addon.layered_mesh_manager
        path = Path(self._tmpdir.name) / "segmented.nii"
        write_nifti_fixture(path, _build_nested_box_segmentation())

        result = lmm.import_layered_head_model_from_nifti(
            [str(path)],
            label_roles={1: "scalp", 2: "skull", 3: "csf", 4: "gray_matter", 5: "white_matter"},
            reference_obj_name="headmesh",
        )

        if not result["success"] and ("CERTIFICATE_VERIFY_FAILED" in result["message"]
                                       or "urlopen error" in result["message"]):
            # iso2mesh fetches its native mesh-tool binaries on first use;
            # some Blender-bundled Python builds lack a working CA cert
            # bundle for that HTTPS download - an environment limitation,
            # not a code regression.
            self.skipTest(f"iso2mesh binary download failed in this environment: {result['message']}")

        self.assertTrue(result["success"], result["message"])
        self.assertEqual(lmm.LAYERED_MESH.num_layers, 5)
        for name in ("Layer1_Scalp", "Layer2_Skull", "Layer3_CSF",
                     "Layer4_Gray_Matter", "Layer5_White_Matter"):
            obj = bpy.data.objects.get(name)
            self.assertIsNotNone(obj, name)
            self.assertGreater(len(obj.data.vertices), 0, name)
            self.assertGreater(len(obj.data.polygons), 0, name)


if __name__ == "__main__":
    unittest.main()
