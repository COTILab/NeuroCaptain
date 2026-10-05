"""Tests for the N-layer ("flexible") mesh import feature added alongside the
standard fixed 5-layer import (see test_light_simulation.py for that one).

Covers, in increasing levels of integration:
  - LayerNameHelpersTest / SidecarLayerJsonTest: pure-logic unit tests for the
    private helpers layered_mesh_manager uses to discover layer names/roles
    embedded in a mesh file or a sidecar JSON.
  - LoadLayeredMeshSourceTest: load_layered_mesh_source() end-to-end against
    real .mat (scipy) and .jmsh (jdata) fixtures, exercising both the
    "names found" and "names not found anywhere" paths.
  - FlexibleLayeredImportTest: the real, scriptable execute() path of
    import_layered_head_model_flexible() and the two new operators
    (NEUROCAPTAIN_OT_import_layered_mesh_flexible,
    NEUROCAPTAIN_OT_define_layers), using a synthetic 3-layer solid ball
    (same Delaunay-tetrahedralization technique as test_light_simulation.py's
    5-layer fixture, just with fewer shells and custom, non-default roles to
    prove this path isn't hardcoded to the 5-layer scalp/skull/csf/gray/white
    convention).

NEUROCAPTAIN_OT_define_layers.invoke() (the "ask the user to name each
layer" dialog) is modal/UI-only and has no scriptable equivalent, so - like
move_optode/rigid_rotate_optodes in test_optode_features.py - it's out of
scope here. Its execute() (what actually runs once the dialog's OK is
clicked) is fully scriptable and is exercised directly below, by calling the
operator with the name_N/role_N properties pre-set instead of going through
invoke_props_dialog.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import bpy
import numpy as np
import jdata as jd
from scipy.io import savemat
from scipy.spatial import Delaunay

from _addon_helpers import import_addon, register_addon


def _fibonacci_shell(radius, n_points):
    indices = np.arange(n_points)
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.stack([x, y, z], axis=1)


def build_layered_sphere_mesh(n_shells, max_radius=0.1, points_per_shell=50):
    """Return (node[N,3] float64, elem[M,5] int32) for a synthetic solid ball
    with n_shells concentric tissue labels (1=outermost .. n_shells=innermost).

    Same technique as test_light_simulation.py's 5-layer fixture (real
    Delaunay tetrahedralization, labeled by each tet centroid's radial band),
    generalized to an arbitrary layer count to prove the flexible importer
    isn't hardcoded to 5 layers.
    """
    shells = [
        _fibonacci_shell(max_radius * (i + 1) / n_shells, points_per_shell)
        for i in range(n_shells)
    ]
    node = np.vstack(shells + [np.zeros((1, 3))]).astype(np.float64)

    tets = Delaunay(node).simplices.astype(np.int64)
    centroid_radius = np.linalg.norm(node[tets].mean(axis=1), axis=1)

    # band_index increases with radius (0=innermost); label = n_shells -
    # band_index maps outermost band -> label 1, innermost -> label n_shells.
    band_edges = np.linspace(0.0, max_radius, n_shells + 1)
    band_index = np.clip(np.digitize(centroid_radius, band_edges[1:-1]), 0, n_shells - 1)
    tissue_labels = (n_shells - band_index).astype(np.int64)

    elem = np.column_stack([tets + 1, tissue_labels]).astype(np.int32)  # 1-indexed verts
    return node, elem


def write_layered_mat_fixture(path, n_shells, layer_names=None):
    """Write a .mat fixture. If layer_names is given, embed it under the
    'layer_name' struct field (an embedded-name mesh); otherwise the struct
    only has node/elem, matching a mesh with no embedded layer names."""
    node, elem = build_layered_sphere_mesh(n_shells)
    mesh_struct = {"node": node, "elem": elem}
    if layer_names is not None:
        mesh_struct["layer_name"] = np.array(layer_names, dtype=object)
    savemat(str(path), {"mesh": mesh_struct})


def write_layered_jmesh_fixture(path, n_shells, layer_names=None):
    node, elem = build_layered_sphere_mesh(n_shells)
    data = {"MeshNode": node, "MeshElem": elem}
    if layer_names is not None:
        data["TissueName"] = list(layer_names)
    jd.save(data, str(path))


class LayerNameHelpersTest(unittest.TestCase):
    """Pure-logic tests for the name-list/definition-building helpers - no
    bpy scene or files involved, just plain Python/NumPy in, dict out."""

    @classmethod
    def setUpClass(cls):
        cls.lmm = import_addon().layered_mesh_manager

    def test_coerce_name_list_from_plain_list(self):
        self.assertEqual(self.lmm._coerce_name_list(["Scalp", "Skull"]), ["Scalp", "Skull"])

    def test_coerce_name_list_from_object_array_of_arrays(self):
        # Mirrors what scipy.io.loadmat actually hands back for a MATLAB
        # cell array of strings: an object array whose elements are each a
        # size-1 numpy array wrapping the string.
        raw = np.array(
            [np.array(["Scalp"]), np.array(["Skull"])], dtype=object
        )
        self.assertEqual(self.lmm._coerce_name_list(raw), ["Scalp", "Skull"])

    def test_coerce_name_list_rejects_blank_entry(self):
        self.assertIsNone(self.lmm._coerce_name_list(["Scalp", "  ", "CSF"]))

    def test_names_to_definitions_builds_role_from_lowercased_name(self):
        defs = self.lmm._names_to_definitions(["Gray Matter", "White Matter"], [4, 5])
        self.assertEqual(
            defs,
            {
                4: {"name": "Gray Matter", "role": "gray_matter"},
                5: {"name": "White Matter", "role": "white_matter"},
            },
        )

    def test_names_to_definitions_returns_none_when_too_few_names(self):
        self.assertIsNone(self.lmm._names_to_definitions(["Scalp"], [1, 2]))

    def test_extract_embedded_layer_names_is_case_insensitive(self):
        # field_lookup uses a differently-cased key than the candidate list.
        field_lookup = {"LayerName": ["Scalp", "Skull"]}
        defs = self.lmm._extract_embedded_layer_names(
            field_lookup, self.lmm._MAT_LAYER_NAME_FIELDS, [1, 2]
        )
        self.assertEqual(defs[1]["name"], "Scalp")
        self.assertEqual(defs[2]["name"], "Skull")

    def test_extract_embedded_layer_names_returns_none_when_no_field_matches(self):
        field_lookup = {"some_other_field": [1, 2, 3]}
        defs = self.lmm._extract_embedded_layer_names(
            field_lookup, self.lmm._MAT_LAYER_NAME_FIELDS, [1, 2]
        )
        self.assertIsNone(defs)


class SidecarLayerJsonTest(unittest.TestCase):
    """File-based tests for _load_sidecar_layer_json()."""

    @classmethod
    def setUpClass(cls):
        cls.lmm = import_addon().layered_mesh_manager

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.mesh_path = Path(self._tmpdir.name) / "sample.mat"

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_string_entries_keyed_by_label(self):
        sidecar = self.mesh_path.with_suffix(".json")
        sidecar.write_text(json.dumps({"1": "Scalp", "2": "Skull"}))

        defs = self.lmm._load_sidecar_layer_json(self.mesh_path, [1, 2])
        self.assertEqual(
            defs,
            {1: {"name": "Scalp", "role": "scalp"}, 2: {"name": "Skull", "role": "skull"}},
        )

    def test_dict_entries_with_explicit_role(self):
        sidecar = self.mesh_path.with_suffix(".json")
        sidecar.write_text(json.dumps({"1": {"name": "Scalp", "role": "outer_skin"}}))

        defs = self.lmm._load_sidecar_layer_json(self.mesh_path, [1])
        self.assertEqual(defs, {1: {"name": "Scalp", "role": "outer_skin"}})

    def test_layers_wrapper_key_is_supported(self):
        sidecar = self.mesh_path.with_suffix(".json")
        sidecar.write_text(json.dumps({"layers": {"1": "Scalp", "2": "Skull"}}))

        defs = self.lmm._load_sidecar_layer_json(self.mesh_path, [1, 2])
        self.assertEqual(defs[1]["name"], "Scalp")
        self.assertEqual(defs[2]["name"], "Skull")

    def test_stem_layers_json_variant_is_found(self):
        sidecar = self.mesh_path.with_name(self.mesh_path.stem + "_layers.json")
        sidecar.write_text(json.dumps({"1": "Scalp"}))

        defs = self.lmm._load_sidecar_layer_json(self.mesh_path, [1])
        self.assertEqual(defs[1]["name"], "Scalp")

    def test_partial_coverage_returns_none(self):
        sidecar = self.mesh_path.with_suffix(".json")
        sidecar.write_text(json.dumps({"1": "Scalp"}))  # missing label 2

        self.assertIsNone(self.lmm._load_sidecar_layer_json(self.mesh_path, [1, 2]))

    def test_missing_sidecar_returns_none(self):
        self.assertIsNone(self.lmm._load_sidecar_layer_json(self.mesh_path, [1, 2]))


class LoadLayeredMeshSourceTest(unittest.TestCase):
    """load_layered_mesh_source() end-to-end against real .mat/.jmsh fixtures."""

    @classmethod
    def setUpClass(cls):
        cls.lmm = import_addon().layered_mesh_manager

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _path(self, name):
        return Path(self._tmpdir.name) / name

    def test_mat_with_embedded_names(self):
        path = self._path("layered.mat")
        write_layered_mat_fixture(path, n_shells=3, layer_names=["Scalp", "Skull", "CSF"])

        source = self.lmm.load_layered_mesh_source(path)
        self.assertEqual(source["unique_labels"], [1, 2, 3])
        self.assertEqual(
            source["layer_definitions"],
            {
                1: {"name": "Scalp", "role": "scalp"},
                2: {"name": "Skull", "role": "skull"},
                3: {"name": "CSF", "role": "csf"},
            },
        )

    def test_mat_without_names_or_sidecar_has_no_layer_definitions(self):
        path = self._path("layered.mat")
        write_layered_mat_fixture(path, n_shells=3, layer_names=None)

        source = self.lmm.load_layered_mesh_source(path)
        self.assertEqual(source["unique_labels"], [1, 2, 3])
        self.assertIsNone(source["layer_definitions"])

    def test_mat_falls_back_to_sidecar_json_when_no_embedded_names(self):
        path = self._path("layered.mat")
        write_layered_mat_fixture(path, n_shells=2, layer_names=None)
        path.with_suffix(".json").write_text(json.dumps({"1": "Scalp", "2": "Brain"}))

        source = self.lmm.load_layered_mesh_source(path)
        self.assertEqual(source["layer_definitions"][1]["name"], "Scalp")
        self.assertEqual(source["layer_definitions"][2]["name"], "Brain")

    def test_jmesh_with_embedded_names(self):
        path = self._path("layered.jmsh")
        write_layered_jmesh_fixture(path, n_shells=3, layer_names=["Scalp", "Skull", "CSF"])

        source = self.lmm.load_layered_mesh_source(path)
        self.assertEqual(source["unique_labels"], [1, 2, 3])
        self.assertEqual(source["layer_definitions"][1]["name"], "Scalp")
        self.assertEqual(source["layer_definitions"][3]["name"], "CSF")

    def test_unsupported_suffix_raises(self):
        path = self._path("layered.txt")
        path.write_text("not a mesh")
        with self.assertRaises(ValueError):
            self.lmm.load_layered_mesh_source(path)


class FlexibleLayeredImportTest(unittest.TestCase):
    """Full pipeline: import_layered_head_model_flexible() and the two new
    operators, driven against a synthetic 3-layer solid ball (fewer than the
    standard 5 layers, with custom roles) via iso2mesh's real volface."""

    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()
        register_addon(cls.addon)

    @classmethod
    def tearDownClass(cls):
        cls.addon.unregister()

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1)
        self.head = bpy.context.active_object
        self.head.name = "headmesh"

    def tearDown(self):
        self._tmpdir.cleanup()

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

        self.addon.layered_mesh_manager.LAYERED_MESH.__init__()

    def _path(self, name):
        return Path(self._tmpdir.name) / name

    def test_import_layered_head_model_flexible_with_explicit_definitions(self):
        lmm = self.addon.layered_mesh_manager
        path = self._path("layered.mat")
        write_layered_mat_fixture(path, n_shells=3, layer_names=None)
        layer_definitions = {
            1: {"name": "Scalp", "role": "scalp"},
            2: {"name": "Skull", "role": "skull"},
            3: {"name": "Brain", "role": "gray_matter"},
        }

        result = lmm.import_layered_head_model_flexible(
            path, reference_obj_name="headmesh", layer_definitions=layer_definitions
        )

        self.assertTrue(result["success"], result["message"])
        self.assertEqual(lmm.LAYERED_MESH.num_layers, 3)

        coll = bpy.data.collections.get("LayeredModel_Visualization")
        self.assertIsNotNone(coll)
        for obj_name in ("Layer1_Scalp", "Layer2_Skull", "Layer3_Brain"):
            obj = bpy.data.objects.get(obj_name)
            self.assertIsNotNone(obj, f"{obj_name} was not created")
            self.assertIn(obj, list(coll.objects))
            self.assertGreater(len(obj.data.polygons), 0)

        self.assertIs(lmm.LAYERED_MESH.head_surface_obj, bpy.data.objects["Layer1_Scalp"])
        self.assertIs(lmm.LAYERED_MESH.cortex_obj, bpy.data.objects["Layer3_Brain"])

    def test_import_layered_head_model_flexible_defaults_unresolved_layers(self):
        # No layer_definitions at all: every layer must still import, using
        # "Layer {label}" / role 'other' defaults rather than failing.
        lmm = self.addon.layered_mesh_manager
        path = self._path("layered.mat")
        write_layered_mat_fixture(path, n_shells=2, layer_names=None)

        result = lmm.import_layered_head_model_flexible(path, reference_obj_name="headmesh")

        self.assertTrue(result["success"], result["message"])
        self.assertEqual(lmm.LAYERED_MESH.num_layers, 2)
        self.assertIsNotNone(bpy.data.objects.get("Layer1_Layer_1"))
        self.assertIsNotNone(bpy.data.objects.get("Layer2_Layer_2"))
        # no role matched 'scalp' anywhere -> falls back to the lowest label
        self.assertIs(lmm.LAYERED_MESH.head_surface_obj, bpy.data.objects["Layer1_Layer_1"])
        # no role in brain_roles -> no cortex object resolved
        self.assertIsNone(lmm.LAYERED_MESH.cortex_obj)

    def test_operator_imports_directly_when_names_are_embedded(self):
        # With embedded layer names present, the operator must import
        # immediately and never fall through to the naming-prompt operator.
        path = self._path("layered.mat")
        write_layered_mat_fixture(path, n_shells=3, layer_names=["Scalp", "Skull", "Gray Matter"])

        result = bpy.ops.neurocaptain.import_layered_mesh_flexible(filepath=str(path))

        self.assertEqual(result, {"FINISHED"})
        lmm = self.addon.layered_mesh_manager
        self.assertEqual(lmm.LAYERED_MESH.num_layers, 3)
        self.assertIsNotNone(bpy.data.objects.get("Layer1_Scalp"))
        self.assertIsNotNone(bpy.data.objects.get("Layer3_Gray_Matter"))

    def test_define_layers_operator_execute_imports_with_chosen_names(self):
        # Exercises NEUROCAPTAIN_OT_define_layers.execute() directly - the
        # part of the "ask the user to name each layer" flow that runs once
        # the (UI-only, unscriptable) props dialog's OK has been clicked.
        path = self._path("layered.mat")
        write_layered_mat_fixture(path, n_shells=2, layer_names=None)

        result = bpy.ops.neurocaptain.define_layers(
            mesh_path=str(path),
            reference_obj_name="headmesh",
            label_ids="1,2",
            name_0="Scalp", role_0="scalp",
            name_1="Brain", role_1="gray_matter",
        )

        self.assertEqual(result, {"FINISHED"})
        lmm = self.addon.layered_mesh_manager
        self.assertEqual(lmm.LAYERED_MESH.num_layers, 2)
        self.assertIsNotNone(bpy.data.objects.get("Layer1_Scalp"))
        self.assertIsNotNone(bpy.data.objects.get("Layer2_Brain"))
        self.assertIs(lmm.LAYERED_MESH.cortex_obj, bpy.data.objects["Layer2_Brain"])

    def test_define_layers_operator_uses_custom_role_text(self):
        path = self._path("layered.mat")
        write_layered_mat_fixture(path, n_shells=1, layer_names=None)

        result = bpy.ops.neurocaptain.define_layers(
            mesh_path=str(path),
            reference_obj_name="headmesh",
            label_ids="1",
            name_0="", role_0="custom", custom_0="Silicone Phantom",
        )

        self.assertEqual(result, {"FINISHED"})
        lmm = self.addon.layered_mesh_manager
        self.assertEqual(lmm.LAYERED_MESH.layer_info[1]["role"], "silicone_phantom")
        self.assertEqual(lmm.LAYERED_MESH.layer_info[1]["name"], "Silicone Phantom")
