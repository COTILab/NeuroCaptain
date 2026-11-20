"""Headless test for the head-mesh unit check and rescale button (head_units.py)."""

import unittest

import bpy

from _addon_helpers import import_addon, register_addon


class _FakeLayout:
    """Records what draw_unit_warning() puts in the panel."""

    def __init__(self):
        self.labels, self.operators, self.alert = [], [], False

    def box(self):
        return self

    def label(self, text="", icon="NONE"):
        self.labels.append(text)

    def operator(self, idname, text="", icon="NONE"):
        props = type("OpProps", (), {})()
        self.operators.append((idname, text, props))
        return props


class HeadUnitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()
        register_addon(cls.addon)
        cls.hu = __import__(f"{cls.addon.__name__}.head_units", fromlist=["head_units"])

    @classmethod
    def tearDownClass(cls):
        cls.addon.unregister()

    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        register_addon(self.addon)

    def _head(self, diameter, name="headmesh"):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=diameter / 2, location=(diameter * 0.05, 0, 0))
        obj = bpy.context.active_object
        obj.name = name
        return obj

    def test_suggested_scale(self):
        s = self.hu.suggested_scale
        self.assertIsNone(s(200.0))           # already mm
        self.assertEqual(s(420.0), 0.5)       # 0.5 mm voxel units (BrainWeb)
        self.assertEqual(s(20.0), 10.0)       # cm
        self.assertEqual(s(0.2), 1000.0)      # m
        self.assertEqual(s(5000.0), 0.0)      # nothing standard fits

    def test_no_warning_for_mm_head(self):
        self._head(200.0)
        layout = _FakeLayout()
        self.hu.draw_unit_warning(layout)
        self.assertEqual(layout.labels, [])

    def test_warning_and_button_rescale_voxel_head(self):
        head = self._head(420.0)
        landmarks = self._head(400.0, name="LandmarkMesh")
        layout = _FakeLayout()
        self.hu.draw_unit_warning(layout)
        self.assertTrue(layout.alert)
        self.assertEqual(len(layout.operators), 1)
        idname, text, props = layout.operators[0]
        self.assertEqual(props.factor, 0.5)

        x_before = head.location.x
        result = bpy.ops.neurocaptain.rescale_head_mesh(factor=props.factor)
        self.assertEqual(result, {"FINISHED"})
        bpy.context.view_layer.update()
        self.assertAlmostEqual(max(head.dimensions), 210.0, delta=1.0)
        self.assertAlmostEqual(max(landmarks.dimensions), 200.0, delta=1.0)
        self.assertAlmostEqual(head.location.x, x_before * 0.5, places=4)   # scaled about the origin
        self.assertEqual(tuple(head.scale), (1.0, 1.0, 1.0))                # applied, not object scale

        layout = _FakeLayout()
        self.hu.draw_unit_warning(layout)
        self.assertEqual(layout.labels, [], "warning should disappear after rescaling")

    def test_warns_for_imports_not_yet_renamed(self):
        nj = self._head(420.0, name="BrainWeb_Subject04_head_surf")
        nj["neurojson_dataset_key"] = "BrainWeb--Subject04--head--surf"
        self._head(20.0, name="importedmodel")
        self._head(10.0, name="Cylinder")            # unrelated small object: no warning
        layout = _FakeLayout()
        self.hu.draw_unit_warning(layout)
        targets = {props.target: props.factor for _, _, props in layout.operators}
        self.assertEqual(targets, {"BrainWeb_Subject04_head_surf": 0.5, "importedmodel": 10.0})

        bpy.ops.neurocaptain.rescale_head_mesh(factor=0.5, target="BrainWeb_Subject04_head_surf")
        bpy.context.view_layer.update()
        self.assertAlmostEqual(max(nj.dimensions), 210.0, delta=1.0)

    def test_landmarks_already_in_mm_are_left_alone(self):
        self._head(420.0)
        landmarks = self._head(200.0, name="LandmarkMesh")
        bpy.ops.neurocaptain.rescale_head_mesh(factor=0.5)
        bpy.context.view_layer.update()
        self.assertAlmostEqual(max(landmarks.dimensions), 200.0, delta=1.0)
