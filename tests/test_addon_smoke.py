"""Smoke test for the NeuroCaptain Blender add-on.

Run via tests/run_tests.py inside Blender's bundled Python. Verifies the
add-on imports cleanly and its register()/unregister() lifecycle works, on
whichever OS/Blender version is running this script. It does not exercise
operators that need iso2mesh/pmcx/pmmc, since those are only imported
lazily when an operator actually runs.
"""

import unittest

import bpy

from _addon_helpers import import_addon, register_addon

# Representative operator classes to check after register(). Checked via
# each class's own bl_idname rather than hardcoded id strings, so this
# stays correct if an id is renamed.
OPERATOR_ATTR_NAMES = [
    "file_import",
    "decimate_mesh",
    "insert_shape",
    "select_model",
    "geo_nodes",
    "dual_mesh_NC",
    "cap_generation",
    "circumference_calc",
    "exportmesh",
    "customLandmarks",
]

SCENE_PROPERTY_NAMES = [
    "neurocaptain",
    "niifile",
    "neurocaptain_flexible_goal_weight",
]


class NeuroCaptainSmokeTest(unittest.TestCase):
    addon = None

    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()

    def test_register_creates_expected_operators_and_properties(self):
        register_addon(self.addon)
        try:
            for scene_prop in SCENE_PROPERTY_NAMES:
                self.assertTrue(
                    hasattr(bpy.types.Scene, scene_prop),
                    f"bpy.types.Scene.{scene_prop} was not registered",
                )

            for attr_name in OPERATOR_ATTR_NAMES:
                operator_cls = getattr(self.addon, attr_name)
                category, name = operator_cls.bl_idname.split(".")
                self.assertTrue(
                    hasattr(getattr(bpy.ops, category), name),
                    f"operator {operator_cls.bl_idname} was not registered",
                )
        finally:
            self.addon.unregister()
