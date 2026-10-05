"""Shared helper for locating and importing the NeuroCaptain add-on package
from within Blender's bundled Python when running tests via
`blender --background --python tests/run_tests.py`.
"""

import importlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADDON_PACKAGE_NAME = os.path.basename(REPO_ROOT)


def import_addon():
    parent_dir = os.path.dirname(REPO_ROOT)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    return importlib.import_module(ADDON_PACKAGE_NAME)


def register_addon(addon):
    """Register the add-on, defensively unregistering any stale state first.

    All test classes share one Blender process (tests/run_tests.py discovers
    and runs every test_*.py file in a single `blender --background` launch),
    so if an earlier test's teardown/unregister() didn't fully complete, its
    leftover registered classes would make every later test's register()
    fail with "already registered" - a single bug cascading into failures
    across unrelated test files. Clearing first keeps each test class
    independent regardless of what ran before it.
    """
    try:
        addon.unregister()
    except Exception:
        pass
    addon.register()


def select_nearest_vertex(obj, world_pos):
    """Select (object mode) only the mesh vertex on `obj` nearest to a given
    world-space position.

    Scriptable equivalent of manually clicking the vertex closest to some
    reference point (e.g. a landmark) in the viewport. Matches the plain
    object-mode `vertex.select = True` pattern already used by
    test_optode_features.py's `_select_only_vertex`, which several operators
    in this add-on read directly (no edit-mode round-trip required).
    """
    from mathutils import Vector

    target = Vector(world_pos)
    inv = obj.matrix_world.inverted()
    local_target = inv @ target

    nearest_index = None
    nearest_dist = None
    for vert in obj.data.vertices:
        vert.select = False
        dist = (vert.co - local_target).length
        if nearest_dist is None or dist < nearest_dist:
            nearest_dist = dist
            nearest_index = vert.index

    obj.data.vertices[nearest_index].select = True
    return nearest_index


def get_view3d_area_and_region():
    """Find a VIEW_3D area/region in the current screen.

    Several operators (e.g. bpy.ops.view3d.snap_selected_to_cursor, used by
    headmodels.py's add_headmesh) poll() against an active VIEW_3D area and
    fail with "context is incorrect" in plain `blender --background` mode,
    even though the default factory-startup screen does have one. Wrap the
    call in `with bpy.context.temp_override(area=area, region=region):`.
    """
    import bpy

    screen = bpy.context.screen
    area = next((a for a in screen.areas if a.type == "VIEW_3D"), None)
    if area is None:
        return None, None
    region = next((r for r in area.regions if r.type == "WINDOW"), None)
    return area, region
