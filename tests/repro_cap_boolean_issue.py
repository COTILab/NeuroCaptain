"""Standalone repro script for the Blender-3.6-only cap-gen boolean face
count bug (CI: 302,514 faces vs the ~247,161 expected on windows-latest/
ubuntu-latest 3.6, but NOT macos-latest 3.6, and NOT any other version).

Not a test_*.py file on purpose - tests/run_tests.py's discover(pattern=
"test_*.py") won't pick this up, so it can't interfere with the real suite.

Usage (run twice, compare the two "FINAL FACE COUNT" lines):

    blender --background --python tests/repro_cap_boolean_issue.py
    blender --background --python tests/repro_cap_boolean_issue.py -- --override-boolean

The first form matches exactly what CI runs today (only the initial
select_model(ADD_HEADMESH) call gets a VIEW_3D context override - see
test_cap_generation_pipeline.py's docstring for why that one call needs it).

The second form additionally wraps ONLY the cap_generation(BOOLEAN_CUT) call
- not the whole pipeline - in the same kind of temp_override. This is
deliberately narrow: wrapping the *entire* pipeline in a persistent override
was already tried in CI and caused a native SIGSEGV (exit code 139) on
Blender 4.2, almost certainly from some later step (decimate_mesh/dual_mesh/
geo_nodes/wireframe/remesh/export_mesh/circumference) touching real
GPU-backed viewport/region state that `--background` can't provide - the
same class of crash this test suite's docstring already documents for
duplicate_move()'s gizmo callback. Only BOOLEAN_CUT itself is a plausible
candidate for actually needing a context (it's the step that internally
calls bpy.ops.object.mode_set()/editmode_toggle()/mesh.select_all()/
mesh.delete()), so isolate it before touching CI again.

Prints face counts and bpy.context state at every stage, matching
test_cap_generation_pipeline.py's real operator sequence.
"""

import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from _addon_helpers import (
    REPO_ROOT,
    get_view3d_area_and_region,
    import_addon,
    register_addon,
    select_nearest_vertex,
)

import bpy

# --- args after Blender's own `--` separator -------------------------------
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OVERRIDE_BOOLEAN = "--override-boolean" in argv

HEAD_MODEL_PATH = os.path.join(REPO_ROOT, "HeadModels", "NDD_35-39Years_scalp.bmsh")
LANDMARK_PATH = os.path.join(REPO_ROOT, "ScalpLandmarks", "NDD_35-39_landmarks.jmsh")
DECIMATE_RATIO = 0.05
EXPECTED_BOOLEAN_CUT_FACE_COUNT = 247161
BOOLEAN_CUT_FACE_COUNT_TOLERANCE = 0.05

CREATED_OBJECT_NAMES = [
    "headmesh", "headmesh.001", "headcopy", "cube_meas", "LandmarkMesh",
    "cutout", "face_cutout", "bottom_cutout", "ear_cutout", "importedmodel",
]
CREATED_NODE_GROUP_NAMES = ["Geometry Nodes", "DualMeshNodeTree"]


def log_stage(head, stage):
    dims = tuple(round(d, 2) for d in head.dimensions)
    print(
        f"[repro] {stage}: dimensions={dims}mm "
        f"verts={len(head.data.vertices)} faces={len(head.data.polygons)}"
    )


def log_context(stage):
    ctx = bpy.context
    print(
        f"[repro][context] {stage}: mode={ctx.mode} "
        f"area={ctx.area.type if ctx.area else None} "
        f"region={ctx.region.type if ctx.region else None} "
        f"active_object={ctx.active_object.name if ctx.active_object else None}"
    )


def cleanup():
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


def main():
    print(f"[repro] Blender version: {bpy.app.version_string}")
    print(f"[repro] --override-boolean: {OVERRIDE_BOOLEAN}")

    addon = import_addon()
    register_addon(addon)
    view3d_area, view3d_region = get_view3d_area_and_region()
    if view3d_area is None:
        print("[repro] FATAL: no VIEW_3D area found in factory-startup screen")
        sys.exit(1)

    cleanup()  # in case a previous run in the same session left state behind

    # select_model's ADD_HEADMESH path calls view3d.snap_selected_to_cursor(),
    # which hard-fails ("context is incorrect") in --background without this.
    with bpy.context.temp_override(area=view3d_area, region=view3d_region):
        result = bpy.ops.braincapgen.select_model(
            action="ADD_HEADMESH",
            filepath=HEAD_MODEL_PATH,
            files=[{"name": os.path.basename(HEAD_MODEL_PATH)}],
        )
    assert result == {"FINISHED"}, f"select_model(ADD_HEADMESH) -> {result}"
    head = bpy.data.objects["headmesh"]
    log_stage(head, "after select_model(ADD_HEADMESH)")
    log_context("after select_model(ADD_HEADMESH)")

    result = bpy.ops.braincapgen.select_model(
        action="ADD_BRAIN1020MESH",
        filepath=LANDMARK_PATH,
        files=[{"name": os.path.basename(LANDMARK_PATH)}],
    )
    assert result == {"FINISHED"}, f"select_model(ADD_BRAIN1020MESH) -> {result}"
    landmark_obj = bpy.data.objects["LandmarkMesh"]

    labels = list(landmark_obj.get("landmark_labels") or [])
    nz_index = labels.index("Nz")
    nz_world = landmark_obj.matrix_world @ landmark_obj.data.vertices[nz_index].co
    print(f"[repro] Nz landmark world position: {tuple(round(c, 2) for c in nz_world)}")

    result = bpy.ops.braincapgen.decimate_mesh(number=DECIMATE_RATIO)
    assert result == {"FINISHED"}, f"decimate_mesh -> {result}"
    head = bpy.data.objects["headmesh"]
    log_stage(head, "after decimate_mesh(0.05)")

    result = bpy.ops.object.dual_mesh()
    assert result == {"FINISHED"}, f"dual_mesh -> {result}"
    head = bpy.data.objects["headmesh"]
    log_stage(head, "after dual_mesh()")

    result = bpy.ops.braincapgen.insert_shape(action="ADD_CYLINDER")
    assert result == {"FINISHED"}, f"insert_shape -> {result}"

    result = bpy.ops.braincapgen.geo_nodes()
    assert result == {"FINISHED"}, f"geo_nodes -> {result}"
    head = bpy.data.objects["headmesh"]
    log_stage(head, "after geo_nodes()")

    nearest_index = select_nearest_vertex(head, nz_world)
    nearest_world = head.matrix_world @ head.data.vertices[nearest_index].co
    print(
        f"[repro] nearest headmesh vertex to Nz: "
        f"{tuple(round(c, 2) for c in nearest_world)} "
        f"(distance={(nearest_world - nz_world).length:.2f}mm)"
    )

    result = bpy.ops.braincapgen.cap_generation(action="PLACE_CUTOUTS", add_cylinder=True)
    assert result == {"FINISHED"}, f"cap_generation(PLACE_CUTOUTS) -> {result}"
    for name in ("face_cutout", "bottom_cutout", "ear_cutout"):
        assert bpy.data.objects.get(name) is not None, f"{name} was not created"

    pre_boolean_z = head.dimensions[2]
    log_context("before cap_generation(BOOLEAN_CUT)")

    if OVERRIDE_BOOLEAN:
        print("[repro] running BOOLEAN_CUT WITH a VIEW_3D context override")
        with bpy.context.temp_override(area=view3d_area, region=view3d_region):
            result = bpy.ops.braincapgen.cap_generation(action="BOOLEAN_CUT", thick=2, voxel=0.5)
    else:
        print("[repro] running BOOLEAN_CUT with NO context override (matches current CI)")
        result = bpy.ops.braincapgen.cap_generation(action="BOOLEAN_CUT", thick=2, voxel=0.5)

    assert result == {"FINISHED"}, f"cap_generation(BOOLEAN_CUT) -> {result}"
    head = bpy.data.objects["headmesh"]
    log_stage(head, "after cap_generation(BOOLEAN_CUT)")
    log_context("after cap_generation(BOOLEAN_CUT)")

    final_face_count = len(head.data.polygons)
    lower = EXPECTED_BOOLEAN_CUT_FACE_COUNT * (1 - BOOLEAN_CUT_FACE_COUNT_TOLERANCE)
    upper = EXPECTED_BOOLEAN_CUT_FACE_COUNT * (1 + BOOLEAN_CUT_FACE_COUNT_TOLERANCE)
    in_range = lower <= final_face_count <= upper
    z_shrank = head.dimensions[2] < pre_boolean_z * 0.9

    print("")
    print("=" * 70)
    print(f"[repro] FINAL FACE COUNT: {final_face_count}")
    print(f"[repro] expected range: [{lower:.0f}, {upper:.0f}] (reference: {EXPECTED_BOOLEAN_CUT_FACE_COUNT})")
    print(f"[repro] within tolerance: {in_range}")
    print(f"[repro] Z dimension before/after BOOLEAN_CUT: {pre_boolean_z:.1f}mm -> {head.dimensions[2]:.1f}mm "
          f"(shrank >=10%: {z_shrank})")
    print("=" * 70)

    for name in ("face_cutout", "bottom_cutout", "ear_cutout"):
        leftover = bpy.data.objects.get(name)
        print(f"[repro] {name} still in scene after BOOLEAN_CUT: {leftover is not None} "
              f"(should be False - it should've been consumed/deleted)")

    cleanup()


if __name__ == "__main__":
    main()
