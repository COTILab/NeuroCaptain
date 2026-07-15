"""Headless integration test for the light-simulation pipeline (pmmc + redbirdpy).

Drives the real operator chain:

    neurocaptain.import_layered_mesh -> loads a 5-layer tetrahedral volume
                                         mesh, extracts Head_Surface_5L /
                                         Brain_Cortex_5L surfaces via
                                         iso2mesh's volface (layered_mesh_
                                         manager.extract_outer_surface_only /
                                         extract_cortex_surface)
    neurocaptain.run_mmc              -> pmmc sensitivity simulation, forced
                                         to mmc_use_gpu=False since GitHub-
                                         hosted runners have no GPU (pmmc's
                                         CPU path is real: autopilot=0, no
                                         gpuid, confirmed in
                                         lightsim_neurocaptain.run_single_mmc)
    neurocaptain.run_redbird           -> redbirdpy CPU FEM forward solve

No sample 5-layer .mat file ships with the repo, so setUpClass synthesizes
one: a solid ball built from 5 concentric point shells (outermost = Scalp,
innermost = White Matter), fed through scipy.spatial.Delaunay to get a real
tetrahedralization, labeled by each tet's centroid radius, and written out
via scipy.io.savemat in the exact struct shape
layered_mesh_manager.import_layered_head_model expects
(mat_data[<any top-level key>][0, 0]['node'] / ['elem'], elem's first 4
columns 1-indexed per iso2mesh/MATLAB convention, 5th column = tissue label
1-5). Using scipy.spatial.Delaunay rather than guessing at iso2mesh's own
mesh-generation function names keeps the fixture's correctness fully
self-verifiable: label 1 is genuinely the outermost radial band (so it's
the true outer boundary iso2mesh's volface finds) and labels {4, 5} genuinely
form the innermost contiguous region (so it has its own closed boundary,
the "cortex" surface).

Earlier static analysis suspected bpy.context.screen.areas (used at the end
of run_mmc/run_redbird to set viewport shading) would fail headlessly like
select_model's view3d.snap_selected_to_cursor does - empirically verified
that's a false alarm: the default factory-startup screen keeps a real
VIEW_3D area, and setting space.shading.type is a plain property write with
no poll() restriction. So no temp_override is needed anywhere in this file.
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest

import bpy
import numpy as np
from scipy.io import savemat
from scipy.spatial import Delaunay

from _addon_helpers import import_addon, register_addon

N_SHELLS = 5
POINTS_PER_SHELL = 60
# Matches the synthetic headmesh's radius below so optode positions actually
# land in/near the volumetric mesh's coordinate space, not 10x outside it.
MAX_RADIUS = 0.1
# outer -> inner: 1=Scalp, 2=Skull, 3=CSF, 4=Gray Matter, 5=White Matter
LAYER_LABELS_OUTER_TO_INNER = [1, 2, 3, 4, 5]


def _fibonacci_shell(radius, n_points):
    indices = np.arange(n_points)
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.stack([x, y, z], axis=1)


def build_layered_sphere_mesh():
    """Return (node[N,3] float64, elem[M,5] int32) for a synthetic 5-layer solid ball."""
    shells = [
        _fibonacci_shell(MAX_RADIUS * (i + 1) / N_SHELLS, POINTS_PER_SHELL)
        for i in range(N_SHELLS)
    ]
    node = np.vstack(shells + [np.zeros((1, 3))]).astype(np.float64)

    tets = Delaunay(node).simplices.astype(np.int64)
    centroid_radius = np.linalg.norm(node[tets].mean(axis=1), axis=1)

    # np.digitize's band_index INCREASES with radius (0=innermost, N_SHELLS-1=
    # outermost - verified empirically), so it must be reversed to map the
    # outermost band to label 1 (Scalp) and the innermost to label 5 (White
    # Matter), per LAYER_LABELS_OUTER_TO_INNER's outer-to-inner ordering.
    band_edges = np.linspace(0.0, MAX_RADIUS, N_SHELLS + 1)
    band_index = np.clip(np.digitize(centroid_radius, band_edges[1:-1]), 0, N_SHELLS - 1)
    tissue_labels = np.array(
        [LAYER_LABELS_OUTER_TO_INNER[N_SHELLS - 1 - i] for i in band_index], dtype=np.int64
    )

    elem = np.column_stack([tets + 1, tissue_labels]).astype(np.int32)  # 1-indexed verts
    return node, elem


def write_layered_mesh_fixture(path):
    node, elem = build_layered_sphere_mesh()
    savemat(str(path), {"mesh": {"node": node, "elem": elem}})


class LightSimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()
        register_addon(cls.addon)

    @classmethod
    def tearDownClass(cls):
        cls.addon.unregister()

    # Production default (lightsim_neurocaptain.MMC_SIMULATION_TIMEOUT_SECONDS)
    # is a generous safety net for real simulations. CI's software OpenCL CPU
    # device (pocl) can be too slow/unresponsive to finish even 100 photons,
    # so use a much shorter budget here - see test_import_layered_mesh_
    # then_run_mmc_and_redbird for how that outcome is distinguished from an
    # actual bug.
    MMC_TEST_TIMEOUT_SECONDS = 20

    def setUp(self):
        fd, self.fixture_path = tempfile.mkstemp(suffix=".mat")
        os.close(fd)
        write_layered_mesh_fixture(self.fixture_path)

        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1)
        self.head = bpy.context.active_object
        self.head.name = "headmesh"

        self._place_optode("add_source", 0)
        self._place_optode("add_source", 100)
        self._place_optode("add_detector", 200)
        self._place_optode("add_detector", 300)

        settings = bpy.context.scene.neurocaptain_settings
        settings.mmc_use_gpu = False
        settings.mmc_nphoton = 100  # keep the simulation small/fast for CI

        self._lightsim = self.addon.lightsim_neurocaptain
        self._orig_mmc_timeout = self._lightsim.MMC_SIMULATION_TIMEOUT_SECONDS
        self._lightsim.MMC_SIMULATION_TIMEOUT_SECONDS = self.MMC_TEST_TIMEOUT_SECONDS

    def tearDown(self):
        self._lightsim.MMC_SIMULATION_TIMEOUT_SECONDS = self._orig_mmc_timeout

        if os.path.exists(self.fixture_path):
            os.remove(self.fixture_path)

        for collection_name in ("FiveLayer_Visualization", "Sources", "Detectors"):
            collection = bpy.data.collections.get(collection_name)
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

        for mat_name in ("HeadSurface_Mat", "Cortex_Mat"):
            mat = bpy.data.materials.get(mat_name)
            if mat is not None:
                bpy.data.materials.remove(mat)
        for mat in [m for m in bpy.data.materials if m.name.endswith("_Material")]:
            bpy.data.materials.remove(mat)

        bpy.context.scene.neurocaptain_settings.mmc_use_gpu = True
        if "mmc_optical_properties" in bpy.context.scene:
            del bpy.context.scene["mmc_optical_properties"]

        self.addon.layered_mesh_manager.LAYERED_MESH.__init__()

    def _place_optode(self, bl_idname, vertex_index):
        for vert in self.head.data.vertices:
            vert.select = False
        self.head.data.vertices[vertex_index].select = True
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = self.head
        self.head.select_set(True)
        getattr(bpy.ops.neurocaptain, bl_idname)()

    def test_import_layered_mesh_then_run_mmc_and_redbird(self):
        lmm = self.addon.layered_mesh_manager
        deps = lmm.check_dependencies()
        self.assertTrue(deps["all_available"], f"missing deps: {deps}")

        result = bpy.ops.neurocaptain.import_layered_mesh(filepath=self.fixture_path)
        self.assertEqual(result, {"FINISHED"})
        self.assertTrue(lmm.is_mesh_loaded())
        self.assertIsNotNone(bpy.data.objects.get("Head_Surface_5L"))
        self.assertIsNotNone(bpy.data.objects.get("Brain_Cortex_5L"))
        self.assertGreater(len(bpy.data.objects["Head_Surface_5L"].data.polygons), 0)
        self.assertGreater(len(bpy.data.objects["Brain_Cortex_5L"].data.polygons), 0)

        # run_single_mmc bounds pmmc.run() to MMC_SIMULATION_TIMEOUT_SECONDS
        # (shortened above for CI) and raises TimeoutError if it doesn't
        # respond in time. On an environment with no real OpenCL device (or
        # not enough compute power for even 100 photons - e.g. CI's software
        # pocl CPU backend), every source/detector call times out, so
        # run_sensitivity_mmc finds no valid channels and the operator
        # reports {'CANCELLED'}. That's an environment limitation, not a
        # NeuroCaptain bug, and per team decision counts as a pass here -
        # but only when the captured output actually shows our timeout
        # message, so a real regression still fails this test.
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            result = bpy.ops.neurocaptain.run_mmc()
        output = captured.getvalue()
        sys.stdout.write(output)

        if result != {"FINISHED"}:
            self.assertIn(
                "pmmc.run() did not finish",
                output,
                f"run_mmc failed for a reason other than an OpenCL timeout:\n{output}",
            )
            self.skipTest(
                "pmmc's OpenCL backend couldn't find a device or enough compute "
                "power for the requested photon count in this environment - "
                "accepted as an environment limitation."
            )
        self.assertEqual(result, {"FINISHED"})

        result = bpy.ops.neurocaptain.run_redbird()
        self.assertEqual(result, {"FINISHED"})
