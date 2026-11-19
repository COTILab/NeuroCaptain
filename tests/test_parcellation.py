"""Tests for parcellation.py (cortical parcellation import + display).

Uses a tiny synthetic labelled octahedron written as text .jmsh. Set
NEUROCAPTAIN_PARC_FILE to a real SIAM DK .bmsh to also run the real-file
check (skipped otherwise).
"""

import json
import os
import tempfile
import unittest

import bpy
import numpy as np

from _addon_helpers import import_addon, register_addon

# Octahedron: 6 nodes, 8 faces (1-based, as JMesh stores them).
NODES = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
FACES = [[1, 3, 5], [3, 2, 5], [2, 4, 5], [4, 1, 5],
         [3, 1, 6], [2, 3, 6], [4, 2, 6], [1, 4, 6]]
LABELS = [1, 1, 24, 24, 58, 58, 0, 0]  # lh-bankssts, lh/rh-precuneus, unlabelled


class ParcellationTest(unittest.TestCase):
    addon = None

    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()
        register_addon(cls.addon)
        cls.parc = __import__(cls.addon.__name__ + ".parcellation",
                              fromlist=["parcellation"])
        cls.tmpdir = tempfile.mkdtemp()
        cls.path = os.path.join(cls.tmpdir, "toy_surf.jmsh")
        surf = [f + [l] for f, l in zip(FACES, LABELS)]
        with open(cls.path, "w") as f:
            json.dump({"MeshNode": NODES, "MeshSurf": surf}, f)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.addon.unregister()
        except Exception:
            pass

    def test_load_is_zero_based(self):
        nodes, faces, labels = self.parc.load_parcellation_surface(self.path)
        self.assertEqual(nodes.shape, (6, 3))
        self.assertEqual(faces.shape, (8, 3))
        self.assertEqual(faces.min(), 0)
        self.assertEqual(faces.max(), 5)
        self.assertEqual(list(labels), LABELS)

    def test_dk_names(self):
        p = self.parc
        self.assertEqual(p.region_info(23, "DK")[0], "lh-precentral")
        self.assertEqual(p.region_info(57, "DK")[0], "rh-precentral")
        self.assertEqual(p.region_info(31, "DK")[0], "lh-frontalpole")
        self.assertEqual(p.region_info(68, "DK")[0], "rh-insula")
        self.assertEqual(p.region_info(1024, "FS_APARC")[0], "lh-precentral")
        self.assertEqual(p.region_info(2035, "FS_APARC")[0], "rh-insula")
        self.assertEqual(len(p.DK_COMPACT), 34)

    def test_detect_atlas(self):
        p = self.parc
        self.assertEqual(p.detect_atlas(np.array([0, 1, 68])), "DK")
        self.assertEqual(p.detect_atlas(np.array([1024, 2024])), "FS_APARC")
        self.assertEqual(p.detect_atlas(np.array([5, 300])), "GENERIC")

    def test_import_builds_object(self):
        scene = bpy.context.scene
        obj, msg = self.parc.import_parcellation(bpy.context, self.path, align=False)
        self.assertIn("Not aligned", msg)
        self.assertEqual(len(obj.data.polygons), 8)
        labels = self.parc.read_face_labels(obj)
        self.assertEqual(list(labels), LABELS)
        # Unique labels 0, 1, 24, 58 -> 4 material slots, in label order.
        self.assertEqual(len(obj.data.materials), 4)
        mi = np.empty(8, dtype=np.int32)
        obj.data.polygons.foreach_get("material_index", mi)
        self.assertEqual(list(mi), [1, 1, 2, 2, 3, 3, 0, 0])
        ps = scene.neurocaptain_parcellation
        self.assertEqual([r.name for r in ps.regions],
                         ["unlabelled", "lh-bankssts", "lh-precuneus", "rh-precuneus"])
        self.assertEqual(ps.atlas, "DK")
        self.assertTrue(all(r.area_mm2 > 0 for r in ps.regions))

    def test_display_modes_grey_out(self):
        scene = bpy.context.scene
        self.parc.import_parcellation(bpy.context, self.path, align=False)
        ps = scene.neurocaptain_parcellation
        bpy.ops.neurocaptain.parc_select(action='LEFT')
        self.assertEqual([r.selected for r in ps.regions], [False, True, True, False])
        ps.display_mode = 'SELECTED'
        rh = bpy.data.materials[ps.regions[3].material]
        lh = bpy.data.materials[ps.regions[1].material]
        self.assertAlmostEqual(rh.diffuse_color[0], self.parc.GREY[0], places=4)
        self.assertAlmostEqual(lh.diffuse_color[0], ps.regions[1].color[0], places=4)
        ps.display_mode = 'ALL'
        self.assertAlmostEqual(rh.diffuse_color[0], ps.regions[3].color[0], places=4)
        ps.display_mode = 'ACTIVE'
        ps.active_index = 3
        self.assertAlmostEqual(lh.diffuse_color[0], self.parc.GREY[0], places=4)
        ps.display_mode = 'ALL'

    def test_reimport_replaces_object(self):
        self.parc.import_parcellation(bpy.context, self.path, align=False)
        self.parc.import_parcellation(bpy.context, self.path, align=False)
        names = [o.name for o in bpy.data.objects if o.name.startswith(self.parc.PARC_OBJ_NAME)]
        self.assertEqual(names, [self.parc.PARC_OBJ_NAME])

    # ── label cleanup / borders ──────────────────────────────────────────
    # Strip of 4 triangles in a row: 0-1-2-3, each sharing one edge with the next.
    STRIP = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]])

    def test_face_neighbors(self):
        nb = self.parc.face_neighbors(self.STRIP)
        self.assertEqual(sorted(x for x in nb[1] if x >= 0), [0, 2])
        self.assertEqual(sorted(x for x in nb[0] if x >= 0), [1])

    def test_fill_and_despeckle(self):
        nb = self.parc.face_neighbors(self.STRIP)
        lab = np.array([7, 0, 0, 9])
        filled = self.parc.fill_labels_from_neighbors(lab, nb, np.array([False, True, True, False]))
        self.assertTrue(np.all(filled > 0))
        self.assertEqual(filled[0], 7)
        self.assertEqual(filled[3], 9)
        # Unfillable faces stay unlabelled.
        kept = self.parc.fill_labels_from_neighbors(lab, nb, np.zeros(4, bool))
        self.assertEqual(kept.tolist(), [7, 0, 0, 9])
        # Face 1 differs from both of its neighbours (which agree) -> despeckled.
        speck = self.parc.despeckle_labels(np.array([5, 6, 5, 5]), nb)
        self.assertEqual(speck.tolist(), [5, 5, 5, 5])

    def test_border_edges_only_between_regions(self):
        edges = self.parc.region_border_edges(self.STRIP, np.array([1, 1, 2, 2]))
        self.assertEqual(sorted(map(tuple, edges.tolist())), [(2, 3)])   # shared by faces 1|2
        self.assertEqual(len(self.parc.region_border_edges(self.STRIP, np.ones(4, int))), 0)

    def test_outline_edges_enclose_kept_region(self):
        lab = np.array([1, 1, 2, 2])
        edges = {tuple(e) for e in self.parc.region_outline_edges(self.STRIP, lab, {1}).tolist()}
        # Region 1 = faces 0,1: its open boundary (0-1, 0-2, 1-3) plus the
        # shared border with region 2 (2-3); not its internal edge 1-2.
        self.assertEqual(edges, {(0, 1), (0, 2), (1, 3), (2, 3)})
        self.assertEqual(len(self.parc.region_outline_edges(self.STRIP, lab, set())), 0)

    def test_overlay_outlines_ticked_thicker(self):
        scene = bpy.context.scene
        obj, _ = self.parc.import_parcellation(bpy.context, self.path, align=False)
        ps = scene.neurocaptain_parcellation
        ps.view_mode = 'OVERLAY'
        ps.overlay_style = 'OUTLINE'
        ps.display_mode = 'ALL'
        bpy.ops.neurocaptain.parc_select(action='NONE')
        self.assertIsNone(bpy.data.objects.get(self.parc.OUTLINE_OBJ_NAME))
        # Thin borders of every region are on in the overlay.
        borders = bpy.data.objects.get(self.parc.BORDER_OBJ_NAME)
        self.assertTrue(borders is not None and not borders.hide_viewport)
        ps.regions[1].selected = True
        outline = bpy.data.objects.get(self.parc.OUTLINE_OBJ_NAME)
        self.assertIsNotNone(outline)
        self.assertEqual(outline.parent, obj)
        self.assertGreater(len(outline.data.polygons), 0)
        # Highlighted-row mode outlines the active row instead.
        ps.display_mode = 'ACTIVE'
        ps.active_index = 3
        self.assertIsNotNone(bpy.data.objects.get(self.parc.OUTLINE_OBJ_NAME))
        ps.display_mode = 'ALL'
        # Heatmap Only: no thick outline.
        ps.overlay_style = 'ONLY'
        self.assertIsNone(bpy.data.objects.get(self.parc.OUTLINE_OBJ_NAME))
        ps.overlay_style = 'OUTLINE'
        ps.view_mode = 'REGIONS'                       # thick outline only in Overlay
        self.assertIsNone(bpy.data.objects.get(self.parc.OUTLINE_OBJ_NAME))
        ps.view_mode = 'OVERLAY'
        bpy.ops.neurocaptain.parc_select(action='NONE')
        self.assertIsNone(bpy.data.objects.get(self.parc.OUTLINE_OBJ_NAME))

    def test_heatmap_only_dims_by_show_filter(self):
        obj, _ = self.parc.import_parcellation(bpy.context, self.path, align=False)
        ps = bpy.context.scene.neurocaptain_parcellation
        ps.view_mode = 'OVERLAY'
        bpy.ops.neurocaptain.parc_select(action='NONE')
        ps.regions[1].selected = True
        ps.display_mode = 'SELECTED'
        names = [m.name for m in obj.data.materials]
        heat = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (len(obj.data.vertices), 1))
        orig = self.parc._heat_colors_on
        self.parc._heat_colors_on = lambda o: heat     # pretend a sensitivity map exists
        try:
            mi = np.repeat([p.material_index for p in obj.data.polygons], 3)
            other = names.index(ps.regions[2].material)
            ps.overlay_style = 'OUTLINE'               # full map: nothing greyed
            self.assertTrue(np.allclose(self._colors(obj)[mi == other], [0, 0, 1], atol=1e-5))
            ps.overlay_style = 'ONLY'                  # unticked region greyed
            self.assertFalse(np.allclose(self._colors(obj)[mi == other], [0, 0, 1], atol=1e-5))
            ticked = names.index(ps.regions[1].material)
            ps.overlay_style = 'FADE'                  # unticked faded 50% to grey
            col = self._colors(obj)
            self.assertTrue(np.allclose(col[mi == other], [0.25, 0.25, 0.75], atol=1e-5))
            self.assertTrue(np.allclose(col[mi == ticked], [0, 0, 1], atol=1e-5))
            self.assertIsNotNone(bpy.data.objects.get(self.parc.OUTLINE_OBJ_NAME))
        finally:
            self.parc._heat_colors_on = orig
            ps.overlay_style = 'OUTLINE'
            ps.display_mode = 'ALL'
            bpy.ops.neurocaptain.parc_select(action='NONE')

    def _colors(self, obj):
        col = np.empty(len(obj.data.loops) * 4, np.float32)
        obj.data.color_attributes[self.parc.COLOR_ATTR].data.foreach_get("color", col)
        return col.reshape(-1, 4)[:, :3]

    def test_cortex_offset_fixed(self):
        obj, _ = self.parc.import_parcellation(bpy.context, self.path, align=False)
        bpy.context.scene.neurocaptain_parcellation.view_mode = 'OVERLAY'
        mod = obj.modifiers[self.parc.OFFSET_MOD]
        self.assertAlmostEqual(mod.strength, self.parc.PARC_SURFACE_OFFSET, places=5)
        self.assertAlmostEqual(bpy.data.objects[self.parc.BORDER_OBJ_NAME]["lift"],
                               self.parc.PARC_SURFACE_OFFSET + self.parc.LINE_OFFSET, places=5)

    def test_blend_with_heatmap_colors(self):
        obj, _ = self.parc.import_parcellation(bpy.context, self.path, align=False)
        names = [m.name for m in obj.data.materials]
        red = {n: (1.0, 0.0, 0.0) for n in names}
        heat = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (len(obj.data.vertices), 1))
        self.parc._write_color_attribute(obj, red, dimmed={names[1]}, heat_rgb=heat)
        col = np.empty(len(obj.data.loops) * 4, np.float32)
        obj.data.color_attributes[self.parc.COLOR_ATTR].data.foreach_get("color", col)
        col = col.reshape(-1, 4)
        mi = np.repeat([p.material_index for p in obj.data.polygons], 3)
        # Shown: heatmap color only (region colors never blended in).
        self.assertTrue(np.allclose(col[mi == 0, :3], [0.0, 0.0, 1.0], atol=1e-5))
        # Dimmed: grey with the heatmap's brightness (0.25 + 0.35 * luminance).
        g = 0.25 + 0.35 * 0.114
        self.assertTrue(np.allclose(col[mi == 1, :3], [g, g, g], atol=1e-5))

    def test_builtin_surface_from_tets(self):
        try:
            import iso2mesh as i2m
        except ImportError:
            self.skipTest("iso2mesh not available")
        node, _face, elem = i2m.meshabox([0, 0, 0], [20, 20, 20], 8)
        node = np.asarray(node)[:, :3]
        elem = np.asarray(elem)[:, :4].astype(np.int64)
        cen = node[elem - 1].mean(axis=1)
        solver = np.full(len(elem), 4)
        region = np.where(cen[:, 0] < 10, 23, 57)
        n, f, lab = self.parc.builtin_parcel_surface(node, elem, solver, region)
        self.assertEqual(set(np.unique(lab).tolist()), {23, 57})
        self.assertEqual(f.max(), len(n) - 1)
        c = n[f].mean(axis=1)
        self.assertGreater(np.mean((c[:, 0] < 10) == (lab == 23)), 0.95)
        # Outward winding: normals point away from the box centre.
        nrm = np.cross(n[f[:, 1]] - n[f[:, 0]], n[f[:, 2]] - n[f[:, 0]])
        self.assertGreater(np.mean(np.einsum('ij,ij->i', nrm, c - 10.0) > 0), 0.99)

    def test_region_labels_persist(self):
        lmm = __import__(self.addon.__name__ + ".layered_mesh_manager",
                         fromlist=["layered_mesh_manager"])
        holder = bpy.data.objects.new("persist_holder", bpy.data.meshes.new("persist_holder"))
        bpy.context.scene.collection.objects.link(holder)
        lm = lmm.LAYERED_MESH
        saved = (lm.head_surface_obj, lm.nodes, lm.elems, lm.tissue_labels,
                 lm.region_labels, lm.region_atlas, lm.mesh_path)
        try:
            lm.head_surface_obj = holder
            lm.nodes = np.zeros((4, 3))
            lm.elems = np.array([[1, 2, 3, 4, 4]])
            lm.tissue_labels = np.array([4])
            lm.mesh_path = "x.bmsh"
            lm.region_labels = np.array([57])
            lm.region_atlas = "DK"
            lmm._persist_layered_mesh_to_object()
            lm.__init__()
            self.assertTrue(lmm.restore_layered_mesh_from_saved_data())
            self.assertEqual(lm.region_labels.tolist(), [57])
            self.assertEqual(lm.region_atlas, "DK")
        finally:
            (lm.head_surface_obj, lm.nodes, lm.elems, lm.tissue_labels,
             lm.region_labels, lm.region_atlas, lm.mesh_path) = saved
            bpy.data.objects.remove(holder, do_unlink=True)

    @unittest.skipUnless(os.path.isfile(os.environ.get("NEUROCAPTAIN_PARC_FILE", "")),
                         "set NEUROCAPTAIN_PARC_FILE to a SIAM DK .bmsh")
    def test_real_dk_file(self):
        path = os.environ["NEUROCAPTAIN_PARC_FILE"]
        nodes, faces, labels = self.parc.load_parcellation_surface(path)
        self.assertEqual(self.parc.detect_atlas(labels), "DK")
        self.assertEqual(len(np.unique(labels)), 68)
        cent = nodes[faces].mean(axis=1)
        # Left hemisphere labels sit at x < 0, right at x > 0 (RAS).
        self.assertLess(cent[labels <= 34, 0].mean(), 0)
        self.assertGreater(cent[labels > 34, 0].mean(), 0)
        obj, _msg = self.parc.import_parcellation(bpy.context, path, align=False)
        self.assertEqual(len(obj.data.polygons), len(faces))
        self.assertEqual(len(bpy.context.scene.neurocaptain_parcellation.regions), 68)


if __name__ == "__main__":
    unittest.main()
