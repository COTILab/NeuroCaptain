"""Tests for lightsim_neurocaptain.prepare_solver_labels / detect_tissue_order.

Solvers expect 1 scalp, 2 skull, 3 CSF, 4 GM, 5 WM. Parcels (> 100) must
become gray matter, and SPM-ordered meshes (1 GM ... 5 scalp) must be
converted. Set NEUROCAPTAIN_DKMERGE_FILE to a real fem5layer_dkmerge .bmsh to
also run the real-file check.
"""

import os
import unittest

import numpy as np

from _addon_helpers import import_addon


def _shells(order_outer_to_inner, n_per=50):
    """Synthetic mesh: each tissue's elements sit on its own radius shell."""
    rng = np.random.default_rng(0)
    nodes, elems, labels = [], [], []
    for depth, lab in enumerate(order_outer_to_inner):
        r = 90.0 - 10.0 * depth
        v = rng.normal(size=(n_per, 3))
        v = r * v / np.linalg.norm(v, axis=1, keepdims=True)
        base = len(nodes)
        nodes.extend(v.tolist())
        for i in range(n_per):
            elems.append([base + i] * 4)
            labels.append(lab)
    return np.array(nodes), np.array(elems), np.array(labels)


class SolverLabelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        addon = import_addon()
        cls.ls = __import__(addon.__name__ + ".lightsim_neurocaptain",
                            fromlist=["lightsim_neurocaptain"])

    def test_detect_spm_and_canonical(self):
        n, e, l = _shells([5, 4, 3, 1, 2])          # SPM: scalp=5 outermost
        self.assertEqual(self.ls.detect_tissue_order(n, e, l), 'SPM')
        n, e, l = _shells([1, 2, 3, 4, 5])          # canonical: scalp=1 outermost
        self.assertEqual(self.ls.detect_tissue_order(n, e, l), 'CANONICAL')

    def test_parcels_become_gm_canonical(self):
        raw = np.array([1, 2, 3, 4, 5, 101, 168, 135])
        out, order = self.ls.prepare_solver_labels(raw, order='CANONICAL')
        self.assertEqual(order, 'CANONICAL')
        self.assertEqual(out.tolist(), [1, 2, 3, 4, 5, 4, 4, 4])
        self.assertEqual(raw.tolist(), [1, 2, 3, 4, 5, 101, 168, 135])  # untouched

    def test_spm_conversion_and_parcels(self):
        raw = np.array([1, 2, 3, 4, 5, 101, 168])
        out, _ = self.ls.prepare_solver_labels(raw, order='SPM')
        # SPM GM, WM, CSF, skull, scalp -> 4, 5, 3, 2, 1; parcels -> GM 4
        self.assertEqual(out.tolist(), [4, 5, 3, 2, 1, 4, 4])

    def test_auto_uses_geometry(self):
        n, e, l = _shells([5, 4, 3, 1, 2])
        l = l.copy()
        l[-10:] = 124                               # a few parcel elements
        out, order = self.ls.prepare_solver_labels(l, n, e, order='AUTO')
        self.assertEqual(order, 'SPM')
        self.assertTrue(np.all(out[l == 5] == 1))   # scalp -> 1
        self.assertTrue(np.all(out[l == 124] == 4))
        self.assertEqual(set(np.unique(out).tolist()), {1, 2, 3, 4, 5})

    def test_auto_without_geometry_falls_back(self):
        out, order = self.ls.prepare_solver_labels(np.array([1, 4, 150]), order='AUTO')
        self.assertEqual(order, 'CANONICAL')
        self.assertEqual(out.tolist(), [1, 4, 4])

    def test_normalize_role(self):
        n = self.ls.normalize_role
        for text, role in (("Gray Matter", "gray_matter"), ("grey_matter", "gray_matter"),
                           ("GM", "gray_matter"), ("White Matter", "white_matter"),
                           ("wm", "white_matter"), ("Skin", "scalp"), ("Scalp", "scalp"),
                           ("bone", "skull"), ("CSF", "csf"), ("soft tissue", "scalp")):
            self.assertEqual(n(text), role, text)
        self.assertIsNone(n("other"))
        self.assertIsNone(n("dura"))
        self.assertIsNone(n(None))

    def test_roles_override_geometry(self):
        # SPM-ordered geometry, but roles say exactly what each label is.
        n, e, l = _shells([5, 4, 3, 1, 2])
        info = {1: {'name': 'Gray Matter', 'role': 'gray_matter'},
                2: {'name': 'White Matter', 'role': 'white_matter'},
                3: {'name': 'CSF', 'role': 'csf'},
                4: {'name': 'Skull', 'role': 'skull'},
                5: {'name': 'Scalp', 'role': 'scalp'}}
        out, how = self.ls.prepare_solver_labels(l, n, e, 'AUTO', info)
        self.assertEqual(how, 'ROLES')
        for src, dst in ((1, 4), (2, 5), (3, 3), (4, 2), (5, 1)):
            self.assertTrue(np.all(out[l == src] == dst))

    def test_roles_any_layer_count(self):
        # 3-layer mesh with arbitrary numbering + one unrecognized layer.
        raw = np.array([7, 7, 9, 12, 20, 150])
        info = {7: {'name': 'Skin', 'role': 'scalp'},
                9: {'name': 'Brain', 'role': 'gray_matter'},
                12: {'name': 'White', 'role': 'custom_white'},
                20: {'name': 'Dura', 'role': 'other'}}
        out, how = self.ls.prepare_solver_labels(raw, order='AUTO', layer_info=info)
        self.assertEqual(how, 'ROLES')
        self.assertEqual(out.tolist(), [1, 1, 4, 5, 6, 4])

    def test_names_only_fall_back_to_geometry(self):
        # Fixed 5-layer importers record names without roles -> not trusted.
        n, e, l = _shells([5, 4, 3, 1, 2])
        info = {k: {'name': v} for k, v in
                {1: 'Scalp', 2: 'Skull', 3: 'CSF', 4: 'Gray Matter', 5: 'White Matter'}.items()}
        _out, how = self.ls.prepare_solver_labels(l, n, e, 'AUTO', info)
        self.assertEqual(how, 'SPM')

    def test_explicit_order_beats_roles(self):
        info = {1: {'role': 'gray_matter'}, 5: {'role': 'scalp'}}
        out, how = self.ls.prepare_solver_labels(np.array([1, 5]), order='CANONICAL',
                                                 layer_info=info)
        self.assertEqual((out.tolist(), how), ([1, 5], 'CANONICAL'))

    @unittest.skipUnless(os.path.isfile(os.environ.get("NEUROCAPTAIN_DKMERGE_FILE", "")),
                         "set NEUROCAPTAIN_DKMERGE_FILE to a fem5layer_dkmerge .bmsh")
    def test_real_dkmerge(self):
        hm = __import__(self.ls.__name__.rsplit(".", 1)[0] + ".headmodels",
                        fromlist=["headmodels"])
        d = hm._load_mesh_direct(os.environ["NEUROCAPTAIN_DKMERGE_FILE"])
        elem = np.asarray(d["MeshElem"])
        nodes, lab = np.asarray(d["MeshNode"]), elem[:, 4].astype(int)
        out, order = self.ls.prepare_solver_labels(lab, nodes, elem[:, :4], 'AUTO')
        self.assertEqual(order, 'SPM')
        self.assertEqual(sorted(np.unique(out).tolist()), [1, 2, 3, 4, 5])
        # GM after merge = old SPM GM (1) + all parcels.
        self.assertEqual(int((out == 4).sum()), int(((lab == 1) | (lab > 100)).sum()))


if __name__ == "__main__":
    unittest.main()
