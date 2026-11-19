"""Headless test: the cloth rest shape is the JSON design shape.

template_rest_shape fits the JSON template positions onto the current
(barycentric) positions with rotation + translation + one uniform scale, so
every spring in the rest shape - not only those in closed triangles - has
its (head-scaled) design length.  fit_to_scalp then puts the relaxed probe
on a smaller head without squashing its stiff springs.
"""

import importlib
import math
import unittest

import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from _addon_helpers import import_addon

TEMPLATE = {"Source_1": (0, 0, 0), "Detector_1": (30, 0, 2), "Source_2": (60, 0, 6),
            "Detector_2": (30, 30, 3), "Detector_3": (0, 8, 0)}       # D3: short-sep, single spring
SPRINGS = [("Source_1", "Detector_1"), ("Detector_1", "Source_2"), ("Detector_1", "Detector_2"),
           ("Source_2", "Detector_2"), ("Source_1", "Detector_3")]


class ClothRestShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addon = import_addon()
        cls.mod = importlib.import_module(f"{cls.addon.__name__}.optodeJSON_blenderGoal")

    def _connections(self, scale):
        return [{"optode1": a, "optode2": b,
                 "rest_length": scale * float(np.linalg.norm(np.subtract(TEMPLATE[a], TEMPLATE[b])))}
                for a, b in SPRINGS]

    def test_rest_shape_has_design_lengths(self):
        names = list(TEMPLATE)
        scale = 0.87
        # current positions: the template rotated, shifted, scaled and distorted
        R = Matrix.Rotation(math.radians(35), 3, Vector((1, 2, 3)).normalized())
        rng = np.random.default_rng(0)
        current = [R @ (Vector(TEMPLATE[n]) * scale) + Vector((10, -5, 80))
                   + Vector(rng.normal(0, 3, 3).tolist()) for n in names]
        rest = self.mod.template_rest_shape(current, names, TEMPLATE, self._connections(scale))
        self.assertIsNotNone(rest)
        idx = {n: i for i, n in enumerate(names)}
        for cd in self._connections(scale):
            got = (rest[idx[cd["optode1"]]] - rest[idx[cd["optode2"]]]).length
            self.assertAlmostEqual(got, cd["rest_length"], places=4,
                                   msg=f"{cd['optode1']}-{cd['optode2']}")
        # placed where the probe currently is
        np.testing.assert_allclose(np.mean([list(v) for v in rest], 0),
                                   np.mean([list(v) for v in current], 0), atol=1e-6)

    def test_missing_template_positions_fall_back(self):
        names = list(TEMPLATE) + ["Detector_9"]
        current = [Vector(TEMPLATE[n]) for n in TEMPLATE] + [Vector((5, 5, 5))]
        self.assertIsNone(self.mod.template_rest_shape(current, names, TEMPLATE, self._connections(1)))

    def test_soft_springs_run_through_chain_vertices(self):
        # 0-1-2 triangle of stiff springs, 2-3 soft (e.g. to a dummy)
        edges = [(0, 1), (1, 2), (0, 2), (2, 3)]
        faces, loose, chains = self.mod.split_soft_springs(4, edges, {(2, 3)})
        self.assertEqual(faces, [(0, 1, 2)])
        self.assertEqual(chains, [(2, 3)])
        self.assertEqual(sorted(loose), [(2, 4), (4, 5), (5, 3)])
        pts = self.mod.chain_points([Vector((0, 0, 0))] * 2 + [Vector((0, 0, 0)), Vector((3, 0, 0))], chains)
        np.testing.assert_allclose([list(p) for p in pts], [(1, 0, 0), (2, 0, 0)], atol=1e-6)

    def test_fit_to_scalp_keeps_design_lengths_on_smaller_head(self):
        # Probe designed on an R=90 sphere, relaxed shape snapped onto an R=78 head
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=128, v_segments=64, radius=78.0)
        bvh = BVHTree.FromBMesh(bm)
        bm.free()

        def on_sphere(theta, phi, r):
            t, p = math.radians(theta), math.radians(phi)
            return Vector((r * math.sin(t) * math.cos(p), r * math.sin(t) * math.sin(p), r * math.cos(t)))
        grid = {f"O{i}{j}": (10 + 18 * i, 18 * j) for i in range(3) for j in range(3)}
        design = {n: on_sphere(t, p, 90.0) for n, (t, p) in grid.items()}
        design["SS"] = design["O11"] + (design["O12"] - design["O11"]).normalized() * 8.0
        names = list(design)
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
                 if (design[a] - design[b]).length < 40]
        states = {f"{a}_{b}": {"optode1": a, "optode2": b, "is_flexible": False,
                               "rest_length": (design[a] - design[b]).length} for a, b in pairs}
        snapped = [bvh.find_nearest(design[n])[0] for n in names]     # squashed onto the head
        squashed = max(abs((snapped[names.index(a)] - snapped[names.index(b)]).length
                           / st["rest_length"] - 1) for st in states.values()
                       for a, b in [(st["optode1"], st["optode2"])])
        self.assertGreater(squashed, 0.08, "test setup should start compressed")

        fitted = self.mod.fit_to_scalp(snapped, names, states, {"O11"}, bvh)
        pos = dict(zip(names, fitted))
        strains = [abs((pos[st["optode1"]] - pos[st["optode2"]]).length / st["rest_length"] - 1)
                   for st in states.values()]
        self.assertLess(np.mean(strains), 0.02, f"mean strain {np.mean(strains):.3f}")
        for n in names:
            self.assertLess((bvh.find_nearest(pos[n])[0] - pos[n]).length, 1e-3, f"{n} off the scalp")
        self.assertEqual(pos["O11"], snapped[names.index("O11")], "fixed optode moved")


if __name__ == "__main__":
    unittest.main()
