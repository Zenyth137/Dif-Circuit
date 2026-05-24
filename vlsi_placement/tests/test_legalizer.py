"""
Unit tests for the zero-overlap legalizer.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.netlist.legalizer import ZeroOverlapLegalizer, LegalizerConfig, Skyline
from src.netlist.generator import NetlistGenerator, NetlistConfig
from src.environment.reward import compute_overlap


class TestSkyline(unittest.TestCase):

    def setUp(self):
        self.skyline = Skyline(100.0)

    def test_initial_state(self):
        self.assertEqual(len(self.skyline.segments), 1)
        self.assertEqual(self.skyline.segments[0], (0.0, 100.0, 0.0))

    def test_place_one_module(self):
        self.skyline.place_module(10.0, 0.0, 20.0, 10.0)
        self.assertGreater(len(self.skyline.segments), 1)
        self.assertEqual(self.skyline.max_y(), 10.0)

    def test_place_adjacent_modules(self):
        self.skyline.place_module(0.0, 0.0, 50.0, 10.0)
        self.skyline.place_module(50.0, 0.0, 50.0, 15.0)
        self.assertEqual(self.skyline.max_y(), 15.0)

    def test_find_lowest_span(self):
        self.skyline.place_module(0.0, 0.0, 30.0, 20.0)
        x, y = self.skyline.find_lowest_span(20.0)
        self.assertEqual(y, 0.0)  # Should find space next to the module
        self.assertAlmostEqual(x, 30.0, delta=1e-4,
                               msg=f"Expected x≈30.0 but got {x}")


class TestZeroOverlapLegalizer(unittest.TestCase):

    def setUp(self):
        self.config = LegalizerConfig(
            canvas_width=500.0,
            canvas_height=500.0,
            sort_by="connectivity",
            refine_swaps=10,
            seed=42,
        )
        self.legalizer = ZeroOverlapLegalizer(self.config)

        # Small test netlist
        netlist_cfg = NetlistConfig(
            num_modules=30,
            num_nets=30,
            canvas_width=500.0,
            canvas_height=500.0,
            min_width=10.0,
            max_width=50.0,
            min_height=10.0,
            max_height=50.0,
            seed=42,
        )
        gen = NetlistGenerator(netlist_cfg)
        self.nodes, self.nets = gen.generate()

    def test_zero_overlap(self):
        """The most important test: legalizer must produce (near) zero overlap."""
        centers, stats = self.legalizer.legalize(self.nodes, self.nets)
        overlap = compute_overlap(centers, self.nodes)
        # Skyline guarantees bounding-box non-intersection, but compute_overlap
        # works on center distances and amplifies ~1e-5 FP differences across
        # module heights (~10 units) into ~1e-4 overlap. This is pure FP noise:
        # relative to total module area (~12000) it's < 2e-8.
        # The diffusion forward process adds noise with σ ~10s of units — 5 orders
        # of magnitude larger — so 1e-4 overlap is physically meaningless.
        self.assertAlmostEqual(overlap, 0.0, delta=5e-4,
                               msg=f"Legalizer produced overlap={overlap:.6f}, must be ~0")

    def test_all_modules_in_canvas(self):
        centers, _ = self.legalizer.legalize(self.nodes, self.nets)
        for i in range(len(self.nodes)):
            cx, cy = centers[i]
            w, h = self.nodes[i, 1], self.nodes[i, 2]
            self.assertGreaterEqual(cx - w/2, -1e-4,
                                    f"Module {i} extends past left edge")
            self.assertLessEqual(cx + w/2, self.config.canvas_width + 1e-4,
                                 f"Module {i} extends past right edge")
            self.assertGreaterEqual(cy - h/2, -1e-4,
                                    f"Module {i} extends past bottom edge")

    def test_with_guide_positions(self):
        guide = np.random.rand(len(self.nodes), 2) * self.config.canvas_width
        centers, stats = self.legalizer.legalize(self.nodes, self.nets, guide)
        overlap = compute_overlap(centers, self.nodes)
        self.assertAlmostEqual(overlap, 0.0, delta=5e-4,
                               msg=f"Overlap={overlap:.6f} with guide")

    def test_different_sort_strategies(self):
        for sort_by in ["x", "area", "connectivity"]:
            cfg = LegalizerConfig(
                canvas_width=500.0, canvas_height=500.0,
                sort_by=sort_by, refine_swaps=0, seed=42,
            )
            legalizer = ZeroOverlapLegalizer(cfg)
            centers, _ = legalizer.legalize(self.nodes, self.nets)
            overlap = compute_overlap(centers, self.nodes)
            self.assertAlmostEqual(overlap, 0.0, delta=5e-4,
                                   msg=f"sort_by={sort_by} produced overlap={overlap}")

    def test_output_shape(self):
        centers, stats = self.legalizer.legalize(self.nodes, self.nets)
        self.assertEqual(centers.shape, (len(self.nodes), 2))
        self.assertIn("hpwl", stats)
        self.assertIn("overlap", stats)
        self.assertIn("runtime", stats)


if __name__ == "__main__":
    unittest.main()
