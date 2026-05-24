"""
Unit tests for evaluation metrics.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.evaluation.metrics import compute_all_metrics, PlacementMetrics


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.positions = np.array([
            [100.0, 100.0],
            [200.0, 150.0],
            [300.0, 200.0],
        ], dtype=np.float32)
        self.nodes = np.array([
            [0, 20.0, 20.0],
            [1, 30.0, 30.0],
            [2, 25.0, 25.0],
        ], dtype=np.float32)
        self.nets = [[0, 1], [1, 2]]

    def test_all_metrics(self):
        metrics = compute_all_metrics(self.positions, self.nodes, self.nets)
        self.assertIsInstance(metrics, PlacementMetrics)
        self.assertGreater(metrics.hpwl, 0)
        self.assertGreaterEqual(metrics.overlap_area, 0)
        self.assertEqual(metrics.num_modules, 3)
        self.assertEqual(metrics.num_nets, 2)

    def test_to_dict(self):
        metrics = compute_all_metrics(self.positions, self.nodes, self.nets)
        d = metrics.to_dict()
        self.assertIn("hpwl", d)
        self.assertIn("overlap_pct", d)
        self.assertIn("congestion", d)

    def test_overlap_pct_bounded(self):
        """Stacked modules: union overlap% must not exceed 100%."""
        nodes = np.array([[0, 10.0, 10.0], [1, 10.0, 10.0]], dtype=np.float32)
        positions = np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float32)
        metrics = compute_all_metrics(positions, nodes, [[0, 1]])
        self.assertLessEqual(metrics.overlap_pct, 100.0)
        self.assertGreater(metrics.overlap_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
