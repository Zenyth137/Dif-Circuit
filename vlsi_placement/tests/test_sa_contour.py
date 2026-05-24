"""SA B*-tree packing smoke test."""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.baselines.simulated_annealing import SimulatedAnnealing, SAConfig


class TestSAContour(unittest.TestCase):

    def test_place_small(self):
        rng = np.random.default_rng(0)
        n = 20
        nodes = np.stack([
            np.arange(n),
            rng.uniform(5, 15, n),
            rng.uniform(5, 15, n),
        ], axis=1)
        nets = [[i, (i + 1) % n] for i in range(n)]
        sa = SimulatedAnnealing(SAConfig(max_iterations=50, seed=0))
        centers, stats = sa.place(nodes, nets, verbose=False)
        self.assertEqual(centers.shape, (n, 2))
        self.assertIn("runtime", stats)


if __name__ == "__main__":
    unittest.main()
