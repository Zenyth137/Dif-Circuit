"""
Unit tests for MDP environment and reward functions.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.environment.mdp_env import MacroPlacementEnv, EnvConfig
from src.environment.reward import compute_hpwl, compute_congestion, compute_overlap


class TestRewardFunctions(unittest.TestCase):

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
        self.nets = [[0, 1], [1, 2], [0, 1, 2]]

    def test_hpwl(self):
        hpwl = compute_hpwl(self.positions, self.nets)
        self.assertGreater(hpwl, 0)
        # Net [0,1]: bb = (200-100) + (150-100) = 100+50 = 150
        # Net [1,2]: bb = (300-200) + (200-150) = 100+50 = 150
        # Net [0,1,2]: bb = (300-100) + (200-100) = 200+100 = 300
        # Total = 600
        self.assertAlmostEqual(hpwl, 600.0, delta=10.0)

    def test_overlap_zero(self):
        # No overlap between these modules
        overlap = compute_overlap(self.positions, self.nodes)
        self.assertEqual(overlap, 0.0)

    def test_overlap_nonzero(self):
        # Overlapping positions
        positions = np.array([
            [100.0, 100.0],
            [110.0, 110.0],  # overlaps with first
        ], dtype=np.float32)
        nodes = np.array([
            [0, 30.0, 30.0],
            [1, 30.0, 30.0],
        ], dtype=np.float32)
        overlap = compute_overlap(positions, nodes)
        self.assertGreater(overlap, 0.0)

    def test_congestion(self):
        cong = compute_congestion(self.positions, self.nodes, bins=4)
        self.assertGreaterEqual(cong, 0.0)


class TestMDPEnv(unittest.TestCase):

    def setUp(self):
        self.config = EnvConfig(
            grid_size=16,
            canvas_width=1000.0,
            canvas_height=1000.0,
            max_modules=10,
        )
        self.env = MacroPlacementEnv(self.config)
        self.nodes = np.array([
            [i, 50.0, 50.0] for i in range(10)
        ], dtype=np.float32)
        self.nets = [[i, (i + 1) % 10] for i in range(10)]

    def test_reset(self):
        state = self.env.reset(self.nodes, self.nets)
        self.assertIn("density_grid", state)
        self.assertEqual(state["density_grid"].shape, (16, 16))
        self.assertFalse(self.env._done)

    def test_step(self):
        self.env.reset(self.nodes, self.nets)
        state, reward, done, info = self.env.step((4, 5))
        self.assertIsInstance(reward, float)
        self.assertFalse(done)

    def test_full_episode(self):
        self.env.reset(self.nodes, self.nets)
        total_reward = 0.0
        for i in range(10):
            _, reward, done, _ = self.env.step((i % 16, i % 16))
            total_reward += reward
        self.assertTrue(done)
        self.assertNotEqual(total_reward, 0.0)

    def test_placement_output(self):
        self.env.reset(self.nodes, self.nets)
        for i in range(10):
            self.env.step((i % 16, i % 16))
        positions = self.env.get_placement()
        self.assertEqual(positions.shape, (10, 2))


if __name__ == "__main__":
    unittest.main()
