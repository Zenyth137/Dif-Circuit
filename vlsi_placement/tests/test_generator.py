"""
Unit tests for synthetic netlist generator.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.netlist.generator import NetlistGenerator, NetlistConfig


class TestNetlistGenerator(unittest.TestCase):

    def setUp(self):
        self.config = NetlistConfig(
            num_modules=100,
            num_nets=100,
            seed=42,
        )
        self.generator = NetlistGenerator(self.config)

    def test_generate_shape(self):
        nodes, nets = self.generator.generate()
        self.assertEqual(nodes.shape, (100, 3))
        self.assertEqual(len(nets), 100)

    def test_generate_node_values(self):
        nodes, _ = self.generator.generate()
        # Check IDs
        np.testing.assert_array_equal(nodes[:, 0], np.arange(100))
        # Check all sizes positive
        self.assertTrue(np.all(nodes[:, 1] > 0))
        self.assertTrue(np.all(nodes[:, 2] > 0))

    def test_nets_valid(self):
        _, nets = self.generator.generate()
        for net in nets:
            self.assertGreaterEqual(len(net), 2)
            self.assertLessEqual(len(net), 10)
            for mid in net:
                self.assertGreaterEqual(mid, 0)
                self.assertLess(mid, 100)

    def test_adjacency(self):
        self.generator.generate()
        adj = self.generator.build_adjacency()
        self.assertEqual(adj.shape, (100, 100))
        self.assertTrue(np.all(adj >= 0))
        self.assertTrue(np.all(adj <= 1))

    def test_edge_index(self):
        self.generator.generate()
        edge_index, edge_attr = self.generator.build_edge_index()
        self.assertEqual(edge_index.shape[0], 2)
        self.assertGreater(edge_index.shape[1], 0)
        self.assertEqual(edge_attr.shape[0], edge_index.shape[1])

    def test_module_edge_index(self):
        self.generator.generate()
        edge_index, edge_attr = self.generator.build_module_edge_index()
        self.assertEqual(edge_index.shape[0], 2)
        self.assertGreater(edge_index.shape[1], 0)
        n = self.config.num_modules
        self.assertTrue(np.all(edge_index < n))
        self.assertEqual(edge_attr.shape[0], edge_index.shape[1])

    def test_truncate_to_max_modules(self):
        self.generator.generate()
        nodes, nets = self.generator.truncate_to_max_modules(50)
        self.assertEqual(len(nodes), 50)
        for net in nets:
            self.assertGreaterEqual(len(net), 2)
            self.assertTrue(all(m < 50 for m in net))

    def test_total_area(self):
        self.generator.generate()
        area = self.generator.get_total_area()
        manual_area = np.sum(
            self.generator.nodes[:, 1] * self.generator.nodes[:, 2]
        )
        self.assertAlmostEqual(area, manual_area)


if __name__ == "__main__":
    unittest.main()
