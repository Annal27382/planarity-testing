"""Tests for the planarity testing library."""

import unittest

from planarity_testing import planar_embedding


class TestPlanarity(unittest.TestCase):
    def test_empty_graph(self):
        self.assertEqual(planar_embedding({}), {})

    def test_single_vertex(self):
        self.assertEqual(planar_embedding({0: []}), {0: []})

    def test_single_edge(self):
        embedding = planar_embedding({0: [1], 1: [0]})
        self.assertIsNotNone(embedding)
        self.assertEqual(set(embedding[0]), {1})
        self.assertEqual(set(embedding[1]), {0})

    def test_path_graph(self):
        graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        embedding = planar_embedding(graph)
        self.assertIsNotNone(embedding)
        for vertex, neighbours in graph.items():
            self.assertEqual(set(embedding[vertex]), set(neighbours))

    def test_cycle_graph(self):
        graph = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [2, 0]}
        embedding = planar_embedding(graph)
        self.assertIsNotNone(embedding)
        for vertex, neighbours in graph.items():
            self.assertEqual(set(embedding[vertex]), set(neighbours))

    def test_k4_is_planar(self):
        graph = {0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2]}
        self.assertIsNotNone(planar_embedding(graph))

    def test_k5_is_nonplanar(self):
        graph = {
            0: [1, 2, 3, 4],
            1: [0, 2, 3, 4],
            2: [0, 1, 3, 4],
            3: [0, 1, 2, 4],
            4: [0, 1, 2, 3],
        }
        self.assertIsNone(planar_embedding(graph))

    def test_k33_is_nonplanar(self):
        graph = {
            0: [3, 4, 5],
            1: [3, 4, 5],
            2: [3, 4, 5],
            3: [0, 1, 2],
            4: [0, 1, 2],
            5: [0, 1, 2],
        }
        self.assertIsNone(planar_embedding(graph))

    def test_disconnected_planar(self):
        graph = {
            0: [1],
            1: [0],
            2: [3, 4],
            3: [2, 4],
            4: [2, 3],
            5: [],
        }
        embedding = planar_embedding(graph)
        self.assertIsNotNone(embedding)
        self.assertEqual(set(embedding[5]), set())

    def test_parallel_edge_rejected(self):
        with self.assertRaises(ValueError):
            planar_embedding({0: [1, 1], 1: [0]})

    def test_self_loop_rejected(self):
        with self.assertRaises(ValueError):
            planar_embedding({0: [0]})

    def test_missing_vertex_key_rejected(self):
        with self.assertRaises(ValueError):
            planar_embedding({0: [1]})


if __name__ == "__main__":
    unittest.main()
