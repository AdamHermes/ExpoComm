import unittest
import torch as th
from multiview_fixtures import args_for
from modules.communication.multiview_graph import graph_views, feature_adjacency, normalize_adjacency, propagate
from modules.communication.multiview_encoder import MultiViewEncoder


class GraphTests(unittest.TestCase):
    def test_knn_excludes_self_and_inactive_and_is_union(self):
        x = th.tensor([[[1., 0], [.99, .01], [-1., 0], [1., 0]]])
        active = th.tensor([[1, 1, 1, 0]], dtype=th.bool)
        a = feature_adjacency(x, active, 1)
        self.assertEqual(a[0, 0, 1], 1)
        self.assertEqual(a[0, 0, 3], 0)
        self.assertEqual(a.diagonal(dim1=-2, dim2=-1).sum(), 0)
        th.testing.assert_close(a, a.transpose(-1, -2))

    def test_normalization_matches_hand_computation(self):
        a = th.tensor([[[0., 1.], [1., 0.]]])
        s = normalize_adjacency(a, th.ones(1, 2, dtype=th.bool), True)
        th.testing.assert_close(s, th.full((1, 2, 2), .5))

    def test_empty_singleton_and_large_k(self):
        for n in (1, 2, 5):
            for alive in (False, True):
                x = th.randn(2, n, 3)
                mask = th.full((2, n), alive)
                s, sf = graph_views(x, mask, 6, 100)
                self.assertTrue(th.isfinite(s).all() and th.isfinite(sf).all())
                if not alive:
                    self.assertEqual((s.abs() + sf.abs()).sum(), 0)

    def test_sgc_equations_and_shared_feature_weights(self):
        args = args_for()
        encoder = MultiViewEncoder(args)
        x, active = th.randn(2, 5, 16), th.ones(2, 5, dtype=th.bool)
        s, sf = graph_views(x, active, args.topk_neighbors, args.mv_feature_k)
        ho, hf, hr, ht = encoder.encode(x, active)
        th.testing.assert_close(ho, th.linalg.matrix_power(s, args.mv_local_hops) @ x @ encoder.w_feature.weight.T)
        th.testing.assert_close(hf, ho + sf @ x @ encoder.w_feature.weight.T)
        th.testing.assert_close(ht, th.linalg.matrix_power(s, args.mv_global_hops) @ x @ encoder.w_topology.weight.T)
        self.assertIsNot(encoder.w_feature.weight, encoder.w_original.weight)


if __name__ == "__main__":
    unittest.main()
