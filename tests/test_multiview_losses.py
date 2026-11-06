import math
import unittest
import torch as th
from multiview_fixtures import args_for
from modules.communication.multiview_losses import mail_contrastive
from modules.communication.multiview_encoder import MultiViewEncoder


class LossTests(unittest.TestCase):
    def test_paper_denominator_not_standard_simclr(self):
        # Two orthogonal agents, identical views: positive exp(1), one
        # cross-view negative exp(0), and two within-view negatives exp(0).
        x = th.eye(2).unsqueeze(0)
        loss = mail_contrastive(x, x, th.ones(1, 2, dtype=th.bool), 1.)
        th.testing.assert_close(loss, th.full((1, 2), math.log(math.e + 3) - 1))

    def test_singleton_and_all_masked_have_finite_backward(self):
        for mask in (th.tensor([[1, 0, 0]]), th.tensor([[0, 0, 0]])):
            x, y = th.randn(1, 3, 4, requires_grad=True), th.randn(1, 3, 4, requires_grad=True)
            loss = mail_contrastive(x, y, mask, .1)
            self.assertTrue(th.isfinite(loss).all())
            loss.sum().backward()
            self.assertTrue(th.isfinite(x.grad).all() and th.isfinite(y.grad).all())
            th.testing.assert_close(loss, th.zeros_like(loss))

    def test_inactive_features_do_not_change_valid_loss(self):
        x, y = th.randn(2, 4, 5), th.randn(2, 4, 5)
        mask = th.tensor([[1, 1, 0, 0], [1, 1, 0, 0]])
        expected = mail_contrastive(x, y, mask, .2)
        x[:, 2:] += 100
        y[:, 2:] -= 100
        th.testing.assert_close(expected, mail_contrastive(x, y, mask, .2))

    def test_adaptive_gradients_and_detached_alignment_teacher(self):
        th.manual_seed(5)
        encoder = MultiViewEncoder(args_for())
        x = th.randn(2, 5, 16, requires_grad=True)
        m = th.randn(2, 5, 16, requires_grad=True)
        gcl, align, reg, metrics = encoder(x, m, th.ones(2, 5, dtype=th.bool))
        align.backward(retain_graph=True)
        self.assertIsNone(encoder.w_feature.weight.grad)
        self.assertIsNone(x.grad)
        self.assertGreater(m.grad.abs().sum(), 0)
        self.assertGreater(encoder.gate[-1].weight.grad.abs().sum(), 0)
        gcl.backward()
        self.assertGreater(encoder.w_feature.weight.grad.abs().sum(), 0)
        self.assertGreater(x.grad.abs().sum(), 0)
        self.assertAlmostEqual(sum(metrics["weight_" + k].item() for k in ("original", "feature", "topology")), 1, places=6)


if __name__ == "__main__":
    unittest.main()
