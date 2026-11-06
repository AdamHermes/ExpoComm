import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .multiview_graph import graph_views, propagate
from .multiview_losses import mail_contrastive, masked_mean


class MultiViewEncoder(nn.Module):
    """MAIL SGC equations plus an explicitly separate message-alignment extension."""

    def __init__(self, args):
        super().__init__()
        self.topk = args.topk_neighbors
        self.k = args.mv_feature_k
        self.p, self.l = args.mv_local_hops, args.mv_global_hops
        self.temperature = args.mv_temperature
        self.mode = args.mv_mode
        self.feature = args.mv_feature
        self.topology = args.mv_topology
        self.cross = args.mv_cross
        self.lambda_t, self.lambda_c = args.mv_lambda_topology, args.mv_lambda_cross
        self.floor = args.mv_weight_floor
        if self.mode not in ("fixed", "adaptive"):
            raise ValueError("mv_mode must be fixed or adaptive")
        if not (0 < self.p < self.l) or self.k < 0:
            raise ValueError("Require 0 < local hops < global hops and feature_k >= 0")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("mv_temperature must be positive and finite")
        if not 0 <= self.floor < 1:
            raise ValueError("mv_weight_floor is a mixture fraction in [0,1)")
        if any(not math.isfinite(v) or v < 0 for v in (self.lambda_t, self.lambda_c)):
            raise ValueError("GCL weights must be finite and nonnegative")
        h, d = args.hidden_dim, args.mv_dim
        # No bias/nonlinearity: H = S^p X W (MAIL Eq. 4).
        self.w_feature = nn.Linear(h, d, bias=False)
        self.w_original = nn.Linear(h, d, bias=False)
        self.w_topology = nn.Linear(h, d, bias=False)
        self.student = nn.Linear(h, d, bias=False)
        self.gate = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 3))
        # Start at the same uniform mixture used by the fixed control.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def encode(self, x, active):
        s, sf = graph_views(x, active, self.topk, self.k if self.feature else 0)
        local = propagate(s, x, self.p)
        ho = self.w_feature(local)
        hf = ho + self.w_feature(sf @ x) if self.feature else ho
        hr = self.w_original(local)
        # S_G = S, one of MAIL Eq. 7's two permitted choices.
        ht = self.w_topology(propagate(s, x, self.l)) if self.topology else hr
        return ho, hf, hr, ht

    def forward(self, x, message, active):
        ho, hf, hr, ht = self.encode(x, active)
        zero = ho.sum(-1) * 0
        lf = mail_contrastive(ho, hf, active, self.temperature) if self.feature else zero
        lt = mail_contrastive(hr, ht, active, self.temperature) if self.topology else zero
        lc = mail_contrastive(ho, hr, active, self.temperature) if self.cross else zero
        gcl = masked_mean(lf + self.lambda_t * lt + self.lambda_c * lc, active)

        enabled = th.tensor([True, self.feature, self.topology], device=x.device)
        uniform = enabled.to(x.dtype) / enabled.sum()
        if self.mode == "adaptive":
            # Local-only input. Detach so gate optimization cannot distort h/m
            # simply to choose an easier view. Gate still receives gradients.
            logits = self.gate(th.cat((x.detach(), message.detach()), dim=-1))
            soft = F.softmax(logits.masked_fill(~enabled, -th.inf), -1)
            weights = (1 - self.floor) * soft + self.floor * uniform
        else:
            weights = uniform.expand(*x.shape[:-1], 3)
        z = F.normalize(self.student(message), dim=-1)
        targets = th.stack((ho, hf, ht), dim=-2).detach()
        targets = F.normalize(targets, dim=-1)
        distances = 1 - (z.unsqueeze(-2) * targets).sum(-1)
        alignment = masked_mean((weights * distances).sum(-1), active)
        # KL(w || uniform) on enabled views, plus a positive uniform floor.
        kl = (weights * (weights.clamp_min(1e-8).log()
                         + enabled.sum().to(x.dtype).log())).sum(-1)
        regularizer = masked_mean(kl, active)
        metrics = {"feature_loss": masked_mean(lf, active),
                   "topology_loss": masked_mean(lt, active),
                   "cross_loss": masked_mean(lc, active)}
        for i, name in enumerate(("original", "feature", "topology")):
            metrics["weight_" + name] = masked_mean(weights[..., i], active)
        return gcl, alignment, regularizer, metrics
