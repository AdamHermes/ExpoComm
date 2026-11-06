"""Graph views for MAIL-inspired training (not execution communication edges).

All adjacency matrices have receiver rows and sender columns. Views are
undirected unions before symmetric degree normalization, as documented.
"""
import torch as th
import torch.nn.functional as F


def normalize_adjacency(adjacency, active, self_loops=False):
    valid = active.bool()
    pair = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    a = adjacency.to(dtype=th.float32) * pair
    if self_loops:
        a = a.clone()
        a.diagonal(dim1=-2, dim2=-1).copy_(valid.to(a.dtype))
    inv = a.sum(-1).clamp_min(1).rsqrt() * valid
    return inv.unsqueeze(-1) * a * inv.unsqueeze(-2)


def exponential_adjacency(n, topk, device):
    if n < 1 or topk < 1:
        raise ValueError("n and topk must be positive")
    # Match ExpoComm's candidate offsets, including its self slot. Collapse
    # duplicate edges and symmetrize only for the training graph encoder.
    offsets = th.tensor([0] + [2**k for k in range(topk - 1)], device=device)
    rows = th.arange(n, device=device)
    cols = (rows[:, None] + offsets) % n
    a = th.zeros(n, n, device=device)
    a[rows[:, None], cols] = 1
    return th.maximum(a, a.T)


def feature_adjacency(x, active, k):
    if k < 0:
        raise ValueError("feature k must be nonnegative")
    n = x.shape[-2]
    a = x.new_zeros(*x.shape[:-1], n)
    if n == 1 or k == 0:
        return a
    # kNN membership is discrete; representation gradients flow through XW.
    with th.no_grad():
        z = F.normalize(x.detach(), dim=-1)
        similarity = z @ z.transpose(-1, -2)
        valid = active.bool()
        allowed = valid.unsqueeze(-1) & valid.unsqueeze(-2)
        allowed &= ~th.eye(n, dtype=th.bool, device=x.device)
        scores = similarity.masked_fill(~allowed, -th.inf)
        _, index = scores.topk(min(k, n - 1), dim=-1)
        keep = allowed.gather(-1, index)
        a.scatter_(-1, index, keep.to(a.dtype))
        a = th.maximum(a, a.transpose(-1, -2))
    return a


def graph_views(x, active, topk, feature_k):
    a = exponential_adjacency(x.size(-2), topk, x.device)
    original = normalize_adjacency(a.expand(*x.shape[:-2], -1, -1), active, True)
    feature = normalize_adjacency(feature_adjacency(x, active, feature_k), active, False)
    return original.to(x.dtype), feature.to(x.dtype)


def propagate(s, x, hops):
    if hops < 1:
        raise ValueError("SGC hops must be positive")
    for _ in range(hops):
        x = s @ x
    return x
