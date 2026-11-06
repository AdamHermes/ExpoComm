"""Stable per-node losses; masks exclude inactive agents from negatives too."""
import math
import torch as th
import torch.nn.functional as F


def masked_mean(values, valid):
    return values.masked_fill(~valid.bool(), 0).sum() / valid.sum().clamp_min(1)


def mail_contrastive(a, b, active, temperature):
    """MAIL Eq. 6/8/9: cross-view candidates + BOTH within-view negatives.

    Returns per-node losses. The positive is the same agent in the same
    snapshot across views. No negatives cross episode/time boundaries.
    Eq. 8's undefined h-bar-o is interpreted as h-r per its preceding text.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("contrastive temperature must be positive and finite")
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    cross = (a @ b.transpose(-1, -2)) / temperature
    aa = (a @ a.transpose(-1, -2)) / temperature
    bb = (b @ b.transpose(-1, -2)) / temperature
    valid = active.bool()
    pair = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    eye = th.eye(a.size(-2), device=a.device, dtype=th.bool)
    within = pair & ~eye
    # All-masked rows need a finite placeholder BEFORE logsumexp, otherwise
    # backward through an unused (-inf)-(-inf) row can still create NaNs.
    cross_mask = pair | (~valid).unsqueeze(-1) & eye
    logits = th.cat((cross.masked_fill(~cross_mask, -th.inf),
                     aa.masked_fill(~within, -th.inf),
                     bb.masked_fill(~within, -th.inf)), dim=-1)
    loss = th.logsumexp(logits, -1) - cross.diagonal(dim1=-2, dim2=-1)
    return loss.masked_fill(~valid, 0)
