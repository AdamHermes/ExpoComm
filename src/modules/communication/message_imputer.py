"""Receiver-side latent-message imputation modules.

The module in this file deliberately knows nothing about packet sampling,
sender tensors, or environment state.  It consumes only information that a
receiver can retain under decentralized execution.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ResidualMessageImputer(nn.Module):
    """Predict a correction to a receiver's cached remote latent message.

    The zero-initialised final projection makes a newly created imputer exactly
    equivalent to using the cache alone.  This is useful both as a safe
    execution fallback and as a clean starting point for offline MSE training.
    """

    def __init__(self, hidden_dim, num_offsets, imputer_hidden_dim):
        super().__init__()
        if hidden_dim <= 0 or num_offsets <= 0 or imputer_hidden_dim <= 0:
            raise ValueError("imputer dimensions must be positive")

        self.hidden_dim = hidden_dim
        self.num_offsets = num_offsets
        input_dim = 3 * hidden_dim + 1 + num_offsets
        self.net = nn.Sequential(
            nn.Linear(input_dim, imputer_hidden_dim),
            nn.ReLU(),
            nn.Linear(imputer_hidden_dim, hidden_dim),
        )

        # Before supervised imputer training, predict a zero residual.  This
        # preserves the TTL-cache baseline exactly rather than injecting noise.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ego_h, ego_msg, cached_msg, normalized_age, offset_index):
        """Return a cached-message residual prediction.

        Args:
            ego_h: ``(..., H)`` receiver local RNN state at the current step.
            ego_msg: ``(..., H)`` receiver message memory before communication.
            cached_msg: ``(..., H)`` last valid remote message for this sender.
            normalized_age: ``(..., 1)`` bounded age of that cache entry.
            offset_index: ``(...)`` exponential-neighbour slot index.
        """
        if ego_h.shape != ego_msg.shape or ego_h.shape != cached_msg.shape:
            raise ValueError("ego_h, ego_msg, and cached_msg must share a shape")
        if ego_h.shape[-1] != self.hidden_dim:
            raise ValueError("unexpected hidden dimension for message imputer")
        if normalized_age.shape != ego_h.shape[:-1] + (1,):
            raise ValueError("normalized_age must have shape (..., 1)")
        if offset_index.shape != ego_h.shape[:-1]:
            raise ValueError("offset_index must have shape (...)")

        offset = F.one_hot(offset_index.long(), num_classes=self.num_offsets)
        offset = offset.to(dtype=ego_h.dtype)
        features = th.cat(
            [ego_h, ego_msg, cached_msg, normalized_age.to(ego_h.dtype), offset],
            dim=-1,
        )
        return cached_msg + self.net(features)
