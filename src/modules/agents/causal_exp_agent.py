"""Local sender routing with an interleaved exponential communication backbone."""
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .ExpoComm_agent import ExpoCommOAgent


class CausalExpAgent(ExpoCommOAgent):
    """One packet per sender; no current peer features are used to choose edges.

    The straight-through estimator is a biased training surrogate. Forward
    routing is hard and deterministic in both replay and evaluation.
    """

    def __init__(self, input_shape, args):
        super().__init__(input_shape, args)
        if self.n_agents < 1:
            raise ValueError("n_agents must be positive")
        self.routing_mode = getattr(args, "routing_mode", "hybrid")
        if self.routing_mode not in ("hybrid", "fixed", "adaptive", "none"):
            raise ValueError("routing_mode must be hybrid, fixed, adaptive or none")
        self.temperature = float(getattr(args, "routing_temperature", 1.0))
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("routing_temperature must be finite and positive")
        self.loss_percent = float(getattr(args, "packet_loss_percent", 0.0))
        if not 0 <= self.loss_percent <= 100:
            raise ValueError("packet_loss_percent must be between 0 and 100")
        offsets = th.tensor([2**k for k in range((self.n_agents - 1).bit_length())]
                            or [0], dtype=th.long)
        self.register_buffer("offsets", offsets, persistent=False)
        destinations = (th.arange(self.n_agents)[:, None] + offsets) % self.n_agents
        self.register_buffer("destinations", destinations, persistent=False)
        # Relative edge metadata only; parameter count does not depend on N.
        edge_features = th.stack((offsets.float() / self.n_agents,
                                 th.log2(offsets.float().clamp_min(1) + 1)
                                 / math.log2(self.n_agents + 1)), dim=-1)
        self.register_buffer("edge_features", edge_features, persistent=False)
        h = args.hidden_dim
        self.router = nn.Sequential(nn.Linear(2 * h + 2, h), nn.ReLU(), nn.Linear(h, 1))
        self.packet_encoder = nn.Sequential(nn.Linear(2 * h, h), nn.Tanh())
        self.last_routing = None

    def route(self, h, previous_message, t):
        """Return [B,N,K] gates. t is episode time, never a mutable counter."""
        b, n, _ = h.shape
        k = self.offsets.numel()
        local = th.cat((h, previous_message), dim=-1).unsqueeze(2).expand(-1, -1, k, -1)
        edges = self.edge_features.view(1, 1, k, 2).expand(b, n, -1, -1)
        logits = self.router(th.cat((local, edges), dim=-1)).squeeze(-1)
        probabilities = F.softmax(logits / self.temperature, dim=-1)
        fixed = self.routing_mode == "fixed" or (self.routing_mode == "hybrid" and t % 2 == 0)
        if fixed:
            # Hybrid's backbone advances on backbone slots, not all env steps.
            phase = t // 2 if self.routing_mode == "hybrid" else t
            selected = th.full((b, n), phase % k, device=h.device, dtype=th.long)
            gates = F.one_hot(selected, k).to(h.dtype)
        else:
            selected = probabilities.argmax(dim=-1)
            hard = F.one_hot(selected, k).to(h.dtype)
            gates = hard + (probabilities - probabilities.detach())
        if self.routing_mode == "none" or self.n_agents == 1:
            gates = gates * 0
        return gates, selected

    def forward(self, inputs, hidden_info, t, test_mode=False):
        old_h, old_m = hidden_info
        n, d = self.n_agents, self.hidden_dim
        h = self.rnn(F.relu(self.fc1(inputs)), old_h.reshape(-1, d)).reshape(-1, n, d)
        old_m = old_m.reshape(-1, n, d)
        b = h.size(0)
        gates, selected = self.route(h, old_m, t)
        # Encode BEFORE receiving: no within-step multi-hop or access to peer h.
        packet = self.packet_encoder(th.cat((h, old_m), dim=-1))
        enabled = getattr(self.args, "packet_loss_test" if test_mode else "packet_loss_train", False)
        keep = th.ones((b, n, 1), device=h.device, dtype=h.dtype)
        if enabled and self.loss_percent:
            keep = (th.rand_like(keep) >= self.loss_percent / 100).to(h.dtype)
        delivered = gates * keep
        index = self.destinations.reshape(1, -1, 1).expand(b, -1, d)
        values = (packet.unsqueeze(2) * delivered.unsqueeze(-1)).reshape(b, -1, d)
        inbox = h.new_zeros(b, n, d).scatter_add(1, index, values)
        counts = h.new_zeros(b, n, 1).scatter_add(
            1, index[:, :, :1], delivered.reshape(b, -1, 1))
        inbox = inbox / counts.clamp_min(1)
        # Losing a packet must not freeze local observation/memory updates.
        message_input = self.msg_processor(th.cat((h, inbox), dim=-1))
        message = self.msg_rnn(message_input.reshape(-1, d), old_m.reshape(-1, d))
        q = self.fc2(th.cat((h.reshape(-1, d), message), dim=-1))
        self.last_routing = {
            "destinations": self.destinations.unsqueeze(0).expand(b, -1, -1)
                .gather(2, selected.unsqueeze(-1)).squeeze(-1).detach(),
            "sent": gates.detach().sum(-1),
            "received": counts.detach().squeeze(-1),
        }
        return q, (h, message.reshape(b, n, d)), message
