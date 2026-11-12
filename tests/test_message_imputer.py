import sys
from pathlib import Path

import torch as th


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.communication import ResidualMessageImputer


def test_new_imputer_equals_cache():
    th.manual_seed(7)
    imputer = ResidualMessageImputer(hidden_dim=4, num_offsets=3, imputer_hidden_dim=8)
    ego_h = th.randn(2, 5, 4)
    ego_msg = th.randn(2, 5, 4)
    cached_msg = th.randn(2, 5, 4)
    age = th.rand(2, 5, 1)
    offset = th.tensor([[0, 1, 2, 0, 1], [2, 1, 0, 2, 1]])

    output = imputer(ego_h, ego_msg, cached_msg, age, offset)

    assert th.equal(output, cached_msg)


def test_imputer_does_not_consume_sender_tensor():
    imputer = ResidualMessageImputer(hidden_dim=2, num_offsets=2, imputer_hidden_dim=4)
    ego_h = th.ones(1, 1, 2)
    ego_msg = th.zeros(1, 1, 2)
    cached_msg = th.tensor([[[3.0, -2.0]]])
    age = th.zeros(1, 1, 1)
    offset = th.zeros(1, 1, dtype=th.long)

    first = imputer(ego_h, ego_msg, cached_msg, age, offset)
    second = imputer(ego_h, ego_msg, cached_msg, age, offset)

    assert th.equal(first, second)
    assert th.equal(first, cached_msg)
