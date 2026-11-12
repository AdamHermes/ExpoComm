# VA-ExpoImpute MLP: execution integration

This is the initial execution-time implementation of the residual latent
message imputer.  It is deliberately separate from the value-aware training
loss, which will be added later.

## Files

- `src/modules/communication/message_imputer.py`: receiver-side residual MLP.
- `src/controllers/ExpoComm_controller.py`: episodic sender-indexed cache,
  packet-mask sampling, and recovery context construction.
- `src/modules/agents/ExpoComm_agent.py`: one-peer recovery insertion point.

## Runtime rule

For a selected remote sender `j`, the controller stores the last genuinely
delivered source message `m_j^(t-1)` in a cache indexed by `(receiver, sender)`.
On loss, if that entry is valid and no older than `packet_cache_ttl`, the agent
uses:

```text
r_hat = cache[i, j] + MLP(h_i, m_i, cache[i, j], normalized_age, offset)
```

Otherwise it performs the self-update fallback, using `m_i` as the remote input.
The cache is reset per episode.  It is updated only for a successful remote
delivery; self links are never cached and a lost packet never reads its sender
message for recovery.

The MLP final layer is zero-initialized.  Therefore, before imputer training,
`mse_imputer` is exactly the TTL-cache method for valid cache entries and
self-update otherwise.

## Enable an evaluation

The existing default remains unchanged (`packet_loss_recovery: hold`).  To
enable this code path:

```bash
python src/main.py ... \
  --packet-loss-percent=50 \
  --packet-loss-mode=test \
  --packet-loss-recovery=mse_imputer \
  --packet-cache-ttl=8
```

This code path can load an old ExpoComm `agent.th`: missing
`message_imputer.*` keys are allowed only in `mse_imputer` mode.  All old
ExpoComm parameter keys remain strict.

## Plain-MSE training

`src/learners/mse_imputer_learner.py` is a separate learner, not a modified
Q-learner. It freezes every original ExpoComm parameter and gives Adam only
`message_imputer.parameters()`. It does not instantiate a target network, does
not compute a TD loss, and deliberately does not load `opt.th` from the RL
checkpoint.

The trainer rolls frozen ExpoComm through replay episodes under sequential 70%
IID message loss. At each valid loss position, centralized training obtains the
target sender latent `m_j^(t-1)`, while the predictor sees only receiver-side
inputs. It optimizes only when all of these hold:

```text
selected link is remote AND packet was lost AND receiver cache is valid/fresh
```

The plain supervised objective is:

```text
L_MSE = mean(|| cache[i,j] + MLP(receiver-side context) - m_j^(t-1) ||²)
```

The trainer logs `imputer_mse`, `imputer_cache_mse`, and
`imputer_eligible_count`. The latter must be non-zero; otherwise no valid
imputation examples were available.

Train on the cluster from the repository root:

```bash
mkdir -p slurm
sbatch run_mse_imputer_train.sh
```

The job uses `mse_ExpoImpute_n6` and
`ExpoComm_Battle25_mse_imputer_5009240`. It saves standalone imputer files as:

```text
work_dirs/ExpoComm_Battle25_mse_imputer/mse_ExpoImpute_n6/
  ExpoComm_Battle25_mse_imputer_5009240/<timestamp>/models/<step>/imputer.th
```

Evaluate a saved imputer with the usual execution job, passing its exact path:

```bash
PACKET_LOSS_TEST_EPISODES=100 \
PACKET_IMPUTER_CHECKPOINT=/absolute/path/to/imputer.th \
sbatch run_mse_imputer_eval.sh \
  iql_ExpoComm_one_peer_n6 MAgent_Battle \
  ExpoComm_Battle25_eval_5009240 70 8
```

The untrained execution path remains a TTL-cache/self-update baseline because
the residual MLP head is initialized to zero. The MSE checkpoint above is the
first learned-predictor comparison. The later Value-Aware method must use the
same architecture, cache TTL, rollout distribution, number of updates, and
evaluation protocol, changing only the per-example MSE weights.
