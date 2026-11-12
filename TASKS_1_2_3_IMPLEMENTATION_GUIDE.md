# ExpoComm Packet-Loss Recovery: Implementation Guide for Tasks 1--3

## Purpose and scope

This document specifies three **evaluation-time recovery baselines** for the
one-peer version of ExpoComm.  They are intentionally ordered from simple to
more ambitious:

1. **Self-update** -- update the receiver's message RNN using its own previous
   message when a remote packet is lost.
2. **Per-sender TTL cache** -- substitute the last packet successfully received
   from the required sender, while it is still fresh.
3. **MSE message imputer** -- a small frozen-policy predictor that refines a
   valid cached packet.  This is supervised learning only; it is *not* an RL
   retraining run.

These tasks must be implemented as alternatives to the current packet-loss
behaviour, not as changes to the trained ExpoComm policy.  The original
checkpoint must load without missing or unexpected model parameters when the
new options are disabled.

The target controller is `src/controllers/ExpoComm_controller.py`; the target
agent is `src/modules/agents/ExpoComm_agent.py`.  The existing packet-loss
switches live in `src/config/default.yaml` and are parsed in
`src/config_overrides.py`.

This guide concerns `ExpoCommOAgent` / `one_peer: true`.  Do not silently apply
these definitions to the static-attention (`ExpoCommSAgent`) path: it has
multiple received packets and needs a separate aggregation design.

## Shared terminology and timing convention

At action step `t`, each receiver `i` has the following quantities:

| Symbol | Code-level meaning |
| --- | --- |
| `h_i^t` | The local recurrent hidden state produced by `self.rnn`. |
| `m_i^(t-1)` | The receiver's message memory before communication at this step (`other_msg` before it is replaced). |
| `j` | Sender selected by ExpoComm's exponential schedule.  In one-peer mode it is `topk_indices[b, i]`. |
| `r_i^t = m_j^(t-1)` | The **remote input**: the sender's learned latent message, not the sender's raw observation. |
| `d_i^t` | Packet-delivery Boolean for link `j -> i`. |
| `m_i^t` | Message memory after the communication update; it is concatenated with `h_i^t` before the Q head. |

The clean, existing one-peer transition is:

```text
u_i^t = msg_processor([h_i^t, m_j^(t-1)])
m_i^t = msg_rnn(u_i^t, m_i^(t-1))
```

The current loss transition is the **hold** baseline:

```text
if d_i^t: m_i^t = clean transition
else:     m_i^t = m_i^(t-1)
```

`packet_loss_percent = L` means a non-self remote link is dropped with
probability `L / 100`.  A zero-offset self slot (`j == i`) is local computation,
not a network packet, and must always be delivered.  This rule is already
implemented by `_packet_keep_mask`.

### Non-negotiable decentralized-execution rule

After a loss is sampled at execution, the receiver may use only:

```text
its own h_i, its own m_i, its own stored cache/predictor state,
the sender identity/offset, and the delivery flag.
```

It must **not** read `m_j`, `h_j`, `obs_j`, the global state, reward, or an
environment-private object after discovering that `d_i^t == False`.  Reading
`m_j` to form a target is allowed only in centralized/offline training for
Task 3, never in test-time recovery.

## Coordination and integration contract

The three developers should work on separate branches.  Each branch should be
rebased onto the current packet-loss branch immediately before hand-off; do not
merge simultaneous edits to the same two Python files without reconciling them
manually.

Use one shared configuration field:

```yaml
packet_loss_recovery: "hold"
```

Permitted values are exactly:

```text
hold | self_update | ttl_cache | mse_imputer
```

The default must be `hold`.  Preserve current results exactly under this value.
Unknown values must raise a clear `ValueError` during startup, rather than
quietly behaving as `hold`.

Recommended shared optional settings:

```yaml
packet_cache_ttl: 8                 # non-negative integer; 0 means only same-step cache use
packet_imputer_hidden_dim: 128
packet_imputer_age_scale: 8.0
packet_imputer_use_cache_only: true # required default for the first version
```

The command-line parser may expose only the recovery mode and TTL if desired,
but YAML configuration is sufficient for the first implementation.  Do not
change the meaning of `--packet-loss-percent` or `--packet-loss-mode`.

### Required recurrent recovery state

Tasks 2 and 3 need state that survives from one environment step to the next
and is reset at every new episode.  Maintain, per batch element and receiver:

```text
cache[b, receiver, sender, hidden_dim]  # last successfully delivered remote message
age[b, receiver, sender]                # number of completed environment steps since that success
valid[b, receiver, sender]              # whether that cache entry exists
```

The required shapes are `(B, N, N, H)`, `(B, N, N)`, and `(B, N, N)`.
The `receiver` dimension is necessary even though agents share parameters:
what agent `i` last received from sender `j` is not necessarily what agent `k`
last received from the same sender.

Store this state together with `h` and `msg` in `ExpoCommMAC.hidden_states`, or
in a clearly named, controller-owned state object.  It must be reset by
`ExpoCommMAC.init_hidden(batch_size)`.  It must not persist across evaluation
episodes, environments in the batch, or random seeds.

For economy, the controller may update only the currently selected sender
entry.  However, `age` still has a precise meaning: increment an already-valid
entry once per environment step; reset it to zero only after a successfully
delivered **remote** packet from that sender.  Do not cache self-slots.

## Task 1 -- Self-update baseline

### Objective

Implement the following counterfactual recovery rule:

```text
if packet is delivered:
    remote_input = m_j^(t-1)
elif the selected slot is remote:
    remote_input = m_i^(t-1)
else:  # self slot; it is always delivered
    remote_input = m_i^(t-1)

m_i^t = msg_rnn(msg_processor([h_i^t, remote_input]), m_i^(t-1))
```

In short: a lost remote packet does **not** freeze the message RNN.  The RNN is
updated using its own prior memory as a deliberate no-new-remote-information
input.

### Why this is a necessary baseline

The existing hold rule conflates two effects:

1. remote information is absent; and
2. the message memory has no recurrent update at all.

Self-update removes (2) while retaining (1).  It is not an imputation method
and does not claim to reconstruct the sender's information.

### Exact code change

In `ExpoCommOAgent._communicate`:

1. Keep the clean path byte-for-byte equivalent when `keep_mask` is true.
2. Build `fallback_input = msg_ego` with shape `(B, N, H)`.
3. For a loss, select `fallback_input` instead of `msg_received` *before*
   `msg_processor` and `msg_rnn`.
4. Always compute the `candidate_msg` and return it under
   `packet_loss_recovery == "self_update"`.
5. Under `"hold"`, preserve the current `torch.where(keep_mask,
   candidate_msg, msg_ego)` exactly.

Do not replace the message with a zero vector.  That is a different baseline.
Do not use the hidden state `h_i` alone as the fallback; the definition above
uses the old message memory as the missing remote input.

### Required tests

Use deterministic tensor tests, preferably in a new `tests/` file.

1. **No loss equivalence.** With recovery set to `hold` and `self_update`, zero
   loss (or `keep_mask=True`) gives identical `q`, `h`, and `m` to the current
   implementation within floating-point tolerance.
2. **Forced remote loss.** For a remote sender and a forced `keep_mask=False`,
   show that `hold` returns `m_ego`, whereas `self_update` returns the result of
   manually calling `msg_rnn(msg_processor([ego_h, msg_ego]), msg_ego)`.
3. **No sender leak.** Change `m_j` while holding `h_i`, `m_i`, the mask, and
   model weights fixed.  The self-update output under loss must not change.
4. **Self-link invariant.** At 100% configured loss, `j == i` remains a clean
   local update and does not take the loss branch.
5. **Checkpoint compatibility.** Load one existing checkpoint and run at least
   one clean evaluation episode using the default configuration.

### Deliverable

A small, focused commit with the new configuration option, implementation, and
tests.  Include a one-paragraph note showing one evaluation command and the
result of tests.  Do not implement caching or a predictor in this task.

## Task 2 -- Per-sender TTL cache baseline

### Objective

When a required remote packet is missing, reuse the last successfully
delivered payload from that *same sender* if it is still fresh; otherwise use
the Task-1 self-update fallback.

```text
if delivered remote packet from j:
    cache[i, j] = m_j^(t-1)
    valid[i, j] = True
    age[i, j] = 0
    remote_input = m_j^(t-1)
elif lost remote packet and valid[i, j] and age[i, j] <= TTL:
    remote_input = cache[i, j]
else:
    remote_input = m_i^(t-1)

m_i^t = msg_rnn(msg_processor([h_i^t, remote_input]), m_i^(t-1))
```

The final fallback deliberately matches Task 1.  This makes the comparison
interpretable: the difference between `ttl_cache` and `self_update` is only
the availability of a fresh, receiver-specific cached remote packet.

### Cache update order (important)

At a delivered remote link, use the incoming source message as it existed
*before* this step's communication update: `m_j^(t-1)`.  Update the cache with
that value and use the same value for this step's clean transition.

At a lost link, do **not** overwrite, refresh, or inspect that sender's cache.
The value `m_j^(t-1)` may exist in the centralized simulation tensor but is
unavailable to the receiver; looking at it is data leakage.

At a self-link (`j == i`), make the usual local clean transition but do not
mark a remote cache entry valid or reset a remote cache age.  Self computation
does not represent network delivery.

Increment ages after selecting the recovery input for a step.  With this
convention, a packet received at step `t` is age `0` at that step and can be
used for the next `TTL + 1` selection events whose age is at most `TTL`.
Document this convention in code comments and do not change it silently.

### Where to implement it

The main state lifecycle belongs in `ExpoCommMAC`:

- `init_hidden(batch_size)` creates zero cache/age/valid tensors on the same
  device and with the same batch dimension as hidden states.
- `forward()` already knows `topk_indices`, current `t`, and the pre-step
  message state.
- The agent should receive only the selected cache entry, validity/freshness
  flag, and recovery mode needed to choose the remote input.  Alternatively,
  the agent can return metadata needed by the controller.  Either is acceptable
  if a delivery mask is sampled exactly once.

Avoid sampling the packet mask separately in the controller and agent: two
independent masks make the cache disagree with the actual communication
transition.  Refactor `_packet_keep_mask` into one shared call or return the
sampled mask as metadata.

The implementation must work with `B > 1`, even though most evaluation runs
currently use `batch_size_run: 1`.

### Required tests

1. **Correct sender index.** Create a schedule in which receiver `i` receives
   sender `j`, then sender `k`.  A cache populated from `j` must never be used
   when `k` is selected.
2. **No update on loss.** Populate `cache[i,j]`, force a loss, then change the
   centralized source `m_j`.  Cache contents and recovered output must remain
   unchanged.
3. **TTL boundary.** Verify that age `TTL` uses the cache and age `TTL + 1`
   falls back to self-update.  Test `TTL=0` explicitly.
4. **Reset.** After `mac.init_hidden()`, every cache entry is invalid.  A loss
   on the first remote communication step must use self-update, never zeros as
   a supposedly valid message.
5. **Perfect channel equivalence.** With 0% loss, cache mode must produce the
   same action/Q/message trajectory as `hold` and current ExpoComm.
6. **One packet mask only.** Add a test or instrumentation proving that the
   packet used for cache update is the packet used for message recovery.

### Deliverable and reporting

Provide `ttl_cache` with `packet_cache_ttl` exposed in YAML, unit tests, and a
short sweep at `TTL in {0, 2, 4, 8}` for at least loss 50% and 70%.  The paper
comparison should use the best TTL selected on a held-out validation seed, then
freeze that TTL for the final test seeds.  Do not select TTL independently on
each test map/loss and call that a single method.

## Task 3 -- Frozen-policy MSE message imputer

### Objective

Implement a lightweight predictor that improves a **valid cached** remote
message before it enters the unchanged ExpoComm message RNN.  This is the
plain-MSE baseline for the later value-aware method; it should not contain any
Q-value, advantage, state-reconstruction, or priority weighting.

The first reliable version is cache-conditioned only:

```text
if a remote packet is lost and cache[i, j] is valid and fresh:
    r_hat = cache[i, j] + g_phi([h_i^t, m_i^(t-1), cache[i, j], age/T])
    remote_input = r_hat
else if remote packet is lost:
    remote_input = m_i^(t-1)     # identical self-update fallback
else:
    remote_input = m_j^(t-1)     # clean delivery
```

`g_phi` predicts a residual, not the entire message.  Initialize the final
linear layer to zero (or very close to zero), so an untrained imputer behaves
like the TTL-cache baseline rather than injecting arbitrary noise.

The predictor must have no access to the true `m_j` at execution.  It may use
the receiver's local hidden state, receiver message memory, cached packet,
normalized cache age, and selected sender/offset encoding.  A two-layer MLP
with ReLU and width 128 is the recommended deadline-safe architecture; an LSTM
is unnecessary because `h_i` and `m_i` already summarize local history.

### Why this differs from Value-Aware MARO

Value-Aware MARO predicts missing shared *observations*.  Here the target is a
sender's recurrent **latent message** `m_j^(t-1)`.  That target is meaningful
only during centralized training/collection.  At execution, the model must
operate solely from receiver-side information.  This task is deliberately
plain MSE; the project owner will later add value-aware weights as Task 4.

### Recommended module and inputs

Add a clearly named module owned by `ExpoCommOAgent`, for example:

```python
self.message_imputer = nn.Sequential(
    nn.Linear(3 * hidden_dim + 1 + offset_dim, imputer_hidden_dim),
    nn.ReLU(),
    nn.Linear(imputer_hidden_dim, hidden_dim),
)
```

The input concatenation is:

```text
[ h_i^t, m_i^(t-1), cache[i,j], normalized_age, offset_or_sender_embedding ]
```

For the initial version, a one-hot index of the current exponential offset is
preferable to absolute sender ID.  It respects parameter sharing and is more
likely to generalize across cyclic agent positions.  If it is inconvenient to
obtain the offset, omit it rather than adding a brittle mapping; document that
choice.

Use `normalized_age = min(age, TTL + 1) / max(TTL, 1)` and only call the
imputer for a valid, fresh cache.  This prevents a first-step all-zero cache
from being treated as an observed remote message.

### Training protocol: no RL retraining

Freeze all existing ExpoComm weights, including `fc1`, `rnn`,
`msg_processor`, `msg_rnn`, `fc2`, and `predict_net`.  Optimize only
`message_imputer`.

1. Load the already trained clean-communication checkpoint.
2. Collect clean trajectories.  Record enough information at each step to
   reconstruct receiver-side predictor inputs and the clean target
   `r_star = m_j^(t-1)`.  The target is collected centrally and is never used
   at execution.
3. During imputer training, simulate IID packet masks (preferably mix 30%,
   50%, and 70% loss) sequentially over episodes so cache ages and loss bursts
   are represented.  Do not train each time step as an unrelated sample.
4. Apply MSE only at positions where the packet is remote, is dropped, and a
   valid fresh cache exists:

   ```text
   L_MSE = mean(|| r_hat - stop_gradient(r_star) ||_2^2)
   ```

5. Keep trajectories with no eligible positions out of the loss denominator.
   Log `eligible_imputation_count` so a small loss cannot be mistaken for a
   good model that rarely ran.
6. Save the imputer separately (recommended) or in a clearly versioned
   checkpoint.  The base ExpoComm checkpoint checksum/path must be recorded.

It is acceptable for the first offline trainer to use teacher-forced clean
actions and recorded local observations.  State this limitation in the report
and evaluate the final imputer in closed-loop environment rollouts.  Do not
claim that supervised MSE alone guarantees return improvement.

### What not to train

- Do not enable `packet_loss_train` and resume full Q-learning as a shortcut.
  That changes the policy and makes the comparison with frozen execution
  recovery ambiguous.
- Do not optimize the Q-learning TD loss, state reconstruction loss, or
  `predict_net` in this task.
- Do not use global state as an imputer input.  It is unavailable under
  decentralized execution.
- Do not train or evaluate on the same random seeds used for final reporting.

### Required tests

1. **Initialization equals cache.** With a zero-initialized residual head,
   `mse_imputer` and `ttl_cache` produce equal recovered inputs/outputs before
   imputer training.
2. **No leak under loss.** Change the true current sender message `m_j` while
   keeping all receiver-side variables fixed.  `r_hat` and the final Q values
   in the loss branch must not change.
3. **Clean-path equivalence.** With packet delivery, output is independent of
   imputer weights and equals current ExpoComm.
4. **Gradient isolation.** A training backward pass produces gradients for
   `message_imputer` but none for all frozen ExpoComm modules.
5. **Checkpoint round trip.** Save/load the imputer and obtain identical
   recovery output for a fixed test tensor.
6. **Closed-loop smoke test.** Run at least five evaluation episodes at 50%
   loss with a trained imputer; report return/win rate and no runtime errors.

### Deliverable

Submit:

- the `mse_imputer` mode and its unit tests;
- a reproducible collection/training command or script;
- a small JSON/CSV training log containing MSE and eligible-position count;
- the imputer checkpoint plus the exact base checkpoint used;
- one short evaluation table comparing `self_update`, `ttl_cache`, and
  `mse_imputer` at identical seeds and loss masks.

## Common evaluation protocol

All three tasks must be compared using the same base checkpoint, map, seed
set, loss rate, evaluation episode count, and action-selection settings.  Use
greedy evaluation.  At minimum report loss rates 0%, 50%, and 70%; retain 100%
as a stress case but do not expect cache-based methods to help if no remote
packet is ever received.

Before making performance claims, verify that the evaluation harness actually
honours distinct seeds and that the environment reset/side protocol is correct.
Past audit notes are in `reports/packet_loss/analysis/AUDIT_VI.md`.

For each run, record:

```text
base checkpoint path and load step
git commit hash
recovery mode and all recovery hyperparameters
packet-loss mode and percent
environment/map, requested seed, effective seed
number of evaluation episodes
mean and standard deviation of return / win rate
```

The final table must include `hold`, `self_update`, `ttl_cache`, and
`mse_imputer`.  `hold` is the primary no-recovery packet-loss baseline, not
the clean no-loss method.

## Final pre-merge checklist

- [ ] Default `packet_loss_recovery: hold` produces the old behavior.
- [ ] Clean communication produces an equivalent trajectory for every mode.
- [ ] Self slots never suffer a packet loss and never populate a remote cache.
- [ ] Lost packets never inspect or update from the real sender tensor.
- [ ] Cache is indexed by both receiver and sender, has a validity flag, and
      resets every episode.
- [ ] A packet delivery mask is sampled once per link/step and reused for the
      recovery decision and cache update.
- [ ] All new tensors follow the batch device and work for `B > 1`.
- [ ] Existing clean checkpoint loads under default configuration.
- [ ] Unit tests and a short closed-loop evaluation run pass.

