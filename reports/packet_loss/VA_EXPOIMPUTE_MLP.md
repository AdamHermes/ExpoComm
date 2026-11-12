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

## Current limitation

The MLP exists and is safely wired into execution, but has not yet been
supervised-trained.  The next step is an offline trainer that freezes ExpoComm,
optimizes only `message_imputer`, and later replaces plain MSE by the
counterfactual communication-value-aware MSE loss.
