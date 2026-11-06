# CausalExp: local adaptive routing with an exponential backbone

Research prototype implemented in ExpoComm for Pursuit/Battle with auxiliary
IQL. No SOPS code is used or changed. This is a testable research hypothesis,
not an established novelty or performance claim. A literature novelty audit
and benchmark training are still required.

## Research question

Can a locally chosen, hard communication graph improve coordination at a
fixed outgoing packet budget, while scheduled exponential edges preserve
temporal communication paths?

The fixed ExpoComm topology does not choose edges from task context. This
variant learns sender decisions without reading potential recipients' current
observations/hidden states. It combines adaptive edges with scheduled coverage.
The learned graph is a directed message-passing GNN: encode node state, select
edges, aggregate incoming packets, then update each node's recurrent memory.
It uses no external graph library, graph clustering or contrastive graph loss.

## Information and tensor flow

For batch B, N agents, hidden width H (default 64), K=ceil(log2(N)):

- Local observation input: `[B*N, O]`, using the existing environment preprocessing.
- Local recurrent hidden h and communication memory m: `[B,N,H]`.
- Outgoing packet z: `[B,N,H]`; **a learned float vector, not a flattened minimap**.
- Candidate destinations: `[N,K]`, `(i + 2^k) mod N`, without self edges for N>1.
- Routing gates: `[B,N,K]`; one nonzero per sender in the forward pass.
- Mean incoming packet and updated memory: `[B,N,H]`.
- Q-values: `[B,N,A]`; auxiliary state predictions: `[B,N,S]`.

With h_i^{-1}=m_i^{-1}=0, one environment step t computes:

```text
h_i^t = GRU_obs(ReLU(W_obs o_i^t), h_i^{t-1})
z_i^t = tanh(MLP_packet([h_i^t, m_i^{t-1}]))
ell_ik^t = MLP_router([h_i^t, m_i^{t-1}, edge_metadata(k,N)])
p_ik^t = softmax_k(ell_ik^t / temperature)
```

`edge_metadata` contains the relative offset/N and log2(offset+1)/log2(N+1).
It contains no recipient observations. It does assume a shared, fixed agent
ordering, known N and addressable recipients; it is not permutation invariant.

In hybrid mode, even t uses candidate `(t//2) mod K`. Odd t uses argmax(p).
On adaptive slots, training uses `one_hot(argmax(p)) + (p - stop_gradient(p))`.
The forward graph is hard; the backward gradient is a biased straight-through
surrogate. There is no claim that this is the true discrete routing gradient.
Routing is deterministic for the same history and weights; action exploration
continues to use the existing epsilon-greedy selector.

```text
inbox_i^t = mean({z_j^t : selected_destination(j,t)=i and packet delivered})
           # zero when no incoming packets
m_i^t = GRU_msg(ReLU(W_msg [h_i^t, inbox_i^t]), m_i^{t-1})
Q_i^t = MLP_Q([h_i^t, m_i^t])
state_prediction_i^t = MLP_aux(m_i^t)
action_i^t = epsilon_greedy(Q_i^t, available_actions_i^t)
```

All outgoing packets are constructed before receiving. Information received
at step t can be forwarded at t+1, not again at t. Local memory still updates
when all packets are dropped. The auxiliary global state is a training target;
it is not input to routing or execution. Rewards enter the existing TD target.

## Training and implementation

The model remains model-free, off-policy recurrent Double DQN/IQL with shared
agent parameters. The existing replay and auxiliary learner unroll complete
episodes, including their final bootstrap observation. Each sampled episode
starts with zero hidden states; routing phase is derived from episode time.

```text
y_i^t = r_i^t + gamma*(1-done_t)*Q_target_i^{t+1}(argmax_a Q_online_i^{t+1}(a))
L = L_masked_TD + aux_coef * L_masked_state_reconstruction
```

Reward normalization, padding/terminal masks, Double-Q, clipping and target
updates are inherited unchanged. The reconstruction term sums squared errors
over state dimensions before averaging valid agent steps, as the original
learner does. Router and packet encoder are agent submodules, so the existing
optimizer, hard/soft target update and checkpoints include them automatically.
Old ExpoComm checkpoints are not architecture-compatible with this agent.

Implementation:

- `src/modules/agents/causal_exp_agent.py`: packet encoding, local routing,
  sparse scatter aggregation, recurrent memory.
- `src/controllers/causal_exp_controller.py`: episode time and existing Q/aux API.
- `src/config/algs/iql_causal_exp*.yaml`: main algorithm and three controls.
- `tests/test_causal_exp.py`: CPU contracts and real auxiliary learner update.

The agent exposes detached `last_routing` diagnostics (destinations, sent and
received counts) for the latest forward. These are not automatically logged
by the existing runner, nor are they a history of an entire rollout.

## Budget, reachability and limitations

For N>1, hybrid/fixed/adaptive send exactly N packets per step before drops.
One sender chooses one destination, but a recipient can receive multiple
packets. Mean aggregation handles collisions; no receiver capacity limit is
implemented. No-communication mode and N=1 send zero network packets.
Each packet carries H floats plus routing metadata. Actual communication is
O(NH); scoring/training work includes all K candidates, O(NKH) intermediate
storage, without a dense N-by-N adjacency. MLP compute additionally depends on H.

The fixed shifts 1,2,4,... allow potential paths from every source to every
destination over K consecutive backbone slots: any displacement 0..N-1 has a
binary representation. In hybrid mode these slots occur within 2K steps.
This is a graph-path guarantee assuming delivery, retention between slots,
fixed membership, shared time and no failed relays. **It does not guarantee
that a learned finite-dimensional GRU preserves every agent's information.**
Adaptive slots cannot erase graph edges already scheduled, but neural memory
can forget information. Lossy links invalidate the delivery guarantee.

The router knows only its local history and edge metadata, not recipient need.
It may converge to an unhelpful constant offset. Dead/inactive agent slots
follow the existing fixed-N wrapper semantics; there is no new alive mask.
Dynamic membership, bandwidth arbitration and physical radio constraints are
outside this implementation. Packet-loss training retains the repository's
resampling-on-replay convention; it does not replay the original link noise.
The current configs target Pursuit/Battle, not IMP's contrastive/QMIX setup.

## Run and ablate

From the repository root, in the project's configured MAgent environment:

```bash
python src/main.py --config=iql_causal_exp --env-config=MAgent_Battle --exp-config=ExpoComm_Battle25_s0
python src/main.py --config=iql_causal_exp --env-config=MAgent_AdvPursuit --exp-config=ExpoComm_AdvPursuit45_s0
```

The existing experiment files provide map/opponent/seed settings; the algorithm
name separates output directories. Full runs use the existing 5,050,000-step
budget. Use separate experiment YAMLs for additional seeds; the current
entrypoint does not generally parse arbitrary `with key=value` overrides.

| Algorithm config | Purpose |
| --- | --- |
| `iql_causal_exp` | Adaptive and backbone slots interleaved |
| `iql_causal_exp_fixed` | Same packet/memory architecture, only scheduled edges |
| `iql_causal_exp_adaptive` | Same architecture, adaptive routing every step |
| `iql_causal_exp_none` | Same local memory, zero network packets |
| Existing ExpoComm one-peer config | Published architecture baseline |

The fixed variant is the essential control: comparison to original ExpoComm
alone confounds learned routing with the new packet encoder, observation timing
and local memory update. Original one-peer also includes a self-neighbor slot;
it does not necessarily use the same number of actual network transmissions.
Report actual transmitted bytes, not just nominal top-k. Use at least 5 seeds,
equal environment-step budgets and opponents; report return/win metrics,
learning curves, routing usage, receiver-load distribution and wall-clock cost.
Evaluate map/agent scaling and packet loss after establishing no-loss learning.
If hybrid does not beat fixed, there is no evidence the learned routing helps.

## Verification status

```bash
python -m unittest discover -s tests -p test_causal_exp.py -v
```

Synthetic CPU tests exercise hard delivery, sender-local dependencies, packet
budget, temporal graph reachability, replay determinism, total packet loss,
real AuxQLearner optimizer updates, router gradients, target synchronization,
action selection and checkpoint restoration. They do not establish benchmark
performance. The local default Python environment lacks MAgent/PettingZoo,
einops, W&B and setproctitle. Therefore full entrypoint/environment training was
not run here; follow the existing Setup.md environment procedure on the cluster.
The test loads the real q_learner module directly to avoid unrelated learners'
optional imports. Baseline agents/controllers/learner behavior is unchanged;
only registry entries were added to existing files.
