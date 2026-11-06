# Adaptive multiview supervision on ExpoComm

Implementation date: 2026-11-04. This is a MAIL-inspired training extension of
ExpoComm one-peer, not a reproduction of MAIL, not CausalExp, and not SOPS.
The adaptive component weights training supervision; it does not learn new
execution edges. Performance/novelty claims require controlled experiments.

## Formulation and information access

Each agent has local observation history, a recurrent local feature h_i^t,
and a communication memory m_i^t. With the original one-peer agent:

```
h_i^t = GRU_obs(ReLU(W_obs input_i^t), h_i^{t-1})
j(i,t) = (i + offset[t mod K]) mod N
offset = [0, 1, 2, 4, ..., 2^(K-2)]
m_i^t = GRU_msg(ReLU(W_msg [h_i^t, m_j(i,t)^{t-1}]), m_i^{t-1})
Q_i^t = MLP_Q([h_i^t, m_i^t])
```

`input` is observation plus optional previous action and agent ID, according to
config. Both memories start at zero for every episode replay/rollout. Runtime
communication and packet-loss semantics are inherited from ExpoCommOAgent:
in particular its packet-loss branch freezes communication memory on a dropped
link. This extension does not silently replace those semantics with CausalExp.

Execution uses no global state, all-agent current feature matrix, kNN graph,
multiview weights or graph encoder. Only the learner accesses those quantities.
This remains model-free off-policy recurrent Double Q-learning, with shared
parameters. Individual-reward MAgent uses IQL; team-reward transfer configs use
QMIX. Communication can couple the Q-functions even with an IQL TD objective.

## File/formula audit

Reference: local `0008.pdf`, MAIL, IJCAI 2025, pp. 66-68, equations 3-12.
Official paper: https://www.ijcai.org/proceedings/2025/0008.pdf

| File | Responsibility | Reference / deliberate adaptation |
| --- | --- | --- |
| `src/modules/communication/multiview_graph.py` | Graph creation, normalization, SGC propagation | MAIL Eq. 3-4; training original graph is the symmetrized union of ExpoComm candidate edges, not the spatial graph of MAIL |
| `src/modules/communication/multiview_encoder.py` | Shared/separate SGC projections and adaptive supervision | Eq. 5 and 7; separate explicit message-alignment extension |
| `src/modules/communication/multiview_losses.py` | Per-agent GCL objective | Eq. 6, 8, 9, including BOTH intra-view negative sets |
| `src/modules/agents/expocomm_multiview_agent.py` | Register all trainable modules inside agent | Existing ExpoComm execution, agent optimizer/checkpoint ownership |
| `src/controllers/expocomm_multiview_controller.py` | Q/aux/features API; active/reset masks | Original routing, explicit lifecycle for absent/reused slots |
| `src/learners/multiview_q_learner.py` | Masked recurrent Double-Q, auxiliary optimization, targets | Existing ExpoComm TD/state loss plus GCL and alignment |

Let X_t stack local GRU features for all agents in one replay snapshot. No view
uses future timestamps. A contains the configured exponential candidate edges,
including the self slot, with duplicates collapsed; it is symmetrized by union.
The feature adjacency A_F is cosine top-k over other active agents, followed by
union symmetrization. A_F has no self-loops. Missing neighbors are not filled
with inactive agents when k exceeds the active population.

```
S   = D^(-1/2) (A with self-loops) D^(-1/2)
S_F = D_F^(-1/2) A_F D_F^(-1/2)
H_o = S^p X W_f
H_f = H_o + S_F X W_f
H_r = S^p X W_r
H_t = S^l X W_t,  l > p >= 1
```

Zero-degree rows normalize to zero. Inactive rows/columns are excluded.
W_f is shared by the original/feature branches; W_r and W_t are independent.
SGC projections have no bias or nonlinearities. Graph powers are computed by
repeated matrix-feature multiplication, not materialized dense matrix powers.
kNN membership is discrete and detached; gradients still flow through XW.
Tie resolution follows PyTorch topk and should not be interpreted as learned
edge selection. The global topology source is S, an allowed MAIL Eq. 7 choice.

For normalized embeddings a,b, temperature tau, the implemented per-node loss is:

```
L(a,b)_i = log[ sum_j exp(sim(a_i,b_j)/tau)
              + sum_(j != i) exp(sim(a_i,a_j)/tau)
              + sum_(j != i) exp(sim(b_i,b_j)/tau) ]
           - sim(a_i,b_i)/tau
```

Only valid agents at the SAME episode/time enter this denominator. There are
no cross-time/cross-episode negatives. The positive is the same agent across
views. This is the printed MAIL denominator, not substituted standard SimCLR.
All-masked rows and one-agent snapshots have finite zero contrastive loss.

```
L_f = L(H_o, H_f)
L_t = L(H_r, H_t)
L_c = L(H_o, H_r)
L_GCL = mean_valid(L_f + lambda_t L_t + lambda_c L_c)
```

**Paper ambiguities:** Eq. 8 prints an undefined h-bar-o while its preceding
text specifies h-r and h-t. This implementation uses H_r,H_t. Eq. 11 prints a
TD residual without a square; the implementation uses squared TD error, as in
ExpoComm. Neither choice is claimed to be verified against MAIL source code.
MAIL's Algorithm 1 places GCL updates under its execution loop; our protocol
explicitly performs these updates only in centralized replay training.

## Alignment and adaptive weighting (our extension)

MAIL's final action input is its original-view embedding. Here the action
input stays ExpoComm's recurrent memory m. The bridge is:

```
z_i = normalize(W_student m_i)
target_iv = stop_gradient(normalize(H_vi)), v in {o, f, t}
d_iv = 1 - dot(z_i, target_iv)
u_v = 1 / number_of_enabled_views
fixed:    w_iv = u_v
adaptive: w_iv = (1-epsilon)*softmax(gate([stopgrad(h_i),stopgrad(m_i)]))_v
                  + epsilon*u_v
L_align = mean_valid sum_v w_iv*d_iv
L_reg = mean_valid KL(w_i || u)
```

The last gate layer initializes to zero so both variants start uniformly.
`mv_weight_floor=0.1` is the uniform **mixture fraction**, not an absolute lower
bound for each view (with 3 views each is at least 0.1/3).
Disabled views have exactly zero weight. Log all three weights and L_reg.

Alignment trains student/message/gate but does NOT move the graph encoders
toward the student; GCL trains graph encoders and local observation features.
The gate receives its optimization signal from weighted alignment and KL,
not a meta-gradient of RL return. The floor/KL limit complete view exclusion,
but do not prove the gate chooses task-relevant supervision. An easier-view
preference remains a research risk and requires the fixed-weight control.

The complete objective is:

```
L = L_TD + aux_coef*L_state
    + mv_gcl_coef*L_GCL + mv_align_coef*L_align + mv_reg_coef*L_reg
```

L_state sums squared errors over state dimensions and averages valid agent
steps, matching ExpoComm scaling. GCL/alignment use up to 8 randomly sampled
valid snapshots per episode per update, to bound O(B*T*N^2) training storage.
This gives equal snapshot sampling opportunities per episode; it is not an
exact full-transition-weighted estimate when episodes have different lengths.
TD and state losses still use all valid transitions.

When state reconstruction is disabled, its decoder is not executed and its
large [B,N,state_dim] predictions are not materialized.

**Information limit:** teacher views use current centralized X_t, whereas
one-peer memory carries delayed information. This is privileged supervision;
not every target is inferable by an executing agent. This version does NOT
implement causal temporal teacher graphs or prove exact recoverability. The
alignment coefficient is tunable and can be zeroed to audit this issue.
Treating other agents as contrastive negatives can also separate agents that
ought to cooperate; keep the no-feature/no-topology controls.

## TD, reward and episode semantics

```
a*_i = argmax over available a of Q_online_i(t+1,a)
y_i  = r_i + gamma*(1-terminal_t)*Q_target_i(t+1,a*_i)  # IQL
y    = r_team + gamma*(1-terminal_t)*Qtot_target(t+1,a*) # QMIX
```

Target unroll and target mixing run under no_grad. An inactive next agent's
individual bootstrap is zero. Current inactive Q values are masked before
QMIX; alive teammates can still bootstrap the global value. Replay padding
and all steps after a true terminal are excluded. A time-limit truncation
retains the bootstrap observation; it is not treated as environment termination.
Reward normalization uses valid samples only and population variance, including
the one-sample case. This differs from legacy learner padding/unbiased variance
handling; `no_multiview` provides a control under the identical new learner.
Return normalization is explicitly unsupported rather than silently misapplied.

`reward_scope: legacy` preserves existing configs (only struct_marl is global).
New configs explicitly select `team` or `individual`. Runner, replay and learner
share this convention. Team multiview requires a mixer; individual requires none.
Only the episode runner is supported by the multiview experiment configs.

Agent-active and agent-reset fields are optional. IC3 traffic uses both to clear
hidden/message/last-action history when a vehicle slot is reused, including
replacement within the same step. Agents that finish Predator-Prey remain
information sources. Original Battle fixed-slot handling is retained.

Checkpoints include online/target agent, online/target mixer, optimizer,
training counters and reward statistics. Replay contents, RNG state and entire
runner state are not checkpointed: this is learner restoration, not an exact
full training-run resume.

## Configs and comparisons

- `iql_expocomm_mv_fixed`, `iql_expocomm_mv_adaptive`: main MAgent comparison.
- `iql_expocomm_mv_no_multiview`: same learner/architecture, all multiview
  coefficients zero. Use this AND the original ExpoComm one-peer baseline.
- `iql_expocomm_mv_no_feature`, `..._no_topology`, `..._no_cross`,
  `..._no_state_aux`: component ablations based on adaptive mode.
- `qmix_expocomm_mv_fixed`, `..._adaptive`, `..._no_multiview`: team rewards,
  state reconstruction disabled because state is concatenated observations.
- Battle25/Battle40 experiment configs supply five seeds. Topology candidate
  count is 6 for Battle25 and overridden to 7 by the Battle40 experiment configs,
  matching the corresponding published one-peer baseline schedules.

The exponential training graph uses ALL configured candidate edges; execution
uses one selected edge at a time. Additional teacher edges are never charged
as execution communication. Training graph compute is O(N^2) for kNN and
contrastive comparisons and must be reported separately.

Communication logs report scheduled non-self packets and H*float-byte-width
payload per episode step, before drops. They include fixed inactive slots and
exclude headers/control traffic. These are modeled payload counts, not measured
network bandwidth. Counts are recorded only for real rollout steps, never replay
or the last bootstrap-only action. The legacy self-neighbor slot costs zero.

## Benchmark adapters and protocol boundaries

The standalone MAIL paper does not fully specify every environment parameter.
No official MAIL implementation was located in the search performed here.
There is intentionally no misleading `MAIL_Battle.yaml` or `mail_reference`
algorithm. Battle runs use ExpoComm's existing pretrained opponent and rewards.
Do not compare those numbers directly with MAIL Table 3.

`IC3_*` configs use the upstream IC3Net source, verified at commit
`69b7e0ce51a79def593abfef1a976f43e5e13f75`:
https://github.com/IC3Net/IC3Net/tree/69b7e0ce51a79def593abfef1a976f43e5e13f75

Traffic: medium 14x14/10 cars/rate .2, hard 18x18/20 cars/rate .05. Fixed rates,
vision 1, horizons 40/80. Observation tuples are flattened preserving upstream
scalar normalization. Team reward is the mean of per-car rewards. Success is
no collision during the horizon. Initial all-inactive snapshots are valid for
team TD but excluded from node losses. Slot reuse resets memory explicitly.

Predator-Prey: 10x10/5 or 20x20/10, one fixed prey, four movement actions,
upstream mixed reward mode, vision 1, horizons 40/80. Completion is all predators
having reached prey. Team reward is mean upstream reward. Vision/horizon/reward
choices are adaptation settings, not verified MAIL hyperparameters.

For these two adapters, the centralized state is concatenated local observations,
not simulator latent state. Simulator RNGs are isolated from replay RNG.

`SMAC_view2_*` configs use a subclass overriding `unit_sight_range` to 2.0, based
on the upstream SMAC method. Supported config names: MMM2, 1c3s5z, 27m_vs_30m.
Attack range/action availability stay as in upstream SMAC; reduced sight does
not prove absence of information in action masks. SC2 binaries/maps/version and
MAIL's other settings must be verified before direct paper comparisons.
SMAC was not executed in the local validation environment.

## Run commands

From the repository root, after installing the project's MAgent dependencies:

```bash
# Short real Battle integration run, deliberately not a paper benchmark.
python src/main.py --config=iql_expocomm_mv_adaptive --env-config=MAgent_Battle --exp-config=Multiview_Battle25_smoke --wandb-mode=disabled

# Full Battle25 training (no extra flags needed for the configured budget).
python src/main.py --config=iql_expocomm_mv_adaptive --env-config=MAgent_Battle --exp-config=Multiview_Battle25_s0

# Print the 3-method x 5-seed sweep; does not launch jobs.
python scripts/run_multiview_sweep.py --slurm --manifest results/multiview_manifest.json
```

Install IC3Net environments from the pinned source in a suitable environment;
the adapter does not vendor or silently change their simulator code. Linux:

```bash
git clone https://github.com/IC3Net/IC3Net.git third_party/IC3Net
git -C third_party/IC3Net checkout 69b7e0ce51a79def593abfef1a976f43e5e13f75
python -m pip install -e third_party/IC3Net/ic3net-envs
python src/main.py --config=qmix_expocomm_mv_adaptive --env-config=IC3_TrafficJunction_medium --exp-config=Multiview_s0 --wandb-mode=disabled --t-max=100 --batch-size=2 --test-nepisode=2
```

Those last overrides are smoke-test settings. Full-budget runs should use the
config's batch size, episode budget and evaluation episode count. Use explicit
`--seed`, `--t-max`, `--buffer-size`, `--batch-size`, `--test-interval` options;
unknown flags now raise errors. `--wandb-mode=disabled` requires no W&B package.
Optional learners/environments load lazily, so IC3 runs do not import SMAC/IMP.

Each run writes config.json, runtime_config.json, source_hashes.json,
revision.txt, metrics.jsonl and (only on success) completed.json. Final policy
evaluation and final checkpoint have the same actual environment-step count.
Whole episodes may overshoot t_max; completed.json records actual_steps.

```bash
python scripts/summarize_multiview.py work_dirs/Multiview_Battle25 --metric test_red_team_win_mean --seeds 0 1 2 3 4
```

Summary rejects missing seeds, duplicate completed seeds, mixed hyperparameters
within a cohort and missing final evaluations. It writes seed-mean/std/bootstrap
intervals and a learning-curve CSV. Intervals resample independent training
seeds, not evaluation episodes. Single-seed runs have no confidence interval.
Do not choose the best seed/checkpoint retrospectively; use the predeclared
budget and final evaluation. Failed runs are not treated as zero performance.

## Validation commands

```
python -m unittest discover -s tests -v
```

Real IC3 adapter tests run only when its package and Gym are importable;
otherwise those tests explicitly skip. Tests cover paper equations against
manual calculations, graph masks, single-agent/empty graphs, gradients and
stop-gradient boundaries, original execution equivalence, no-multiview TD
equivalence (without reward normalization), padding, IQL/QMIX/VDN updates,
target schedules, learner checkpoint continuation, and slot reuse. See
multiview_validation.md for the actual runs performed and outstanding checks.
