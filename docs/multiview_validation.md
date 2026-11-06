# Multiview implementation verification

Verified on **2026-11-06 (Asia/Saigon)**. These are correctness/integration
checks, not benchmark performance results. No full training sweep ran.

## Checks completed

- **40 tests passed**, including real IC3Net adapter contracts:
  `python -m unittest discover -s tests -v`.
- Python compilation and `git diff --check` passed.
- Sweep generation produced valid Slurm commands without submitting jobs.
- Summary generation processed the real Predator-Prey smoke run and produced
  no confidence interval for its single seed, as intended.

Windows tests used isolated dependencies on PYTHONPATH from
`tmp/multiview_deps` and `tmp/ic3net_source/ic3net-envs`.

## Formula/file audit

| File | Verified property |
| --- | --- |
| `multiview_graph.py` | Cosine neighbors exclude self/inactive nodes; union symmetry; hand-checked normalization; oversized k, empty/singleton graphs remain finite |
| `multiview_encoder.py` | Eq. 5/7 outputs match explicit matrix powers; shared W_f/separate W_r; detached alignment teachers; GCL gradients reach local features; adaptive gate has gradients |
| `multiview_losses.py` | Analytic two-node loss is log(e+3)-1 at temperature 1; BOTH within-view negative sets; masked negatives excluded; empty backward is finite |
| `expocomm_multiview_agent.py` | Auxiliary parameters belong to the agent's optimizer/checkpoint |
| `expocomm_multiview_controller.py` | Identical execution weights reproduce original ExpoComm Q-values; no multiview call in execution; reused slots clear memory/last action |
| `multiview_q_learner.py` | IQL/QMIX/VDN; padding/terminal masks; valid reward statistics; hard/soft targets; no target gradients; checkpoint continuation with identical RNG/replay |

Additional tests cover Double-Q online argmax and target evaluation,
unavailable actions, terminal targets, slot-reset bootstrap removal, disabled
state decoder, and MAgent input widths (Battle 98, Pursuit 128). With multiview
and reward normalization disabled, TD/state losses and parameter updates match
the original AuxQLearner within floating-point tolerance. Masked reward
normalization is a documented change controlled by the no-multiview variant.

## Real simulator smoke runs

These used the actual entrypoint, runner, replay, learner, final evaluation and
checkpoint code. No simulator was mocked.

| Environment | Algorithm | Horizon | Requested / actual training steps | Result |
| --- | --- | --- | --- | --- |
| Battle25, 20 controlled agents, pretrained opponent | IQL MV-adaptive | 20 | 60 / 80 | Passed: 4 episodes, 3 optimizer updates |
| IC3 Traffic Junction medium, 10 slots | QMIX MV-adaptive | 40 | 100 / 120 | Passed: 3 episodes, 2 optimizer updates |
| IC3 Predator-Prey easy, 5 predators | QMIX MV-adaptive | 40 | 100 / 120 | Passed November 4; adapter tests passed again November 6 |

The repository collects whole episodes with `while t_env <= t_max`;
completed.json explicitly records the resulting overshoot.

Evidence directories relative to the repository root:

- Battle: `work_dirs/Multiview_Smoke/iql_expocomm_mv_adaptive/Battle25_smoke/20261106_085023`
- Traffic: `work_dirs/Multiview_Transfer/qmix_expocomm_mv_adaptive/Multiview_s0_verified_traffic/20261106_154927`
- Predator: `work_dirs/Multiview_Transfer/qmix_expocomm_mv_adaptive/Multiview_s0_smoke_predator/20261104_143813`

The WSL Battle directory timestamp is UTC. Its three GCL losses were roughly
8.3198, 8.0635, 7.8154; alignment losses were 1.0104, 0.9631, 0.9318.
Topology supervision weight changed from 0.3333 to 0.3346, demonstrating actual
gate updates. Three updates do not establish useful behavior or superiority.

Battle dependencies were installed only under `tmp/multiview_linux_deps`:
MAgent 0.1.14, PettingZoo 1.12.0, SuperSuit 3.3.2, Gym 0.23.1 and NumPy 1.23.5,
using WSL Python 3.8 and its existing Torch. Repo scenario files were copied
only into that isolated PettingZoo package. An initial MAgent `np.bool` failure
was resolved by the project's NumPy pin; no simulator source was patched.
The cluster setup uses Gym 0.21.0; the smoke used 0.23.1.

IC3Net revision: `69b7e0ce51a79def593abfef1a976f43e5e13f75`. Windows tests
used Gym 0.26.2 with direct simulator classes. These are IC3 adaptations,
not verified MAIL protocol reproductions.

## Outstanding empirical checks

- SMAC adapter/configs exist, but SC2 runtime/maps were not launched.
- MAIL's complete benchmark protocols and official implementation have not
  been verified. There is no claimed `mail_reference` reproduction.
- Full budgets, five-seed comparisons, scale-up and packet-loss experiments
  remain to be run before scientific performance claims.
- Current centralized teachers may contain information unavailable to sparse
  recurrent memory. The implementation does not solve temporal recoverability.
- Adaptive supervision minimizes alignment error plus KL regularization; it
  does not use a meta-gradient of task return. Compare fixed weights and
  no-multiview controls before interpreting its benefit.

See `multiview_experiments.md` for full equations and explicit deviations
from MAIL, file mapping, configurations, and commands.
