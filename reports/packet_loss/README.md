# ExpoComm packet-loss evaluation

**Start here when writing the report. These are reported W&B metrics from one
training checkpoint per map and one actual evaluation seed (0). The evaluation
audit remains open; do not describe the Battle win rates as independently
validated outcomes.**

This folder contains the complete report bundle, including loss 100%. The main
slide figures deliberately cover only partial loss 0–50%; full-sweep figures and
all measurements remain available in the appendix and data folders.

## Where to find things

| Need | Location |
|---|---|
| Full-precision results for all 36 conditions | [data/metrics_0_to100.csv](data/metrics_0_to100.csv) |
| Run IDs, actual seeds, episode counts, W&B links and local JSON paths | [data/run_index.csv](data/run_index.csv) |
| Unmodified W&B configs and summaries, including repeated runs | [data/raw/](data/raw/) |
| Main graphs for slides, loss 0–50%, Figure 4 metrics | [figures/main_0_to50/](figures/main_0_to50/) |
| Complete comparison including loss 100% | [figures/appendix_0_to100/](figures/appendix_0_to100/) |
| Confirmed issues, diagnostic evidence and interpretation limits | [analysis/AUDIT_VI.md](analysis/AUDIT_VI.md) |
| Updated presentation wording and speaker notes | [analysis/SLIDE_NOTES_VI.md](analysis/SLIDE_NOTES_VI.md) |
| Synthetic audit probe results | [analysis/probe_results.json](analysis/probe_results.json) |
| Latest English slide, editable charts | [slides/ExpoComm_Message_Importance.pptx](slides/ExpoComm_Message_Importance.pptx) |
| Preview of that slide | [slides/preview.png](slides/preview.png) |
| Export, download, plotting and audit helper snapshots | [code/](code/) |
| Source revision, original input paths and file checksums | [PROVENANCE.json](PROVENANCE.json) |

PNG figures are 3840×2160 (16:9). SVG versions support resizing without loss of
resolution. The CSV files retain full precision; labels round for readability.

## Protocol and metrics

- Models: ExpoComm one-peer, trained with seed 0, one checkpoint per map.
- Battle25/35/40 have 20/42/64 learning agents. AdvPursuit45/60/70 have 25/45/61.
- Rates: 0%, 10%, 20%, 30%, 50%, 100%. Each condition declares 100 episodes.
- Battle reports `test_red_team_win_mean` on the 0–1 scale.
- AdvPursuit reports raw `test_return_mean`.
- These follow the metric types in Figure 4 of the paper. The x-axis here is
  packet loss, whereas Figure 4 in the paper uses training timesteps.
- Loss is an independent probability of dropping a complete message from another
  agent at each communication call. Self-links remain. One-peer retains its
  previous message memory on failed delivery. Hidden/message states reset at
  each episode start.

The bundle contains **59 raw run JSONs but only 36 distinct map/rate conditions**.
The older Battle25 export contains repeated runs with identical metrics. Do not
count those as independent seeds or extra independent episodes. The five names
`seed0`–`seed4` in older runs did not represent five effective seeds; actual
config.seed was 0. Check `run_index.csv` and the raw configs.

## Audit caveats that must accompany a report

1. **Seed CLI:** the audited CLI ignores `--eval-seed` and `--eval-result-path`.
   Recent runs explicitly used actual seed 0. The latter option does not provide
   a dependable local export in that code revision; this bundle uses W&B data.
2. **Double reset:** Battle resets twice per episode, cancelling the intended
   alternating starting sides. The starting-side protocol needs correction or
   an explicit controlled comparison.
3. **Terminal survivor counting:** observation dictionary membership may still
   include newly dead agents in PettingZoo 1.12.0. Wrapper-derived winner counts
   need comparison with native MAgent alive counts. The synthetic counterexample
   is a logic test, not proof that a specific W&B episode was misclassified.
4. Local tensor probes found no remote-message influence on the tested agent at
   loss100, but no trained-checkpoint replay on the cluster was performed during
   the audit. One-seed results do not support between-seed confidence intervals.

The complete loss100 data are retained even though the current main slide
excludes that point. Do not claim universal communication dependence or
irrelevance based on a selected plot range. A policy trained with communication
and evaluated with delivery disabled is not a baseline trained without it.

## Reproduce the figures

From the repository root, in a Python environment with the pinned plotting
dependencies:

```powershell
python -m pip install -r reports/packet_loss/requirements.txt
python reports/packet_loss/reproduce.py
```

The same commands work in Bash. Outputs go to `reports/packet_loss/reproduced/`,
which is ignored by Git. No W&B account or network request is required to plot
the included data. To verify file integrity without plotting or installing
plotting dependencies:

```powershell
python reports/packet_loss/reproduce.py --verify-only
```

The audit helper requires PyTorch and the evaluation source in the surrounding
ExpoComm checkout. It uses the actual local methods with synthetic inputs:

```powershell
python reports/packet_loss/code/audit_packet_loss_eval.py --out reports/packet_loss/reproduced/probe_results.json
```

Audit findings describe the source revision recorded in `PROVENANCE.json`.
Later source fixes can intentionally change or invalidate its diagnostic probes.

## Scope

Checkpoint weights, videos, full training histories and temporary slide-build
files are not included. This package reproduces figures from the exported logs;
rerunning evaluation requires the matching checkpoints and cluster environment.
Raw configs retain original checkpoint paths as provenance, not portable paths
that are expected to exist on every collaborator's machine.

Published figure titles are descriptive summaries of logged data and are subject
to the audit above. For conclusions and limitations, use the audit and updated
slide notes rather than treating automatically generated captions as final claims.
