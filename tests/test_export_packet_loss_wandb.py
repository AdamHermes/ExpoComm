import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exporter", ROOT / "scripts/export_packet_loss_wandb.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def run(loss=0, seed=0, name_seed=None, **extra):
    config = dict(packet_loss_percent=loss, seed=seed, evaluate=True,
                  packet_loss_test=True, packet_loss_train=False, test_nepisode=100,
                  checkpoint_path="models", load_step=5009240,
                  name="iql_ExpoComm_one_peer_n6", env="magent", env_args={"seed": seed, "map_size": 25})
    config.update(extra)
    return SimpleNamespace(
        id=f"r{loss}s{seed}",
        name=f"ExpoComm_Battle25_eval_5009240_loss{loss}_seed{seed if name_seed is None else name_seed}-iql-20261103",
        state="finished", config=config, url="https://wandb.ai/test/project/runs/id",
        created_at="2026-11-03T14:00:00Z",
        summary={"test_return_mean": 5.0 + seed / 10 - loss / 50,
                 "test_red_team_win_mean": 0.9, "test_return_std": 1.0,
                 "test_ep_length_mean": 150, "timestep": 5009240},
        scan_history=lambda: iter([{"test_return_mean": 5.0}, {"test_red_team_win_mean": 0.9}]),
    )


class FakeAPI:
    def __init__(self, runs):
        self.data = runs

    def runs(self, project, **kwargs):
        return iter(self.data)

    def run(self, path):
        return next(item for item in self.data if path.endswith("/" + item.id))


def options(directory, **extra):
    args = dict(project="test/ExpoComm_Battle25_eval", run_ids=None,
                name_prefix="ExpoComm_Battle25_eval_5009240", created_after=None,
                created_before=None, loss_rates=[0, 10, 20, 30, 50],
                expected_seeds=[0, 1, 2, 3, 4], history=False, out=directory)
    args.update(extra)
    return SimpleNamespace(**args)


class ExportTests(unittest.TestCase):
    def export(self, api, args):
        with contextlib.redirect_stdout(io.StringIO()):
            exporter.export(api, args)

    def test_complete_sweep_exports_exact_metrics_and_history(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            runs = [run(loss, seed) for loss in [0, 10, 20, 30, 50] for seed in range(5)]
            self.export(FakeAPI(runs), options(directory, history=True))
            self.assertEqual(len(list((directory / "runs").glob("*.json"))), 25)
            self.assertEqual(len(list((directory / "runs").glob("*.jsonl"))), 25)
            import csv
            with (directory / "summary.csv").open(encoding="utf-8-sig") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row["status"] == "ok" for row in rows))
            self.assertAlmostEqual(float(rows[0]["return_mean"]), 5.2)
            self.assertEqual(float(rows[0]["win_rate_percent_mean"]), 90)
            self.assertAlmostEqual(float(rows[0]["return_seed_std"]), 0.158113883)
            raw = json.loads((directory / "runs/r0s0.json").read_text())
            self.assertEqual(raw["config"]["seed"], 0)
            self.assertEqual(raw["summary"]["test_return_std"], 1.0)
            self.assertEqual((directory / "warnings.txt").read_text().strip(), "")

    def test_seed_in_name_does_not_replace_actual_seed(self):
        row, _ = exporter.extract_run(run(seed=0, name_seed=4))
        self.assertEqual(row["config_seed"], 0)
        self.assertEqual(row["name_seed"], 4)
        self.assertIn("name says seed=4", row["issues"])

    def test_duplicates_and_missing_rates_flagged(self):
        rows = []
        for _ in range(2):
            row, _ = exporter.extract_run(run())
            row["comparison_config"] = "{}"
            rows.append(row)
        summary, warnings = exporter.aggregate(rows, [0, 50], [0, 1])
        self.assertIn("duplicate actual seed", summary[0]["status"])
        self.assertIsNone(summary[0]["return_seed_std"])
        self.assertEqual(summary[1]["n_runs"], 0)
        self.assertTrue(warnings)

    def test_mixed_checkpoints_do_not_get_averaged(self):
        rows = []
        for seed in [0, 1]:
            row, _ = exporter.extract_run(run(seed=seed, checkpoint_path=f"model{seed}"))
            row["comparison_config"] = "{}"
            rows.append(row)
        summary, _ = exporter.aggregate(rows, [0], [0, 1])
        self.assertIsNone(summary[0]["return_mean"])
        self.assertIn("mixed checkpoint", summary[0]["status"])

    def test_date_and_id_filters_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            args = options(directory, run_ids=["r0s1"], loss_rates=[0], expected_seeds=[1],
                           created_after=exporter.timestamp("2026-11-03T20:00:00+07:00"))
            self.export(FakeAPI([run(seed=0), run(seed=1)]), args)
            self.assertEqual(len(list((directory / "runs").glob("*.json"))), 1)
            with self.assertRaisesRegex(ValueError, "empty"):
                self.export(FakeAPI([run(seed=1)]), args)
            args.created_before = exporter.timestamp("2026-11-03T21:00:00+07:00")
            with self.assertRaisesRegex(ValueError, "No matching"):
                self.export(FakeAPI([run(seed=1)]), args)

    def test_missing_metric_is_preserved_and_flagged(self):
        item = run()
        del item.summary["test_return_mean"]
        row, _ = exporter.extract_run(item)
        self.assertIsNone(row["return"])
        self.assertIn("missing/non-finite", row["issues"])

    def test_pursuit_does_not_require_battle_win_rate(self):
        item = run(env="MAgent_AdvPursuit")
        del item.summary["test_red_team_win_mean"]
        row, _ = exporter.extract_run(item)
        self.assertIsNone(row["win_rate_percent"])
        self.assertEqual(row["issues"], "")

    def test_battle_still_requires_win_rate(self):
        item = run(env="MAgent_Battle")
        del item.summary["test_red_team_win_mean"]
        row, _ = exporter.extract_run(item)
        self.assertIn("missing/non-finite win-rate", row["issues"])


if __name__ == "__main__":
    unittest.main()
