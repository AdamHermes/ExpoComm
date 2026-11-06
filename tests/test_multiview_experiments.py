import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from multiview_fixtures import ROOT
from config_overrides import apply_experiment_cli_overrides


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExperimentTests(unittest.TestCase):
    def test_override_validation(self):
        params = ["--seed=4", "--t-max=100", "--wandb-mode=disabled"]
        config = apply_experiment_cli_overrides({}, params)
        self.assertEqual(config["seed"], 4)
        self.assertEqual(params, [])
        with self.assertRaises(ValueError):
            apply_experiment_cli_overrides({}, ["--seed=-1"])

    def test_sweep_matrix(self):
        runs = load("run_multiview_sweep").build_runs(
            ["iql_expocomm_mv_fixed", "iql_expocomm_mv_adaptive"],
            ["MAgent_Battle"], ["Multiview_Battle25_s0"], [0, 1], 100, "disabled")
        self.assertEqual(len(runs), 4)
        self.assertIn("--seed=1", runs[-1]["command"])

    def test_summary_missing_seeds_and_duplicate_fail(self):
        summarize = load("summarize_multiview").summarize
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (0, 1):
                directory = root / str(seed)
                directory.mkdir()
                (directory / "config.json").write_text(json.dumps(dict(
                    name="method", env="battle", env_args={"seed": seed}, seed=seed,
                    t_max=100, test_nepisode=10)))
                (directory / "completed.json").write_text(json.dumps(dict(actual_steps=100, wall_seconds=2)))
                (directory / "metrics.jsonl").write_text(json.dumps(dict(metric="test_return_mean", value=seed, step=100)) + "\n")
            rows, _ = summarize(root, "test_return_mean", [0, 1])
            self.assertEqual(rows[0]["mean"], .5)
            with self.assertRaises(ValueError):
                summarize(root, "test_return_mean", [0, 1, 2])
