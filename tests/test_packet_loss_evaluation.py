import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


overrides = load("config_overrides", "src/config_overrides.py")
summary = load("summarize_packet_loss", "scripts/summarize_packet_loss.py")


class EvaluationTests(unittest.TestCase):
    def test_override_forces_eval_and_sets_seed(self):
        config = dict(checkpoint_path="models", runner="episode", evaluate=False)
        params = ["--eval-seed=42", "--eval-result-path=results.json"]
        result = overrides.apply_packet_loss_cli_overrides(config, params)
        self.assertTrue(result["evaluate"])
        self.assertEqual(result["seed"], 42)
        self.assertEqual(params, [])

    def test_rejects_invalid_seed_and_missing_checkpoint(self):
        for value in ["-1", "4294967296", "1.5"]:
            with self.assertRaises(ValueError):
                overrides.apply_packet_loss_cli_overrides({}, [f"--eval-seed={value}"])
        with self.assertRaises(ValueError):
            overrides.apply_packet_loss_cli_overrides({}, ["--eval-result-path=x.json"])

    def fixture(self, directory):
        with (directory / "manifest.tsv").open("w", newline="") as output:
            writer = csv.writer(output, delimiter="\t")
            writer.writerow(["loss", "seed", "episodes", "file"])
            for loss in [0, 20]:
                for seed in [0, 1]:
                    name = f"loss{loss}_seed{seed}.json"
                    writer.writerow([loss, seed, 100, name])
                    result = dict(
                        seed=seed, packet_loss_percent=loss, test_nepisode=100,
                        checkpoint="models/5009240", algorithm="one_peer", environment="battle",
                        metrics={"test_return_mean": 5 + seed - loss / 20},
                    )
                    (directory / name).write_text(json.dumps(result))

    def test_summary_and_baseline_difference(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.fixture(directory)
            rows = summary.summarize(directory)
            self.assertEqual([row["mean"] for row in rows], [5.5, 4.5])
            self.assertEqual(rows[1]["delta_vs_loss0"], -1)
            self.assertAlmostEqual(rows[0]["seed_std"], 2**-0.5)
            self.assertLessEqual(rows[0]["bootstrap95_low"], 5.5)
            self.assertGreaterEqual(rows[0]["bootstrap95_high"], 5.5)

    def test_missing_run_is_not_silently_averaged(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.fixture(directory)
            (directory / "loss20_seed1.json").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                summary.summarize(directory)

    def test_mixed_checkpoints_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.fixture(directory)
            path = directory / "loss20_seed1.json"
            result = json.loads(path.read_text())
            result["checkpoint"] = "models/100"
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "different checkpoints"):
                summary.summarize(directory)

    def test_single_seed_has_no_confidence_interval(self):
        self.assertEqual(summary.interval([5.0]), (None, None))


if __name__ == "__main__":
    unittest.main()
