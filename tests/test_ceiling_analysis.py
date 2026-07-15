import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_hit_classifier_ceiling import (
    remove_tpad_tokens,
    summarize_paired_tpad_ablation,
)
from ml_ldmx.viz.training import plot_tpad_ablation_comparison


class CeilingAnalysisTest(unittest.TestCase):
    def test_remove_tpad_tokens_preserves_hit_targets(self):
        view = {
            "x": torch.arange(24, dtype=torch.float32).reshape(3, 8),
            "ecal_mask": torch.tensor([True, False, True]),
            "tpad_mask": torch.tensor([False, True, False]),
            "tpad": torch.tensor([[2.0, 3.0]]),
            "tpad_raw_pe": torch.tensor([10.0]),
            "y": torch.tensor([0, 1]),
        }

        ablated = remove_tpad_tokens(view)

        self.assertEqual(tuple(ablated["x"].shape), (2, 8))
        self.assertEqual(ablated["ecal_mask"].tolist(), [True, True])
        self.assertEqual(ablated["tpad_mask"].tolist(), [False, False])
        self.assertEqual(ablated["tpad"].shape[0], 0)
        self.assertTrue(torch.equal(ablated["y"], view["y"]))

    def test_summarize_paired_tpad_ablation_is_hit_weighted(self):
        reference = [
            {
                "event_idx": 1,
                "num_hits": 2,
                "correct_hits": 2,
                "accuracy": 1.0,
                "has_complete_tpad_context": True,
            },
            {
                "event_idx": 2,
                "num_hits": 8,
                "correct_hits": 4,
                "accuracy": 0.5,
                "has_complete_tpad_context": False,
            },
        ]
        ablated = [
            {"event_idx": 1, "num_hits": 2, "correct_hits": 1, "accuracy": 0.5},
            {"event_idx": 2, "num_hits": 8, "correct_hits": 3, "accuracy": 0.375},
        ]

        summary = summarize_paired_tpad_ablation(reference, ablated)

        self.assertAlmostEqual(summary["reference"]["hit_accuracy"], 0.6)
        self.assertAlmostEqual(summary["tpad_ablated"]["hit_accuracy"], 0.4)
        self.assertAlmostEqual(summary["tpad_hit_accuracy_gain"], 0.2)
        self.assertEqual(summary["events_helped_by_tpad"], 2)
        self.assertAlmostEqual(
            summary["by_tpad_completeness"]["complete"]["tpad_hit_accuracy_gain"],
            0.5,
        )

    def test_tpad_ablation_plot_writes_file(self):
        reference = [
            {
                "event_idx": 1,
                "num_hits": 2,
                "correct_hits": 2,
                "accuracy": 1.0,
                "has_complete_tpad_context": True,
                "num_tpad_tokens": 2,
            },
            {
                "event_idx": 2,
                "num_hits": 4,
                "correct_hits": 2,
                "accuracy": 0.5,
                "has_complete_tpad_context": False,
                "num_tpad_tokens": 1,
            },
        ]
        ablated = [
            {"event_idx": 1, "num_hits": 2, "correct_hits": 1, "accuracy": 0.5},
            {"event_idx": 2, "num_hits": 4, "correct_hits": 1, "accuracy": 0.25},
        ]

        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "tpad_ablation.png"
            plotted = plot_tpad_ablation_comparison(
                reference,
                ablated,
                output_path,
                "TPad ablation test",
            )
            self.assertTrue(plotted)
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
