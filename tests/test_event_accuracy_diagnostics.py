from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

import torch
import torch.nn as nn

from ml_ldmx.eval.event_diagnostics import (
    detector_context_summary,
    optimal_label_permutation_summary,
    select_representative_events,
    truth_fraction_margin_summary,
)
from ml_ldmx.eval.hit_classifier_baseline import collect_event_metrics
from ml_ldmx.viz.training import (
    plot_assignment_ceiling_diagnostics,
    plot_event_accuracy_overview,
    plot_event_diagnostic_correlations,
    plot_shower_separation_profiles,
)


class IdentityLogitModel(nn.Module):
    def forward(self, x):
        return x


def _view(logits, target, event_idx, pos=None, energy=None, fractions=None, electron_count=None):
    num_hits = len(target)
    if pos is None:
        pos = [[float(idx), float(label), float(idx % 2)] for idx, label in enumerate(target)]
    if energy is None:
        energy = [1.0 for _idx in range(num_hits)]
    view = {
        "x": torch.as_tensor(logits, dtype=torch.float32),
        "ecal_mask": torch.ones((num_hits,), dtype=torch.bool),
        "ecal_pos": torch.as_tensor(pos, dtype=torch.float32),
        "ecal_input_energy": torch.as_tensor(energy, dtype=torch.float32),
        "ecal_raw_energy": torch.as_tensor(energy, dtype=torch.float32),
        "y": torch.as_tensor(target, dtype=torch.long),
        "source_file": f"event_{event_idx}.root",
        "source_entry": torch.tensor(event_idx),
    }
    if fractions is not None:
        view["origin_id_fraction_target"] = torch.as_tensor(fractions, dtype=torch.float32)
    if electron_count is not None:
        view["electron_count"] = torch.tensor(electron_count)
    return view


class EventAccuracyDiagnosticsTest(unittest.TestCase):
    def test_optimal_label_permutation_recovers_global_class_swap(self):
        summary = optimal_label_permutation_summary(
            true_class=torch.tensor([0, 0, 1, 1]),
            pred_class=torch.tensor([1, 1, 0, 0]),
            num_classes=2,
            energy_weights=torch.tensor([4.0, 3.0, 2.0, 1.0]),
        )

        self.assertEqual(summary["permutation_invariant_correct_hits"], 4)
        self.assertEqual(summary["permutation_invariant_accuracy"], 1.0)
        self.assertEqual(summary["label_permutation_gain"], 1.0)
        self.assertEqual(summary["optimal_prediction_label_mapping"], [1, 0])
        self.assertFalse(summary["optimal_prediction_label_mapping_is_identity"])
        self.assertEqual(summary["permutation_invariant_energy_weighted_accuracy"], 1.0)

    def test_energy_weighted_permutation_is_optimized_independently(self):
        summary = optimal_label_permutation_summary(
            true_class=torch.tensor([0, 0, 0, 1]),
            pred_class=torch.tensor([0, 0, 1, 0]),
            num_classes=2,
            energy_weights=torch.tensor([1.0, 1.0, 100.0, 100.0]),
        )

        self.assertEqual(summary["optimal_prediction_label_mapping"], [0, 1])
        self.assertEqual(
            summary["optimal_energy_weighted_prediction_label_mapping"],
            [1, 0],
        )
        self.assertAlmostEqual(
            summary["permutation_invariant_energy_weighted_accuracy"],
            200.0 / 202.0,
        )

    def test_truth_fraction_margin_summary_bins_accuracy(self):
        summary = truth_fraction_margin_summary(
            fraction_target=torch.tensor(
                [
                    [0.51, 0.49],
                    [0.60, 0.40],
                    [0.90, 0.10],
                    [1.00, 0.00],
                ]
            ),
            true_class=torch.tensor([0, 0, 0, 0]),
            pred_class=torch.tensor([1, 0, 0, 0]),
            energy_weights=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )

        self.assertEqual(summary["truth_fraction_num_valid_hits"], 4)
        self.assertAlmostEqual(summary["mean_truth_dominance_margin"], 0.505)
        self.assertAlmostEqual(summary["truth_ambiguous_hit_fraction_margin_le_0p1"], 0.25)
        self.assertAlmostEqual(summary["truth_ambiguous_hit_fraction_margin_le_0p25"], 0.5)
        self.assertEqual(sum(summary["truth_margin_bin_total_hits"]), 4)
        self.assertEqual(sum(summary["truth_margin_bin_correct_hits"]), 3)
        self.assertAlmostEqual(sum(summary["truth_margin_bin_total_energy"]), 10.0)
        self.assertAlmostEqual(sum(summary["truth_margin_bin_correct_energy"]), 9.0)

    def test_detector_context_summary_reports_missing_tracks(self):
        summary = detector_context_summary(
            {
                "tpad_mask": torch.tensor([False, True, False, True]),
                "electron_count": torch.tensor(3),
            }
        )

        self.assertEqual(summary["num_tpad_tokens"], 2)
        self.assertEqual(summary["tpad_token_deficit"], 1)
        self.assertEqual(summary["tpad_token_excess"], 0)
        self.assertFalse(summary["has_complete_tpad_context"])

    def test_collect_event_metrics_reports_per_event_accuracy(self):
        events = [
            _view(
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 0.0, 0.0]],
                [0, 1, 2],
                10,
                pos=[[0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [40.0, 0.0, 10.0]],
                energy=[10.0, 1.0, 1.0],
            ),
            _view(
                [[0.0, 0.0, 5.0], [0.0, 5.0, 0.0]],
                [2, 1],
                11,
            ),
        ]
        args = SimpleNamespace(batch_size=2, valid_labels=[0, 1, 2])

        records = collect_event_metrics(
            IdentityLogitModel(),
            events,
            [0, 1],
            None,
            args,
            torch.device("cpu"),
        )

        self.assertEqual([record["event_idx"] for record in records], [0, 1])
        self.assertEqual(records[0]["correct_hits"], 2)
        self.assertEqual(records[0]["incorrect_hits"], 1)
        self.assertAlmostEqual(records[0]["accuracy"], 2 / 3)
        self.assertAlmostEqual(records[0]["energy_weighted_accuracy"], 11 / 12)
        self.assertGreater(records[0]["mean_confidence"], 0.0)
        self.assertEqual(records[0]["num_truth_classes"], 3)
        self.assertAlmostEqual(records[0]["min_origin_centroid_distance_xy"], 10.0)
        self.assertEqual(records[1]["correct_hits"], 2)
        self.assertEqual(records[1]["incorrect_hits"], 0)
        self.assertEqual(records[1]["accuracy"], 1.0)
        self.assertEqual(records[1]["source_file"], "event_11.root")
        self.assertEqual(records[1]["source_entry"], 11)

    def test_normalized_shower_separation_and_ambiguity_metrics(self):
        events = [
            _view(
                [[5.0, 0.0], [5.0, 0.0], [0.0, 5.0], [0.0, 5.0]],
                [0, 0, 1, 1],
                12,
                pos=[
                    [-1.0, 0.0, 10.0],
                    [1.0, 0.0, 10.0],
                    [9.0, 0.0, 10.0],
                    [11.0, 0.0, 10.0],
                ],
            )
        ]
        args = SimpleNamespace(batch_size=1, valid_labels=[0, 1])

        record = collect_event_metrics(
            IdentityLogitModel(),
            events,
            [0],
            None,
            args,
            torch.device("cpu"),
        )[0]

        self.assertAlmostEqual(
            record["min_normalized_shower_separation_xy"],
            10.0 / (2.0**0.5),
        )
        self.assertAlmostEqual(
            record["early_min_normalized_shower_separation_xy"],
            10.0 / (2.0**0.5),
        )
        self.assertEqual(record["ambiguous_hit_fraction_xy"], 0.0)
        self.assertEqual(record["early_layer_count"], 1)
        self.assertEqual(record["shower_overlap_weighting"], "uniform_binary_membership")
        self.assertEqual(record["contributor_membership_source"], "dominant_origin_fallback")

    def test_any_contributor_membership_is_default_and_dominant_is_retained(self):
        event = _view(
            [[5.0, 0.0], [5.0, 0.0], [0.0, 5.0], [0.0, 5.0]],
            [0, 0, 1, 1],
            13,
            pos=[
                [-1.0, 0.0, 10.0],
                [1.0, 0.0, 10.0],
                [9.0, 0.0, 10.0],
                [11.0, 0.0, 10.0],
            ],
            fractions=[
                [1.0, 0.0],
                [0.6, 0.4],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            electron_count=2,
        )
        args = SimpleNamespace(batch_size=1, valid_labels=[0, 1])

        record = collect_event_metrics(
            IdentityLogitModel(),
            [event],
            [0],
            None,
            args,
            torch.device("cpu"),
        )[0]

        expected_contributor = 7.0 / ((59.0 / 3.0) ** 0.5)
        expected_dominant = 10.0 / (2.0**0.5)
        self.assertAlmostEqual(
            record["contributor_min_normalized_shower_separation_xy"],
            expected_contributor,
        )
        self.assertAlmostEqual(
            record["dominant_min_normalized_shower_separation_xy"],
            expected_dominant,
        )
        self.assertEqual(
            record["min_normalized_shower_separation_xy"],
            record["contributor_min_normalized_shower_separation_xy"],
        )
        self.assertEqual(
            record["early_contributor_min_normalized_shower_separation_xy"],
            record["contributor_min_normalized_shower_separation_xy"],
        )
        self.assertAlmostEqual(
            record["first_layer_contributor_min_centroid_distance_xy"],
            7.0,
        )
        self.assertAlmostEqual(
            record["first_layer_dominant_min_centroid_distance_xy"],
            10.0,
        )
        self.assertEqual(record["contributor_num_showers"], 2)
        self.assertEqual(record["dominant_num_showers"], 2)
        self.assertEqual(record["shower_separation_default"], "binary_any_contributor")
        self.assertEqual(record["contributor_membership_source"], "origin_id_fraction_target")

    def test_three_electron_separation_uses_minimum_of_three_pairs(self):
        event = _view(
            [[5.0, 0.0, 0.0]] * 2 + [[0.0, 5.0, 0.0]] * 2 + [[0.0, 0.0, 5.0]] * 2,
            [0, 0, 1, 1, 2, 2],
            14,
            pos=[
                [-1.0, 0.0, 10.0],
                [1.0, 0.0, 10.0],
                [4.0, 0.0, 10.0],
                [6.0, 0.0, 10.0],
                [19.0, 0.0, 10.0],
                [21.0, 0.0, 10.0],
            ],
            fractions=[
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            electron_count=3,
        )
        args = SimpleNamespace(batch_size=1, valid_labels=[0, 1, 2])

        record = collect_event_metrics(
            IdentityLogitModel(),
            [event],
            [0],
            None,
            args,
            torch.device("cpu"),
        )[0]

        self.assertAlmostEqual(
            record["contributor_min_normalized_shower_separation_xy"],
            5.0 / (2.0**0.5),
        )
        self.assertAlmostEqual(
            record["contributor_mean_normalized_shower_separation_xy"],
            (5.0 + 15.0 + 20.0) / (3.0 * (2.0**0.5)),
        )

    def test_plot_event_accuracy_overview_writes_file(self):
        records = [
            {
                "event_idx": 12,
                "split_position": 0,
                "num_hits": 100,
                "correct_hits": 95,
                "incorrect_hits": 5,
                "accuracy": 0.95,
            },
            {
                "event_idx": 15,
                "split_position": 1,
                "num_hits": 120,
                "correct_hits": 72,
                "incorrect_hits": 48,
                "accuracy": 0.60,
            },
        ]
        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "event_accuracy.png"
            plot_event_accuracy_overview(records, output_path, "validation accuracy")

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_plot_event_diagnostic_correlations_writes_file(self):
        records = [
            {
                "event_idx": 12,
                "split_position": 0,
                "num_hits": 100,
                "incorrect_hits": 5,
                "accuracy": 0.95,
                "loss": 0.1,
                "mean_confidence": 0.88,
                "min_origin_centroid_distance_xy": 55.0,
            },
            {
                "event_idx": 15,
                "split_position": 1,
                "num_hits": 120,
                "incorrect_hits": 48,
                "accuracy": 0.60,
                "loss": 1.4,
                "mean_confidence": 0.62,
                "min_origin_centroid_distance_xy": 12.0,
            },
            {
                "event_idx": 20,
                "split_position": 2,
                "num_hits": 80,
                "incorrect_hits": 12,
                "accuracy": 0.85,
                "loss": 0.4,
                "mean_confidence": 0.75,
                "min_origin_centroid_distance_xy": 30.0,
            },
        ]
        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "event_diagnostics.png"
            plot_event_diagnostic_correlations(records, output_path, "validation diagnostics")

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_plot_shower_separation_profiles_writes_both_accuracy_targets(self):
        records = []
        for event_idx in range(16):
            separation = 0.5 + 0.2 * event_idx
            records.append(
                {
                    "event_idx": event_idx,
                    "electron_count": 2 if event_idx < 8 else 3,
                    "accuracy": min(1.0, 0.45 + 0.03 * event_idx),
                    "energy_weighted_accuracy": min(1.0, 0.5 + 0.03 * event_idx),
                    "incorrect_hits": max(0, 16 - event_idx),
                    "contributor_min_normalized_shower_separation_xy": separation,
                    "early_contributor_min_normalized_shower_separation_xy": separation * 0.8,
                    "dominant_min_normalized_shower_separation_xy": separation * 1.1,
                    "early_dominant_min_normalized_shower_separation_xy": separation * 0.9,
                }
            )

        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "separation_profiles.png"
            plotted = plot_shower_separation_profiles(
                records,
                output_path,
                "shower separation",
                num_bins=3,
                bootstrap_samples=10,
            )

            self.assertTrue(plotted)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_plot_assignment_ceiling_diagnostics_writes_file(self):
        records = []
        edges = [0.0, 0.1, 0.5, 1.000001]
        for event_idx in range(6):
            records.append(
                {
                    "event_idx": event_idx,
                    "num_hits": 10,
                    "correct_hits": 5 + event_idx,
                    "accuracy": (5 + event_idx) / 10,
                    "permutation_invariant_correct_hits": 7 + event_idx // 2,
                    "permutation_invariant_accuracy": (7 + event_idx // 2) / 10,
                    "label_permutation_gain": (2 - event_idx // 2) / 10,
                    "truth_margin_bin_edges": edges,
                    "truth_margin_bin_total_hits": [2, 3, 5],
                    "truth_margin_bin_correct_hits": [1, 2, 4],
                    "truth_margin_bin_total_energy": [1.0, 2.0, 7.0],
                    "truth_margin_bin_correct_energy": [0.5, 1.5, 6.5],
                    "tpad_token_deficit": event_idx % 2,
                }
            )

        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "assignment_ceiling.png"
            plotted = plot_assignment_ceiling_diagnostics(
                records,
                output_path,
                "assignment ceiling",
            )

            self.assertTrue(plotted)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_select_representative_events_returns_worst_median_best(self):
        records = [
            {"event_idx": 1, "accuracy": 0.2, "incorrect_hits": 8},
            {"event_idx": 2, "accuracy": 0.6, "incorrect_hits": 4},
            {"event_idx": 3, "accuracy": 0.9, "incorrect_hits": 1},
        ]

        selection = select_representative_events(records, limit_per_group=1)

        self.assertEqual(selection["worst"][0]["event_idx"], 1)
        self.assertEqual(selection["median"][0]["event_idx"], 2)
        self.assertEqual(selection["best"][0]["event_idx"], 3)


if __name__ == "__main__":
    unittest.main()
