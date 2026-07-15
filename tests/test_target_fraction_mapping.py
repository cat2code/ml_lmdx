import unittest

import torch

from ml_ldmx.datasets.ecal_tpad_loading import apply_variable_count_target_mode


class TargetFractionMappingTest(unittest.TestCase):
    def test_subset_run_uses_stored_physical_fraction_column_labels(self):
        fractions = torch.tensor(
            [
                [0.8, 0.2, 0.0],
                [0.1, 0.9, 0.0],
            ],
            dtype=torch.float32,
        )
        event = {
            "physical_y": torch.tensor([1, 2]),
            "ecal_pos": torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
            "fraction_target": fractions.clone(),
            "target_label_order": [1, 2, 3],
        }

        apply_variable_count_target_mode(
            event,
            valid_labels=(1, 2),
            target_mode="canonical-y",
            max_electrons=2,
        )

        self.assertTrue(torch.equal(event["fraction_target"], fractions[:, :2]))
        self.assertTrue(torch.equal(event["origin_id_fraction_target"], fractions))
        self.assertEqual(event["origin_id_fraction_label_order"], [1, 2, 3])

    def test_canonical_fraction_columns_follow_spatial_order(self):
        fractions = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32)
        event = {
            "physical_y": torch.tensor([1, 2]),
            "ecal_pos": torch.tensor([[0.0, 2.0, 0.0], [0.0, -2.0, 0.0]]),
            "fraction_target": fractions.clone(),
            "target_label_order": [1, 2],
        }

        apply_variable_count_target_mode(
            event,
            valid_labels=(1, 2),
            target_mode="canonical-y",
            max_electrons=2,
        )

        self.assertEqual(event["target_label_order"], [2, 1])
        self.assertTrue(torch.equal(event["fraction_target"], fractions[:, [1, 0]]))


if __name__ == "__main__":
    unittest.main()
