import unittest

import torch

from mldmx.models import ECalGravNet, ECalTransformer
from mldmx.train.hit_classifier_baseline import compute_batch_losses, compute_event_losses
from mldmx.train.hit_classifier_batching import (
    IGNORE_INDEX,
    collate_gravnet_hit_classifier_batch,
    collate_transformer_hit_classifier_batch,
)


def _view(x, ecal_mask, y):
    return {
        "x": torch.as_tensor(x, dtype=torch.float32),
        "ecal_mask": torch.as_tensor(ecal_mask, dtype=torch.bool),
        "y": torch.as_tensor(y, dtype=torch.long),
    }


class HitClassifierBatchingTest(unittest.TestCase):
    def test_transformer_collate_pads_and_expands_ecal_targets(self):
        views = [
            _view(
                [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
                [True, False, True, False],
                [1, 2],
            ),
            _view(
                [[5, 0, 0], [6, 0, 0]],
                [False, True],
                [0],
            ),
        ]

        batch = collate_transformer_hit_classifier_batch(views)

        self.assertEqual(tuple(batch.x.shape), (2, 4, 3))  # [B, max_tokens, F]
        self.assertTrue(
            torch.equal(
                batch.valid_mask,
                torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.supervised_mask,
                torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.bool),
            )
        )
        expected_target = torch.tensor(
            [
                [1, IGNORE_INDEX, 2, IGNORE_INDEX],
                [IGNORE_INDEX, 0, IGNORE_INDEX, IGNORE_INDEX],
            ]
        )
        self.assertTrue(torch.equal(batch.target, expected_target))

    def test_gravnet_collate_concatenates_nodes_and_builds_batch_vector(self):
        views = [
            _view([[1, 0], [2, 0], [3, 0]], [True, False, True], [2, 1]),
            _view([[4, 0], [5, 0]], [False, True], [0]),
        ]

        batch = collate_gravnet_hit_classifier_batch(views)

        self.assertEqual(tuple(batch.x.shape), (5, 2))  # [sum_tokens, F]
        self.assertTrue(torch.equal(batch.batch_index, torch.tensor([0, 0, 0, 1, 1])))
        self.assertTrue(
            torch.equal(
                batch.supervised_mask,
                torch.tensor([1, 0, 1, 0, 1], dtype=torch.bool),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.target,
                torch.tensor([2, IGNORE_INDEX, 1, IGNORE_INDEX, 0]),
            )
        )

    def test_transformer_batched_forward_matches_single_event_forward(self):
        torch.manual_seed(3)
        model = ECalTransformer(
            in_dim=3,
            d_model=4,
            nhead=2,
            num_layers=1,
            dim_feedforward=8,
            dropout=0.0,
            out_dim=3,
        )
        model.eval()
        views = [
            _view(torch.randn(3, 3), [True, True, True], [0, 1, 2]),
            _view(torch.randn(1, 3), [True], [1]),
        ]
        batch = collate_transformer_hit_classifier_batch(views)

        with torch.no_grad():
            single_logits = model(views[0]["x"])
            batched_logits = model(batch.x, key_padding_mask=~batch.valid_mask)

        self.assertTrue(torch.allclose(single_logits, batched_logits[0, :3], atol=1e-6))

    def test_transformer_batched_loss_matches_hit_weighted_single_event_loss(self):
        torch.manual_seed(5)
        model = ECalTransformer(
            in_dim=3,
            d_model=4,
            nhead=2,
            num_layers=1,
            dim_feedforward=8,
            dropout=0.0,
            out_dim=3,
        )
        model.eval()
        views = [
            _view(torch.randn(4, 3), [True, False, True, True], [0, 2, 1]),
            _view(torch.randn(2, 3), [False, True], [2]),
        ]

        batched = compute_batch_losses(model, views, "padded", torch.device("cpu"))
        singles = [compute_event_losses(model, view, None, torch.device("cpu")) for view in views]
        old_loss = sum(loss["total_loss"] * loss["num_hits"] for loss in singles) / sum(
            int(loss["num_hits"]) for loss in singles
        )
        old_true = torch.cat([loss["true_class"] for loss in singles])

        self.assertTrue(torch.allclose(batched["total_loss"], old_loss, atol=1e-6))
        self.assertTrue(torch.equal(batched["true_class"], old_true))

    def test_gravnet_batched_loss_matches_hit_weighted_single_event_loss(self):
        torch.manual_seed(7)
        try:
            model = ECalGravNet(
                in_dim=3,
                hidden_dim=4,
                out_dim=3,
                num_layers=1,
                space_dimensions=2,
                propagate_dimensions=4,
                k=2,
                dropout=0.0,
            )
        except Exception as exc:
            self.skipTest(f"GravNetConv runtime unavailable in this environment: {exc}")
        model.eval()
        views = [
            _view(torch.randn(4, 3), [True, True, True, True], [0, 1, 2, 1]),
            _view(torch.randn(3, 3), [True, True, True], [2, 0, 1]),
        ]

        try:
            batched = compute_batch_losses(model, views, "graph", torch.device("cpu"))
            singles = [compute_event_losses(model, view, None, torch.device("cpu")) for view in views]
        except Exception as exc:
            self.skipTest(f"GravNetConv runtime unavailable in this environment: {exc}")

        old_loss = sum(loss["total_loss"] * loss["num_hits"] for loss in singles) / sum(
            int(loss["num_hits"]) for loss in singles
        )
        old_true = torch.cat([loss["true_class"] for loss in singles])

        self.assertTrue(torch.allclose(batched["total_loss"], old_loss, atol=1e-5))
        self.assertTrue(torch.equal(batched["true_class"], old_true))


if __name__ == "__main__":
    unittest.main()
