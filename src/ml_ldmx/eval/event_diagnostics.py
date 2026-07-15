"""Per-event diagnostics for hit-origin classifier evaluation."""

import itertools
import math

import torch
import torch.nn.functional as F


TRUTH_MARGIN_BIN_EDGES = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.999999, 1.000001)


def _to_1d_cpu_tensor(value, dtype=None):
    if value is None:
        return None
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.reshape(-1)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _to_2d_cpu_tensor(value, dtype=None):
    if value is None:
        return None
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _finite_float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _mean_or_none(values):
    if values is None or values.numel() == 0:
        return None
    return _finite_float(values.to(dtype=torch.float64).mean().item())


def _quantile_or_none(values, quantile):
    if values is None or values.numel() == 0:
        return None
    return _finite_float(torch.quantile(values.to(dtype=torch.float64), float(quantile)).item())


def _optional_aligned_hit_vector(view, names, num_hits, dtype=torch.float32):
    for name in names:
        if name not in view:
            continue
        values = _to_1d_cpu_tensor(view[name], dtype=dtype)
        if values is not None and values.shape[0] == num_hits:
            return values
    return None


def prediction_metric_summary(
    true_class,
    pred_class,
    logits=None,
    energy_weights=None,
    high_confidence_threshold=0.8,
):
    """Return scalar per-event prediction diagnostics."""
    true_class = _to_1d_cpu_tensor(true_class, dtype=torch.long)
    pred_class = _to_1d_cpu_tensor(pred_class, dtype=torch.long)
    if true_class is None or pred_class is None or true_class.shape != pred_class.shape:
        raise ValueError("true_class and pred_class must be aligned one-dimensional vectors.")

    num_hits = int(true_class.numel())
    correct_mask = true_class == pred_class
    incorrect_mask = ~correct_mask
    correct_hits = int(correct_mask.sum().item())
    record = {
        "num_hits": num_hits,
        "correct_hits": correct_hits,
        "incorrect_hits": num_hits - correct_hits,
        "accuracy": None if num_hits == 0 else correct_hits / num_hits,
    }

    logits = _to_2d_cpu_tensor(logits, dtype=torch.float32)
    if logits is not None:
        if logits.ndim != 2 or logits.shape[0] != num_hits:
            raise ValueError(
                "logits must have shape [num_hits, num_classes], "
                f"got {tuple(logits.shape)} for {num_hits} hit(s)."
            )
        probabilities = F.softmax(logits, dim=1)
        confidence, _prob_pred_class = probabilities.max(dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
        if probabilities.shape[1] > 1:
            entropy = entropy / math.log(probabilities.shape[1])
            top2 = probabilities.topk(k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = torch.ones_like(confidence)

        valid_true = (true_class >= 0) & (true_class < probabilities.shape[1])
        true_probability = torch.full_like(confidence, float("nan"))
        if bool(valid_true.any().item()):
            true_probability[valid_true] = probabilities[valid_true, true_class[valid_true]]

        record.update(
            {
                "mean_confidence": _mean_or_none(confidence),
                "mean_true_probability": _mean_or_none(true_probability[torch.isfinite(true_probability)]),
                "mean_margin": _mean_or_none(margin),
                "mean_entropy": _mean_or_none(entropy),
                "correct_mean_confidence": _mean_or_none(confidence[correct_mask]),
                "wrong_mean_confidence": _mean_or_none(confidence[incorrect_mask]),
                "high_confidence_error_fraction": (
                    None
                    if num_hits == 0
                    else float(((confidence >= high_confidence_threshold) & incorrect_mask).sum().item())
                    / num_hits
                ),
            }
        )

    weights = _to_1d_cpu_tensor(energy_weights, dtype=torch.float32)
    if weights is not None and weights.shape[0] == num_hits:
        weights = weights.clamp_min(0.0)
        total_weight = float(weights.sum().item())
        if total_weight > 0.0:
            correct_weight = float(weights[correct_mask].sum().item())
            incorrect_weight = total_weight - correct_weight
            record.update(
                {
                    "total_energy_weight": total_weight,
                    "correct_energy_weight": correct_weight,
                    "incorrect_energy_weight": incorrect_weight,
                    "energy_weighted_accuracy": correct_weight / total_weight,
                    "wrong_energy_fraction": incorrect_weight / total_weight,
                }
            )
        else:
            record.update(
                {
                    "total_energy_weight": total_weight,
                    "correct_energy_weight": None,
                    "incorrect_energy_weight": None,
                    "energy_weighted_accuracy": None,
                    "wrong_energy_fraction": None,
                }
            )

    return record


def optimal_label_permutation_summary(
    true_class,
    pred_class,
    num_classes=None,
    energy_weights=None,
):
    """Measure event accuracy after the best global relabeling of predictions."""
    true_class = _to_1d_cpu_tensor(true_class, dtype=torch.long)
    pred_class = _to_1d_cpu_tensor(pred_class, dtype=torch.long)
    if true_class is None or pred_class is None or true_class.shape != pred_class.shape:
        raise ValueError("true_class and pred_class must be aligned one-dimensional vectors.")

    if num_classes is None:
        nonnegative = torch.cat([true_class[true_class >= 0], pred_class[pred_class >= 0]])
        num_classes = int(nonnegative.max().item()) + 1 if nonnegative.numel() else 0
    num_classes = int(num_classes)
    if num_classes < 1:
        return {}
    if num_classes > 8:
        raise ValueError("Permutation diagnostics are limited to at most eight classes.")

    valid = (
        (true_class >= 0)
        & (true_class < num_classes)
        & (pred_class >= 0)
        & (pred_class < num_classes)
    )
    if not bool(valid.any().item()):
        return {}
    true_valid = true_class[valid]
    pred_valid = pred_class[valid]
    identity_correct = int((true_valid == pred_valid).sum().item())

    valid_weights = None
    total_weight = None
    identity_weight = None
    weights = _to_1d_cpu_tensor(energy_weights, dtype=torch.float64)
    if weights is not None and weights.shape == true_class.shape:
        valid_weights = weights[valid].clamp_min(0.0)
        total_weight = float(valid_weights.sum().item())
        if total_weight <= 0.0:
            valid_weights = None
            total_weight = None
        else:
            identity_weight = float(valid_weights[true_valid == pred_valid].sum().item())

    best_correct = -1
    best_mapping = tuple(range(num_classes))
    best_weight = -1.0
    best_weight_mapping = tuple(range(num_classes))
    for mapping in itertools.permutations(range(num_classes)):
        mapping_tensor = torch.tensor(mapping, dtype=torch.long)
        mapped_prediction = mapping_tensor[pred_valid]
        correct_mask = true_valid == mapped_prediction
        correct = int(correct_mask.sum().item())
        if correct > best_correct:
            best_correct = correct
            best_mapping = mapping
        if valid_weights is not None:
            correct_weight = float(valid_weights[correct_mask].sum().item())
            if correct_weight > best_weight:
                best_weight = correct_weight
                best_weight_mapping = mapping

    num_valid = int(valid.sum().item())
    summary = {
        "permutation_invariant_correct_hits": best_correct,
        "permutation_invariant_accuracy": best_correct / num_valid,
        "label_permutation_gain": (best_correct - identity_correct) / num_valid,
        "optimal_prediction_label_mapping": list(best_mapping),
        "optimal_prediction_label_mapping_is_identity": best_mapping
        == tuple(range(num_classes)),
    }

    if valid_weights is not None:
        summary.update(
            {
                "permutation_invariant_energy_weighted_accuracy": best_weight
                / total_weight,
                "energy_weighted_label_permutation_gain": (
                    best_weight - identity_weight
                )
                / total_weight,
                "optimal_energy_weighted_prediction_label_mapping": list(
                    best_weight_mapping
                ),
                "optimal_energy_weighted_prediction_label_mapping_is_identity": (
                    best_weight_mapping == tuple(range(num_classes))
                ),
            }
        )
    return summary


def truth_fraction_margin_summary(
    fraction_target,
    true_class,
    pred_class,
    energy_weights=None,
    bin_edges=TRUTH_MARGIN_BIN_EDGES,
):
    """Summarize how prediction accuracy changes with truth-composition ambiguity."""
    fractions = _to_2d_cpu_tensor(fraction_target, dtype=torch.float64)
    true_class = _to_1d_cpu_tensor(true_class, dtype=torch.long)
    pred_class = _to_1d_cpu_tensor(pred_class, dtype=torch.long)
    if fractions is None:
        return {}
    if (
        fractions.ndim != 2
        or fractions.shape[0] != true_class.shape[0]
        or true_class.shape != pred_class.shape
        or fractions.shape[1] < 2
    ):
        return {}

    finite_rows = torch.isfinite(fractions).all(dim=1)
    fractions = fractions.clamp_min(0.0)
    row_sums = fractions.sum(dim=1)
    valid = finite_rows & (row_sums > 0.0)
    if not bool(valid.any().item()):
        return {}
    normalized = fractions[valid] / row_sums[valid].unsqueeze(1)
    top_two = normalized.topk(k=2, dim=1).values
    margins = top_two[:, 0] - top_two[:, 1]
    correct = true_class[valid] == pred_class[valid]

    edges = torch.as_tensor(bin_edges, dtype=torch.float64)
    if edges.ndim != 1 or edges.numel() < 2 or not bool((edges[1:] > edges[:-1]).all().item()):
        raise ValueError("Truth-margin bin edges must be a strictly increasing vector.")
    bin_index = torch.bucketize(margins, edges[1:-1], right=False)
    num_bins = int(edges.numel() - 1)
    total_hits = torch.bincount(bin_index, minlength=num_bins)
    correct_hits = torch.bincount(bin_index[correct], minlength=num_bins)

    summary = {
        "truth_fraction_num_valid_hits": int(valid.sum().item()),
        "mean_truth_dominance_margin": _mean_or_none(margins),
        "median_truth_dominance_margin": _quantile_or_none(margins, 0.5),
        "mixed_truth_hit_fraction": float((margins < 0.999999).double().mean().item()),
        "truth_ambiguous_hit_fraction_margin_le_0p1": float(
            (margins <= 0.1).double().mean().item()
        ),
        "truth_ambiguous_hit_fraction_margin_le_0p25": float(
            (margins <= 0.25).double().mean().item()
        ),
        "truth_ambiguous_hit_fraction_margin_le_0p5": float(
            (margins <= 0.5).double().mean().item()
        ),
        "truth_margin_bin_edges": [float(value) for value in edges.tolist()],
        "truth_margin_bin_total_hits": [int(value) for value in total_hits.tolist()],
        "truth_margin_bin_correct_hits": [int(value) for value in correct_hits.tolist()],
    }

    weights = _to_1d_cpu_tensor(energy_weights, dtype=torch.float64)
    if weights is not None and weights.shape == true_class.shape:
        weights = weights[valid].clamp_min(0.0)
        total_weight = float(weights.sum().item())
        if total_weight > 0.0:
            weighted_total = torch.zeros(num_bins, dtype=torch.float64)
            weighted_correct = torch.zeros(num_bins, dtype=torch.float64)
            weighted_total.scatter_add_(0, bin_index, weights)
            weighted_correct.scatter_add_(0, bin_index[correct], weights[correct])
            summary.update(
                {
                    "truth_margin_bin_total_energy": [
                        float(value) for value in weighted_total.tolist()
                    ],
                    "truth_margin_bin_correct_energy": [
                        float(value) for value in weighted_correct.tolist()
                    ],
                    "energy_weighted_mixed_truth_fraction": float(
                        weights[margins < 0.999999].sum().item()
                    )
                    / total_weight,
                }
            )
    return summary


def detector_context_summary(view):
    """Report available TPad context without inventing values for ECal-only views."""
    if "tpad_mask" not in view and "tpad" not in view:
        return {}
    if "tpad_mask" in view:
        mask = _to_1d_cpu_tensor(view["tpad_mask"], dtype=torch.bool)
        num_tpad = int(mask.sum().item())
    else:
        tpad = _to_2d_cpu_tensor(view.get("tpad"))
        num_tpad = 0 if tpad is None else int(tpad.shape[0])

    summary = {"num_tpad_tokens": num_tpad}
    electron_count = view.get("electron_count")
    if electron_count is not None:
        electron_count = int(_to_1d_cpu_tensor(electron_count, dtype=torch.long)[0].item())
        summary.update(
            {
                "expected_electron_count": electron_count,
                "tpad_token_deficit": max(0, electron_count - num_tpad),
                "tpad_token_excess": max(0, num_tpad - electron_count),
                "has_complete_tpad_context": num_tpad == electron_count,
            }
        )
    return summary


def _centroids_by_label(pos, labels, dims):
    valid = labels >= 0
    if not bool(valid.any().item()):
        return None, []
    present_labels = sorted({int(label) for label in labels[valid].tolist()})
    centroids = []
    for label in present_labels:
        mask = labels == label
        if bool(mask.any().item()):
            centroids.append(pos[mask][:, dims].mean(dim=0))
    if not centroids:
        return None, []
    return torch.stack(centroids, dim=0), present_labels


def _centroid_distance_stats(centroids, radius):
    if centroids is None or centroids.shape[0] < 2:
        return None, None, None
    pairwise = torch.pdist(centroids.to(dtype=torch.float64))
    min_distance = _finite_float(pairwise.min().item()) if pairwise.numel() else None
    mean_distance = _finite_float(pairwise.mean().item()) if pairwise.numel() else None
    within_radius = None
    if radius is not None and radius > 0:
        distances = torch.cdist(centroids.to(dtype=torch.float64), centroids.to(dtype=torch.float64))
        within_radius = int((distances <= float(radius)).sum(dim=1).max().item())
    return min_distance, mean_distance, within_radius


def contributor_shower_membership(fraction_target):
    """Return binary hit membership for every origin with a positive contribution."""
    fractions = _to_2d_cpu_tensor(fraction_target, dtype=torch.float64)
    if fractions is None or fractions.ndim != 2:
        return None
    membership = torch.isfinite(fractions) & (fractions > 0.0)
    active_columns = membership.any(dim=0)
    return membership[:, active_columns]


def dominant_shower_membership(labels):
    """Return exclusive binary hit membership from dominant-origin labels."""
    labels = _to_1d_cpu_tensor(labels, dtype=torch.long)
    if labels is None:
        return None
    present_labels = sorted({int(label) for label in labels[labels >= 0].tolist()})
    if not present_labels:
        return torch.zeros((labels.shape[0], 0), dtype=torch.bool)
    return torch.stack([labels == label for label in present_labels], dim=1)


def _shower_moments_from_membership(pos_xy, membership):
    if membership is None:
        return None, None
    pos_xy = _to_2d_cpu_tensor(pos_xy, dtype=torch.float64)
    membership = _to_2d_cpu_tensor(membership, dtype=torch.bool)
    if pos_xy is None or pos_xy.ndim != 2 or pos_xy.shape[1] != 2:
        raise ValueError("pos_xy must have shape [num_hits, 2].")
    if membership.ndim != 2 or membership.shape[0] != pos_xy.shape[0]:
        raise ValueError("membership must have shape [num_hits, num_showers].")

    active_columns = membership.any(dim=0)
    membership = membership[:, active_columns]
    centroids = []
    widths = []
    for shower_idx in range(membership.shape[1]):
        shower_pos = pos_xy[membership[:, shower_idx]]
        if shower_pos.numel() == 0:
            continue
        centroid = shower_pos.mean(dim=0)
        squared_radius = ((shower_pos - centroid) ** 2).sum(dim=1)
        width = torch.sqrt(squared_radius.mean())
        centroids.append(centroid)
        widths.append(width)
    if not centroids:
        return None, None
    return torch.stack(centroids), torch.stack(widths)


def _normalized_separation_stats(centroids, widths):
    if centroids is None or widths is None or centroids.shape[0] < 2:
        return None, None
    separations = []
    for first_idx in range(centroids.shape[0] - 1):
        for second_idx in range(first_idx + 1, centroids.shape[0]):
            combined_width = torch.sqrt(
                widths[first_idx] ** 2 + widths[second_idx] ** 2
            )
            if float(combined_width.item()) <= 1e-12:
                continue
            distance = torch.linalg.vector_norm(
                centroids[first_idx] - centroids[second_idx]
            )
            separations.append(distance / combined_width)
    if not separations:
        return None, None
    values = torch.stack(separations)
    return _finite_float(values.min().item()), _mean_or_none(values)


def projected_shower_geometry(pos_xy, membership, centroid_radius_mm=None):
    """Summarize projected shower centroids, radial widths, and pair separations."""
    centroids, widths = _shower_moments_from_membership(pos_xy, membership)
    min_distance, mean_distance, within_radius = _centroid_distance_stats(
        centroids,
        radius=centroid_radius_mm,
    )
    minimum, mean = _normalized_separation_stats(centroids, widths)
    return {
        "num_showers": 0 if widths is None else int(widths.numel()),
        "min_centroid_distance_xy": min_distance,
        "mean_centroid_distance_xy": mean_distance,
        "max_centroids_within_radius_xy": within_radius,
        "min_normalized_shower_separation_xy": minimum,
        "mean_normalized_shower_separation_xy": mean,
        "mean_shower_width_xy": _mean_or_none(widths),
        "max_shower_width_xy": (
            None if widths is None or widths.numel() == 0 else _finite_float(widths.max().item())
        ),
    }


def _named_shower_geometry(geometry, membership_name, layer_prefix=""):
    return {
        f"{layer_prefix}{membership_name}_{key}": value
        for key, value in geometry.items()
    }


def _default_geometry_aliases(summary, contributor_geometry, layer_prefix=""):
    aliases = {
        "min_centroid_distance_xy": "min_origin_centroid_distance_xy",
        "mean_centroid_distance_xy": "mean_origin_centroid_distance_xy",
        "max_centroids_within_radius_xy": "max_origin_centroids_within_radius_xy",
        "min_normalized_shower_separation_xy": "min_normalized_shower_separation_xy",
        "mean_normalized_shower_separation_xy": "mean_normalized_shower_separation_xy",
        "mean_shower_width_xy": "mean_shower_width_xy",
        "max_shower_width_xy": "max_shower_width_xy",
    }
    for source_key, alias_key in aliases.items():
        summary[f"{layer_prefix}{alias_key}"] = contributor_geometry[source_key]


def _hit_centroid_margin_summary(pos_xy, labels, weights=None):
    centroids, present_labels = _centroids_by_label(pos_xy, labels, dims=[0, 1])
    if centroids is None or centroids.shape[0] < 2:
        return {}
    label_to_idx = {label: idx for idx, label in enumerate(present_labels)}
    label_indices = torch.tensor(
        [label_to_idx.get(int(label), -1) for label in labels.tolist()],
        dtype=torch.long,
    )
    valid = label_indices >= 0
    if not bool(valid.any().item()):
        return {}

    distances = torch.cdist(pos_xy[valid].to(dtype=torch.float64), centroids.to(dtype=torch.float64))
    row = torch.arange(distances.shape[0])
    own_indices = label_indices[valid]
    own_distance = distances[row, own_indices]
    other_distances = distances.clone()
    other_distances[row, own_indices] = float("inf")
    nearest_other_distance = other_distances.min(dim=1).values
    finite_other = torch.isfinite(nearest_other_distance)
    if not bool(finite_other.any().item()):
        return {}
    margin = nearest_other_distance[finite_other] - own_distance[finite_other]
    ambiguous = margin < 0.0
    summary = {
        "mean_hit_distance_to_origin_centroid_xy": _mean_or_none(own_distance),
        "p90_hit_distance_to_origin_centroid_xy": _quantile_or_none(own_distance, 0.9),
        "mean_hit_centroid_margin_xy": _mean_or_none(margin),
        "min_hit_centroid_margin_xy": _finite_float(margin.min().item()),
        "ambiguous_hit_fraction_xy": float(ambiguous.to(dtype=torch.float64).mean().item()),
    }
    if weights is not None and weights.shape[0] == labels.shape[0]:
        valid_weights = weights[valid][finite_other].to(dtype=torch.float64).clamp_min(0.0)
        total_weight = float(valid_weights.sum().item())
        summary["energy_weighted_ambiguous_hit_fraction_xy"] = (
            float(valid_weights[ambiguous].sum().item()) / total_weight
            if total_weight > 0.0
            else None
        )
    return summary


def geometry_metric_summary(
    view,
    true_class,
    centroid_radius_mm=25.0,
    first_layer_tolerance_mm=1e-3,
):
    """Return event geometry and shower-overlap summaries from ECal positions."""
    true_class = _to_1d_cpu_tensor(true_class, dtype=torch.long)
    num_hits = 0 if true_class is None else int(true_class.numel())
    pos = _to_2d_cpu_tensor(view.get("ecal_pos"), dtype=torch.float32)
    if pos is None or pos.ndim != 2 or pos.shape != (num_hits, 3):
        return {}

    labels = _optional_aligned_hit_vector(
        view,
        names=("origin_id_y", "physical_y", "canonical_y", "y"),
        num_hits=num_hits,
        dtype=torch.long,
    )
    if labels is None:
        labels = true_class

    fraction_target = None
    fraction_target_name = None
    for name in ("origin_id_fraction_target", "fraction_target"):
        candidate = _to_2d_cpu_tensor(view.get(name), dtype=torch.float32)
        if candidate is not None and candidate.ndim == 2 and candidate.shape[0] == num_hits:
            fraction_target = candidate
            fraction_target_name = name
            break

    dominant_memberships = dominant_shower_membership(labels)
    contributor_memberships = contributor_shower_membership(fraction_target)
    if contributor_memberships is None:
        contributor_memberships = dominant_memberships
        contributor_membership_source = "dominant_origin_fallback"
    else:
        contributor_membership_source = fraction_target_name

    raw_energy_weights = _optional_aligned_hit_vector(
        view,
        names=("ecal_raw_energy",),
        num_hits=num_hits,
        dtype=torch.float32,
    )
    input_energy_weights = _optional_aligned_hit_vector(
        view,
        names=("ecal_input_energy", "ecal_energy"),
        num_hits=num_hits,
        dtype=torch.float32,
    )
    if raw_energy_weights is not None:
        energy_weights = raw_energy_weights
        ambiguity_weighting = "raw_reconstructed_energy"
    elif input_energy_weights is not None:
        energy_weights = input_energy_weights
        ambiguity_weighting = "model_input_energy_fallback"
    else:
        energy_weights = None
        ambiguity_weighting = "uniform"

    valid_labels = labels >= 0
    present_labels = sorted({int(label) for label in labels[valid_labels].tolist()})
    z_values = pos[:, 2]
    xy = pos[:, :2]

    summary = {
        "diagnostic_centroid_radius_mm": float(centroid_radius_mm),
        "shower_separation_default": "binary_any_contributor",
        "shower_overlap_weighting": "uniform_binary_membership",
        "contributor_membership_source": contributor_membership_source,
        "ambiguity_weighting": ambiguity_weighting,
        "num_truth_classes": len(present_labels),
        "num_ecal_layers": int(torch.unique(z_values).numel()),
        "ecal_z_min": _finite_float(z_values.min().item()) if num_hits else None,
        "ecal_z_max": _finite_float(z_values.max().item()) if num_hits else None,
        "ecal_z_span": _finite_float((z_values.max() - z_values.min()).item()) if num_hits else None,
    }

    centroids_3d, _labels_3d = _centroids_by_label(pos, labels, dims=[0, 1, 2])
    min_distance, mean_distance, within_radius = _centroid_distance_stats(
        centroids_3d,
        radius=centroid_radius_mm,
    )
    summary.update(
        {
            "min_origin_centroid_distance_3d": min_distance,
            "mean_origin_centroid_distance_3d": mean_distance,
            "max_origin_centroids_within_radius_3d": within_radius,
        }
    )

    contributor_geometry = projected_shower_geometry(
        xy,
        contributor_memberships,
        centroid_radius_mm=centroid_radius_mm,
    )
    dominant_geometry = projected_shower_geometry(
        xy,
        dominant_memberships,
        centroid_radius_mm=centroid_radius_mm,
    )
    summary.update(_named_shower_geometry(contributor_geometry, "contributor"))
    summary.update(_named_shower_geometry(dominant_geometry, "dominant"))
    _default_geometry_aliases(summary, contributor_geometry)
    summary.update(_hit_centroid_margin_summary(xy, labels, weights=energy_weights))

    if num_hits:
        unique_z = torch.unique(z_values, sorted=True)
        early_z = unique_z[: min(3, unique_z.numel())]
        early_mask = torch.isin(z_values, early_z)
        early_labels = labels[early_mask]
        early_pos = pos[early_mask]
        early_valid_labels = early_labels >= 0
        early_present_labels = sorted(
            {int(label) for label in early_labels[early_valid_labels].tolist()}
        )
        early_contributor_geometry = projected_shower_geometry(
            early_pos[:, :2],
            contributor_memberships[early_mask],
            centroid_radius_mm=None,
        )
        early_dominant_geometry = projected_shower_geometry(
            early_pos[:, :2],
            dominant_memberships[early_mask],
            centroid_radius_mm=None,
        )
        summary.update(
            {
                "early_layer_count": int(early_z.numel()),
                "early_layer_num_hits": int(early_mask.sum().item()),
                "early_layer_num_truth_classes": len(early_present_labels),
            }
        )
        summary.update(
            _named_shower_geometry(
                early_contributor_geometry,
                "contributor",
                layer_prefix="early_",
            )
        )
        summary.update(
            _named_shower_geometry(
                early_dominant_geometry,
                "dominant",
                layer_prefix="early_",
            )
        )
        _default_geometry_aliases(summary, early_contributor_geometry, layer_prefix="early_")

        first_z = z_values.min()
        first_layer_mask = torch.isclose(
            z_values,
            first_z,
            atol=float(first_layer_tolerance_mm),
            rtol=0.0,
        )
        first_labels = labels[first_layer_mask]
        first_pos = pos[first_layer_mask]
        first_valid_labels = first_labels >= 0
        first_present_labels = sorted(
            {int(label) for label in first_labels[first_valid_labels].tolist()}
        )
        first_contributor_geometry = projected_shower_geometry(
            first_pos[:, :2],
            contributor_memberships[first_layer_mask],
            centroid_radius_mm=centroid_radius_mm,
        )
        first_dominant_geometry = projected_shower_geometry(
            first_pos[:, :2],
            dominant_memberships[first_layer_mask],
            centroid_radius_mm=centroid_radius_mm,
        )
        summary.update(
            {
                "first_layer_z": _finite_float(first_z.item()),
                "first_layer_num_hits": int(first_layer_mask.sum().item()),
                "first_layer_num_truth_classes": len(first_present_labels),
            }
        )
        summary.update(
            _named_shower_geometry(
                first_contributor_geometry,
                "contributor",
                layer_prefix="first_layer_",
            )
        )
        summary.update(
            _named_shower_geometry(
                first_dominant_geometry,
                "dominant",
                layer_prefix="first_layer_",
            )
        )
        _default_geometry_aliases(summary, first_contributor_geometry, layer_prefix="first_layer_")

    return summary


def event_diagnostic_record(
    event_idx,
    split_position,
    view,
    true_class,
    pred_class,
    loss=None,
    logits=None,
    centroid_radius_mm=25.0,
):
    """Build one JSON/CSV-safe event diagnostic record."""
    true_class = _to_1d_cpu_tensor(true_class, dtype=torch.long)
    pred_class = _to_1d_cpu_tensor(pred_class, dtype=torch.long)
    energy_weights = _optional_aligned_hit_vector(
        view,
        names=("ecal_raw_energy", "ecal_input_energy", "ecal_energy"),
        num_hits=int(true_class.numel()),
        dtype=torch.float32,
    )
    record = {
        "event_idx": int(event_idx),
        "split_position": int(split_position),
        **prediction_metric_summary(
            true_class=true_class,
            pred_class=pred_class,
            logits=logits,
            energy_weights=energy_weights,
        ),
        "loss": None if loss is None else _finite_float(loss.detach().cpu().item()),
    }
    num_classes = None
    if logits is not None:
        logits_tensor = _to_2d_cpu_tensor(logits)
        if logits_tensor is not None and logits_tensor.ndim == 2:
            num_classes = int(logits_tensor.shape[1])
    record.update(
        optimal_label_permutation_summary(
            true_class=true_class,
            pred_class=pred_class,
            num_classes=num_classes,
            energy_weights=energy_weights,
        )
    )
    fraction_target = view.get("origin_id_fraction_target", view.get("fraction_target"))
    record.update(
        truth_fraction_margin_summary(
            fraction_target=fraction_target,
            true_class=true_class,
            pred_class=pred_class,
            energy_weights=energy_weights,
        )
    )
    record.update(detector_context_summary(view))
    record.update(
        geometry_metric_summary(
            view=view,
            true_class=true_class,
            centroid_radius_mm=centroid_radius_mm,
        )
    )
    return record


def select_representative_events(records, limit_per_group=3, metric="accuracy"):
    """Select worst, median, and best events by one scalar metric."""
    valid_records = []
    for record in records:
        value = record.get(metric)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            valid_records.append(record)
    if not valid_records:
        return {"metric": metric, "worst": [], "median": [], "best": []}

    limit = max(0, int(limit_per_group))
    median_value = sorted(float(record[metric]) for record in valid_records)[len(valid_records) // 2]

    def ascending_key(record):
        return (
            float(record[metric]),
            -int(record.get("incorrect_hits", 0)),
            int(record.get("event_idx", 0)),
        )

    def descending_key(record):
        return (
            -float(record[metric]),
            int(record.get("incorrect_hits", 0)),
            int(record.get("event_idx", 0)),
        )

    def median_key(record):
        return (
            abs(float(record[metric]) - median_value),
            int(record.get("incorrect_hits", 0)),
            int(record.get("event_idx", 0)),
        )

    return {
        "metric": metric,
        "median_value": median_value,
        "worst": sorted(valid_records, key=ascending_key)[:limit],
        "median": sorted(valid_records, key=median_key)[:limit],
        "best": sorted(valid_records, key=descending_key)[:limit],
    }
