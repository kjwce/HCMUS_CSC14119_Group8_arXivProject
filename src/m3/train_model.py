"""Fit classical pair classifiers on Train, select on Valid, evaluate Test once."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .evaluate import evaluate_predictions, validation_objective
from .model_data import feature_manifest, handoff_labels, iter_feature_batches, matrix, sha256
from .prediction import rank_candidates, write_predictions
from .run_m3 import _atomic_jsonl, _atomic_text


def collect_training(features, labels, columns, negative_cap, seed):
    """Train-only scaler and uniform negative reservoir for bounded tree fitting."""
    rng = np.random.default_rng(seed)
    scaler = StandardScaler()
    positives = []
    negative_x = np.empty((0, len(columns)), dtype=np.float32)
    priorities = np.empty(0)
    counts = np.zeros(2, dtype=np.int64)
    for frame in iter_feature_batches(features, labels, "train"):
        values = matrix(frame, columns)
        y = frame["label"].to_numpy(dtype=np.int32)
        scaler.partial_fit(values)
        counts += np.bincount(y, minlength=2)
        positives.append(values[y == 1])
        incoming = values[y == 0]
        negative_x = np.concatenate([negative_x, incoming])
        priorities = np.concatenate([priorities, rng.random(len(incoming))])
        if len(priorities) > negative_cap:
            keep = np.argpartition(priorities, negative_cap - 1)[:negative_cap]
            negative_x, priorities = negative_x[keep], priorities[keep]
    if not np.all(counts > 0):
        raise ValueError("Training requires both positive and negative pairs")
    positive_x = np.concatenate(positives)
    tree_x = np.concatenate([positive_x, negative_x])
    tree_y = np.concatenate([np.ones(len(positive_x), dtype=np.int32), np.zeros(len(negative_x), dtype=np.int32)])
    return scaler, counts, tree_x, tree_y


def train_sgd(features, labels, columns, scaler, counts, alpha, epochs, seed):
    model = SGDClassifier(loss="log_loss", alpha=alpha, average=True, random_state=seed)
    weights = counts.sum() / (2 * counts.astype(float))
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        for frame in iter_feature_batches(features, labels, "train"):
            values = scaler.transform(matrix(frame, columns))
            y = frame["label"].to_numpy(dtype=np.int32)
            order = rng.permutation(len(frame))
            model.partial_fit(values[order], y[order], classes=np.array([0, 1]), sample_weight=weights[y[order]])
        print(f"Logistic SGD alpha={alpha}: epoch {epoch + 1}/{epochs}", flush=True)
    return model


def plot_validation_comparison(comparison, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([row["name"] for row in comparison], [row["validation_objective"] for row in comparison], color="#2469A0")
    ax.set_ylabel("Validation MRR@5\n(source-balanced)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def write_plots(output, columns, x, y, comparison, results):
    """Training distributions justify features; held-out scores are descriptive."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output / "plots"
    plots.mkdir()
    for offset in range(0, len(columns), 6):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for axis, name in zip(axes.flat, columns[offset:offset + 6]):
            index = columns.index(name)
            low, high = np.quantile(x[:, index], [.01, .99])
            if high <= low:
                low, high = low - .5, high + .5
            for value, label, color in ((0, "nonmatch", "#D28A35"), (1, "match", "#2469A0")):
                values = np.clip(x[y == value, index], low, high)
                axis.hist(values, bins=20, density=True, histtype="step", linewidth=1.4, label=label, color=color)
            axis.set_title(name, fontsize=10)
            axis.set_ylabel("Density")
            axis.legend(fontsize=8)
        for axis in list(axes.flat)[len(columns[offset:offset + 6]):]:
            axis.set_visible(False)
        fig.suptitle("Train pair feature distributions (uniform negative sample; 1–99% clipped)")
        fig.tight_layout()
        fig.savefig(plots / f"feature_distributions_{offset // 6 + 1:02d}.png", dpi=140)
        plt.close(fig)
    plot_validation_comparison(comparison, plots / "validation_model_comparison.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    groups = results["by_partition_source"]
    positions = np.arange(3)
    for offset, source in ((-.18, "manual"), (.18, "automatic")):
        ax.bar(positions + offset, [groups[part][source]["mrr_at_5"] or 0 for part in ("train", "valid", "test")], width=.36, label=source)
    ax.set_xticks(positions, ["Train", "Valid", "Test"])
    ax.set_ylabel("MRR@5 (all labelled entries)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "selected_model_mrr.png", dpi=150)
    plt.close(fig)


def train(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError("Use a new model output directory")
    if min(args.epochs, args.negative_cap, args.tree_iterations) < 1:
        raise ValueError("Training limits must be positive")
    manifest = feature_manifest(args.features)
    for name in ("labels.jsonl", "partitions.jsonl"):
        if manifest["input_sha256"][name] != sha256(args.handoff / name):
            raise ValueError(f"Frozen handoff changed: {name}")
    labels, parts = handoff_labels(args.handoff)
    expected_train = sorted(pid for pid, part in parts.items() if part == "train")
    if manifest["tfidf_fit_partition"] != "train" or manifest["tfidf_allowed_training_publications"] != expected_train:
        raise ValueError("TF-IDF fit provenance includes held-out publications")
    columns = manifest["feature_columns"]
    args.output.mkdir(parents=True)
    models_dir = args.output / "models"
    models_dir.mkdir()
    scaler, counts, tree_x, tree_y = collect_training(args.features, labels, columns, args.negative_cap, args.seed)
    print(f"Train pairs: negative={counts[0]:,}, positive={counts[1]:,}; tree sample={len(tree_y):,}", flush=True)
    comparison = []
    baseline = rank_candidates(args.features, labels, columns, partition="valid")
    baseline_report = evaluate_predictions(baseline, labels, "valid")
    comparison.append({"name": "fixed_combo_baseline", "validation_objective": validation_objective(baseline_report), "report": baseline_report})
    bundles = {}
    for alpha in (1e-4, 1e-5):
        name = f"logistic_sgd_alpha_{alpha:g}"
        model = train_sgd(args.features, labels, columns, scaler, counts, alpha, args.epochs, args.seed)
        bundle = {"name": name, "model": model, "scaler": scaler, "feature_columns": columns,
                  "features_manifest_sha256": sha256(args.features / "features_manifest.json")}
        joblib.dump(bundle, models_dir / f"{name}.joblib")
        bundles[name] = bundle
        valid_rows = rank_candidates(args.features, labels, columns, model, scaler, "valid")
        report = evaluate_predictions(valid_rows, labels, "valid")
        comparison.append({"name": name, "validation_objective": validation_objective(report), "report": report})
    tree_counts = np.bincount(tree_y, minlength=2)
    sample_weights = len(tree_y) / (2 * tree_counts[tree_y])
    tree = HistGradientBoostingClassifier(max_iter=args.tree_iterations, max_leaf_nodes=15, learning_rate=.08, early_stopping=False, random_state=args.seed)
    with threadpool_limits(limits=2):
        tree.fit(tree_x, tree_y, sample_weight=sample_weights)
    name = "hist_gradient_boosting"
    bundles[name] = {"name": name, "model": tree, "scaler": None, "feature_columns": columns,
                     "features_manifest_sha256": sha256(args.features / "features_manifest.json")}
    joblib.dump(bundles[name], models_dir / f"{name}.joblib")
    valid_rows = rank_candidates(args.features, labels, columns, tree, partition="valid")
    report = evaluate_predictions(valid_rows, labels, "valid")
    comparison.append({"name": name, "validation_objective": validation_objective(report), "report": report})
    # Baseline is reported, but a trained supervised model is required for submission.
    chosen = max([row for row in comparison if row["name"] in bundles], key=lambda row: row["validation_objective"])
    bundle = bundles[chosen["name"]]
    selection = {
        "selected_model": chosen["name"], "criterion": "mean manual/automatic Validation MRR@5 over all labelled entries",
        "comparison": comparison, "seed": args.seed,
        "train_pair_counts": {"negative": int(counts[0]), "positive": int(counts[1])},
        "tree_sampling": {"all_train_positives": int(tree_counts[1]), "uniform_train_negatives": int(tree_counts[0])},
        "tfidf_fit": "train only", "scaler_fit": "train only", "test_used_for_selection": False,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__},
    }
    # Freeze and persist selection BEFORE accessing Test predictions/metrics.
    _atomic_text(args.output / "selection.json", json.dumps(selection, indent=2) + "\n")
    joblib.dump(bundle, args.output / "best_model.joblib")
    print(f"Selected by Validation: {chosen['name']}. Producing final Train/Valid/Test predictions...", flush=True)
    predictions = rank_candidates(args.features, labels, columns, bundle["model"], bundle["scaler"])
    write_predictions(args.output, predictions)
    metrics = evaluate_predictions(predictions, labels)
    metrics["selected_model"] = chosen["name"]
    metrics["limitations"] = ["Automatic labels are silver labels from string heuristics.", "Manual labels include AI assistance and human review; not an independent purely human benchmark.", "Only five manual and five automatic publications form each holdout; estimates have limited support."]
    _atomic_text(args.output / "metrics.json", json.dumps(metrics, indent=2) + "\n")
    errors = [row for row in predictions if row["partition"] == "test" and row["groundtruth_arxiv_id"] is not None and row["cand_1"] != row["groundtruth_arxiv_id"]]
    _atomic_jsonl(args.output / "test_error_cases.jsonl", errors)
    write_plots(args.output, columns, tree_x, tree_y, comparison, metrics)
    _atomic_text(args.output / "training_report.json", json.dumps({"status": "complete", "wall_time_s": round(time.perf_counter() - started, 3), "predictions": len(predictions), "test_top1_errors_on_matched_entries": len(errors), **selection}, indent=2) + "\n")
    print(json.dumps({"selected_model": chosen["name"], "validation_objective": chosen["validation_objective"], "test": metrics["by_partition_source"]["test"], "predictions": len(predictions)}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=14119)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--negative-cap", type=int, default=200000)
    parser.add_argument("--tree-iterations", type=int, default=100)
    return train(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
