#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""UHD Generation Tool CLI.

Train a LightGBM model from benchmark data and export to FlatBuffer format
for use with hipDNN's TreeDataAdapter.

Usage:
    python -m uhd_gen \\
        --input benchmark_results.csv \\
        --features M N K tile_m tile_n cu_count \\
        --target tflops \\
        --group-by M N K \\
        --output-dir ./uhd_output \\
        --name "GEMM UHD"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

import pandas as pd

from .features import build_features_signature, compute_features_hash
from .lgbm_to_flatbuffer import convert
from .train_uhd import evaluate_regret, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


#: Substrings that mark a target as something you want less of. Used only to warn
#: about an objective/target mismatch, never to override the caller's choice.
_COST_METRIC_MARKERS = (
    "latency",
    "time",
    "duration",
    "elapsed",
    "_ms",
    "_us",
    "_ns",
    "sec",
    "cost",
    "error",
    "loss",
)


def _looks_like_cost_metric(target: str) -> bool:
    """Heuristic: does this target name describe something to minimize?"""
    lowered = target.lower()
    return any(marker in lowered for marker in _COST_METRIC_MARKERS)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate UHD heuristic from benchmark data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Benchmark CSV/JSON with feature columns and target",
    )
    parser.add_argument(
        "--features",
        required=True,
        nargs="+",
        help="Feature column names to train on",
    )
    parser.add_argument(
        "--target",
        default="tflops",
        help="Target column name (default: tflops)",
    )
    parser.add_argument(
        "--objective",
        choices=("max", "min"),
        default="max",
        help="Whether the runtime should maximize or minimize the score "
        "(default: max, correct for throughput targets like tflops). Pass 'min' "
        "for a cost target such as latency_ms.",
    )
    parser.add_argument(
        "--score-units",
        default=None,
        dest="score_units",
        help="Units the score is expressed in (default: the --target column name).",
    )
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="Declare the score cross-engine comparable (RFC 0019 §12.3). Only pass "
        "this if the target really is calibrated across engines; it is not verified "
        "here, and an unwarranted claim silently corrupts cross-engine comparison.",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=None,
        dest="group_by",
        help="Columns for GroupKFold (prevents problem leakage)",
    )
    parser.add_argument(
        "--report-regret",
        nargs="+",
        default=None,
        dest="report_regret",
        metavar="COL",
        help=(
            "Columns identifying one problem (e.g. the q.* columns). Reports "
            "out-of-fold top-1 regret of the ranking the model induces, which is what "
            "RFC 0019.13 §11 asks for and what CV RMSE cannot answer."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        dest="output_dir",
        help="Output directory for model artifacts",
    )
    parser.add_argument(
        "--name",
        default="UHD",
        help="UHD display name",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=500,
        dest="num_boost_round",
        help="Maximum number of boosting rounds (default: 500)",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=50,
        dest="early_stopping",
        help="Early stopping patience (default: 50)",
    )
    parser.add_argument(
        "--keep-lgbm",
        action="store_true",
        dest="keep_lgbm",
        help="Keep intermediate .lgbm file",
    )
    parser.add_argument(
        "--training-arches",
        nargs="+",
        dest="training_arches",
        help="GPU architectures the model was trained on (e.g., gfx942 gfx1100). "
        "Embedded in the model for RFC 0019 §9.2 out-of-distribution detection.",
    )
    parser.add_argument(
        "--model-version",
        dest="model_version",
        help="Semantic version for the model (e.g., 1.0.0). Embedded in model metadata.",
    )

    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from %s", input_path)
    if input_path.suffix == ".parquet":
        # What tools/results_import publishes (RFC 0019.13 §8.3). Preferred over the collected
        # CSV: the dataset carries its own types, so a column that is empty in one shard and
        # populated in another cannot concatenate to `object` and change what the trainer sees.
        df = pd.read_parquet(input_path)
    elif input_path.suffix == ".json":
        df = pd.read_json(input_path)
    else:
        # A collected CSV, read directly. Everything §8.3 checks is unchecked on this path --
        # that is what the importer exists for -- so it is the escape hatch for a quick local
        # run rather than the route a trained model should come by.
        df = pd.read_csv(input_path)
    logger.info("Loaded %d rows", len(df))

    missing = set(args.features) - set(df.columns)
    if missing:
        logger.error("Missing feature columns: %s", missing)
        return 1
    if args.target not in df.columns:
        logger.error("Missing target column: %s", args.target)
        return 1

    if args.objective == "max" and _looks_like_cost_metric(args.target):
        logger.warning(
            "Target '%s' looks like a cost metric but --objective is 'max', so the "
            "runtime will prefer the WORST kernel. Pass --objective min if that is "
            "not what you want.",
            args.target,
        )

    logger.info("Training on features: %s", args.features)
    logger.info("Target column: %s (objective: %s)", args.target, args.objective)
    if args.group_by:
        logger.info("GroupKFold columns: %s", args.group_by)

    model = train_model(
        df,
        args.features,
        args.target,
        args.group_by,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping,
    )

    if args.report_regret:
        # Reported after training and measured independently of it: this scores the
        # ranking the model induces on problems it did not see, which is the question
        # the heuristic exists to answer. RMSE says how close the numbers are.
        metrics = evaluate_regret(
            df,
            args.features,
            args.target,
            args.report_regret,
            num_boost_round=args.num_boost_round,
        )
        logger.info(
            "Out-of-fold top-1 accuracy %.1f%% over %d problems "
            "(%d single-variant excluded, %d unusable)",
            metrics["top1_accuracy"] * 100.0,
            metrics["problems_scored"],
            metrics["problems_single_variant"],
            metrics["problems_unusable"],
        )
        logger.info(
            "Regret mean %.4f, median %.4f, p90 %.4f, p99 %.4f, max %.4f",
            metrics["mean_regret"],
            metrics["median_regret"],
            metrics["p90_regret"],
            metrics["p99_regret"],
            metrics["max_regret"],
        )
        with (output_dir / "regret.json").open("w") as handle:
            json.dump(metrics, handle, indent=2)

    lgbm_path = output_dir / "model.lgbm"
    model.save_model(str(lgbm_path))
    logger.info("Saved LightGBM model to %s", lgbm_path)

    features_signature = build_features_signature(args.features)
    features_hash = compute_features_hash(features_signature)
    fb_path = output_dir / "model.bin"
    convert(
        lgbm_path,
        features_hash,
        fb_path,
        num_training_samples=len(df),
        training_arches=args.training_arches,
        model_version=args.model_version,
    )
    logger.info("Converted to FlatBuffer: %s", fb_path)

    if not args.keep_lgbm:
        lgbm_path.unlink()
        logger.info("Removed intermediate %s", lgbm_path)

    # Generate UHD identifier
    uhd_id = str(uuid.uuid4())

    # The whole UHD, in the descriptor. RFC 0019 §4 always specified JSON; an earlier
    # design put these fields in a FlatBuffer that a four-field stub pointed at, which
    # made the UHD the only descriptor in the family a human could not read, diff or
    # review -- to save 134 bytes on a file read once per engine.
    #
    # `model.bin` stays binary. It is read once per candidate score, and at a realistic
    # 500 trees it is 3.7 MB; that one earns its format.
    descriptor = {
        "version": "1.0",
        "id": uhd_id,
        "name": args.name,
        "adapter": "tree_data",
        "features_signature": features_signature,
        "features_hash": features_hash,
        "objective": args.objective,
        "score": {
            "units": args.score_units or args.target,
            "calibrated": args.calibrated,
            # log1p because train_uhd.train_model always fits on log1p(target); the
            # runtime inverts it to recover the declared units.
            "transform": "log1p",
        },
        # The body key equals the adapter value (RFC 0019 §4). `artifact` is relative to
        # this file, which is where the loader resolves it from, so the pair relocates
        # together.
        "tree_data": {"artifact": fb_path.name},
    }
    # Named `<stem>.uhd.json`, not a bare `uhd.json`: DescriptorLoader discovers a
    # heuristic by that suffix, so a bare name would be invisible to it.
    descriptor_path = output_dir / "heuristic.uhd.json"
    with open(descriptor_path, "w") as f:
        json.dump(descriptor, f, indent=2)
        f.write("\n")
    logger.info("Generated UHD descriptor: %s", descriptor_path)

    manifest = {
        "features": args.features,
        "target": args.target,
        "group_by": args.group_by or [],
        "features_hash": features_hash,
        "num_trees": model.num_trees(),
        "num_samples": len(df),
        "input_file": str(input_path),
        "training_arches": args.training_arches or [],
        "model_version": args.model_version,
    }
    manifest_path = output_dir / "train_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote training manifest: %s", manifest_path)

    print(f"\nUHD Generation Complete")
    print(f"  UHD descriptor: {descriptor_path}")
    print(f"  Model artifact: {fb_path} ({model.num_trees()} trees)")
    print(f"  Features hash:  {features_hash}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
