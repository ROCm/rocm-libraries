#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""UHD Generation Tool CLI.

Two subcommands, one per stage of RFC 0019.13's pipeline that exists today:

    export-benchmarks   ingestor benchmark log -> §8.3 training CSV
    train               training CSV -> UHD descriptor + model artifact

Collect and train:

    HIPDNN_LOG_LEVEL=info HIPDNN_LOG_FILE=sweep.log <run the graphs you care about>

    python -m uhd_gen export-benchmarks sweep.log -o bench.csv

    python -m uhd_gen train \\
        --input bench.csv \\
        --features q.seqlen_q kernel.block_size device.cu_count \\
        --target tflops \\
        --group-by q.seqlen_q \\
        --output-dir ./uhd_output \\
        --descriptor-name pointwise \\
        --name "Pointwise UHD"

Features must be namespace-qualified (`q.`, `kernel.`, `device.`); an unqualified
name produces a descriptor that loads but never scores.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

import pandas as pd

from .benchmark_log import main as benchmark_log_main
from .features import build_features_signature, compute_features_hash
from .lgbm_to_flatbuffer import convert
from .train_uhd import train_model

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
    """Dispatch to a subcommand."""
    parser = argparse.ArgumentParser(
        prog="uhd_gen",
        description="Generate a UHD heuristic from benchmark data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Thin delegation: the exporter owns its own arguments, and duplicating them
    # here would be a second place for them to drift.
    subparsers.add_parser(
        "export-benchmarks",
        add_help=False,
        help="convert an ingestor benchmark log into the §8.3 training CSV",
    )

    train = subparsers.add_parser(
        "train",
        help="train a UHD from a benchmark CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_train_arguments(train)

    # export-benchmarks parses its own argv tail, so it is split off before the
    # main parser sees flags it does not declare.
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "export-benchmarks":
        return benchmark_log_main(argv[1:])

    args = parser.parse_args(argv)
    return _run_train(args)


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument(
        "--descriptor-name",
        dest="descriptor_name",
        default="heuristic",
        help=(
            "Stem for the emitted descriptor, producing "
            "<stem>.uhd.json (default: heuristic). DescriptorLoader discovers a "
            "heuristic by that suffix, so a bare 'uhd.json' is invisible to it."
        ),
    )


def _run_train(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from %s", input_path)
    if input_path.suffix == ".json":
        df = pd.read_json(input_path)
    else:
        df = pd.read_csv(input_path)
    logger.info("Loaded %d row(s)", len(df))

    # RFC 0019.13 §10.2 step 1. A §8.3 CSV records the pairs that failed to run so
    # coverage can be audited; they carry no timings, so training on them would fit
    # the model to empty cells. Absent column means the CSV predates the envelope.
    if "is_valid" in df.columns:
        before = len(df)
        df = df[df["is_valid"].astype(str).str.lower() == "true"]
        skipped = before - len(df)
        if skipped:
            logger.info("Dropped %d row(s) with is_valid=False", skipped)
        if df.empty:
            logger.error("Every row in %s is is_valid=False; nothing to train on", input_path)
            return 1

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

    uhd_id = str(uuid.uuid4())
    stem = args.descriptor_name

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
    descriptor_path = output_dir / f"{stem}.uhd.json"
    with open(descriptor_path, "w", encoding="utf-8") as handle:
        json.dump(descriptor, handle, indent=2)
        handle.write("\n")
    logger.info("Generated descriptor: %s", descriptor_path)

    manifest = {
        "uhd_id": uhd_id,
        "features": args.features,
        "features_signature": features_signature,
        "features_hash": features_hash,
        "target": args.target,
        "objective": args.objective,
        "score_units": args.score_units or args.target,
        "score_calibrated": args.calibrated,
        "score_transform": "log1p",
        "group_by": args.group_by or [],
        "num_trees": model.num_trees(),
        "num_samples": len(df),
        "input_file": str(input_path),
        "training_arches": args.training_arches or [],
        "model_version": args.model_version,
    }
    manifest_path = output_dir / "train_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("Wrote training manifest: %s", manifest_path)

    print("\nUHD generation complete")
    print(f"  descriptor:     {descriptor_path}")
    print(f"  model artifact: {fb_path} ({model.num_trees()} trees)")
    print(f"  features hash:  {features_hash}")
    print(
        "\nThe engine's UED must name this heuristic by id:\n"
        f'  "heuristic": "{uhd_id}"'
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
