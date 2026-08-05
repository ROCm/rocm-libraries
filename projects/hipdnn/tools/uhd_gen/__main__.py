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
import hashlib
import json
import logging
import sys
import uuid
from pathlib import Path

import pandas as pd

from .lgbm_to_flatbuffer import convert
from .train_uhd import train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_features_hash(feature_cols: list[str]) -> str:
    """Compute SHA-256 hash of canonical feature column list.

    The hash is used to validate that inference uses the same features
    as training (feature contract enforcement).
    """
    canonical = json.dumps(sorted(feature_cols), separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return f"sha256:{digest}"


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

    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from %s", input_path)
    if input_path.suffix == ".json":
        df = pd.read_json(input_path)
    else:
        df = pd.read_csv(input_path)
    logger.info("Loaded %d rows", len(df))

    missing = set(args.features) - set(df.columns)
    if missing:
        logger.error("Missing feature columns: %s", missing)
        return 1
    if args.target not in df.columns:
        logger.error("Missing target column: %s", args.target)
        return 1

    logger.info("Training on features: %s", args.features)
    logger.info("Target column: %s", args.target)
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

    features_hash = compute_features_hash(args.features)
    fb_path = output_dir / "model.bin"
    convert(lgbm_path, features_hash, fb_path, num_training_samples=len(df))
    logger.info("Converted to FlatBuffer: %s", fb_path)

    if not args.keep_lgbm:
        lgbm_path.unlink()
        logger.info("Removed intermediate %s", lgbm_path)

    uhd = {
        "schema": "hipdnn.uhd/v1",
        "id": str(uuid.uuid4()),
        "name": args.name,
        "adapter": "tree_data",
        "features_signature": [f"${col}" for col in args.features],
        "features_hash": features_hash,
        "objective": "max",
        "score": {"units": "tflops", "calibrated": True, "transform": "log1p"},
        "model": {"artifact": "model.bin"},
    }
    uhd_path = output_dir / "uhd.json"
    with open(uhd_path, "w") as f:
        json.dump(uhd, f, indent=2)
    logger.info("Generated UHD descriptor: %s", uhd_path)

    manifest = {
        "features": args.features,
        "target": args.target,
        "group_by": args.group_by or [],
        "features_hash": features_hash,
        "num_trees": model.num_trees(),
        "num_samples": len(df),
        "input_file": str(input_path),
    }
    manifest_path = output_dir / "train_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote training manifest: %s", manifest_path)

    print(f"\nUHD Generation Complete")
    print(f"  UHD descriptor: {uhd_path}")
    print(f"  Model artifact: {fb_path} ({model.num_trees()} trees)")
    print(f"  Features hash:  {features_hash}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
