#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""UHD Generation Tool CLI.

Four subcommands, one per stage of RFC 0019.13's pipeline that exists today:

    export-benchmarks   ingestor benchmark log -> §8.3 training CSV
    train               training CSV -> UHD descriptor + model artifact
    evaluate            corpus + trained UHD -> §11.2 regret report
    promote             install that pair into a descriptor tree and point a UED at it

Collect, train and install:

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

    python -m uhd_gen promote \\
        --model-dir ./uhd_output \\
        --descriptor-tree ./descriptors \\
        --engine hipkernel:pointwise

    python -m uhd_gen evaluate \\
        --input bench.csv \\
        --model-dir ./uhd_output

Features must be namespace-qualified (`q.`, `kernel.`, `device.`); an unqualified
name produces a descriptor that loads but never scores.

Without the `promote` step the engine keeps its previous `heuristic` id and the new
model is never consulted -- silently, since ranking by priority is a valid state.
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
from .evaluate import add_evaluate_arguments, run_evaluate
from .features import build_features_signature, compute_features_hash
from .lgbm_to_flatbuffer import convert
from .promote import add_promote_arguments, run_promote
from .train_uhd import evaluate_regret, find_constant_feature_columns, train_model

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

#: Fraction of the requested feature set that, being constant, stops looking like
#: pinned knobs and starts looking like a thin corpus.
#:
#: 2/3 is picked against the case this tool exists for: a rocKE attention corpus has 8
#: of 14 kernel fields pinned by the kernel matcher before ranking begins -- 57%, the
#: ORDINARY reading -- so a threshold at or below that would fire on every normal run
#: and be learned as noise, which is worse than not warning at all. 2/3 clears it with
#: margin and still catches the shapes that really are suspicious (2 of 3, 3 of 4).
#:
#: The proportion is a smell, not a diagnosis: nothing in a CSV distinguishes
#: pinned-by-construction from under-sampled, so the message says which two readings
#: are possible and points at the corpus rather than asserting one.
CONSTANT_FEATURE_WARN_FRACTION = 2 / 3


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

    evaluate = subparsers.add_parser(
        "evaluate",
        help="score a trained UHD against the best measured kernel (RFC 0019.13 §11.2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_evaluate_arguments(evaluate)

    promote = subparsers.add_parser(
        "promote",
        help="install a trained UHD into a descriptor tree and point a UED at it",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_promote_arguments(promote)

    # export-benchmarks parses its own argv tail, so it is split off before the
    # main parser sees flags it does not declare.
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "export-benchmarks":
        return benchmark_log_main(argv[1:])

    args = parser.parse_args(argv)
    if args.command == "promote":
        return run_promote(args)
    if args.command == "evaluate":
        return run_evaluate(args)
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
        "--keep-constant-features",
        action="store_true",
        dest="keep_constant_features",
        help="Keep feature columns that never vary in the input. By default such a "
        "column is dropped from features_signature: it cannot separate candidates, and "
        "it costs a feature extraction per candidate score at runtime. Pass this when "
        "the column does vary in the world and this corpus is merely thin, so the "
        "signature matches the richer corpus you intend to retrain on.",
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
    parser.add_argument(
        "--uhd-id",
        dest="uhd_id",
        default=None,
        help=(
            "UUID for the emitted descriptor instead of a fresh one. Pass the id the "
            "engine's UED already names and a retrain needs no descriptor edit at all: "
            "the pair is simply overwritten in place."
        ),
    )


def _resolve_uhd_id(requested: str | None) -> str:
    """The descriptor's identity: the caller's id, or a fresh one.

    A typo'd id is not caught anywhere downstream -- it becomes the descriptor's
    identity, the UED points at the id the author meant, nothing resolves, and the
    engine loads with no heuristic. That is precisely the silence --uhd-id exists to
    end, so it is rejected here instead.
    """
    if requested is None:
        return str(uuid.uuid4())
    try:
        parsed = uuid.UUID(requested)
    except (ValueError, AttributeError, TypeError) as error:
        raise ValueError(
            f"--uhd-id {requested!r} is not a UUID ({error}); the UED's `heuristic` "
            "field is resolved by id, so a malformed one would leave the engine with "
            "no heuristic and no error"
        ) from error
    canonical = str(parsed)
    if canonical != requested:
        # Braced/urn/undashed spellings parse, but the descriptor is written canonical.
        # Say so, or the id in the file quietly differs from the one that was typed.
        logger.warning("--uhd-id %r normalized to canonical form %s", requested, canonical)
    return canonical


def _run_train(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    # Before the output directory exists and long before training spends minutes: a bad
    # id is an argument error, and finding it after the model is written wastes the run.
    try:
        uhd_id = _resolve_uhd_id(args.uhd_id)
    except ValueError as error:
        logger.error("%s", error)
        return 1

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

    # RFC 0019 §7.2: features_hash is computed over the signature actually trained, so
    # the drop decision has to happen here, before training and before the signature is
    # built -- not as a post-hoc edit of either.
    constants = find_constant_feature_columns(df, args.features)
    constant_listing = ", ".join(f"{name}={value!r}" for name, value in constants)
    features = list(args.features)
    dropped: list[str] = []

    if constants and len(constants) == len(args.features):
        # Fails with the override too. The flag chooses a signature; it cannot make a
        # column vary, and the degenerate model is the same either way.
        logger.error(
            "Every requested feature column is constant in %s, so there is nothing to "
            "train on: %s. A model over columns that never vary scores every candidate "
            "identically; shipping one is worse than shipping none, because the engine "
            "would rank by a model that cannot discriminate instead of falling back to "
            "its declared order. --keep-constant-features does not help -- it changes "
            "the signature, not the fact that no column varies. Widen the corpus, or "
            "pass --features that vary in it.",
            input_path,
            constant_listing,
        )
        return 1

    if constants:
        if args.keep_constant_features:
            logger.warning(
                "KEPT %d constant feature column(s) at --keep-constant-features: %s. "
                "They stay in features_signature so the hash matches the richer corpus "
                "you intend to retrain on, but they inform nothing in this model and "
                "cost a feature extraction per candidate score at runtime.",
                len(constants),
                constant_listing,
            )
        else:
            for name, value in constants:
                logger.warning(
                    "DROPPED constant feature column %s: every row is %r. It cannot "
                    "separate one candidate from another, so it is NOT in the trained "
                    "features_signature and features_hash is over the smaller set. If "
                    "this column does vary in the world and this corpus is merely thin, "
                    "retrain with --keep-constant-features -- better, widen the corpus.",
                    name,
                    value,
                )
            dropped = [name for name, _ in constants]
            features = [name for name in args.features if name not in set(dropped)]

        if len(constants) / len(args.features) >= CONSTANT_FEATURE_WARN_FRACTION:
            logger.warning(
                "%d of %d requested feature columns are constant (%.0f%%, at or above "
                "the %.0f%% thin-corpus threshold). Kernels that bake their geometry in "
                "pin knobs by construction, and a pinned knob really is uninformative -- "
                "but nothing in a CSV tells that apart from a corpus that only ever "
                "sampled one value, and for a thin corpus dropping is the WRONG fix. "
                "Check that %s spans the problems and kernels you expect to rank before "
                "trusting this model.",
                len(constants),
                len(args.features),
                100.0 * len(constants) / len(args.features),
                100.0 * CONSTANT_FEATURE_WARN_FRACTION,
                input_path,
            )

    if args.objective == "max" and _looks_like_cost_metric(args.target):
        logger.warning(
            "Target '%s' looks like a cost metric but --objective is 'max', so the "
            "runtime will prefer the WORST kernel. Pass --objective min if that is "
            "not what you want.",
            args.target,
        )

    logger.info("Training on features: %s", features)
    logger.info("Target column: %s (objective: %s)", args.target, args.objective)
    if args.group_by:
        logger.info("GroupKFold columns: %s", args.group_by)

    # An unencodable categorical value is an input error like a missing column, and is
    # reported like one. Letting build_feature_matrix's ValueError escape would print a
    # traceback through LightGBM's call stack, burying the column/row/value it names --
    # the only three facts an author needs to fix the CSV.
    try:
        model = train_model(
            df,
            features,
            args.target,
            args.group_by,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping,
        )
    except ValueError as error:
        logger.error("%s", error)
        return 1

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

    features_signature = build_features_signature(features)
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

    # `uhd_id` was resolved at entry, from --uhd-id or a fresh uuid4.
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
        # What was asked for and what was trained, separately: a manifest that recorded
        # only one of them would hide that the emitted signature is not the caller's.
        "requested_features": list(args.features),
        "features": features,
        "constant_features": [{"column": name, "value": value} for name, value in constants],
        "dropped_constant_features": dropped,
        "keep_constant_features": bool(args.keep_constant_features),
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
    if dropped:
        print(f"  dropped (constant): {', '.join(dropped)}")
    print(
        "\nThe engine's UED must name this heuristic by id:\n"
        f'  "heuristic": "{uhd_id}"\n'
        "\nInstall the pair and write that id for you:\n"
        f"  python -m uhd_gen promote --model-dir {output_dir} --descriptor-tree <TREE>"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
