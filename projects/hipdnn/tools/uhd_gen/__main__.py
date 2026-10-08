#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""UHD Generation Tool CLI.

Subcommands for collection, training, evaluation and descriptor installation:

    export-benchmarks   ingestor benchmark log -> §8.3 training CSV
    train               training CSV -> UHD descriptor + model artifact
    evaluate            corpus + trained UHD -> §11.2 regret report
    promote             install that pair into a descriptor tree and point a UED at it
    generate            graph corpus -> public collection -> train/evaluate/promote

Collect, train and install:

    HIPDNN_LOG_LEVEL=info HIPDNN_LOG_FILE=sweep.log <run the graphs you care about>

    python -m uhd_gen export-benchmarks sweep.log -o bench.csv

    python -m uhd_gen train \\
        --input bench.csv \\
        --descriptor-tree ./descriptors --engine hipkernel:pointwise \\
        --features pointwise.elements kernel.block_size device.cu_count \\
        --target tflops \\
        --group-by benchmark device \\
        --output-dir ./uhd_output \\
        --descriptor-name pointwise \\
        --name "Pointwise UHD"

    python -m uhd_gen promote \\
        --model-dir ./uhd_output \\
        --descriptor-tree ./descriptors \\
        --arch gfx942 \\
        --engine hipkernel:pointwise

    python -m uhd_gen evaluate \\
        --input bench.csv \\
        --model-dir ./uhd_output

Feature columns are the full published names without a leading `$`; no namespace
is inserted. Use --feature-signature for canonical inline expression objects.
Computed features require the shared hipdnn_uhd_features executable.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import logging
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark_log import main as benchmark_log_main
from .catalog import require_rankable
from .evaluate import (
    BENCHMARK_COLUMN,
    Grouping,
    add_evaluate_arguments,
    problem_keys,
    resolve_grouping,
    run_evaluate,
)
from .corpus_io import read_corpus_frame
from .correctness import known_wrong
from .coverage import device_field_coverage, enforce_device_coverage
from .knobs import add_knob_arguments, run_knobs
from .merge import add_merge_arguments, run_merge
from .features import (
    build_features_signature,
    compute_features_hash,
    derive_categorical_encoding,
    evaluate_feature_rows,
    evaluator_feature_semantics_revision,
    feature_reference,
    parse_signature_entry,
    signature_references,
)
from .lgbm_to_flatbuffer import convert
from .promote import add_promote_arguments, run_promote
from .ranking_metrics import DEFAULT_RANKING_METRIC, RANKING_METRICS, ranking_metric
from . import score_transform
from .train_uhd import build_feature_matrix, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


#: Substrings that mark a target as a cost; used only to warn on an objective mismatch.
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

#: Constant fraction of the requested features that suggests a thin corpus, not pinned
#: knobs. Above rocKE attention's normal 8/14 (57%) so ordinary runs stay quiet.
CONSTANT_FEATURE_WARN_FRACTION = 2 / 3


def _looks_like_cost_metric(target: str) -> bool:
    """Heuristic: does this target name describe something to minimize?"""
    lowered = target.lower()
    return any(marker in lowered for marker in _COST_METRIC_MARKERS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="uhd_gen",
        description="Generate a UHD heuristic from benchmark data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # The exporter owns its arguments; it parses its own argv below.
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

    immediate_import = subparsers.add_parser(
        "import-immediate",
        help="import hipdnn_bench --collect-immediate JSON without candidate enumeration",
    )
    immediate_import.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Immediate JSON responses or normalized CSV corpora",
    )
    immediate_import.add_argument(
        "--output", required=True, help="Output .json or .csv corpus"
    )

    knobs = subparsers.add_parser(
        "knobs",
        help="measure what each knob is worth, and how few AOT variants suffice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_knob_arguments(knobs)

    merge = subparsers.add_parser(
        "merge",
        help="join sweeps from several machines of one arch into one corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_merge_arguments(merge)

    from .generate import add_generate_arguments, run_generate

    generate = subparsers.add_parser(
        "generate", help="collect, train, evaluate and promote from a graph corpus"
    )
    add_generate_arguments(generate)

    from .sizing import add_size_arguments, run_size

    size = subparsers.add_parser(
        "size",
        help="how many more unique shapes an L1 model needs, and from which regimes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_size_arguments(size)

    # Split off before the main parser sees flags it does not declare.
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "export-benchmarks":
        return benchmark_log_main(argv[1:])

    args = parser.parse_args(argv)
    if args.command == "import-immediate":
        from .immediate import normalize_corpus, read_corpus

        try:
            frame = normalize_corpus(
                pd.concat(
                    [read_corpus(Path(path)) for path in args.input], ignore_index=True
                )
            )
            destination = Path(args.output)
            if destination.suffix not in (".json", ".csv"):
                raise ValueError("--output must have .json or .csv suffix")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.suffix == ".json":
                records = (
                    frame.astype(object)
                    .where(pd.notna(frame), None)
                    .to_dict(orient="records")
                )
                destination.write_text(
                    json.dumps(records, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
            else:
                frame.to_csv(destination, index=False, lineterminator="\n")
            return 0
        except (OSError, TypeError, ValueError, KeyError) as error:
            logger.error("%s", error)
            return 1
    if args.command == "generate":
        return run_generate(args)
    if args.command == "size":
        return run_size(args)
    if args.command == "promote":
        return run_promote(args)
    if args.command == "evaluate":
        return run_evaluate(args)
    if args.command == "knobs":
        return run_knobs(args)
    if args.command == "merge":
        return run_merge(args)
    return _run_train(args)


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    from .provenance import ROLES

    parser.add_argument("--role", choices=ROLES, default="sort_kernel_catalog")
    parser.add_argument(
        "--arch", help="UED role-map architecture, or default for a multi-arch model"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Training corpus with feature columns and target: the .parquet dataset "
        "uhd_gen/dataset publishes, or a collected .csv/.json corpus",
    )
    feature_source = parser.add_mutually_exclusive_group(required=True)
    feature_source.add_argument(
        "--features", nargs="+", help="Full published feature column names"
    )
    feature_source.add_argument(
        "--feature-signature",
        help="JSON file containing a canonical inline features_signature array",
    )
    provenance = parser.add_mutually_exclusive_group()
    provenance.add_argument(
        "--descriptor-tree",
        help="Descriptors whose revisions are captured before fitting",
    )
    provenance.add_argument(
        "--provenance", help="Explicit recorded trained_against JSON snapshot"
    )
    parser.add_argument(
        "--engine", help="UED name/UUID, or immediate engine canonical name/public ID"
    )
    parser.add_argument(
        "--feature-evaluator", help="Path to the shared hipdnn_uhd_features executable"
    )
    parser.add_argument(
        "--metric",
        choices=tuple(RANKING_METRICS),
        default=None,
        help="Ranking metric the score estimates (RFC 0019 §4.4). It fixes the label "
        "column (tflops: the tflops column; time: avgTimeMs), the objective and the units, "
        "so a label cannot be paired with the wrong direction. Default: tflops when "
        "--target is omitted or tflops; otherwise the model declares no metric and ranks "
        "its own catalog only (a metric-less default ranker).",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target column name (default: the metric's label column, else tflops). "
        "With a metric it must be that metric's label.",
    )
    parser.add_argument(
        "--objective",
        choices=("max", "min"),
        default=None,
        help="Whether the runtime should maximize or minimize the score. A metric fixes "
        "it and a conflicting value is refused; a metric-less model defaults to max. Pass "
        "'min' for a cost target such as latency_ms.",
    )
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="Declare the score cross-engine comparable: RFC 0019 §4.1's "
        "score.calibrated header, which RFC 0019 §11.3 reads when it compares "
        "predicted values across engines. Requires a metric. Only pass this if the target "
        "really is calibrated across engines; it is not verified here, and an unwarranted "
        "claim silently corrupts cross-engine comparison. RFC 0019.13 §11.2 additionally "
        "requires --timing-statistic avgTimeMs alongside it.",
    )
    parser.add_argument(
        "--timing-statistic",
        default=None,
        dest="timing_statistic",
        help="Which measured timing the target was derived from (avgTimeMs, "
        "minTimeMs, robustMeanMs). Recorded in the manifest per RFC 0019.13 §10.5, "
        "because §11.2 refuses cross-engine comparison between models trained on "
        "different statistics. Required with --calibrated, which §11.2 pins to "
        "avgTimeMs.",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=None,
        dest="group_by",
        help="Columns for GroupKFold (prevents problem leakage)",
    )
    parser.add_argument(
        "--group-by-feature",
        default=None,
        dest="group_by_feature",
        metavar="FEATURE",
        help=(
            "Train two layers into one artifact: this feature names a candidate's group "
            "(e.g. kernel.solver_id). Layer 1 ranks groups, layer 2 ranks candidates within "
            "the chosen one. Ranking a whole catalog then takes the group decision first, "
            "which a single ensemble over all candidates cannot express."
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

    A malformed id is rejected here; downstream it would silently resolve to nothing.
    """
    if requested is None:
        return str(uuid.uuid4())
    try:
        parsed = uuid.UUID(requested)
    except (ValueError, AttributeError, TypeError) as error:
        raise ValueError(f"--uhd-id {requested!r} is not a UUID ({error})") from error
    canonical = str(parsed)
    if canonical != requested:
        # Braced/urn/undashed spellings parse, but the descriptor is written canonical.
        logger.warning(
            "--uhd-id %r normalized to canonical form %s", requested, canonical
        )
    return canonical


def _resolve_score(args: argparse.Namespace, immediate: bool) -> None:
    """Settle the metric, label column and objective before any data is read.

    RFC 0019 §4.4/§13.4: a metric fixes label and direction, so --target and --objective
    can only contradict it. A metric-less model is a within-engine ranker.
    """
    from .immediate import LABEL_STATISTIC

    if args.metric is None and args.target in (None, "tflops"):
        args.metric = DEFAULT_RANKING_METRIC
    if args.metric is None:
        if immediate:
            raise ValueError(
                f"{args.role} requires --metric: engine selection compares its "
                "score across engines, so it must name a registered metric"
            )
        if args.calibrated:
            raise ValueError(
                "--calibrated requires --metric: a calibrated score must name the "
                "quantity it is calibrated in (RFC 0019 §4.1)"
            )
        args.objective = args.objective or "max"
        return
    metric = ranking_metric(args.metric)
    if args.target not in (None, metric.label):
        raise ValueError(
            f"--metric {metric.name} trains on the {metric.label!r} column; "
            f"--target {args.target} contradicts it"
        )
    if args.objective not in (None, metric.objective):
        raise ValueError(
            f"--metric {metric.name} fixes objective {metric.objective}; "
            f"--objective {args.objective} contradicts it"
        )
    args.target, args.objective = metric.label, metric.objective
    if metric.label == LABEL_STATISTIC:
        # This metric's label is itself a timing statistic.
        if args.timing_statistic not in (None, LABEL_STATISTIC):
            raise ValueError(
                f"--metric {metric.name} is labelled by {LABEL_STATISTIC}; "
                f"--timing-statistic {args.timing_statistic} contradicts it"
            )
        args.timing_statistic = LABEL_STATISTIC


def _run_train(args: argparse.Namespace) -> int:
    from .provenance import (
        record_feature_semantics,
        snapshot_provenance,
        validate_provenance,
    )
    from .immediate import (
        LABEL_STATISTIC,
        ROLE,
        check_signature_evaluates,
        read_corpus,
        require_binding_metric,
        training_binding,
        validate_signature,
    )

    immediate = args.role == ROLE
    binding = None

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    try:
        uhd_id = _resolve_uhd_id(args.uhd_id)
        if Path(
            args.descriptor_name
        ).name != args.descriptor_name or args.descriptor_name in ("", ".", ".."):
            raise ValueError("--descriptor-name must be a file stem, not a path")
        # Settled before data preparation or fitting, never stamped later.
        _resolve_score(args, immediate)
        if immediate:
            if args.group_by_feature:
                # The runtime scores only an engine-level model's root ensemble, so a
                # grouped L1 artifact would not be scored the way it was fitted.
                raise ValueError(
                    f"--group-by-feature is not supported for {ROLE}: a grouped "
                    "artifact has no engine-level (per-row) scoring contract"
                )
            df, binding = training_binding(read_corpus(input_path), args.engine)
            # RFC 0019 §13.2: a pick checked wrong is never a label; null is unknown.
            wrong = known_wrong(df)
            if wrong.any():
                logger.info(
                    "Dropped %d row(s) checked numerically wrong (numerically_valid=False)",
                    int(wrong.sum()),
                )
                df = df[~wrong]
            # Check every row: rows of another metric describe another selector.
            for index, row_binding in enumerate(df["binding"]):
                require_binding_metric(
                    json.loads(row_binding), args.metric, f"{input_path} row {index}"
                )
            trained_against = binding["trained_against"]
            if args.provenance:
                recorded = validate_provenance(
                    json.loads(Path(args.provenance).read_text(encoding="utf-8"))
                )
                if recorded != trained_against:
                    raise ValueError(
                        "recorded provenance differs from immediate engine binding"
                    )
            elif args.descriptor_tree:
                recorded = snapshot_provenance(
                    Path(args.descriptor_tree), args.engine, args.arch
                )
                if recorded != trained_against:
                    raise ValueError(
                        "descriptor provenance differs from immediate engine binding"
                    )
            if args.group_by is not None and args.group_by != ["benchmark", "device"]:
                raise ValueError(
                    "L1 training groups must be benchmark device, never engine/candidate rows"
                )
            args.group_by = ["benchmark", "device"]
            args.calibrated = True
            # RFC 0019.13 §11.2: L1 is always calibrated, so its label comes from
            # `avgTimeMs`.
            if args.timing_statistic not in (None, LABEL_STATISTIC):
                raise ValueError(
                    f"{ROLE} labels are derived from {LABEL_STATISTIC}; "
                    f"--timing-statistic {args.timing_statistic} contradicts the corpus"
                )
            args.timing_statistic = LABEL_STATISTIC
            observed_arches = sorted(df["arch"].unique())
            if (
                args.training_arches
                and sorted(set(args.training_arches)) != observed_arches
            ):
                raise ValueError(
                    "--training-arches must match the measured immediate corpus"
                )
            args.training_arches = observed_arches
            if args.arch is None:
                if len(observed_arches) != 1:
                    raise ValueError(
                        "multi-architecture L1 training requires --arch default"
                    )
                args.arch = observed_arches[0]
            if args.arch != "default" and (
                len(observed_arches) != 1 or args.arch != observed_arches[0]
            ):
                raise ValueError(
                    "L1 role-map arch must cover the complete training corpus"
                )
        else:
            if args.provenance:
                trained_against = validate_provenance(
                    json.loads(Path(args.provenance).read_text(encoding="utf-8"))
                )
            elif args.descriptor_tree:
                arch = (
                    args.training_arches[0]
                    if args.training_arches and len(args.training_arches) == 1
                    else None
                )
                trained_against = snapshot_provenance(
                    Path(args.descriptor_tree), args.engine, arch
                )
            else:
                raise ValueError("training requires --descriptor-tree or --provenance")
            # The suffix picks the reader. Only the `.parquet` dataset is §8.3-checked; a
            # collected CSV is a local escape hatch. The §11.2 rule below applies to both.
            df = read_corpus_frame(input_path)
            # A row with no measurement is `is_valid=False` (collected CSV) or carries an
            # `error` (published dataset). Drop both, as `evaluate` does, so training and
            # evaluation agree on which rows exist.
            invalid = errored = unmeasured = 0
            # RFC 0019 §13.2: a candidate shown wrong is never a label (a wrong-but-fast
            # kernel would teach the ranker to prefer it). False is dropped; null is kept.
            wrong = known_wrong(df)
            numerically_wrong = int(wrong.sum())
            df = df[~wrong]
            if "is_valid" in df.columns:
                keep = df["is_valid"].astype(str).str.strip().str.lower() == "true"
                invalid = int((~keep).sum())
                df = df[keep]
            if "error" in df.columns:
                failed = df["error"].fillna("").astype(str).str.strip().str.len() > 0
                errored = int(failed.sum())
                df = df[~failed]
            if args.target in df.columns:
                # Coerced: one empty CSV cell reads the column back as `object`.
                finite = np.isfinite(pd.to_numeric(df[args.target], errors="coerce"))
                unmeasured = int((~finite).sum())
                df = df[finite]
            if numerically_wrong or invalid or errored or unmeasured:
                # Counted apart: each points at a different producer.
                logger.info(
                    "Dropped %d row(s) checked numerically wrong, %d row(s) with "
                    "is_valid=False, %d row(s) carrying a collection error, and %d "
                    "row(s) whose %s is not a finite number",
                    numerically_wrong,
                    invalid,
                    errored,
                    unmeasured,
                    args.target,
                )
            # RFC 0019.13 §11.2: a calibrated score MUST train on `avgTimeMs`. Uncalibrated
            # models may use any statistic but must record which.
            if args.calibrated and args.timing_statistic != LABEL_STATISTIC:
                raise ValueError(
                    "--calibrated requires --timing-statistic avgTimeMs (RFC 0019.13 "
                    f"§11.2); got {args.timing_statistic!r}"
                )
        if df.empty:
            raise ValueError("No valid rows to train on")
        if not immediate:
            # Census after row filtering, so it sees exactly the rows that train. Problems
            # are grouped as training groups them; a corpus with no problem identity is
            # trained ungrouped and the census is skipped.
            census_grouping = None
            if args.group_by is not None:
                census_grouping = Grouping(
                    columns=tuple(args.group_by), degraded=False, detail="--group-by"
                )
            if census_grouping is not None or BENCHMARK_COLUMN in df.columns:
                density = require_rankable(
                    df, engine=args.engine, grouping=census_grouping
                )
                thin = density.near_deterministic_warning()
                if thin:
                    logger.warning("%s", thin)
            else:
                logger.info(
                    "catalog census skipped: the corpus has no %r column and no --group-by, "
                    "so rows carry no problem identity to count candidates over",
                    BENCHMARK_COLUMN,
                )
        signature = (
            json.loads(Path(args.feature_signature).read_text(encoding="utf-8"))
            if args.feature_signature
            else build_features_signature(args.features)
        )
        if not isinstance(signature, list) or not signature:
            raise ValueError("features_signature must be a nonempty JSON array")
        signature = [parse_signature_entry(entry) for entry in signature]
        requested_signature = list(signature)
        references = signature_references(signature)
        if immediate:
            validate_signature(signature)
        if not any(isinstance(entry, dict) for entry in signature):
            # A raw reference is a column gather; expression inputs may be legally absent
            # (value_or_default/present), so the shared evaluator decides for those.
            missing = {reference[1:] for reference in references} - set(df.columns)
            if missing:
                raise ValueError(f"Missing feature columns: {sorted(missing)}")
        if args.target not in df.columns:
            raise ValueError(f"Missing target column: {args.target}")
        coverage = device_field_coverage(df)
        enforce_device_coverage(signature, coverage)
        categorical_encoding = derive_categorical_encoding(
            df, [ref[1:] for ref in references]
        )
        if immediate:
            # After the encoding: a published string is only evaluable through it.
            check_signature_evaluates(
                signature, df["features"], categorical_encoding, args.feature_evaluator
            )
        # The digest always comes from the shared evaluator (RFC 0019 §6.3); raw references
        # are gathered in-process rather than piped through it.
        if any(isinstance(entry, dict) for entry in signature):
            features_hash, values = evaluate_feature_rows(
                df, signature, categorical_encoding, args.feature_evaluator
            )
            matrix = np.asarray(values, dtype=np.float64)
        else:
            features_hash = compute_features_hash(
                signature, categorical_encoding, args.feature_evaluator
            )
            matrix = build_feature_matrix(
                df, [entry[1:] for entry in signature], categorical_encoding
            )
        # The loader refuses a model recording another feature-semantics revision
        # (FeatureSemantics.hpp), so record this evaluator's before fitting.
        trained_against = record_feature_semantics(
            trained_against,
            evaluator_feature_semantics_revision(args.feature_evaluator),
        )
        names = [
            entry[1:] if isinstance(entry, str) else f"expression_{index}"
            for index, entry in enumerate(signature)
        ]
        constant_indices = [
            index
            for index in range(matrix.shape[1])
            if np.all(matrix[:, index] == matrix[0, index])
        ]
        constants = []
        for index in constant_indices:
            value = (
                df[signature[index][1:]].iloc[0]
                if isinstance(signature[index], str)
                else float(matrix[0, index])
            )
            constants.append(
                (names[index], value.item() if hasattr(value, "item") else value)
            )
        if len(constants) == len(signature):
            raise ValueError(
                "Every requested feature column is constant: "
                + ", ".join(f"{name}={value!r}" for name, value in constants)
            )
        dropped = [{"column": name, "value": value} for name, value in constants]
        if constants:
            # A constant column cannot be split on, yet RFC 0019 §6.3 hashes it into the
            # descriptor's identity. Constancy is measured in THIS corpus, not by name
            # (one arch spans boards with differing memory fields), and each drop is
            # logged because only the author can tell under-sampled from pinned.
            logger.warning(
                "Dropping %d feature column(s) that never vary in this corpus: %s. "
                "RFC 0019.13 §10.4: this prunes model inputs only, and leaves the "
                "engine's authored public knobs untouched.",
                len(constants),
                ", ".join(f"{name}={value!r}" for name, value in constants),
            )
            if len(constants) / len(signature) >= CONSTANT_FEATURE_WARN_FRACTION:
                logger.warning(
                    "High constant-feature proportion: check device and problem coverage"
                )
            keep = [
                index
                for index in range(len(signature))
                if index not in constant_indices
            ]
            signature = [signature[index] for index in keep]
            names = [names[index] for index in keep]
            matrix = matrix[:, keep]
            remaining_refs = set(signature_references(signature))
            categorical_encoding = {
                key: value
                for key, value in categorical_encoding.items()
                if key in remaining_refs
            }
            # Pruning changed the signature, so restate its digest (§6.3).
            features_hash = compute_features_hash(
                signature, categorical_encoding, args.feature_evaluator
            )
        if (
            args.metric is None
            and args.objective == "max"
            and _looks_like_cost_metric(args.target)
        ):
            logger.warning(
                "Target '%s' looks like a cost; use --objective min to prefer faster candidates",
                args.target,
            )
        groups = args.group_by
        if groups is None and "benchmark" in df.columns:
            groups = ["benchmark"] + (["device"] if "device" in df.columns else [])
        # Layer 1 ranks groups, so fit it on one row per (problem, group) with that
        # group's best target, not on every candidate.
        layer_one_df, layer_one_matrix = df, matrix
        if args.group_by_feature:
            if not groups:
                raise ValueError(
                    "--group-by-feature needs --group-by: layer 1 is fitted on one row per "
                    "(problem, group), so it has to know which columns identify a problem. "
                    "Without them every group would collapse to a single row."
                )
            keys = list(groups) + [args.group_by_feature]
            positions = (
                df.assign(_uhd_row=np.arange(len(df)))
                .sort_values(args.target, ascending=(args.objective == "min"))
                .drop_duplicates(subset=keys, keep="first")["_uhd_row"]
                .to_numpy()
            )
            positions = np.sort(positions)
            layer_one_df = df.iloc[positions].reset_index(drop=True)
            layer_one_matrix = None if matrix is None else matrix[positions]
            logger.info(
                "Layer 1 fitted on %d group rows (from %d candidates)",
                len(layer_one_df),
                len(df),
            )

        model = train_model(
            layer_one_df,
            names,
            args.target,
            groups,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping,
            categorical_encoding=categorical_encoding,
            feature_matrix=layer_one_matrix,
        )

        # Layer 2: one ensemble per group on that group's rows, over the same feature
        # columns so one signature describes both layers.
        group_models: list[tuple[float, object]] = []
        group_index = -1
        if args.group_by_feature:
            if args.group_by_feature not in names:
                raise ValueError(
                    f"--group-by-feature {args.group_by_feature!r} is not among --features; "
                    "the runtime reads the group from a slot in the feature row, so it has "
                    "to be one of them"
                )
            group_index = names.index(args.group_by_feature)
            # The runtime compares the group's feature SLOT value, which for a string
            # column is its categorical code, so export codes rather than raw values.
            codes = categorical_encoding.get(feature_reference(args.group_by_feature))
            tagged = df.assign(_uhd_row=np.arange(len(df)))
            for value, rows in tagged.groupby(args.group_by_feature, sort=True):
                at = rows["_uhd_row"].to_numpy()
                # Cross-validation is over problems, so a group may be unfittable; it then
                # ranks by layer 1, which the adapter handles.
                try:
                    group_models.append(
                        (
                            float(codes[value]) if codes is not None else float(value),
                            train_model(
                                df.iloc[at].reset_index(drop=True),
                                names,
                                args.target,
                                groups,
                                num_boost_round=args.num_boost_round,
                                early_stopping_rounds=args.early_stopping,
                                categorical_encoding=categorical_encoding,
                                feature_matrix=None if matrix is None else matrix[at],
                            ),
                        )
                    )
                except (ValueError, RuntimeError) as error:
                    # One unfittable group must not cost every other group's layer 2.
                    logger.warning(
                        "group %s not fitted (%s); it will rank by layer 1",
                        value,
                        str(error).split(chr(10))[0][:90],
                    )
            logger.info(
                "Trained %d group model(s) on %s",
                len(group_models),
                args.group_by_feature,
            )
    except (OSError, TypeError, ValueError, KeyError) as error:
        logger.error("%s", error)
        return 1

    # No descriptor or model artifact is published before all input checks and fitting.
    output_dir.mkdir(parents=True, exist_ok=True)
    lgbm_path = output_dir / "model.lgbm"
    model.save_model(str(lgbm_path))

    fb_path = output_dir / "model.bin"
    model_sha256 = convert(
        lgbm_path,
        features_hash,
        fb_path,
        num_training_samples=len(df),
        training_arches=args.training_arches,
        model_version=args.model_version,
        group_by_feature_index=group_index,
        group_models=group_models or None,
    )
    if not args.keep_lgbm:
        lgbm_path.unlink()
    descriptor = {
        "version": "1.0",
        "id": uhd_id,
        "name": args.name,
        "adapter": "tree_data",
        "features_signature": signature,
        "features_hash": features_hash,
        "trained_against": trained_against,
        "objective": args.objective,
        # RFC 0019 §4.1: `metric` names the registered quantity the score estimates, and
        # is absent from a within-engine ranker that estimates none of them.
        "score": {
            **({"metric": args.metric} if args.metric else {}),
            "calibrated": args.calibrated,
            "transform": score_transform.TRAINED,
        },
        # RFC 0019 §7.2: TreeDataAdapter verifies this artifact digest before parsing.
        # Bare hex, matching `sha256(buffer, size)` on the runtime side.
        "tree_data": {"artifact": fb_path.name, "hash": model_sha256},
    }
    if categorical_encoding:
        descriptor["categorical_encoding"] = categorical_encoding
    descriptor_path = output_dir / f"{args.descriptor_name}.uhd.json"
    descriptor_document = json.dumps(descriptor, indent=2) + "\n"
    # LF on every platform: `uhd_sha256` below hashes these exact bytes.
    descriptor_path.write_text(descriptor_document, encoding="utf-8", newline="\n")
    manifest = {
        "uhd_id": uhd_id,
        "requested_features": args.features or requested_signature,
        "features": names,
        "features_signature": signature,
        "features_hash": features_hash,
        "trained_against": trained_against,
        "device_coverage": coverage,
        "dropped_constant_features": dropped,
        # RFC 0019.13 §10.5: content hashes of the UHD document and model artifact;
        # conversion is deterministic, so both are reproducible from sources.
        "uhd_sha256": hashlib.sha256(descriptor_document.encode("utf-8")).hexdigest(),
        "model_sha256": model_sha256,
        "categorical_encoding": categorical_encoding,
        "target": args.target,
        "objective": args.objective,
        "score_metric": args.metric,
        "score_calibrated": args.calibrated,
        # RFC 0019.13 §10.5/§11.2: models trained on different statistics are not
        # comparable, so the artifact records which one it used.
        "timing_statistic": args.timing_statistic,
        "score_transform": score_transform.TRAINED,
        "group_by": groups or [],
        # The artifact holds only a slot index; the name lets a report attribute regret
        # to the group choice versus the member choice.
        "group_by_feature": args.group_by_feature,
        "group_models": len(group_models or []),
        "num_trees": model.num_trees(),
        "feature_importance": {
            name: {"gain": float(gain), "split": int(split)}
            for name, gain, split in zip(
                names,
                model.feature_importance(importance_type="gain"),
                model.feature_importance(importance_type="split"),
            )
        },
        "num_samples": len(df),
        "input_file": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "training_arches": args.training_arches or [],
        "model_version": args.model_version,
        "training_options": {
            "num_boost_round": args.num_boost_round,
            "early_stopping_rounds": args.early_stopping,
        },
    }
    if immediate:
        manifest.update(
            role=ROLE,
            binding=binding,
            arch=args.arch,
            training_problem_keys=sorted(set(zip(df["benchmark"], df["device"]))),
        )
    elif BENCHMARK_COLUMN in df.columns:
        # In `evaluate`'s identity, so a later evaluation can prove its slice held out.
        manifest["training_problem_keys"] = sorted(
            set(problem_keys(df, resolve_grouping(df)))
        )
    (output_dir / "train_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"UHD generated: {descriptor_path}\nModel: {fb_path}\nFeatures hash: {features_hash}"
    )
    print(
        f"Install: python -m uhd_gen promote --model-dir {output_dir} --descriptor-tree <TREE> --role {args.role} --arch {args.arch or '<ARCH>'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
