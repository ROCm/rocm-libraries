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
from .evaluate import add_evaluate_arguments, run_evaluate
from .corpus_io import read_corpus_frame
from .coverage import device_field_coverage, enforce_device_coverage
from .knobs import add_knob_arguments, run_knobs
from .merge import add_merge_arguments, run_merge
from .features import (
    build_features_signature,
    compute_features_hash,
    derive_categorical_encoding,
    evaluate_feature_rows,
    parse_signature_entry,
    signature_references,
)
from .lgbm_to_flatbuffer import convert
from .promote import add_promote_arguments, run_promote
from .train_uhd import build_feature_matrix, evaluate_regret, train_model

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

    immediate_import = subparsers.add_parser("import-immediate", help="import hipdnn_bench --collect-immediate JSON without candidate enumeration")
    immediate_import.add_argument("--input", nargs="+", required=True, help="Immediate JSON responses or normalized CSV corpora")
    immediate_import.add_argument("--output", required=True, help="Output .json or .csv corpus")

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
    generate = subparsers.add_parser("generate", help="collect, train, evaluate and promote from a graph corpus")
    add_generate_arguments(generate)

    # export-benchmarks parses its own argv tail, so it is split off before the
    # main parser sees flags it does not declare.
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "export-benchmarks":
        return benchmark_log_main(argv[1:])

    args = parser.parse_args(argv)
    if args.command == "import-immediate":
        from .immediate import normalize_corpus, read_corpus
        try:
            frame = normalize_corpus(pd.concat([read_corpus(Path(path)) for path in args.input], ignore_index=True))
            destination = Path(args.output)
            if destination.suffix not in (".json", ".csv"):
                raise ValueError("--output must have .json or .csv suffix")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.suffix == ".json":
                records = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
                destination.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            else:
                frame.to_csv(destination, index=False)
            return 0
        except (OSError, TypeError, ValueError, KeyError) as error:
            logger.error("%s", error)
            return 1
    if args.command == "generate":
        return run_generate(args)
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
    parser.add_argument("--arch", help="UED role-map architecture, or default for a multi-arch model")
    parser.add_argument(
        "--input",
        required=True,
        help="Training corpus with feature columns and target: the .parquet dataset "
        "uhd_gen/dataset publishes, or a collected .csv/.json corpus",
    )
    feature_source = parser.add_mutually_exclusive_group(required=True)
    feature_source.add_argument("--features", nargs="+", help="Full published feature column names")
    feature_source.add_argument("--feature-signature", help="JSON file containing a canonical inline features_signature array")
    provenance = parser.add_mutually_exclusive_group()
    provenance.add_argument("--descriptor-tree", help="Descriptors whose revisions are captured before fitting")
    provenance.add_argument("--provenance", help="Explicit recorded trained_against JSON snapshot")
    parser.add_argument("--engine", help="UED name/UUID, or immediate engine canonical name/public ID")
    parser.add_argument("--feature-evaluator", help="Path to the shared hipdnn_uhd_features executable")
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
        help="Declare the score cross-engine comparable: RFC 0019 §4.1's "
        "score.calibrated header, which RFC 0019 §11.3 reads when it compares "
        "predicted throughput across engines. Only pass this if the target really is "
        "calibrated across engines; it is not verified here, and an unwarranted claim "
        "silently corrupts cross-engine comparison. RFC 0019.13 §11.2 additionally "
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
        "--report-regret",
        nargs="+",
        default=None,
        dest="report_regret",
        metavar="COL",
        help=(
            "Columns identifying one problem (e.g. benchmark device). Reports "
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
            f"--uhd-id {requested!r} is not a UUID ({error})"
        ) from error
    canonical = str(parsed)
    if canonical != requested:
        # Braced/urn/undashed spellings parse, but the descriptor is written canonical.
        # Say so, or the id in the file quietly differs from the one that was typed.
        logger.warning("--uhd-id %r normalized to canonical form %s", requested, canonical)
    return canonical


def _run_train(args: argparse.Namespace) -> int:
    from .provenance import snapshot_provenance, validate_provenance
    from .immediate import LABEL_STATISTIC, ROLE, read_corpus, training_binding, validate_signature

    immediate = args.role == ROLE
    binding = None

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    try:
        uhd_id = _resolve_uhd_id(args.uhd_id)
        if Path(args.descriptor_name).name != args.descriptor_name or args.descriptor_name in ("", ".", ".."):
            raise ValueError("--descriptor-name must be a file stem, not a path")
        # This is captured before data preparation or fitting, never stamped later.
        if immediate:
            if args.target != "tflops" or args.objective != "max" or args.score_units not in (None, "tflops"):
                raise ValueError("predict_engine_tflops requires --target tflops --objective max --score-units tflops")
            if args.report_regret:
                raise ValueError("L1 evaluation compares immediate engines, not within-engine candidate regret")
            df, binding = training_binding(read_corpus(input_path), args.engine)
            trained_against = binding["trained_against"]
            if args.provenance:
                recorded = validate_provenance(json.loads(Path(args.provenance).read_text(encoding="utf-8")))
                if recorded != trained_against:
                    raise ValueError("recorded provenance differs from immediate engine binding")
            elif args.descriptor_tree:
                recorded = snapshot_provenance(Path(args.descriptor_tree), args.engine, args.arch)
                if recorded != trained_against:
                    raise ValueError("descriptor provenance differs from immediate engine binding")
            if args.group_by is not None and args.group_by != ["benchmark", "device"]:
                raise ValueError("L1 training groups must be benchmark device, never engine/candidate rows")
            args.group_by = ["benchmark", "device"]
            args.calibrated = True
            # RFC 0019.13 §11.2 (:2003) and §10.6.2 (:1914-1916): L1 always declares a
            # calibrated score, so its label is `avgTimeMs` and the manifest says so.
            if args.timing_statistic not in (None, LABEL_STATISTIC):
                raise ValueError(
                    f"predict_engine_tflops labels are derived from {LABEL_STATISTIC}; "
                    f"--timing-statistic {args.timing_statistic} contradicts the corpus")
            args.timing_statistic = LABEL_STATISTIC
            observed_arches = sorted(df["arch"].unique())
            if args.training_arches and sorted(set(args.training_arches)) != observed_arches:
                raise ValueError("--training-arches must match the measured immediate corpus")
            args.training_arches = observed_arches
            if args.arch is None:
                if len(observed_arches) != 1:
                    raise ValueError("multi-architecture L1 training requires --arch default")
                args.arch = observed_arches[0]
            if args.arch != "default" and (len(observed_arches) != 1 or args.arch != observed_arches[0]):
                raise ValueError("L1 role-map arch must cover the complete training corpus")
        else:
            if args.provenance:
                trained_against = validate_provenance(json.loads(Path(args.provenance).read_text(encoding="utf-8")))
            elif args.descriptor_tree:
                arch = args.training_arches[0] if args.training_arches and len(args.training_arches) == 1 else None
                trained_against = snapshot_provenance(Path(args.descriptor_tree), args.engine, arch)
            else:
                raise ValueError("training requires --descriptor-tree or --provenance")
            # The suffix decides, in `corpus_io.read_corpus_frame` for every command
            # alike. `.parquet` is what uhd_gen/dataset publishes (RFC 0019.13
            # §8.3) and is the route a model anyone ships should come by: the dataset
            # carries its own types, so a column empty in one shard and populated in
            # another cannot concatenate to `object` and quietly change what the
            # trainer sees. A collected CSV is read directly, and nothing §8.3
            # specifies is checked on it -- that is what the importer exists for -- so
            # it is the escape hatch for a quick local run. The §11.2 label rule below
            # is applied to all three alike: the published dataset earns no exemption
            # from it.
            df = read_corpus_frame(input_path)
            # What "this row has no measurement" looks like, in both spellings, because
            # a corpus arrives in both. A collected CSV says `is_valid=False`, which is
            # what the runtime record carries at the moment of failure. §8.3's published
            # dataset has no validity flag at all: the measurement is null and `error`
            # carries the reason, so that no two columns can disagree about one row.
            # Both mean the candidate never ran, and a candidate that never ran cannot
            # be fitted -- its target is NaN, and without this the NaN reaches
            # `train_uhd.train_model`, whose "target must contain finite nonnegative
            # values" then reports a missing filter as a corrupt corpus. `evaluate`
            # excludes exactly these rows, by both spellings, for the same reason
            # (§5.6.3 and `Exclusions`), and training and evaluation must not disagree
            # about which rows exist.
            invalid = errored = unmeasured = 0
            if "is_valid" in df.columns:
                keep = df["is_valid"].astype(str).str.strip().str.lower() == "true"
                invalid = int((~keep).sum())
                df = df[keep]
            if "error" in df.columns:
                failed = df["error"].fillna("").astype(str).str.strip().str.len() > 0
                errored = int(failed.sum())
                df = df[~failed]
            if args.target in df.columns:
                # Coerced rather than trusted: a CSV column holding one empty cell reads
                # back as `object`, so the target can be a string here even when every
                # populated row is a number.
                finite = np.isfinite(pd.to_numeric(df[args.target], errors="coerce"))
                unmeasured = int((~finite).sum())
                df = df[finite]
            if invalid or errored or unmeasured:
                # Counted apart because they are three different producer facts, and the
                # one that fires says which end to look at: the collector's flag, the
                # published dataset's error, or a target column that is neither.
                logger.info(
                    "Dropped %d row(s) with is_valid=False, %d row(s) carrying a "
                    "collection error, and %d row(s) whose %s is not a finite number",
                    invalid, errored, unmeasured, args.target)
            # RFC 0019.13 §11.2 (:2003): "A UHD declaring `calibrated: true` MUST train
            # its score on `avgTimeMs`". A calibrated model is the one whose absolute
            # value gets compared across engines, and minimum- or robust-mean-derived
            # throughput is optimistically biased, so the claim is checked rather than
            # trusted. Uncalibrated ranking may use any statistic; it just has to say
            # which, because §11.2 refuses to compare models trained on different ones.
            if args.calibrated and args.timing_statistic != LABEL_STATISTIC:
                raise ValueError(
                    "--calibrated requires --timing-statistic avgTimeMs (RFC 0019.13 "
                    f"§11.2); got {args.timing_statistic!r}")
        if df.empty:
            raise ValueError("No valid rows to train on")
        if not immediate:
            # After the row filtering above, so the census sees exactly the rows that would
            # train -- a candidate that failed to run is not a candidate the ranker gets to
            # choose between, and counting it would report ranking density that measurement
            # already destroyed. `generate` checks the same thing earlier and at greater
            # value, before a GPU sweep rather than after; this catches the corpus handed
            # to `train` directly, which is the route a re-train off collected data takes.
            density = require_rankable(df, engine=args.engine)
            thin = density.near_deterministic_warning()
            if thin:
                logger.warning("%s", thin)
        signature = (
            json.loads(Path(args.feature_signature).read_text(encoding="utf-8"))
            if args.feature_signature else build_features_signature(args.features)
        )
        if not isinstance(signature, list) or not signature:
            raise ValueError("features_signature must be a nonempty JSON array")
        signature = [parse_signature_entry(entry) for entry in signature]
        requested_signature = list(signature)
        references = signature_references(signature)
        if immediate:
            for published in df["features"].unique():
                validate_signature(signature, set(json.loads(published)))
        missing = {reference[1:] for reference in references} - set(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        if args.target not in df.columns:
            raise ValueError(f"Missing target column: {args.target}")
        coverage = device_field_coverage(df)
        enforce_device_coverage(signature, coverage)
        categorical_encoding = derive_categorical_encoding(df, [ref[1:] for ref in references])
        # The branch decides where the VALUES come from, never where the digest comes from:
        # RFC 0019 §6.3 gives features_hash one definition and both kinds of signature take
        # it from the shared evaluator. Values still split, because a raw reference is a
        # column gather that pandas does in-process, and pushing a whole training corpus
        # through the evaluator's JSON pipe to re-derive it would buy nothing.
        if any(isinstance(entry, dict) for entry in signature):
            features_hash, values = evaluate_feature_rows(df, signature, categorical_encoding, args.feature_evaluator)
            matrix = np.asarray(values, dtype=np.float64)
        else:
            features_hash = compute_features_hash(signature, categorical_encoding, args.feature_evaluator)
            matrix = build_feature_matrix(df, [entry[1:] for entry in signature], categorical_encoding)
        names = [entry[1:] if isinstance(entry, str) else f"expression_{index}"
                 for index, entry in enumerate(signature)]
        constant_indices = [index for index in range(matrix.shape[1])
                            if np.all(matrix[:, index] == matrix[0, index])]
        constants = []
        for index in constant_indices:
            value = df[signature[index][1:]].iloc[0] if isinstance(signature[index], str) else float(matrix[0, index])
            constants.append((names[index], value.item() if hasattr(value, "item") else value))
        if len(constants) == len(signature):
            raise ValueError("Every requested feature column is constant: " +
                             ", ".join(f"{name}={value!r}" for name, value in constants))
        dropped = [{"column": name, "value": value} for name, value in constants]
        if constants:
            # A column with one value is a column no tree can split on, so it buys the
            # model nothing -- and it is not free. RFC 0019 §6.3 hashes the whole
            # signature into features_hash, so the dead column enlarges the contract the
            # runtime must reproduce and bakes itself into the descriptor's identity: a
            # later, more correct retrain that omits it reads as a contract break rather
            # than as a better model. The test is variance in THIS corpus, never the
            # column's name -- GenericPlanBuilder::candidateFeatures merges a sweep
            # across several boards of one arch, and gfx942 spans MI300X and MI325X,
            # whose total_global_mem, memory_clock_rate and peak_memory_bandwidth
            # genuinely differ. Every drop is named with its value, because constancy is
            # measured against the corpus that was collected: a field the sweep failed to
            # cover is indistinguishable here from one the kernels pin, and only the
            # author can tell those apart.
            logger.warning("Dropping %d feature column(s) that never vary in this corpus: %s. "
                           "RFC 0019.13 §10.4: this prunes model inputs only, and leaves the "
                           "engine's authored public knobs untouched.",
                           len(constants), ", ".join(f"{name}={value!r}" for name, value in constants))
            if len(constants) / len(signature) >= CONSTANT_FEATURE_WARN_FRACTION:
                logger.warning("High constant-feature proportion: check device and problem coverage")
            keep = [index for index in range(len(signature)) if index not in constant_indices]
            signature = [signature[index] for index in keep]
            names = [names[index] for index in keep]
            matrix = matrix[:, keep]
            remaining_refs = set(signature_references(signature))
            categorical_encoding = {key: value for key, value in categorical_encoding.items() if key in remaining_refs}
            # Pruning changed the signature, so the descriptor's identity changed with
            # it (§6.3). Only the digest is restated -- the kept columns of `matrix`
            # are already the values for the surviving entries.
            features_hash = compute_features_hash(signature, categorical_encoding, args.feature_evaluator)
        if args.objective == "max" and _looks_like_cost_metric(args.target):
            logger.warning("Target '%s' looks like a cost; use --objective min to prefer faster candidates", args.target)
        groups = args.group_by
        if groups is None and "benchmark" in df.columns:
            groups = ["benchmark"] + (["device"] if "device" in df.columns else [])
        # Layer 1 ranks groups, so it is fitted on one row per (problem, group) carrying
        # that group's best achievable target -- not on every candidate. Fitted on the raw
        # rows it would instead rank individual candidates, and taking the group of the best
        # one answers a different question: the error over a group's whole candidate set
        # propagates into what should be a choice among a handful of groups.
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
            logger.info("Layer 1 fitted on %d group rows (from %d candidates)",
                        len(layer_one_df), len(df))

        model = train_model(
            layer_one_df, names, args.target, groups, num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping, categorical_encoding=categorical_encoding,
            feature_matrix=layer_one_matrix,
        )

        # Layer 2: one ensemble per group, fitted on that group's rows alone. Trained on the
        # same feature columns, so one signature describes both layers; the slots a layer
        # does not read are simply unused by its trees.
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
            tagged = df.assign(_uhd_row=np.arange(len(df)))
            for value, rows in tagged.groupby(args.group_by_feature, sort=True):
                at = rows["_uhd_row"].to_numpy()
                # train_model cross-validates over problems, not rows, so a group with more
                # rows than problems can still be unfittable. Skipping leaves the group to
                # layer 1, which the adapter already handles -- a group it chose but layer 2
                # does not describe ranks by layer 1 rather than being discarded.
                try:
                    group_models.append((
                        float(value),
                        train_model(
                            df.iloc[at].reset_index(drop=True), names, args.target, groups,
                            num_boost_round=args.num_boost_round,
                            early_stopping_rounds=args.early_stopping,
                            categorical_encoding=categorical_encoding,
                            feature_matrix=None if matrix is None else matrix[at],
                        ),
                    ))
                except (ValueError, RuntimeError) as error:
                    # Warn rather than fail: one unfittable group must not cost the artifact
                    # every other group's layer 2, and the degradation is visible here.
                    logger.warning("group %s not fitted (%s); it will rank by layer 1",
                                   value, str(error).split(chr(10))[0][:90])
            logger.info("Trained %d group model(s) on %s",
                        len(group_models), args.group_by_feature)

        metrics = None
        if args.report_regret:
            metrics = evaluate_regret(
                df, names, args.target, args.report_regret,
                num_boost_round=args.num_boost_round, categorical_encoding=categorical_encoding,
                feature_matrix=matrix, objective=args.objective,
            )
    except (OSError, TypeError, ValueError, KeyError) as error:
        logger.error("%s", error)
        return 1

    # No descriptor or model artifact is published before all input checks and fitting.
    output_dir.mkdir(parents=True, exist_ok=True)
    lgbm_path = output_dir / "model.lgbm"
    model.save_model(str(lgbm_path))

    fb_path = output_dir / "model.bin"
    model_sha256 = convert(lgbm_path, features_hash, fb_path, num_training_samples=len(df),
                           training_arches=args.training_arches,
                           model_version=args.model_version,
                           group_by_feature_index=group_index,
                           group_models=group_models or None)
    if not args.keep_lgbm:
        lgbm_path.unlink()
    descriptor = {
        "version": "1.0", "id": uhd_id, "name": args.name, "adapter": "tree_data",
        "features_signature": signature, "features_hash": features_hash,
        "trained_against": trained_against, "objective": args.objective,
        "score": {"units": args.score_units or args.target, "calibrated": args.calibrated, "transform": "log1p"},
        # RFC 0019 §7.2: the body naming the artifact carries the digest of its bytes,
        # which TreeDataAdapter recomputes before parsing and refuses on mismatch. It
        # answers the question features_hash does not -- that one fingerprints the input
        # contract, so two models with identical signatures and different training hash
        # identically. Emitted bare-hex because that is what `sha256(buffer, size)`
        # returns on the other side of the comparison.
        "tree_data": {"artifact": fb_path.name, "hash": model_sha256},
    }
    if categorical_encoding:
        descriptor["categorical_encoding"] = categorical_encoding
    descriptor_path = output_dir / f"{args.descriptor_name}.uhd.json"
    descriptor_document = json.dumps(descriptor, indent=2) + "\n"
    descriptor_path.write_text(descriptor_document, encoding="utf-8")
    manifest = {
        "uhd_id": uhd_id, "requested_features": args.features or requested_signature,
        "features": names, "features_signature": signature, "features_hash": features_hash,
        "trained_against": trained_against, "device_coverage": coverage,
        "dropped_constant_features": dropped,
        # RFC 0019.13 §10.5: a content hash over the UHD document AND the model artifact.
        # Conversion is deterministic (`lgbm_to_flatbuffer.resolve_training_date`), so
        # these are checkable against the sources rather than merely recorded.
        "uhd_sha256": hashlib.sha256(descriptor_document.encode("utf-8")).hexdigest(),
        "model_sha256": model_sha256,
        "categorical_encoding": categorical_encoding, "target": args.target, "objective": args.objective,
        "score_units": args.score_units or args.target, "score_calibrated": args.calibrated,
        # RFC 0019.13 §10.5/§11.2: which measured timing the target came from. §11.2
        # refuses cross-engine comparison between models trained on different ones, so
        # a consumer has to be able to read it off the artifact rather than infer it.
        "timing_statistic": args.timing_statistic,
        "score_transform": "log1p", "group_by": groups or [],
        # The feature layer 1 groups on, and the count it produced. Recorded because the
        # artifact alone gives an evaluator only a slot index, and a slot index cannot say
        # which column it came from -- without the name, a report cannot attribute a regret
        # to choosing the wrong group rather than the wrong member of the right one.
        "group_by_feature": args.group_by_feature,
        "group_models": len(group_models or []),
        "num_trees": model.num_trees(),
        "feature_importance": {
            name: {"gain": float(gain), "split": int(split)}
            for name, gain, split in zip(names, model.feature_importance(importance_type="gain"),
                                         model.feature_importance(importance_type="split"))
        },
        "num_samples": len(df), "input_file": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "training_arches": args.training_arches or [], "model_version": args.model_version,
        "training_options": {"num_boost_round": args.num_boost_round, "early_stopping_rounds": args.early_stopping},
    }
    if immediate:
        manifest.update(role=ROLE, binding=binding, arch=args.arch,
                        training_problem_keys=sorted(set(zip(df["benchmark"], df["device"]))))
    (output_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if metrics is not None:
        (output_dir / "regret.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"UHD generated: {descriptor_path}\nModel: {fb_path}\nFeatures hash: {features_hash}")
    print(f"Install: python -m uhd_gen promote --model-dir {output_dir} --descriptor-tree <TREE> --role {args.role} --arch {args.arch or '<ARCH>'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
