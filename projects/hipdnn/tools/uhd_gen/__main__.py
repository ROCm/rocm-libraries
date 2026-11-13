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
from .correctness import known_wrong, measured_rows
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
from .lgbm_to_flatbuffer import (
    ENCODINGS,
    LEAF_PRECISIONS,
    SPLIT_LEAF,
    convert,
    resolve_training_date,
)
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


def install_hint(
    model_dir, role: str, arch: str | None, uhd_id: str | None = None
) -> str:
    """The `promote` command that installs a trained model, as one line to copy."""
    hint = (
        f"Install: python3 -m uhd_gen promote --model-dir {model_dir} "
        f"--descriptor-tree <TREE> --role {role} --arch {arch or '<ARCH>'}"
    )
    if uhd_id:
        hint += f" --uhd-id {uhd_id}"
    return hint


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
    code = _run_train(args)
    if code == 0:
        print(install_hint(args.output_dir, args.role, args.arch, args.uhd_id))
    return code


def train(argv: list[str]) -> int:
    """`uhd_gen train <argv>` for a caller that places and installs the model itself:
    `generate` trains into a staging directory and prints the model's final path."""
    parser = argparse.ArgumentParser(prog="uhd_gen train")
    _add_train_arguments(parser)
    return _run_train(parser.parse_args(argv))


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
        help="Columns identifying a problem, shape column first. Cross-validation folds "
        "by the shape, so no shape is in both train and validation on any device",
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
    add_model_encoding_arguments(parser)
    add_adapter_arguments(parser)
    parser.add_argument(
        "--key-features",
        nargs="+",
        default=None,
        dest="key_features",
        metavar="FEATURE",
        help="linear_data only: the features whose values together identify a kernel "
        "configuration (full published names, among --features). Default: every "
        "`$kernel.*` reference in the signature.",
    )
    parser.add_argument(
        "--family-features",
        nargs="+",
        default=None,
        dest="family_features",
        metavar="FEATURE",
        help="linear_data only: a coarser key for configurations never seen in training; "
        "they are scored by their family's fit. Default: none (the global fit).",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=None,
        help="linear_data only: ridge strength of every fit (default: rellin.DEFAULT_RIDGE)",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=None,
        dest="min_rows",
        help="linear_data only: a configuration with fewer rows is shrunk harder toward its "
        "family (default: rellin.DEFAULT_MIN_ROWS)",
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


def add_model_encoding_arguments(parser: argparse.ArgumentParser) -> None:
    """How model.bin stores its trees: `train` and `generate` take the same two flags."""
    parser.add_argument(
        "--model-encoding",
        choices=ENCODINGS,
        default=SPLIT_LEAF,
        help="Tree encoding of model.bin (gbdt_model.fbs). split_leaf (default) stores each "
        "array in the narrowest type that holds it; node_arrays is the original encoding, for "
        "a runtime that predates split_leaf",
    )
    parser.add_argument(
        "--leaf-precision",
        choices=LEAF_PRECISIONS,
        default="float32",
        help="Storage of split_leaf leaf values: float32 (default; each within 2**-24 of "
        "itself) or float64 (exact)",
    )
    parser.add_argument(
        "--early-exit",
        default="auto",
        help="Early-exit schedule of a tree_data catalog ranker (README, 'Early exit'): "
        "'auto' (default) chooses one from out-of-fold predictions for an uncalibrated, "
        "single-layer sort_kernel_catalog model with problem identity, and writes none "
        "otherwise; 'none' writes none; TREES:KEEP[,TREES:KEEP...] writes that schedule "
        "(refused for a calibrated, grouped or predict_engine model). A schedule changes "
        "model.bin, and with it the engine's selector revision",
    )


#: The model families `train` fits, one per UHD adapter that ships a trained artifact.
TRAINED_ADAPTERS = ("tree_data", "linear_data", "mlp_data")

#: `train`'s mlp_data options, by flag and argparse destination. They default to None so an
#: unset one is told from a set one: a flag of another adapter is refused, and `generate`
#: forwards only what its caller gave.
_MLP_OPTIONS = (
    ("--mlp-hidden", "mlp_hidden"),
    ("--mlp-members", "mlp_members"),
    ("--mlp-epochs", "mlp_epochs"),
    ("--mlp-seed", "mlp_seed"),
    ("--mlp-max-rows", "mlp_max_rows"),
)


def add_adapter_arguments(parser: argparse.ArgumentParser) -> None:
    """Which model family `train` fits, and its options: `train` and `generate` take the
    same flags."""
    from .mlp import DEFAULT_EPOCHS, DEFAULT_HIDDEN, DEFAULT_MEMBERS, DEFAULT_SEED

    parser.add_argument(
        "--adapter",
        choices=TRAINED_ADAPTERS,
        default="tree_data",
        help="UHD adapter, and so model, to train. tree_data (default): a LightGBM ensemble. "
        "linear_data: a per-configuration linear ranker (RelLin) of the log slowdown against "
        "the best candidate of each problem, for a small catalog; it needs problem identity "
        "(--group-by or a benchmark column), serves no --metric or --calibrated score and no "
        "predict_engine role, and ignores the tree-only options (--num-boost-round, "
        "--early-stopping, --model-encoding, --leaf-precision, --group-by-feature). "
        "mlp_data: an ensemble of small neural networks fitted to the log of one metric; "
        "an estimate of one engine's metric for --role predict_engine, a catalog ranker "
        "(like an ungrouped tree_data model) for --role sort_kernel_catalog, where "
        "evaluation decides whether it is picked (README, 'The mlp_data adapter'). It "
        "refuses --group-by-feature, ignores the other tree-only options and takes the "
        "--mlp-* ones.",
    )
    parser.add_argument(
        "--mlp-hidden",
        default=None,
        dest="mlp_hidden",
        metavar="WIDTHS",
        help="mlp_data only: comma-separated hidden layer widths of each network, e.g. "
        f"64,64 (default: {','.join(str(w) for w in DEFAULT_HIDDEN)})",
    )
    parser.add_argument(
        "--mlp-members",
        type=int,
        default=None,
        dest="mlp_members",
        help=f"mlp_data only: networks in the ensemble (default: {DEFAULT_MEMBERS})",
    )
    parser.add_argument(
        "--mlp-epochs",
        type=int,
        default=None,
        dest="mlp_epochs",
        help="mlp_data only: epochs of the cosine schedule; a member stops earlier when "
        f"its validation loss stops improving (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--mlp-seed",
        type=int,
        default=None,
        dest="mlp_seed",
        help="mlp_data only: seed of the first member; member k uses seed + k, for its "
        f"initial weights, validation split and shuffles (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--mlp-max-rows",
        type=int,
        default=None,
        dest="mlp_max_rows",
        help="mlp_data only: subsample a corpus of more rows than this, seeded; 0 uses "
        "every row (default: 0)",
    )


def adapter_args(args: argparse.Namespace) -> list[str]:
    """The `train` flags that pick the adapter `args` asked for, and its options."""
    flags = ["--adapter", args.adapter]
    for flag, dest in _MLP_OPTIONS:
        if getattr(args, dest) is not None:
            flags += [flag, str(getattr(args, dest))]
    return flags


def model_encoding_args(args: argparse.Namespace) -> list[str]:
    """The `train` flags that write the encoding `args` asked for."""
    return [
        "--model-encoding",
        args.model_encoding,
        "--leaf-precision",
        args.leaf_precision,
        "--early-exit",
        args.early_exit,
    ]


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


def check_adapter_options(args: argparse.Namespace) -> None:
    """Refuse an adapter option that does not apply: `train` and `generate` share this, so
    `generate` refuses before it measures anything."""
    if args.adapter != "mlp_data":
        given = [flag for flag, dest in _MLP_OPTIONS if getattr(args, dest) is not None]
        if given:
            raise ValueError(
                f"{', '.join(given)} apply only to --adapter mlp_data, not {args.adapter}"
            )
        return
    if args.group_by_feature:
        raise ValueError(
            "--group-by-feature is a tree_data option: an mlp_data model is one "
            "ensemble per engine"
        )


def _resolve_score(args: argparse.Namespace, immediate: bool) -> None:
    """Settle the metric, label column and objective before any data is read.

    RFC 0019 §4.4/§13.4: a metric fixes label and direction, so --target and --objective
    can only contradict it. A metric-less model is a within-engine ranker.
    """
    from .immediate import LABEL_STATISTIC

    check_adapter_options(args)
    if args.adapter == "linear_data":
        _resolve_linear_score(args, immediate)
        return
    if args.metric is None and args.target in (None, "tflops"):
        args.metric = DEFAULT_RANKING_METRIC
    if args.metric is None:
        if immediate:
            raise ValueError(
                f"{args.role} requires --metric: engine selection compares its "
                "score across engines, so it must name a registered metric"
            )
        if args.adapter == "mlp_data":
            raise ValueError(
                "mlp_data predicts the log of one registered metric, and its artifact "
                "names it: pass --metric (--target alone names a column, not a metric)"
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


def _resolve_linear_score(args: argparse.Namespace, immediate: bool) -> None:
    """The label and its direction for a linear_data model.

    Its score is a log slowdown, lower is better, in no registered unit (RFC 0019 §7.5), so
    the UHD declares `objective: min` whatever the corpus holds. `--objective` here names
    the direction of the corpus's own target column, which the slowdown is measured in.
    """
    if immediate:
        raise ValueError(
            f"linear_data cannot train {args.role}: its score is a within-catalog slowdown, "
            "not a calibrated estimate engine selection can compare across engines"
        )
    if args.metric is not None or args.calibrated:
        raise ValueError(
            "linear_data scores a log slowdown against the catalog's best candidate, which "
            "is no registered metric and is not comparable across engines: drop --metric "
            "and --calibrated"
        )
    if args.group_by_feature:
        raise ValueError("--group-by-feature is a tree_data option")
    args.target = args.target or "tflops"
    args.objective = args.objective or "max"


def _immediate_training_rows(
    args: argparse.Namespace, input_path: Path
) -> tuple[pd.DataFrame, dict, dict]:
    """An engine-immediate (L1) corpus's rows that train, its binding, and the provenance it
    was measured under. Settles `args`' grouping, calibration, timing statistic and arches
    from the corpus, refusing any that contradict it."""
    from .immediate import (
        LABEL_STATISTIC,
        ROLE,
        read_corpus,
        require_binding_metric,
        training_binding,
    )
    from .provenance import snapshot_provenance, validate_provenance

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
    if args.training_arches and sorted(set(args.training_arches)) != observed_arches:
        raise ValueError("--training-arches must match the measured immediate corpus")
    args.training_arches = observed_arches
    if args.arch is None:
        if len(observed_arches) != 1:
            raise ValueError("multi-architecture L1 training requires --arch default")
        args.arch = observed_arches[0]
    if args.arch != "default" and (
        len(observed_arches) != 1 or args.arch != observed_arches[0]
    ):
        raise ValueError("L1 role-map arch must cover the complete training corpus")
    return df, binding, trained_against


def _catalog_training_rows(
    args: argparse.Namespace, input_path: Path
) -> tuple[pd.DataFrame, dict]:
    """A catalog corpus's rows that train, and the provenance it was measured under.

    Rows with no measurement, a collection error or a pick checked wrong are dropped, as
    `evaluate` and `knobs` drop them, so the tools agree on which rows exist."""
    from .immediate import LABEL_STATISTIC
    from .provenance import snapshot_provenance, validate_provenance

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
    # A row with no measurement is `is_valid=False` (collected CSV) or carries an
    # `error` (published dataset); a row checked wrong is never a label (RFC 0019
    # §13.2). Dropped as `evaluate` and `knobs` drop them, so the tools agree on
    # which rows exist.
    df, excluded = measured_rows(read_corpus_frame(input_path), args.target)
    if excluded:
        # Counted apart: each points at a different producer.
        logger.info(
            "Dropped %d row(s) checked numerically wrong, %d row(s) with "
            "is_valid=False, %d row(s) carrying a collection error, and %d "
            "row(s) whose %s is not a finite number",
            excluded.numerically_wrong,
            excluded.invalid,
            excluded.errored,
            excluded.unmeasured,
            args.target,
        )
    # RFC 0019.13 §11.2: a calibrated score MUST train on `avgTimeMs`. Uncalibrated
    # models may use any statistic but must record which.
    if args.calibrated and args.timing_statistic != LABEL_STATISTIC:
        raise ValueError(
            "--calibrated requires --timing-statistic avgTimeMs (RFC 0019.13 "
            f"§11.2); got {args.timing_statistic!r}"
        )
    return df, trained_against


def _drop_constant_features(
    df: pd.DataFrame,
    signature: list,
    names: list,
    matrix: np.ndarray,
    categorical_encoding: dict,
    feature_evaluator: str | None,
):
    """Remove the features that never vary in this corpus.

    Returns the pruned signature, names, matrix and encoding, the pruned signature's
    features_hash (None when nothing was pruned), and the dropped columns."""
    features_hash = None
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
            index for index in range(len(signature)) if index not in constant_indices
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
            signature, categorical_encoding, feature_evaluator
        )
    return signature, names, matrix, categorical_encoding, features_hash, dropped


def _group_models(
    args: argparse.Namespace,
    df: pd.DataFrame,
    names: list,
    groups: list | None,
    matrix: np.ndarray,
    categorical_encoding: dict,
) -> tuple[int, list[tuple[float, object]]]:
    """Layer 2 of a grouped ranker: the slot of `--group-by-feature` in the feature row (-1
    when there is none) and one ensemble per group value, each fitted on that group's rows
    over the same feature columns, so one signature describes both layers."""
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
    return group_index, group_models


def _fit_trees(
    args: argparse.Namespace,
    df: pd.DataFrame,
    names: list,
    matrix: np.ndarray,
    groups: list | None,
    categorical_encoding: dict,
):
    """The LightGBM fit of tree_data: layer 1, and layer 2 when the model is grouped.
    Returns the layer-1 model, the group feature's slot (-1 if none), the group models, and
    the early-exit schedule with what the manifest records about it."""
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

    early_exit = _early_exit_request(args, groups)
    cross_validation: list | None = [] if early_exit == "auto" else None
    model = train_model(
        layer_one_df,
        names,
        args.target,
        groups,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping,
        categorical_encoding=categorical_encoding,
        feature_matrix=layer_one_matrix,
        cross_validation=cross_validation,
    )

    group_index, group_models = _group_models(
        args, df, names, groups, matrix, categorical_encoding
    )
    return (
        model,
        group_index,
        group_models,
        _early_exit_schedule(
            args, early_exit, layer_one_df, groups, cross_validation, model.num_trees()
        ),
    )


def _early_exit_request(args: argparse.Namespace, groups: list | None):
    """`--early-exit` resolved: 'auto' (choose), [] (none) or the stages asked for.

    `auto` chooses only where the runtime ranks catalogs with the whole ensemble and the
    choice can be measured: a single-layer sort_kernel_catalog model with problem identity.
    A calibrated score keeps every candidate fully scored until pruned candidates' missing
    estimates are reviewed against every consumer (README, 'Early exit')."""
    from .early_exit import parse_schedule
    from .immediate import ROLE

    text = (args.early_exit or "auto").strip()
    if text == "none":
        return []
    if text != "auto":
        if args.role == ROLE or args.group_by_feature:
            raise ValueError(
                "--early-exit applies to a single-layer sort_kernel_catalog model: "
                "predict_engine scores one row with every tree, and a grouped model's "
                "runtime ignores the schedule"
            )
        if args.calibrated:
            raise ValueError(
                "--early-exit does not apply to a --calibrated model: a pruned candidate "
                "carries no estimate, and a calibrated ranking needs every candidate's"
            )
        return parse_schedule(text)
    if args.role == ROLE or args.group_by_feature or args.calibrated or not groups:
        return []
    return "auto"


def _early_exit_schedule(
    args: argparse.Namespace,
    request,
    df: pd.DataFrame,
    groups: list | None,
    cross_validation: list | None,
    tree_count: int,
) -> tuple[list, dict]:
    """The schedule to write and what the manifest records about it."""
    from .early_exit import check_schedule, choose_schedule, out_of_fold_catalogs

    if request != "auto":
        check_schedule(request, tree_count)
        return request, {
            "schedule": [list(stage) for stage in request],
            "chosen": False,
        }
    problem_ids = df[list(groups)].astype(str).agg("\x1f".join, axis=1).to_numpy()
    catalogs = out_of_fold_catalogs(
        cross_validation[0], problem_ids, df[args.target].to_numpy(dtype=np.float64)
    )
    choice = choose_schedule(catalogs, tree_count, args.objective)
    logger.info(
        "Early exit: %s (out-of-fold mean regret %.4f with every tree)",
        choice.schedule or "none",
        choice.full["mean_regret"],
    )
    return choice.schedule, {**choice.to_manifest(), "chosen": True}


def _fit_linear(
    args: argparse.Namespace,
    df: pd.DataFrame,
    signature: list,
    names: list,
    matrix: np.ndarray,
    groups: list | None,
):
    """The RelLin fit of linear_data. Returns the ranker and what the manifest records."""
    from .rellin import (
        DEFAULT_MIN_ROWS,
        DEFAULT_RIDGE,
        fit_rellin,
        relative_log_slowdown,
        select_basis,
    )

    if not groups or any(column not in df.columns for column in groups):
        raise ValueError(
            "linear_data trains on each candidate's slowdown against the best candidate of "
            "its problem, so it needs problem identity: pass --group-by (columns present in "
            "the corpus) or use a corpus with a 'benchmark' column"
        )
    problems = df.groupby(list(groups), sort=False, dropna=False).ngroup().to_numpy()
    slowdown = relative_log_slowdown(
        df[args.target].to_numpy(dtype=np.float64), problems, args.objective
    )

    def slots(requested: list[str], flag: str) -> list[int]:
        missing = [name for name in requested if name not in names]
        if missing:
            raise ValueError(
                f"{flag} {missing} is not among the model's features {names}; a feature "
                "that never varies in this corpus is dropped, so it cannot key a config"
            )
        return [names.index(name) for name in requested]

    if args.key_features:
        key = slots(args.key_features, "--key-features")
    else:
        key = [
            index
            for index, entry in enumerate(signature)
            if isinstance(entry, str) and entry.startswith("$kernel.")
        ]
        if not key:
            raise ValueError(
                "linear_data keys its per-config models on the signature's `$kernel.*` "
                "references, and this one has none that vary: name the features with "
                "--key-features"
            )
    family = (
        slots(args.family_features, "--family-features") if args.family_features else []
    )
    basis, transforms = select_basis(matrix, key)
    if not basis:
        logger.warning(
            "no feature besides the config key varies, so the model is one constant per config"
        )
    ranker = fit_rellin(
        matrix,
        slowdown,
        key_features=key,
        family_features=family,
        basis_features=basis,
        transforms=transforms,
        ridge=DEFAULT_RIDGE if args.ridge is None else args.ridge,
        min_rows=DEFAULT_MIN_ROWS if args.min_rows is None else args.min_rows,
    )
    logger.info(
        "Fitted %d config(s), %d family(ies), %d basis term(s) on %d rows (%d problems)",
        len(ranker.config_floors),
        len(ranker.family_floors),
        len(basis),
        len(df),
        int(problems.max()) + 1,
    )
    return ranker, {
        "adapter": "linear_data",
        "key_features": [names[index] for index in key],
        "family_features": [names[index] for index in family],
        "basis": [
            {"feature": names[slot], "transform": "log1p" if transform else "identity"}
            for slot, transform in zip(basis, transforms)
        ],
        "num_configs": int(len(ranker.config_floors)),
        "num_families": int(len(ranker.family_floors)),
        "ridge": DEFAULT_RIDGE if args.ridge is None else args.ridge,
        "min_rows": DEFAULT_MIN_ROWS if args.min_rows is None else args.min_rows,
    }


def _parse_mlp_hidden(text: str) -> tuple[int, ...]:
    """`--mlp-hidden`: comma-separated positive widths; empty means no hidden layer."""
    try:
        widths = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError:
        raise ValueError(
            f"--mlp-hidden {text!r} must be comma-separated integers, e.g. 64,64"
        ) from None
    if any(width < 1 for width in widths):
        raise ValueError(f"--mlp-hidden {text!r} widths must be positive")
    return widths


def _fit_mlp(
    args: argparse.Namespace,
    df: pd.DataFrame,
    matrix: np.ndarray,
    groups: list | None = None,
):
    """The ensemble fit of mlp_data on log(target). Returns the ensemble and what the
    manifest records. `groups` are the columns identifying a catalog problem (None for an
    L1 corpus or one without problem identity)."""
    from .mlp import (
        DEFAULT_EPOCHS,
        DEFAULT_HIDDEN,
        DEFAULT_MEMBERS,
        DEFAULT_SEED,
        fit_mlp,
    )

    def option(value, default):
        return default if value is None else value

    target = df[args.target].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or (target <= 0).any():
        # RFC 0019 §8.3: a non-positive measurement is no measurement.
        raise ValueError(
            f"target {args.target!r} must contain finite, strictly positive values"
        )
    # A catalog's rows are candidates of a problem: hold out whole problems for early
    # stopping, keyed as the tree trainer's GroupKFold keys them (shape first, so every
    # device's measurements of one shape fall on one side). Rows without problem identity,
    # and an L1 corpus (one row per graph), split by row.
    problems = None
    if groups:
        if groups[0] not in df.columns:
            raise ValueError(
                f"problem column {groups[0]!r} is not in the corpus, so mlp_data cannot "
                "hold out whole problems for early stopping"
            )
        problems = df.groupby(groups[0]).ngroup().to_numpy()
    ensemble, fit = fit_mlp(
        matrix,
        score_transform.forward(target),
        hidden=(
            DEFAULT_HIDDEN
            if args.mlp_hidden is None
            else _parse_mlp_hidden(args.mlp_hidden)
        ),
        members=option(args.mlp_members, DEFAULT_MEMBERS),
        epochs=option(args.mlp_epochs, DEFAULT_EPOCHS),
        seed=option(args.mlp_seed, DEFAULT_SEED),
        max_rows=option(args.mlp_max_rows, 0),
        groups=problems,
    )
    logger.info(
        "Fitted %d network(s) on %d of %d rows; held-out L1 %.4f in log units",
        len(ensemble.networks),
        fit["rows"],
        fit["rows_available"],
        fit["validation_l1_log"],
    )
    return ensemble, fit


def _run_train(args: argparse.Namespace) -> int:
    from .provenance import record_feature_semantics
    from .immediate import ROLE, check_signature_evaluates, validate_signature

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
            df, binding, trained_against = _immediate_training_rows(args, input_path)
        else:
            df, trained_against = _catalog_training_rows(args, input_path)
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
        signature, names, matrix, categorical_encoding, pruned_hash, dropped = (
            _drop_constant_features(
                df,
                signature,
                names,
                matrix,
                categorical_encoding,
                args.feature_evaluator,
            )
        )
        features_hash = pruned_hash or features_hash
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
        if args.adapter == "linear_data":
            ranker, linear_fit = _fit_linear(args, df, signature, names, matrix, groups)
            model, group_index, group_models = None, -1, []
            early_exit, early_exit_record = [], None
        elif args.adapter == "mlp_data":
            ensemble, mlp_fit = _fit_mlp(
                args, df, matrix, None if immediate else groups
            )
            model, group_index, group_models = None, -1, []
            early_exit, early_exit_record = [], None
        else:
            model, group_index, group_models, (early_exit, early_exit_record) = (
                _fit_trees(args, df, names, matrix, groups, categorical_encoding)
            )
        # Read before anything is written: a bad SOURCE_DATE_EPOCH or an unreadable input
        # fails the run here, with no partial output.
        training_date = resolve_training_date()
        input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError, KeyError) as error:
        logger.error("%s", error)
        return 1

    # No descriptor or model artifact is published before all input checks and fitting.
    output_dir.mkdir(parents=True, exist_ok=True)
    fb_path = output_dir / "model.bin"
    if args.adapter == "linear_data":
        from .linear_to_flatbuffer import write_linear_model

        model_sha256 = write_linear_model(
            ranker,
            features_hash,
            fb_path,
            num_training_samples=len(df),
            training_arches=args.training_arches,
            model_version=args.model_version,
            training_date=training_date,
        )
    elif args.adapter == "mlp_data":
        from .mlp_to_flatbuffer import write_mlp_model

        model_sha256 = write_mlp_model(
            ensemble,
            args.metric,
            features_hash,
            fb_path,
            num_training_samples=mlp_fit["rows"],
            training_arches=args.training_arches,
            model_version=args.model_version,
            training_date=training_date,
        )
    else:
        lgbm_path = output_dir / "model.lgbm"
        model.save_model(str(lgbm_path))
        model_sha256 = convert(
            lgbm_path,
            features_hash,
            fb_path,
            num_training_samples=len(df),
            training_arches=args.training_arches,
            model_version=args.model_version,
            training_date=training_date,
            group_by_feature_index=group_index,
            group_models=group_models or None,
            encoding=args.model_encoding,
            leaf_precision=args.leaf_precision,
            early_exit=early_exit,
        )
        if not args.keep_lgbm:
            lgbm_path.unlink()
    descriptor = {
        # Slot provenance (`slot`, or `kmds` for a slotted engine's L1 model) is a 1.1 feature.
        "version": "1.1" if {"slot", "kmds"} & set(trained_against) else "1.0",
        "id": uhd_id,
        "name": args.name,
        "adapter": args.adapter,
        "features_signature": signature,
        "features_hash": features_hash,
        "trained_against": trained_against,
        # A linear_data score is a log slowdown, so lower is better whatever the corpus's
        # own direction (RFC 0019 §7.5); the manifest keeps the corpus's.
        "objective": "min" if args.adapter == "linear_data" else args.objective,
        # RFC 0019 §4.1: `metric` names the registered quantity the score estimates, and
        # is absent from a within-engine ranker that estimates none of them; linear_data
        # declares no score at all.
        **(
            {}
            if args.adapter == "linear_data"
            else {
                "score": {
                    **({"metric": args.metric} if args.metric else {}),
                    "calibrated": args.calibrated,
                    "transform": score_transform.TRAINED,
                }
            }
        ),
        # RFC 0019 §7.2: the adapter verifies this artifact digest before parsing.
        # Bare hex, matching `sha256(buffer, size)` on the runtime side.
        args.adapter: {"artifact": fb_path.name, "hash": model_sha256},
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
        "score_transform": (
            "identity" if args.adapter == "linear_data" else score_transform.TRAINED
        ),
        "group_by": groups or [],
        # The artifact holds only a slot index; the name lets a report attribute regret
        # to the group choice versus the member choice.
        "group_by_feature": args.group_by_feature,
        "group_models": len(group_models or []),
        **(
            linear_fit
            if args.adapter == "linear_data"
            else (
                {"adapter": "mlp_data", "mlp_fit": mlp_fit}
                if args.adapter == "mlp_data"
                else {
                    "num_trees": model.num_trees(),
                    "feature_importance": {
                        name: {"gain": float(gain), "split": int(split)}
                        for name, gain, split in zip(
                            names,
                            model.feature_importance(importance_type="gain"),
                            model.feature_importance(importance_type="split"),
                        )
                    },
                }
            )
        ),
        "num_samples": len(df),
        "input_file": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "training_arches": args.training_arches or [],
        "model_version": args.model_version,
        **(
            {
                "training_options": {
                    "num_boost_round": args.num_boost_round,
                    "early_stopping_rounds": args.early_stopping,
                },
                "early_exit": early_exit_record,
            }
            if args.adapter == "tree_data"
            else {}
        ),
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
