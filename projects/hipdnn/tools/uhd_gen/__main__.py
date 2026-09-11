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
        --features q.seqlen_q kernel.block_size device.cu_count \\
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
from .evaluate import add_evaluate_arguments, run_evaluate
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
        "tools/results_import publishes, or a collected .csv/.json corpus",
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
        "--drop-constant-features",
        action="store_true",
        dest="drop_constant_features",
        help="Omit constant model inputs; never changes the engine's authored public knobs.",
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
            # The suffix decides. `.parquet` is what tools/results_import publishes (RFC
            # 0019.13 §8.3) and is the route a model anyone ships should come by: the
            # dataset carries its own types, so a column empty in one shard and populated
            # in another cannot concatenate to `object` and quietly change what the
            # trainer sees. A collected CSV is read directly, and nothing §8.3 specifies
            # is checked on it -- that is what the importer exists for -- so it is the
            # escape hatch for a quick local run. The §11.2 label rule below is applied
            # to all three alike: the published dataset earns no exemption from it.
            if input_path.suffix == ".parquet":
                df = pd.read_parquet(input_path)
            elif input_path.suffix == ".json":
                df = pd.DataFrame(json.loads(input_path.read_text(encoding="utf-8")))
            else:
                df = pd.read_csv(input_path, dtype={"benchmark": str, "device": str})
            if "is_valid" in df.columns:
                before = len(df)
                df = df[df["is_valid"].astype(str).str.lower() == "true"]
                logger.info("Dropped %d row(s) with is_valid=False", before - len(df))
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
        if any(isinstance(entry, dict) for entry in signature):
            features_hash, values = evaluate_feature_rows(df, signature, categorical_encoding, args.feature_evaluator)
            matrix = np.asarray(values, dtype=np.float64)
        else:
            features_hash = compute_features_hash(signature, categorical_encoding)
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
        dropped = []
        if constants:
            logger.warning("%d feature column(s) never vary: %s", len(constants),
                           ", ".join(f"{name}={value!r}" for name, value in constants))
            if len(constants) / len(signature) >= CONSTANT_FEATURE_WARN_FRACTION:
                logger.warning("High constant-feature proportion: check device and problem coverage")
            if args.drop_constant_features:
                dropped = [names[index] for index in constant_indices]
                keep = [index for index in range(len(signature)) if index not in constant_indices]
                signature = [signature[index] for index in keep]
                names = [names[index] for index in keep]
                matrix = matrix[:, keep]
                remaining_refs = set(signature_references(signature))
                categorical_encoding = {key: value for key, value in categorical_encoding.items() if key in remaining_refs}
                if any(isinstance(entry, dict) for entry in signature):
                    features_hash, _ = evaluate_feature_rows(df.iloc[:0], signature, categorical_encoding, args.feature_evaluator)
                else:
                    features_hash = compute_features_hash(signature, categorical_encoding)
                logger.warning("Dropping constant model inputs %s; authored knobs are unchanged", dropped)
        if args.objective == "max" and _looks_like_cost_metric(args.target):
            logger.warning("Target '%s' looks like a cost; use --objective min to prefer faster candidates", args.target)
        groups = args.group_by
        if groups is None and "benchmark" in df.columns:
            groups = ["benchmark"] + (["device"] if "device" in df.columns else [])
        model = train_model(
            df, names, args.target, groups, num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping, categorical_encoding=categorical_encoding,
            feature_matrix=matrix,
        )
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
    convert(lgbm_path, features_hash, fb_path, num_training_samples=len(df),
            training_arches=args.training_arches, model_version=args.model_version)
    if not args.keep_lgbm:
        lgbm_path.unlink()
    descriptor = {
        "version": "1.0", "id": uhd_id, "name": args.name, "adapter": "tree_data",
        "features_signature": signature, "features_hash": features_hash,
        "trained_against": trained_against, "objective": args.objective,
        "score": {"units": args.score_units or args.target, "calibrated": args.calibrated, "transform": "log1p"},
        "tree_data": {"artifact": fb_path.name},
    }
    if categorical_encoding:
        descriptor["categorical_encoding"] = categorical_encoding
    descriptor_path = output_dir / f"{args.descriptor_name}.uhd.json"
    descriptor_path.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "uhd_id": uhd_id, "requested_features": args.features or requested_signature,
        "features": names, "features_signature": signature, "features_hash": features_hash,
        "trained_against": trained_against, "device_coverage": coverage,
        "constant_features": [{"column": name, "value": value} for name, value in constants],
        "dropped_constant_features": dropped, "drop_constant_features": bool(args.drop_constant_features),
        "categorical_encoding": categorical_encoding, "target": args.target, "objective": args.objective,
        "score_units": args.score_units or args.target, "score_calibrated": args.calibrated,
        # RFC 0019.13 §10.5/§11.2: which measured timing the target came from. §11.2
        # refuses cross-engine comparison between models trained on different ones, so
        # a consumer has to be able to read it off the artifact rather than infer it.
        "timing_statistic": args.timing_statistic,
        "score_transform": "log1p", "group_by": groups or [], "num_trees": model.num_trees(),
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
