# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Collect engine-immediate or enrolled-candidate measurements and generate UHDs."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import pandas as pd

from .coverage import device_field_coverage, enforce_device_coverage, propose_features
from .evaluate import problem_keys, resolve_grouping, split_problems
from .features import build_features_signature, signature_references
from .provenance import snapshot_provenance
from .immediate import ROLE, normalize_row, normalize_corpus, training_binding, validate_signature

logger = logging.getLogger(__name__)


def add_generate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graphs", nargs="+", required=True, help="Graph JSON files or corpus directories (recursive)")
    parser.add_argument("--descriptor-tree", required=True, help="Shipping descriptor tree; authored knobs are preserved")
    parser.add_argument("--engine", help="UED name/UUID or canonical immediate engine name")
    parser.add_argument("--engine-id", required=True, type=int, help="Public hipDNN engine ID used by hipdnn_bench")
    parser.add_argument("--bench", default="hipdnn_bench", help="Public hipdnn_bench executable")
    parser.add_argument("--plugin-dir")
    parser.add_argument("--device", action="append", help="HIP_VISIBLE_DEVICES selection; repeat to collect multiple devices")
    parser.add_argument("--knob", action="append", default=[], help="Explicit NAME=INTEGER collection pin (repeatable)")
    parser.add_argument("--workspace-limit", type=int, help="L1 immediate workspace constraint in bytes")
    parser.add_argument("--output-dir", required=True, help="New output directory for reproducible collection and model artifacts")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--features", nargs="+", help="Explicit full published raw feature names")
    source.add_argument("--feature-signature", help="Explicit canonical inline JSON feature array")
    parser.add_argument("--dim-tile", action="append", default=[], metavar="DIMENSION=KERNEL_FIELD",
                        help="Author-declared dimension-to-tile pair used to propose ceil_div/remainder features")
    parser.add_argument("--feature-evaluator", help="Shared hipdnn_uhd_features executable")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--name", default="Generated UHD")
    parser.add_argument("--uhd-id")
    parser.add_argument("--arch", help="Promotion arch; otherwise infer one observed architecture")
    parser.add_argument("--role", default="sort_kernel_catalog", choices=["sort_kernel_catalog", "predict_engine_tflops", "predict_applicable_kernels"])
    parser.add_argument("--no-promote", action="store_true", help="Validate installation but leave shipping descriptors untouched")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _descriptor(tree: Path, suffix: str, identity: str) -> tuple[Path, dict]:
    matches = []
    for path in sorted(tree.rglob("*" + suffix)):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("id") == identity:
            matches.append((path, document))
    if not matches:
        raise ValueError(f"descriptor {identity} ({suffix}) does not resolve in {tree}")
    if any(document != matches[0][1] for _, document in matches[1:]):
        raise ValueError(f"descriptor {identity} has conflicting definitions")
    return matches[0]


def _run_json(command: list[str], environment: dict, log_dir: Path, ordinal: int, commands: list) -> dict:
    result = subprocess.run(command, env=environment, capture_output=True, text=True, encoding="utf-8", check=False)
    entry = {"argv": command, "returncode": result.returncode,
             "environment": {key: environment.get(key) for key in
                             ("HIPDNN_DESCRIPTOR_PATH", "HIPDNN_DESCRIPTOR_DIR", "HIPDNN_DESCRIPTOR_RUNTIME_DIR", "HIP_VISIBLE_DEVICES")}}
    commands.append(entry)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{ordinal:06d}.stdout.json").write_text(result.stdout, encoding="utf-8")
    (log_dir / f"{ordinal:06d}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise ValueError(f"hipdnn_bench failed ({result.returncode}): {result.stderr.strip()}")
    try:
        response = json.loads(result.stdout)
    except ValueError as error:
        raise ValueError("hipdnn_bench did not emit one JSON response; see captured command output") from error
    if not isinstance(response, dict):
        raise ValueError("hipdnn_bench response must be an object")
    return response


def _identity(response: dict) -> tuple:
    identity = tuple(response.get(key) for key in
                     ("engine_id", "graph_id", "device_id", "device_arch", "engine_descriptor_id", "engine_name"))
    if any(value is None or value == "" for value in identity):
        raise ValueError("benchmark response lacks engine/graph/device identity")
    return identity


def _feature_map(response: dict, key: str) -> dict:
    mapping = response.get(key)
    if not isinstance(mapping, dict) or any(not isinstance(name, str) or not name or name.startswith("$") for name in mapping):
        raise ValueError(f"{key} must contain canonical published names without '$'")
    return mapping


def _knob_tuple(candidate: dict) -> tuple:
    knobs = candidate.get("knob_settings")
    if not isinstance(knobs, dict) or any(not isinstance(name, str) or not name or
                                          isinstance(value, bool) or not isinstance(value, int)
                                          for name, value in knobs.items()):
        raise ValueError("candidate lacks an integer-valued enrolled knob tuple")
    return tuple(sorted(knobs.items()))


def collect_graph(command: list[str], environment: dict, log_dir: Path, commands: list,
                  *, engine_descriptor_id: str) -> tuple[list[dict], set[str]]:
    """Enumeration and timing must agree on identity, bindings and the exact tuple."""
    candidates = []
    seen_ids, seen_tuples = set(), set()
    offset = 0
    first = None
    total = None
    while True:
        page = _run_json([command[0], "enumerate", *command[1:], "--offset", str(offset), "--limit", "10000"],
                         environment, log_dir, len(commands), commands)
        identity = _identity(page)
        if str(identity[0]) != command[command.index("--engine-id") + 1]:
            raise ValueError("enumeration returned another engine's catalog")
        if identity[4] != engine_descriptor_id:
            raise ValueError("enumerated engine does not own the recorded UED provenance")
        if first is None:
            first = page
            total = page.get("total_count")
            if not isinstance(total, int) or total < 0:
                raise ValueError("enumeration lacks a bounded total_count")
        elif identity != _identity(first) or page.get("total_count") != total:
            raise ValueError("candidate enumeration identity/count changed between pages")
        for key in ("problem_features", "device_features"):
            if _feature_map(page, key) != _feature_map(first, key):
                raise ValueError(f"{key} changed between enumeration pages")
        batch = page.get("candidates")
        if not isinstance(batch, list):
            raise ValueError("enumeration lacks a candidates array")
        for candidate in batch:
            candidate_id = candidate.get("id")
            knobs = _knob_tuple(candidate)
            if not candidate_id or candidate_id in seen_ids or knobs in seen_tuples:
                raise ValueError("candidate identities and complete enrolled knob tuples must be unique")
            seen_ids.add(candidate_id)
            seen_tuples.add(knobs)
            _feature_map(candidate, "kernel_features")
            candidates.append(candidate)
        next_offset = page.get("next_offset")
        if next_offset is None:
            if len(candidates) != total:
                raise ValueError("candidate enumeration ended before total_count; refusing silent truncation")
            break
        if not batch or next_offset != offset + len(batch) or next_offset >= total:
            raise ValueError("candidate enumeration returned an invalid continuation offset")
        offset = next_offset
    if not candidates:
        raise ValueError(f"no matched candidates for graph {first['graph_id']}")
    rows = []
    published = set(_feature_map(first, "problem_features")) | set(_feature_map(first, "device_features"))
    for candidate in candidates:
        # Complete enrolled settings replace collection pins; no Cartesian combinations.
        timing_command = list(command)
        while "--knob" in timing_command:
            index = timing_command.index("--knob")
            del timing_command[index:index + 2]
        for name, value in _knob_tuple(candidate):
            timing_command.extend(["--knob", f"{name}={value}"])
        measured = _run_json([*timing_command, "--json"], environment, log_dir, len(commands), commands)
        if _identity(measured) != _identity(first):
            raise ValueError("timing response belongs to another graph/device/engine")
        for key in ("problem_features", "device_features"):
            if _feature_map(measured, key) != _feature_map(first, key):
                raise ValueError(f"timing changed the enumerated {key}")
        results = measured.get("results", [])
        if len(results) != 1:
            raise ValueError("one enrolled tuple must time exactly one candidate")
        result = results[0]
        if result.get("candidate_id") != candidate["id"] or _knob_tuple(result) != _knob_tuple(candidate):
            raise ValueError("timed knobs did not resolve to the enrolled candidate")
        if _feature_map(result, "kernel_features") != _feature_map(candidate, "kernel_features"):
            raise ValueError("timing kernel metadata differs from enrolled candidate")
        if not isinstance(result.get("is_valid"), bool):
            raise ValueError("timing response must preserve the benchmark's is_valid verdict")
        elapsed = result.get("robust_time_ms")
        if result.get("succeeded") and (not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0):
            raise ValueError("successful timing requires a positive finite robust_time_ms")
        row = {"benchmark": first["graph_id"], "device": first["device_id"],
               "arch": first["device_arch"].split(":", 1)[0], "device_arch": first["device_arch"],
               "engine": first["engine_id"], "kernel": candidate["id"], "is_valid": result["is_valid"],
               "succeeded": result.get("succeeded"), "skip_reason": result.get("skip_reason"),
               "robustMeanMs": elapsed, "minTimeMs": result.get("min_time_ms"), "avgTimeMs": result.get("avg_time_ms"),
               "knob_settings": json.dumps(candidate["knob_settings"], sort_keys=True)}
        for mapping in (first["problem_features"], first["device_features"], candidate["kernel_features"]):
            collision = set(row) & set(mapping)
            if collision:
                raise ValueError(f"published feature names collide with envelope fields: {sorted(collision)}")
            row.update(mapping)
            published.update(mapping)
        rows.append(row)
    return rows, published


def collect_immediate_graph(command: list[str], environment: dict, log_dir: Path,
                            commands: list) -> tuple[list[dict], set[str]]:
    """Measure one engine's ordinary no-search selection without inspecting its catalog."""
    if "--knob" in command or "enumerate" in command:
        raise ValueError("L1 collection cannot pin knobs or enumerate candidates")
    response = _run_json([*command, "--collect-immediate", "--json"],
                         environment, log_dir, len(commands), commands)
    row = normalize_row(response)
    if str(row["engine"]) != command[command.index("--engine-id") + 1]:
        raise ValueError("immediate measurement returned another engine")
    return [row], set(json.loads(row["features"]))


def run_generate(args: argparse.Namespace) -> int:
    from .__main__ import main
    from .promote import PromoteError, build_plan, run_promote, add_promote_arguments

    stage = None
    immediate = args.role == ROLE
    if args.workspace_limit is not None and (not immediate or args.workspace_limit < 0):
        logger.error("--workspace-limit requires predict_engine_tflops and a nonnegative byte count")
        return 1
    try:
        output = Path(args.output_dir).resolve()
        tree = Path(args.descriptor_tree).resolve()
        if output.exists():
            raise ValueError("--output-dir must not exist; generation never overwrites a previous run")
        if tree == output or tree in output.parents:
            raise ValueError("--output-dir must be outside the shipping descriptor tree")
        if not 0 < args.eval_fraction < 1:
            raise ValueError("generate requires a true problem holdout: 0 < --eval-fraction < 1")
        bench = shutil.which(args.bench)
        if bench is None:
            raise ValueError(f"hipdnn_bench executable {args.bench!r} was not found")
        graphs = set()
        for supplied in args.graphs:
            path = Path(supplied).resolve()
            graphs.update(path.rglob("*.json") if path.is_dir() else [path])
        if not graphs or any(not path.is_file() for path in graphs):
            raise ValueError("--graphs must identify existing graph JSON files")
        if immediate:
            if args.knob or args.dim_tile:
                raise ValueError("L1 generation cannot use kernel knobs or dimension/tile candidate features")
            if not tree.is_dir():
                raise ValueError("--descriptor-tree must be an existing descriptor root (it may be empty)")
            provenance, ued, exposed = None, {}, {}
            kernel_fields = set()
        else:
            provenance = snapshot_provenance(tree, args.engine, args.arch)
            ued_path, ued = _descriptor(tree, ".ued.json", provenance["ued"]["id"])
            _, kmd = _descriptor(tree, ".kmd.json", provenance["kmd"]["id"])
            kernel_fields = {"kernel." + field["name"] for field in kmd["fields"]}
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".uhd-generate-", dir=output.parent))
        environment = dict(os.environ)
        if immediate:
            environment["HIPDNN_DESCRIPTOR_PATH"] = str(tree)
        else:
            collection_tree = stage / "collection_descriptors"
            shutil.copytree(tree, collection_tree)
            exposed = dict(ued)
            exposed["knobs"] = [field["name"] for field in kmd["fields"] if field["type"] == "int"]
            _write_json(collection_tree / ued_path.relative_to(tree), exposed)
            _write_json(stage / "shipping_ued.json", ued)
            environment["HIPDNN_DESCRIPTOR_DIR"] = str(collection_tree)
            environment.pop("HIPDNN_DESCRIPTOR_RUNTIME_DIR", None)
        rows, commands, graph_inputs = [], [], []
        published = set()
        for graph_index, graph in enumerate(sorted(graphs)):
            saved_graph = stage / "graphs" / f"{graph_index:06d}.json"
            saved_graph.parent.mkdir(exist_ok=True)
            if immediate:
                graph_document = json.loads(graph.read_text(encoding="utf-8"))
                if not isinstance(graph_document, dict):
                    raise ValueError("graph input must be a JSON object")
                if not graph_document.get("id"):
                    canonical = json.dumps(graph_document, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    graph_document["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "hipdnn:graph:" + canonical))
                _write_json(saved_graph, graph_document)
            else:
                shutil.copyfile(graph, saved_graph)
            graph_inputs.append({"source": str(graph), "copy": str(saved_graph.relative_to(stage)),
                                 "sha256": hashlib.sha256(graph.read_bytes()).hexdigest()})
            command = [bench, "--graph", str(saved_graph), "--engine-id", str(args.engine_id)]
            if args.plugin_dir:
                command.extend(["--plugin-dir", str(Path(args.plugin_dir).resolve())])
            if args.workspace_limit is not None:
                command.extend(["--workspace-limit", str(args.workspace_limit)])
            for knob in args.knob:
                command.extend(["--knob", knob])
            for device in args.device or [environment.get("HIP_VISIBLE_DEVICES")]:
                run_env = dict(environment)
                if device is not None:
                    run_env["HIP_VISIBLE_DEVICES"] = device
                if immediate:
                    collected, names = collect_immediate_graph(command, run_env, stage / "commands", commands)
                else:
                    collected, names = collect_graph(command, run_env, stage / "commands", commands,
                                                     engine_descriptor_id=ued["id"])
                rows.extend(collected)
                published.update(names)
        frame = pd.DataFrame(rows)
        if immediate:
            frame, binding = training_binding(normalize_corpus(frame))
            provenance = binding["trained_against"]
            if args.engine and args.engine not in (binding["engine"], provenance.get("ued", {}).get("id")):
                raise ValueError("--engine does not match the collected engine binding")
        elif frame.duplicated(["benchmark", "device", "kernel"]).any():
            raise ValueError("the graph/device corpus contains duplicate candidate measurements")
        _write_json(stage / "provenance.json", provenance)
        frame.to_csv(stage / "corpus.csv", index=False)
        _write_json(stage / "corpus.json", rows)
        usable = frame.copy() if immediate else frame[frame["is_valid"] & frame["succeeded"].eq(True)].copy()
        if usable.empty:
            raise ValueError("the benchmark produced no successful valid timings")
        grouping = resolve_grouping(frame)
        split = split_problems(problem_keys(frame, grouping), args.eval_fraction, args.seed)
        train_frame = usable[~problem_keys(usable, grouping).isin(split.eval_problems)]
        if len(set(problem_keys(train_frame, grouping))) < 5:
            raise ValueError("generation needs at least five training graph/device groups plus held-out problems")
        pairs = []
        for pair in args.dim_tile:
            parts = pair.split("=", 1)
            if len(parts) != 2:
                raise ValueError("--dim-tile requires DIMENSION=KERNEL_FIELD")
            pairs.append(tuple(part.removeprefix("$") for part in parts))
        omitted = []
        if args.feature_signature:
            signature = json.loads(Path(args.feature_signature).read_text(encoding="utf-8"))
        elif args.features:
            signature = build_features_signature(args.features)
        else:
            scalar_columns = [name for name in sorted(published)
                              if train_frame[name].notna().all()
                              and train_frame[name].map(lambda value: isinstance(value, (str, int, float, bool))).all()]
            legal_kernel_fields = {name for name in scalar_columns
                                   if name.split("[", 1)[0] in kernel_fields}
            signature, omitted = propose_features(train_frame[scalar_columns], legal_kernel_fields, pairs)
        if not isinstance(signature, list) or not signature:
            raise ValueError("the feature recipe must be a nonempty canonical array")
        if immediate:
            validate_signature(signature, published)
        unknown = {ref[1:] for ref in signature_references(signature)} - published
        if unknown:
            raise ValueError(f"features are not published by this engine: {sorted(unknown)}")
        coverage = device_field_coverage(train_frame)
        enforce_device_coverage(signature, coverage)
        arches = sorted(usable["arch"].unique())
        if args.arch not in (None, "default") and args.arch not in arches:
            raise ValueError("promotion arch is absent from the observed device architectures")
        if args.arch is None and len(arches) != 1:
            raise ValueError("multiple observed architectures require an explicit --arch promotion target")
        _write_json(stage / "features.json", signature)
        train_frame.to_csv(stage / "train.csv", index=False)
        _write_json(stage / "train.json", train_frame.to_dict(orient="records"))
        train_args = ["train", "--input", str(stage / "train.json"), "--feature-signature", str(stage / "features.json"),
                      "--provenance", str(stage / "provenance.json"),
                      "--target", "tflops" if immediate else "robustMeanMs",
                      "--objective", "max" if immediate else "min",
                      "--score-units", "tflops" if immediate else "ms",
                      "--role", args.role, "--group-by", *grouping.columns, "--output-dir", str(stage / "model"),
                      "--name", args.name, "--num-boost-round", str(args.num_boost_round),
                      "--early-stopping", str(args.early_stopping), "--training-arches", *arches]
        if immediate:
            train_args.extend(["--calibrated", "--arch", args.arch or arches[0]])
        if args.uhd_id:
            train_args.extend(["--uhd-id", args.uhd_id])
        if args.feature_evaluator:
            train_args.extend(["--feature-evaluator", args.feature_evaluator])
        if main(train_args):
            raise ValueError("training failed; no generated model was published")
        eval_args = ["evaluate", "--input", str(stage / "corpus.json"), "--model-dir", str(stage / "model"),
                     "--eval-fraction", str(args.eval_fraction), "--seed", str(args.seed), "--include-per-problem"]
        if args.feature_evaluator:
            eval_args.extend(["--feature-evaluator", args.feature_evaluator])
        if main(eval_args):
            raise ValueError("artifact evaluation failed; no generated model was published")
        report_path = stage / "model" / "eval_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not report["metrics"]["problems_scored"]:
            raise ValueError("held-out corpus has no evaluable immediate predictions" if immediate
                             else "held-out corpus has no evaluable candidate ranking")
        evaluated_keys = {tuple(key) for key in report["split"]["eval_problem_keys"]}
        training_keys = set(problem_keys(train_frame, grouping))
        if training_keys & evaluated_keys:
            raise ValueError("evaluation includes a problem seen during training")
        report["holdout_integrity"] = {"status": "held_out", "detail": "Verified disjoint graph/device identities in recorded training and evaluation slices"}
        _write_json(report_path, report)
        _write_json(stage / "generation_manifest.json", {
            "schema": "uhd_gen.generation/1", "trained_against": provenance, "graphs": graph_inputs,
            "commands": commands, "features_signature": signature, "omitted_proposals": omitted,
            "training_arguments": train_args, "evaluation_arguments": eval_args,
            "device_coverage": coverage, "training_problem_keys": sorted(training_keys),
            "eval_problem_keys": sorted(evaluated_keys), "seed": args.seed, "eval_fraction": args.eval_fraction,
            "shipping_knobs": ued.get("knobs", []), "collection_knobs": exposed.get("knobs", []),
            "engine_id": args.engine_id, "training_arches": arches, "promotion_role": args.role,
            "promotion_arch": args.arch or arches[0],
        })
        # Validate installation against the original tree before publishing any artifacts.
        build_plan(stage / "model", tree, args.engine, role=args.role, arch=args.arch or arches[0])
        # Recorded paths must refer to the final output rather than the staging directory.
        old_root = str(stage)
        for path in stage.rglob("*.json"):
            if "collection_descriptors" in path.relative_to(stage).parts or "graphs" in path.relative_to(stage).parts:
                continue
            document = json.loads(path.read_text(encoding="utf-8"))
            def relocate(value):
                if isinstance(value, str):
                    return str(output) + value[len(old_root):] if value.startswith(old_root) else value
                if isinstance(value, list):
                    return [relocate(item) for item in value]
                if isinstance(value, dict):
                    return {key: relocate(item) for key, item in value.items()}
                return value
            _write_json(path, relocate(document))
        stage.rename(output)
        stage = None
        if not args.no_promote:
            parser = argparse.ArgumentParser()
            add_promote_arguments(parser)
            promote_args = ["--model-dir", str(output / "model"), "--descriptor-tree", str(tree),
                            "--role", args.role, "--arch", args.arch or arches[0]]
            if args.engine:
                promote_args.extend(["--engine", args.engine])
            if run_promote(parser.parse_args(promote_args)):
                raise ValueError(f"promotion failed; validated model and reproducible collection remain at {output}")
        print(f"Generated {'installable' if args.no_promote else 'installed'} UHD: {output / 'model'}")
        return 0
    except (OSError, TypeError, ValueError, KeyError, PromoteError) as error:
        logger.error("%s", error)
        return 1
    finally:
        if stage is not None:
            shutil.rmtree(stage)
