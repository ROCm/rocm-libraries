# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`uhd_gen rewrite-signature`: make an installed UHD tolerate absent optional features.

A model trained before signatures read optional features through value_or_default fails on
any valid graph that leaves one unbound (RFC 0019 §6.2). Wrapping those references changes no
value on a graph that binds them, so the trees, weights and every prediction the model could
already make are kept; only the signature, its `features_hash`, the hash the artifact embeds
and the artifact's digest are restated. Before writing anything, the rewrite proves both halves
through the runtime's own evaluator: identical features on rows that bind every reference, and
an evaluable row where each wrapped reference is absent, which the old signature refused.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import random
import sys
from pathlib import Path

import flatbuffers
import numpy as np

from .artifact import GBDT_MODEL_IDENTIFIER, LINEAR_MODEL_IDENTIFIER
from .features import compute_features_hash, evaluate_feature_maps, signature_references
from .optional import require_safe_signature, safe_signature
from .promote import PromoteError, _score_metric, _serialize_json, _verify_artifact
from .provenance import ROLES

logger = logging.getLogger(__name__)

#: The flatbuffers root an adapter's artifact holds, and its file identifier. An adapter not
#: listed embeds no features_hash in its artifact.
_ARTIFACT_ROOTS = {
    "tree_data": ("GbdtModel", GBDT_MODEL_IDENTIFIER),
    "linear_data": ("LinearModel", LINEAR_MODEL_IDENTIFIER),
}


def add_rewrite_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "descriptors",
        nargs="+",
        type=Path,
        help="installed heuristic.uhd.json files; each artifact beside its descriptor",
    )
    parser.add_argument(
        "--role",
        choices=ROLES,
        default=None,
        help="the role each model is bound to; default: the role directory in its path",
    )
    parser.add_argument(
        "--feature-evaluator",
        default=None,
        help="hipdnn_uhd_features to evaluate and hash with (default: as for train)",
    )
    parser.add_argument(
        "--proof-rows",
        type=int,
        default=64,
        help="random rows binding every reference, on which old and new features must agree",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="prove and report, writing nothing",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="also write the JSON report here",
    )


def _role_of(path: Path, requested: str | None) -> str:
    if requested is not None:
        return requested
    found = [part for part in path.parts if part in ROLES]
    if len(found) != 1:
        raise ValueError(
            f"{path}: name its role with --role; its path names {found or 'none'}"
        )
    return found[0]


def _bound_rows(
    references, encoding: dict, count: int, rng: random.Random
) -> list[dict]:
    """Rows binding every reference: an encoded categorical takes one of its strings, any
    other reference a positive integer, which every operator a signature uses accepts.
    """
    rows = []
    for _ in range(count):
        row = {}
        for reference in references:
            codes = encoding.get(reference)
            row[reference[1:]] = (
                rng.choice(sorted(codes)) if codes else rng.randint(1, 4096)
            )
        rows.append(row)
    return rows


def _same(left, right) -> bool:
    """Whether two unpacked flatbuffers objects hold the same values."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left, right = np.asarray(left), np.asarray(right)
        return left.shape == right.shape and bool(
            np.array_equal(left, right, equal_nan=left.dtype.kind == "f")
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(map(_same, left, right))
    if hasattr(left, "__dict__") and hasattr(right, "__dict__"):
        return type(left) is type(right) and _same(vars(left), vars(right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same(left[k], right[k]) for k in left
        )
    return left == right


def restamp_artifact(
    data: bytes, adapter: str, features_hash: str
) -> tuple[bytes, bool]:
    """`data` embedding `features_hash` instead, every other value unchanged (checked), and
    whether repacking the unchanged model reproduced `data` byte for byte."""
    root_name, identifier = _ARTIFACT_ROOTS[adapter]
    module = importlib.import_module(f"hipdnn_flatbuffers_sdk.data_objects.{root_name}")
    root_type, object_type = getattr(module, root_name), getattr(
        module, root_name + "T"
    )

    def unpack(buffer: bytes):
        return object_type.InitFromObj(root_type.GetRootAs(bytearray(buffer), 0))

    def pack(model) -> bytes:
        builder = flatbuffers.Builder(len(data) + 1024)
        builder.Finish(model.Pack(builder), file_identifier=identifier)
        return bytes(builder.Output())

    model = unpack(data)
    reproduced = pack(model) == data
    model.featuresHash = features_hash
    restamped = pack(model)
    check = unpack(restamped)
    recorded = check.featuresHash
    if isinstance(recorded, bytes):
        recorded = recorded.decode("utf-8")
    if recorded != features_hash:
        raise ValueError("the repacked artifact does not record the new features_hash")
    original = unpack(data)
    check.featuresHash = original.featuresHash = None
    if not _same(check, original):
        raise ValueError("repacking changed a value other than features_hash")
    return restamped, reproduced


def _prove(
    signature: list,
    rewritten: list,
    wrapped: list[dict],
    encoding: dict,
    args: argparse.Namespace,
) -> dict:
    """Old and new features agree wherever every reference is bound; each wrapped reference
    absent leaves every row evaluable. How many of those rows the old signature refused is
    reported, not required: a bare read under a conditional may short-circuit on some rows,
    which is no proof it always does."""
    references = signature_references(signature)
    rows = _bound_rows(references, encoding, args.proof_rows, random.Random(args.seed))
    _, old = evaluate_feature_maps(rows, signature, encoding, args.feature_evaluator)
    _, new = evaluate_feature_maps(rows, rewritten, encoding, args.feature_evaluator)
    if old != new:
        differing = next(index for index, (a, b) in enumerate(zip(old, new)) if a != b)
        raise ValueError(
            f"the rewritten signature changes features on row {rows[differing]}: "
            f"{old[differing]} -> {new[differing]}"
        )
    absent_rows = {}
    for item in wrapped:
        reference = item["reference"]
        absent = [{**row, reference[1:]: None} for row in rows]
        # Raises when any row is unscorable: the rewrite must score them all.
        evaluate_feature_maps(absent, rewritten, encoding, args.feature_evaluator)
        refused = 0
        for row in absent:
            try:
                evaluate_feature_maps(
                    [row], signature, encoding, args.feature_evaluator
                )
            except ValueError:
                refused += 1
        absent_rows[reference] = refused
    return {
        "rows_compared": len(rows),
        "features_identical": True,
        "absent_now_evaluable": [item["reference"] for item in wrapped],
        # Of `rows_compared` rows lacking each reference, how many the old signature refused.
        "old_refused_rows_when_absent": absent_rows,
    }


def rewrite_descriptor(path: Path, args: argparse.Namespace) -> dict:
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    role = _role_of(path, args.role)
    adapter = descriptor.get("adapter")
    signature = descriptor.get("features_signature") or []
    metric = _score_metric(descriptor)
    report = {"descriptor": str(path), "role": role, "metric": metric}
    if not signature:
        return {**report, "status": "no_features"}
    rewritten, wrapped = safe_signature(signature, metric, args.feature_evaluator)
    if not wrapped:
        return {**report, "status": "already_safe"}
    encoding = descriptor.get("categorical_encoding") or {}
    categorical = sorted(
        item["reference"] for item in wrapped if item["reference"] in encoding
    )
    if categorical:
        raise ValueError(
            f"{path}: an encoded categorical cannot take a default: {categorical}"
        )
    proof = _prove(signature, rewritten, wrapped, encoding, args)
    features_hash = compute_features_hash(rewritten, encoding, args.feature_evaluator)
    updated = {
        **descriptor,
        "features_signature": rewritten,
        "features_hash": features_hash,
    }
    report.update(
        status="rewritten",
        wrapped=wrapped,
        features_hash={"old": descriptor["features_hash"], "new": features_hash},
        proof=proof,
    )
    artifact = None
    restamped = None
    if adapter in _ARTIFACT_ROOTS:
        artifact = path.parent / descriptor[adapter]["artifact"]
        data = artifact.read_bytes()
        restamped, reproduced = restamp_artifact(data, adapter, features_hash)
        digest = hashlib.sha256(restamped).hexdigest()
        updated[adapter] = {**descriptor[adapter], "hash": digest}
        report["artifact"] = {
            "path": str(artifact),
            "old_sha256": hashlib.sha256(data).hexdigest(),
            "new_sha256": digest,
            "repack_reproduces_original": reproduced,
        }
    elif "features_hash" in json.dumps(descriptor.get(adapter, {})):
        raise ValueError(
            f"{path}: adapter {adapter} has no rewrite rule for its artifact"
        )
    require_safe_signature(rewritten, metric, str(path), args.feature_evaluator)
    if args.dry_run:
        return report
    original = path.read_bytes()
    original_artifact = artifact.read_bytes() if artifact is not None else None
    try:
        if artifact is not None:
            artifact.write_bytes(restamped)
        path.write_bytes(_serialize_json(updated))
        # The loader's own checks of the artifact against the descriptor.
        declared = _verify_artifact(updated, artifact, role)
        if artifact is not None and declared != updated[adapter]["hash"]:
            raise ValueError(
                "the artifact's digest is not the one the descriptor records"
            )
    except (ValueError, PromoteError, OSError):
        path.write_bytes(original)
        if artifact is not None:
            artifact.write_bytes(original_artifact)
        raise
    return report


def run_rewrite(args: argparse.Namespace) -> int:
    reports = []
    failed = False
    for path in args.descriptors:
        try:
            reports.append(rewrite_descriptor(path, args))
        except (ValueError, PromoteError, OSError, KeyError) as error:
            failed = True
            reports.append(
                {"descriptor": str(path), "status": "failed", "error": str(error)}
            )
    text = json.dumps(reports, indent=2)
    print(text)
    if args.report is not None:
        args.report.write_text(text + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_rewrite_arguments(parser)
    sys.exit(run_rewrite(parser.parse_args()))
