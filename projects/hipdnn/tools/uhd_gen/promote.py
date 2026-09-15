#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Plan a role/architecture-scoped model installation before writing any files."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .provenance import (
    ROLES, ProvenanceError, compare_provenance, descriptor_id, load_descriptor_tree,
    provenance_for_engine, revision, select_engine, validate_provenance,
)
from .immediate import ROLE, validate_model

logger = logging.getLogger(__name__)
UHD_SUFFIX = ".uhd.json"
_ARCH = re.compile(r"^gfx[a-z0-9_-]+$")
_ADAPTERS = ("static_order", "native", "tree_data", "table", "onnx", "custom_library")


class PromoteError(ValueError):
    """A planning refusal; no installed file has been touched."""


@dataclass
class PromotePlan:
    descriptor_path: Path
    descriptor_id: str
    artifact_path: Path | None
    ued_path: Path
    ued_document: dict
    engine_name: str
    old_heuristic: str | None
    role: str
    arch: str
    destination_descriptor: Path
    descriptor_document: dict
    copies: list[tuple[Path, Path]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dropped_knobs: list[str] = field(default_factory=list)
    write_descriptor: bool = True


def add_promote_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", required=True, help="Directory containing one trained *.uhd.json")
    parser.add_argument("--descriptor-tree", required=True, help="Destination descriptor tree")
    parser.add_argument("--engine", help="UED name or UUID that owns the role being promoted")
    parser.add_argument("--role", choices=ROLES, default="sort_kernel_catalog")
    parser.add_argument("--arch", help="Target gfx architecture or explicit 'default'; otherwise infer a unique training_arches value")
    parser.add_argument("--remove-knob", action="append", default=[], dest="remove_knobs",
                        help="Explicitly remove an authored knob; requires a model trained against the prospective major revision")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing")


def run_promote(args: argparse.Namespace) -> int:
    try:
        plan = build_plan(Path(args.model_dir), Path(args.descriptor_tree), args.engine,
                          role=args.role, arch=args.arch, remove_knobs=args.remove_knobs)
        for warning in plan.warnings:
            logger.warning("%s", warning)
        if not args.dry_run:
            _apply(plan)
        _report(plan, dry_run=args.dry_run)
    except (PromoteError, OSError) as error:
        logger.error("%s", error)
        return 1
    return 0


def build_plan(model_dir: Path, descriptor_tree: Path, engine: str | None = None, *,
               role: str = "sort_kernel_catalog", arch: str | None = None,
               remove_knobs: tuple[str, ...] | list[str] = ()) -> PromotePlan:
    """Resolve all dependencies, ownership and destination collisions without writes."""
    try:
        return _build_plan(Path(model_dir), Path(descriptor_tree), engine, role, arch, remove_knobs)
    except ValueError as error:
        raise PromoteError(str(error)) from error


def _build_plan(model_dir, descriptor_tree, engine, role, arch, remove_knobs):
    if role not in ROLES:
        raise PromoteError(f"unknown heuristic role {role!r}")
    descriptor_path = _find_descriptor(model_dir)
    _contained(descriptor_path, model_dir, "source descriptor")
    descriptor = _load_json(descriptor_path, "UHD")
    _validate_descriptor(descriptor, descriptor_path)
    identity = descriptor_id(descriptor.get("id"), str(descriptor_path))
    manifest_path = model_dir / "train_manifest.json"
    manifest = _load_json(manifest_path, "training manifest") if manifest_path.is_file() else {}
    arches = manifest.get("training_arches", [])
    if not isinstance(arches, list) or any(not isinstance(item, str) or not _ARCH.fullmatch(item) for item in arches):
        raise PromoteError("training_arches must be an array of bare gfx targets")
    if arch is None:
        if len(set(arches)) != 1:
            raise PromoteError("pass --arch (or explicit --arch default): training_arches does not identify one target")
        arch = arches[0]
    if arch != "default" and (not isinstance(arch, str) or not _ARCH.fullmatch(arch)):
        raise PromoteError(f"invalid target architecture {arch!r}")
    if arches and arch != "default" and arch not in arches:
        raise PromoteError(f"target {arch} is not in training_arches {arches}")
    if "trained_against" in manifest and validate_provenance(manifest["trained_against"]) != validate_provenance(descriptor.get("trained_against")):
        raise PromoteError("training manifest provenance differs from the UHD's trained_against")
    recorded_role = manifest.get("role")
    if recorded_role is not None and recorded_role != role:
        raise PromoteError("incoming model was trained for another role")
    provenance = descriptor.get("trained_against", {})
    if role == ROLE:
        validate_model(descriptor)

    index = load_descriptor_tree(descriptor_tree)
    if any(identity in entries for entries in index.values()):
        raise PromoteError(f"incoming UHD identity {identity} conflicts with a dependency descriptor")
    ued_path, original_ued = select_engine(index, engine)
    engine_name = str(original_ued.get("name", ""))
    destination_dir = ued_path.parent / "heuristics" / original_ued["id"] / role / arch
    ueds = list(index["ued"].values())
    references = [ref for path, doc in ueds for ref in _role_references(path, doc)]
    ued = copy.deepcopy(original_ued)
    old = ued.get(role, {}).get(arch)
    artifact_path, artifact_key = _artifact_path(descriptor, descriptor_path, model_dir)
    installed_descriptor = copy.deepcopy(descriptor)
    destination_descriptor = destination_dir / descriptor_path.name
    _contained(destination_descriptor, descriptor_tree, "destination descriptor")
    plan = PromotePlan(descriptor_path, identity, artifact_path, ued_path, ued,
                       engine_name, old, role, arch,
                       destination_descriptor, installed_descriptor)
    if remove_knobs:
        exposed = ued.get("knobs", [])
        if not isinstance(exposed, list) or any(not isinstance(item, str) for item in exposed):
            raise PromoteError("UED knobs must be an array of strings")
        unknown = set(remove_knobs) - set(exposed)
        if unknown:
            raise PromoteError(f"cannot remove unauthored knobs: {sorted(unknown)}")
        plan.dropped_knobs = sorted(set(remove_knobs))
        still_used = set(plan.dropped_knobs) & _kernel_references(descriptor.get("features_signature", []))
        if still_used:
            raise PromoteError(f"incoming model still consumes removed knobs: {sorted(still_used)}")
        major, _ = revision(ued.get("revision", "1.0"), "UED")
        ued["revision"] = f"{major + 1}.0"
        ued["knobs"] = [knob for knob in exposed if knob not in plan.dropped_knobs]

    actual = provenance_for_engine(index, ued, arch)
    if actual is not None and "trained_against" in descriptor:
        try:
            compare_provenance(provenance, actual)
        except ProvenanceError as error:
            suffix = "; retrain against the intended revised UED before removing knobs" if remove_knobs else ""
            raise PromoteError(f"{error}{suffix}") from error
    elif remove_knobs:
        raise PromoteError("explicit knob removal requires training provenance for the intended revised UED; retrain first")

    installed = []
    installed_ids = {}
    for path in sorted(descriptor_tree.rglob(f"*{UHD_SUFFIX}")):
        if path.name == UHD_SUFFIX:
            continue
        doc = _load_json(path, "installed UHD")
        current_id = descriptor_id(doc.get("id"), str(path))
        if current_id in installed_ids:
            raise PromoteError(f"duplicate installed UHD identity {current_id}: {installed_ids[current_id]} and {path}")
        installed_ids[current_id] = path
        installed.append((path, doc, current_id))
    target = (ued_path.resolve(), role, arch)
    others = [(path, slot, target_arch, ref) for path, slot, target_arch, ref in references
              if (path.resolve(), slot, target_arch) != target]
    # A UUID shared by another role or architecture is immutable for this promotion,
    # even if the destination filename happens to be the model it already references.
    descriptor_changes = not _same_file(descriptor_path, destination_descriptor)
    destination_artifact = None
    if artifact_path is not None:
        destination_artifact = destination_dir / artifact_path.name
        _contained(destination_artifact, descriptor_tree, "destination artifact")
        installed_descriptor[descriptor["adapter"]][artifact_key] = artifact_path.name
        if destination_artifact.name.endswith(tuple(
            f".{kind}.json" for kind in ("ued", "umd", "kmd", "kdp", "ukd", "udd", "uhd")
        )):
            raise PromoteError("artifact would overwrite or masquerade as a descriptor")
        if destination_artifact.exists() and not destination_artifact.is_file():
            raise PromoteError(f"destination artifact is not a regular file: {destination_artifact}")
        if not _same_file(artifact_path, destination_artifact):
            plan.copies.append((artifact_path, destination_artifact))
    descriptor_changes = descriptor_changes or installed_descriptor != descriptor
    for path, doc, current_id in installed:
        same_destination = _same_file(path, destination_descriptor)
        if current_id == identity and not same_destination:
            raise PromoteError(f"incoming UHD id {identity} is already installed at {path}; refusing duplicate identity")
        if not same_destination:
            continue
        holders = [(p, r, a) for p, r, a, ref in others if ref in (current_id, identity)]
        if holders and (descriptor_changes or plan.copies):
            raise PromoteError(f"would clobber UHD {current_id} used by other role/arch entries: {holders}")
        if current_id != identity and current_id != old:
            plan.warnings.append(f"OVERWRITING unreferenced UHD {current_id} at {path}")
    if any(ref == identity for _, _, _, ref in others) and (descriptor_changes or plan.copies):
        raise PromoteError(f"incoming UHD id {identity} is owned by another role/arch entry")
    if destination_artifact is not None and plan.copies:
        for path, doc, current_id in installed:
            if _same_file(path, destination_descriptor):
                continue
            adapter = doc.get("adapter")
            body = doc.get(adapter) if isinstance(adapter, str) else None
            key = "library" if adapter == "custom_library" else "artifact"
            payload = body.get(key) if isinstance(body, dict) else None
            if isinstance(payload, str) and _same_file(path.parent / payload, destination_artifact):
                raise PromoteError(f"artifact collision: {destination_artifact} belongs to {path} ({current_id})")
        if destination_artifact.exists() and not destination_descriptor.exists():
            raise PromoteError(f"unowned destination artifact already exists: {destination_artifact}")
    if remove_knobs:
        for path, other_role, other_arch, ref in others:
            if path.resolve() != ued_path.resolve():
                continue
            model_path = installed_ids.get(ref)
            if model_path is None:
                raise PromoteError(f"cannot prove knob removal safe: missing model {ref} for {other_role}/{other_arch}")
            model = _load_json(model_path, "other role UHD")
            if model.get("features_signature"):
                try:
                    compare_provenance(model.get("trained_against", {}), provenance_for_engine(index, ued, other_arch))
                except ProvenanceError as error:
                    raise PromoteError(f"knob removal would invalidate {other_role}/{other_arch}: {error}") from error
            if set(plan.dropped_knobs) & _kernel_references(model.get("features_signature", [])):
                raise PromoteError(f"knob removal would affect {other_role}/{other_arch}")
    ued.setdefault(role, {})[arch] = identity
    plan.write_descriptor = descriptor_changes
    return plan


def _role_references(path: Path, document: dict):
    if "heuristic" in document:
        raise PromoteError(f"{path}: legacy heuristic field is not supported; use role/arch maps")
    for role in ROLES:
        if role not in document:
            continue
        entries = document[role]
        if not isinstance(entries, dict) or not entries:
            raise PromoteError(f"{path}.{role} must be a nonempty arch-to-UUID map")
        for arch, identity in entries.items():
            if arch != "default" and not _ARCH.fullmatch(arch):
                raise PromoteError(f"{path}.{role}: invalid architecture {arch!r}")
            yield path, role, arch, descriptor_id(identity, f"{path}.{role}.{arch}")


def _validate_descriptor(document: dict, path: Path) -> None:
    descriptor_id(document.get("id"), str(path))
    known = {"version", "id", "name", "adapter", "objective", "score",
             "features_signature", "features_hash", "categorical_encoding",
             "trained_against", *_ADAPTERS}
    if set(document) - known:
        raise PromoteError(f"{path}: unknown UHD fields {sorted(set(document) - known)}")
    if document.get("version") != "1.0":
        raise PromoteError(f"{path}: unsupported UHD file-format version {document.get('version')!r}")
    if not isinstance(document.get("name"), str) or not document["name"]:
        raise PromoteError(f"{path}: missing required name")
    adapter = document.get("adapter")
    if adapter not in _ADAPTERS:
        raise PromoteError(f"{path}: unsupported adapter {adapter!r}")
    bodies = [key for key in _ADAPTERS if key in document]
    if bodies != [adapter] or not isinstance(document[adapter], dict):
        raise PromoteError(f"{path}: requires exactly one body matching adapter {adapter}")
    if (adapter != "static_order" or "objective" in document) and document.get("objective") not in ("min", "max"):
        raise PromoteError(f"{path}: scorer requires objective min or max")
    signature = document.get("features_signature")
    if "features_signature" in document or adapter in ("tree_data", "table", "onnx"):
        if not isinstance(signature, list) or not signature:
            raise PromoteError(f"{path}: requires nonempty features_signature")
        for entry in signature:
            if not ((isinstance(entry, str) and entry.startswith("$") and len(entry) > 1)
                    or (isinstance(entry, dict) and len(entry) == 1)):
                raise PromoteError(f"{path}: features_signature requires bare references or inline expressions")
        if "features_hash" not in document:
            raise PromoteError(f"{path}: missing features_hash")
        validate_provenance(document.get("trained_against"))
    elif "trained_against" in document:
        validate_provenance(document["trained_against"])
    if adapter in ("native", "custom_library"):
        symbol = document[adapter].get("symbol")
        if not isinstance(symbol, str) or not symbol:
            raise PromoteError(f"{path}: {adapter}.symbol is required")
    if "features_hash" in document and (
        not isinstance(document["features_hash"], str)
        or not re.fullmatch(r"sha256:[0-9a-f]{16}", document["features_hash"])
    ):
        raise PromoteError(f"{path}: invalid features_hash")
    if "categorical_encoding" in document:
        encoding = document["categorical_encoding"]
        if not isinstance(encoding, dict) or any(
            not isinstance(codes, dict) or not codes
            or any(type(code) is not int for code in codes.values())
            for codes in encoding.values()
        ):
            raise PromoteError(f"{path}: categorical_encoding requires value-to-integer maps")
    if "score" in document:
        score = document["score"]
        if not isinstance(score, dict) or set(score) - {"units", "calibrated", "transform"}:
            raise PromoteError(f"{path}: invalid score header")
        if any(key in score and (not isinstance(score[key], str) or not score[key])
               for key in ("units", "transform")):
            raise PromoteError(f"{path}: score units/transform must be nonempty strings")
        if "calibrated" in score and not isinstance(score["calibrated"], bool):
            raise PromoteError(f"{path}: score.calibrated must be a boolean")
    body = document[adapter]
    allowed = ({"order"} if adapter == "static_order" else {"symbol"} if adapter == "native"
               else {"library", "symbol", "hash", "config"} if adapter == "custom_library"
               else {"artifact", "hash"})
    if set(body) - allowed:
        raise PromoteError(f"{path}: unknown {adapter} body fields")
    if "hash" in body and (not isinstance(body["hash"], str) or not body["hash"]):
        raise PromoteError(f"{path}: model hash must be a nonempty string")
    if adapter == "static_order" and "order" in body and (
        not isinstance(body["order"], list) or any(not isinstance(item, str) for item in body["order"])
    ):
        raise PromoteError(f"{path}: static_order.order must be an array of strings")
    if adapter == "custom_library" and "config" in body and not isinstance(body["config"], dict):
        raise PromoteError(f"{path}: custom_library.config must be an object")


def _artifact_path(descriptor, descriptor_path, model_dir):
    adapter = descriptor["adapter"]
    if adapter in ("static_order", "native"):
        return None, None
    key = "library" if adapter == "custom_library" else "artifact"
    payload = descriptor[adapter].get(key)
    if not isinstance(payload, str) or not payload:
        raise PromoteError(f"{descriptor_path}: missing {adapter}.{key}")
    resolved = (descriptor_path.parent / payload).resolve()
    _contained(resolved, model_dir, "model artifact")
    if not resolved.is_file():
        raise PromoteError(f"model artifact does not exist: {resolved}")
    return resolved, key


def _contained(path, root, what):
    if not path.resolve().is_relative_to(root.resolve()):
        raise PromoteError(f"{what} escapes root {root}: {path}")


def _kernel_references(value):
    if isinstance(value, str):
        return {value[len("$kernel."):]} if value.startswith("$kernel.") else set()
    if isinstance(value, (dict, list, tuple)):
        children = value.values() if isinstance(value, dict) else value
        return set().union(*(_kernel_references(item) for item in children))
    return set()


def _find_descriptor(model_dir):
    if not model_dir.is_dir():
        raise PromoteError(f"--model-dir {model_dir} is not a directory")
    paths = sorted(path for path in model_dir.glob(f"*{UHD_SUFFIX}") if path.name != UHD_SUFFIX)
    if len(paths) != 1:
        raise PromoteError(f"{model_dir} must contain exactly one *{UHD_SUFFIX}; found {len(paths)}")
    return paths[0]


def _load_json(path, what):
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PromoteError(f"cannot read {what} {path}: {error}") from error
    if not isinstance(document, dict):
        raise PromoteError(f"{what} {path} is not an object")
    return document


def _same_file(left, right):
    return left.resolve() == right.resolve() or (left.exists() and right.exists() and left.samefile(right))


def _write_json(path, document):
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _apply(plan: PromotePlan) -> None:
    """Execute only the operations approved by build_plan; never restamp provenance."""
    plan.destination_descriptor.parent.mkdir(parents=True, exist_ok=True)
    for source, destination in plan.copies:
        shutil.copy2(source, destination)
    if plan.write_descriptor:
        _write_json(plan.destination_descriptor, plan.descriptor_document)
    _write_json(plan.ued_path, plan.ued_document)


def _report(plan: PromotePlan, dry_run: bool) -> None:
    print("UHD promotion plan (dry run, nothing written)" if dry_run else "UHD promoted")
    print(f"  engine: {plan.engine_name}\n  binding: {plan.ued_path}\n  role/arch: {plan.role}/{plan.arch}")
    print(f"  heuristic was: {plan.old_heuristic or '(none)'}\n  heuristic now: {plan.descriptor_id}")
    for source, destination in plan.copies:
        print(f"  {'would copy' if dry_run else 'copy'}: {source} -> {destination}")
    if plan.write_descriptor:
        print(f"  descriptor: {plan.destination_descriptor}")
    for knob in plan.dropped_knobs:
        print(f"  removed knob: {knob}; UED revision {plan.ued_document['revision']}")
