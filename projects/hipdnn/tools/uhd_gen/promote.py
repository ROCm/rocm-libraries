#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Plan a role/architecture-scoped model installation before writing any files."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .artifact import (
    artifact_digest,
    is_contained_relative_path,
    is_grouped_tree,
    verify_tree_artifact,
)
from .correctness import REASON, VERDICT, numerical_reason, numerical_verdict
from .features import (
    compute_features_hash,
    evaluator_feature_semantics_revision,
    kernel_axes,
    require_admissible_kernel_axes,
)
from .provenance import (
    ROLES,
    ProvenanceError,
    compare_provenance,
    descriptor_id,
    foreign_matcher_ids,
    load_descriptor_tree,
    provenance_for_engine,
    require_feature_semantics,
    revision,
    select_engine,
    validate_provenance,
)
from .immediate import ROLE, validate_model
from .ranking_metrics import RANKING_METRICS

logger = logging.getLogger(__name__)
UHD_SUFFIX = ".uhd.json"
_ARCH = re.compile(r"^gfx[a-z0-9_-]+$")
_ADAPTERS = ("static_order", "native", "tree_data", "table", "onnx", "custom_library")
#: A model `hash`: the SHA-256 of the file it names, as UhdParser and the adapters spell it.
_SHA256 = re.compile(r"[0-9a-f]{64}")
#: Roles whose arch key binds a list of UHDs, one per ranking metric (RFC 0019 §3.1).
_METRIC_ROLES = ("sort_kernel_catalog", ROLE)


class PromoteError(ValueError):
    """A planning refusal; no installed file has been touched."""


@dataclass
class PromotePlan:
    descriptor_path: Path
    descriptor_id: str
    artifact_path: Path | None
    #: The UED that binds this model and its rewritten document; both None for an engine
    #: with no UED, which binds by a UUID declared in provider code (RFC 0019 §4.1).
    ued_path: Path | None
    ued_document: dict | None
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
    #: The incoming UHD's `score.metric`; None for a metric-less ranker.
    metric: str | None = None


def add_promote_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-dir", required=True, help="Directory containing one trained *.uhd.json"
    )
    parser.add_argument(
        "--descriptor-tree", required=True, help="Destination descriptor tree"
    )
    parser.add_argument(
        "--engine", help="UED name or UUID that owns the role being promoted"
    )
    parser.add_argument("--role", choices=ROLES, default="sort_kernel_catalog")
    parser.add_argument(
        "--arch",
        help="Target gfx architecture or explicit 'default'; otherwise infer a unique training_arches value",
    )
    parser.add_argument(
        "--remove-knob",
        action="append",
        default=[],
        dest="remove_knobs",
        help="Explicitly remove an authored knob; requires a model trained against the prospective major revision",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and report without writing"
    )
    parser.add_argument(
        "--corpus",
        help="The corpus.json this model was trained from; "
        "defaults to the one generate stages beside --model-dir",
    )
    parser.add_argument(
        "--uhd-id",
        action="append",
        default=[],
        dest="uhd_ids",
        metavar="METRIC=UUID",
        help="The UHD id the engine reads for a metric (repeatable; a bare UUID "
        "names the one model being promoted). The model must already carry "
        "it: promote never renames a model. Required for an engine with no "
        "UED unless the collection recorded the id its provider declares.",
    )
    parser.add_argument(
        "--feature-evaluator",
        help="Path to the shared hipdnn_uhd_features executable of the build the "
        "model is promoted for; its feature-semantics revision must be the "
        "one the model recorded",
    )


def parse_uhd_ids(values: list[str] | None) -> dict[str | None, str]:
    """`--uhd-id` values as metric -> canonical UUID; a bare UUID is keyed by None.

    A bare UUID names exactly one model, so it cannot be mixed with metric-keyed ones.
    """
    ids: dict[str | None, str] = {}
    for value in values or []:
        metric, separator, text = value.rpartition("=")
        key = metric if separator else None
        if key is not None and key not in RANKING_METRICS:
            raise ValueError(
                f"--uhd-id {value!r}: {key!r} is not a registered ranking metric "
                f"({', '.join(RANKING_METRICS)})"
            )
        if key in ids:
            raise ValueError(
                f"--uhd-id names {'a bare id' if key is None else key} twice"
            )
        ids[key] = descriptor_id(text, f"--uhd-id {value!r}")
    if None in ids and len(ids) > 1:
        raise ValueError(
            "a bare --uhd-id names one model; name each metric's id as METRIC=UUID instead"
        )
    return ids


def run_promote(args: argparse.Namespace) -> int:
    try:
        plan = build_plan(
            Path(args.model_dir),
            Path(args.descriptor_tree),
            args.engine,
            role=args.role,
            arch=args.arch,
            remove_knobs=args.remove_knobs,
            corpus=Path(args.corpus) if args.corpus else None,
            uhd_ids=parse_uhd_ids(args.uhd_ids),
            feature_evaluator=args.feature_evaluator,
        )
        for warning in plan.warnings:
            logger.warning("%s", warning)
        if not args.dry_run:
            _apply(plan)
        _report(plan, dry_run=args.dry_run)
    except (ValueError, OSError) as error:
        logger.error("%s", error)
        return 1
    return 0


def build_plan(
    model_dir: Path,
    descriptor_tree: Path,
    engine: str | None = None,
    *,
    role: str = "sort_kernel_catalog",
    arch: str | None = None,
    remove_knobs: tuple[str, ...] | list[str] = (),
    corpus: Path | None = None,
    uhd_ids: dict[str | None, str] | None = None,
    feature_evaluator: str | None = None,
) -> PromotePlan:
    """Resolve all dependencies, ownership and destination collisions without writes."""
    try:
        return _build_plan(
            Path(model_dir),
            Path(descriptor_tree),
            engine,
            role,
            arch,
            remove_knobs,
            corpus,
            uhd_ids or {},
            feature_evaluator,
        )
    except ValueError as error:
        raise PromoteError(str(error)) from error


def _corpus_path(model_dir: Path, corpus: Path | None) -> Path | None:
    """`--corpus`, else the `corpus.json` that `generate` stages beside the model, else None."""
    if corpus is not None:
        if not corpus.is_file():
            raise PromoteError(f"--corpus is not a readable file: {corpus}")
        return corpus
    for sibling in (model_dir / "corpus.json", model_dir.parent / "corpus.json"):
        if sibling.is_file():
            return sibling
    return None


def _correctness_gate(model_dir: Path, corpus: Path | None) -> list[str]:
    """Refuse emission while any candidate carries an invalid marker (RFC 0019 §13.4).

    The model can only demote a kernel, not exclude it, so this is the gate. Undecided
    verdicts are not refused but come back as warnings.
    """
    path = _corpus_path(model_dir, corpus)
    if path is None:
        return [
            "no corpus.json beside the model, so RFC 0019 §13.2's correctness markers "
            "could not be read; this promotion is not gated on them. Pass --corpus to "
            "point at the corpus this model was trained from."
        ]
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PromoteError(f"cannot read training corpus {path}: {error}") from error
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise PromoteError(f"{path}: the corpus must be an array of measured rows")
    verdicts = [numerical_verdict(row.get(VERDICT)) for row in rows]
    invalid = [row for row, verdict in zip(rows, verdicts) if verdict is False]
    if invalid:
        named = "\n".join(
            f"  {row.get('kernel', '<unnamed candidate>')} on {row.get('benchmark', '<unnamed problem>')}: "
            f"{numerical_reason(row.get(REASON)) or 'no reason recorded'}"
            for row in invalid[:5]
        )
        more = f"\n  ... and {len(invalid) - 5} more" if len(invalid) > 5 else ""
        raise PromoteError(
            f"{len(invalid)} candidate(s) in {path} carry an unresolved RFC 0019 §13.2 "
            f"invalid marker, so package emission is refused (§13.4):\n{named}{more}\n"
            "The model cannot fix this: a ranking demotes a kernel, and every path that "
            "does not consult the model still selects it. Narrow the pack's UMD so the "
            "kernel is no longer applicable for those problems, or withdraw the UKD, then "
            "regenerate."
        )
    unchecked = verdicts.count(None)
    if unchecked:
        return [
            f"{unchecked} of {len(rows)} corpus row(s) carry no decided correctness "
            f"verdict (RFC 0019 Open Question 19 leaves the per-op reference open), so "
            f"their timings were trained on without being shown correct"
        ]
    return []


def _opaque_plan(
    descriptor_path,
    descriptor,
    installed_descriptor,
    identity,
    artifact_path,
    artifact_key,
    descriptor_tree,
    engine,
    role,
    arch,
    remove_knobs,
    uhd_ids,
    manifest,
):
    """Install a model for an engine that owns no descriptor set.

    The model's id must be the UUID the provider declares for its metric. An incumbent
    under that UUID serving the same metric and arches is replaced where it lies; any
    other incumbent is refused.
    """
    if remove_knobs:
        raise PromoteError(
            "knob removal applies to authored UED knobs; this engine has no UED"
        )
    if role != ROLE:
        raise PromoteError(
            f"an engine with no descriptor set has only the {ROLE} role; {role} ranks a "
            "catalog it does not own"
        )
    engine_name = str(engine or descriptor.get("engine") or "")
    if not engine_name:
        raise PromoteError(
            "pass --engine: the canonical engine name the provider declares"
        )
    metric = _score_metric(descriptor)
    declared = _requested_identity(uhd_ids, metric)
    binding = manifest.get("binding")
    recorded = binding.get("uhd_id") if isinstance(binding, dict) else None
    if recorded is not None:
        # The id the provider declared for this metric at collection time.
        recorded = descriptor_id(recorded, "train_manifest.json binding.uhd_id")
        if declared is not None and declared != recorded:
            raise PromoteError(
                f"--uhd-id names {declared} for metric {metric}, but the engine declared "
                f"{recorded} when this corpus was collected; the engine reads only its own"
            )
        declared = recorded
    if declared is None:
        raise PromoteError(
            f"{engine_name} owns no UED, so it reads only the UHD id its provider declares "
            f"for each metric, and this model's collection recorded none. Pass --uhd-id "
            f"{metric}=<uuid> naming the id the provider declares for {metric}; installing "
            "under any other id ships a model the engine never reads."
        )
    if declared != identity:
        raise PromoteError(
            f"model {identity} is not the id the provider declares for metric {metric} "
            f"({declared}); retrain with --uhd-id {metric}={declared} -- promote never "
            "renames a model"
        )
    # One directory per metric: one UHD per role, arch and metric (RFC 0019 §3.1).
    destination_dir = (
        Path(descriptor_tree) / "heuristics" / _slug(engine_name) / role / arch / metric
    )
    destination_descriptor = destination_dir / descriptor_path.name
    _contained(destination_descriptor, descriptor_tree, "destination descriptor")
    plan = PromotePlan(
        descriptor_path,
        identity,
        artifact_path,
        None,
        None,
        engine_name,
        None,
        role,
        arch,
        destination_descriptor,
        installed_descriptor,
        metric=metric,
    )
    if artifact_path is not None:
        destination_artifact = destination_dir / artifact_path.name
        _contained(destination_artifact, descriptor_tree, "destination artifact")
        installed_descriptor[descriptor["adapter"]][artifact_key] = artifact_path.name
        if not _same_file(artifact_path, destination_artifact):
            plan.copies.append((artifact_path, destination_artifact))
    incumbents = []
    for path in sorted(Path(descriptor_tree).rglob(f"*{UHD_SUFFIX}")):
        if path.name == UHD_SUFFIX:
            continue
        installed = _load_json(path, "installed UHD")
        if descriptor_id(installed.get("id"), str(path)) == identity:
            incumbents.append((path, installed))
    if len(incumbents) > 1:
        raise PromoteError(
            f"duplicate installed UHD identity {identity}: "
            f"{', '.join(str(path) for path, _ in incumbents)}; the provider reads one "
            "model per id"
        )
    for path, installed in incumbents:
        if _same_file(path, destination_descriptor):
            # Promote's own layout for this slot: overwritten as planned.
            continue
        if _same_model(
            path, installed, installed_descriptor, artifact_path, artifact_key
        ):
            _reuse(plan, path)
            continue
        incumbent_artifact, payload = _same_slot_artifact(
            path,
            installed,
            descriptor["adapter"],
            metric,
            {arch} if arch != "default" else set(manifest.get("training_arches", [])),
            descriptor_tree,
        )
        # Overwrite the incumbent in place, keeping its artifact file name.
        plan.destination_descriptor = path
        installed_descriptor["tree_data"][artifact_key] = payload
        plan.copies[:] = (
            []
            if _same_file(artifact_path, incumbent_artifact)
            else [(artifact_path, incumbent_artifact)]
        )
        plan.warnings.append(
            f"replacing {identity} in place at {path}: the installed model serves the "
            f"same metric ({metric}) and architectures"
        )
    return plan


def _same_slot_artifact(
    path: Path,
    installed: dict,
    adapter: str,
    metric: str | None,
    served: set,
    descriptor_tree,
) -> tuple[Path, str]:
    """The incumbent's artifact and its name, once proven to be this slot's model.

    It must estimate `metric` and its recorded training arches must all be in `served`;
    anything else, including unreadable coverage, is refused.
    """

    def refuse(reason: str):
        raise PromoteError(
            f"UHD id {installed.get('id')} is already installed at {path} with different "
            f"content, and {reason}; one UUID names one model under every arch key that "
            "binds it (D2), so it is not replaced. Promote to the arch and metric that "
            "model serves to replace it there"
        )

    if _score_metric(installed) != metric:
        refuse(f"it estimates {_score_metric(installed) or 'no metric'}, not {metric}")
    body = installed.get("tree_data")
    payload = body.get("artifact") if isinstance(body, dict) else None
    if (
        adapter != "tree_data"
        or installed.get("adapter") != "tree_data"
        or not isinstance(payload, str)
    ):
        refuse(
            "the installed and incoming models are not both tree_data models whose "
            "training architectures can be read"
        )
    artifact = path.parent / payload
    _contained(artifact, descriptor_tree, "installed artifact")
    try:
        data = verify_tree_artifact(artifact, None)
    except (OSError, ValueError) as error:
        refuse(f"its artifact cannot be read for its training architectures: {error}")
    import uhd_gen  # noqa: F401  puts _generated/ on sys.path

    from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModel

    model = GbdtModel.GetRootAs(bytearray(data), 0)
    trained = {
        model.TrainingArches(index).decode("utf-8", "replace")
        for index in range(model.TrainingArchesLength())
    }
    if not trained or not trained <= served:
        refuse(
            f"it was trained for {sorted(trained) or 'unrecorded architectures'}, "
            f"which this promotion for {sorted(served) or 'no recorded architecture'} "
            "does not cover"
        )
    for other in sorted(Path(descriptor_tree).rglob(f"*{UHD_SUFFIX}")):
        if other.name == UHD_SUFFIX or _same_file(other, path):
            continue
        document = _load_json(other, "installed UHD")
        kind = document.get("adapter")
        other_body = document.get(kind) if isinstance(kind, str) else None
        key = "library" if kind == "custom_library" else "artifact"
        other_payload = other_body.get(key) if isinstance(other_body, dict) else None
        if isinstance(other_payload, str) and _same_file(
            other.parent / other_payload, artifact
        ):
            raise PromoteError(
                f"artifact collision: {artifact} also belongs to {other} "
                f"({document.get('id')})"
            )
    return artifact, payload


def _requested_identity(uhd_ids: dict, metric: str | None) -> str | None:
    """The id `--uhd-id` names for a model of `metric`, or None when it names none."""
    if not uhd_ids:
        return None
    if None in uhd_ids:
        return uhd_ids[None]
    if metric not in uhd_ids:
        raise PromoteError(
            f"--uhd-id names ids for {', '.join(sorted(uhd_ids))}, but this model "
            f"estimates {metric or 'no metric'}"
        )
    return uhd_ids[metric]


def _same_model(
    path: Path,
    installed: dict,
    incoming: dict,
    artifact_path: Path | None,
    artifact_key: str | None,
) -> bool:
    """Whether the UHD at `path` is the incoming model: same document, same bytes."""
    same = installed == incoming
    if artifact_path is not None:
        adapter = incoming["adapter"]
        body = installed.get(adapter)
        payload = body.get(artifact_key) if isinstance(body, dict) else None
        installed_artifact = path.parent / payload if isinstance(payload, str) else None
        same = installed_artifact is not None and installed_artifact.is_file()
        if same:
            data = installed_artifact.read_bytes()
            normalized = copy.deepcopy(installed)
            normalized[adapter][artifact_key] = incoming[adapter][artifact_key]
            normalized[adapter].setdefault("hash", artifact_digest(installed_artifact))
            same = data == artifact_path.read_bytes() and normalized == incoming
    return same


def _require_same_model(
    path: Path,
    installed: dict,
    incoming: dict,
    artifact_path: Path | None,
    artifact_key: str | None,
) -> None:
    """Refuse unless the UHD at `path` is the incoming model (D2: one UUID, one model)."""
    if not _same_model(path, installed, incoming, artifact_path, artifact_key):
        raise PromoteError(
            f"UHD id {installed.get('id')} is already installed at {path} with different "
            "content; one UUID names one model under every arch key that binds it (D2). "
            "Train the replacement under a new id, or promote it to the arch that model is "
            "installed for to replace it there"
        )


def _reuse(plan: PromotePlan, installed: Path) -> None:
    """Bind the identical model already installed at `installed` instead of a second copy."""
    plan.destination_descriptor = installed
    plan.copies.clear()
    plan.write_descriptor = False


def _verify_artifact(
    descriptor: dict, artifact_path: Path | None, role: str
) -> str | None:
    """The artifact's digest, after the checks the runtime's loader applies to it."""
    if artifact_path is None:
        return None
    adapter = descriptor["adapter"]
    declared = descriptor[adapter].get("hash")
    try:
        if adapter != "tree_data":
            digest = artifact_digest(artifact_path)
            if declared is not None and declared != digest:
                raise ValueError(
                    f"{artifact_path}: model hash mismatch - declared {declared!r}, "
                    f"actual {digest!r}"
                )
            return digest
        # The runtime checks num_features against the signature length; a matching
        # features_hash does not imply it.
        data = verify_tree_artifact(
            artifact_path,
            declared,
            feature_count=len(descriptor["features_signature"]),
        )
    except ValueError as error:
        raise PromoteError(f"the engine would refuse this artifact: {error}") from error
    import uhd_gen  # noqa: F401  puts _generated/ on sys.path

    from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModel

    recorded = GbdtModel.GetRootAs(bytearray(data), 0).FeaturesHash()
    recorded = recorded.decode("utf-8") if recorded is not None else ""
    if recorded != descriptor.get("features_hash"):
        raise PromoteError(
            f"{artifact_path}: artifact features_hash {recorded!r} differs from the "
            f"UHD's {descriptor.get('features_hash')!r}; the engine would refuse it"
        )
    if role == ROLE and is_grouped_tree(data):
        # Grouped artifacts have no engine-level scoring contract yet; refuse, don't mis-score.
        raise PromoteError(
            f"{artifact_path} is a grouped (two-layer) tree_data artifact, which "
            f"{ROLE} cannot bind; retrain without --group-by-feature"
        )
    return hashlib.sha256(data).hexdigest()


def _slug(name: str) -> str:
    """A directory name from an engine name; the loader keys on content, not layout."""
    return "".join(
        character if character.isalnum() else "_" for character in name
    ).strip("_")


def _build_plan(
    model_dir,
    descriptor_tree,
    engine,
    role,
    arch,
    remove_knobs,
    corpus=None,
    uhd_ids=None,
    feature_evaluator=None,
):
    uhd_ids = uhd_ids or {}
    if role not in ROLES:
        raise PromoteError(f"unknown heuristic role {role!r}")
    # First: a correctness defect refuses regardless of the descriptors.
    gate_warnings = _correctness_gate(model_dir, corpus)
    descriptor_path = _find_descriptor(model_dir)
    _contained(descriptor_path, model_dir, "source descriptor")
    descriptor = _load_json(descriptor_path, "UHD")
    _validate_descriptor(descriptor, descriptor_path)
    identity = descriptor_id(descriptor.get("id"), str(descriptor_path))
    manifest_path = model_dir / "train_manifest.json"
    manifest = (
        _load_json(manifest_path, "training manifest")
        if manifest_path.is_file()
        else {}
    )
    arches = manifest.get("training_arches", [])
    if not isinstance(arches, list) or any(
        not isinstance(item, str) or not _ARCH.fullmatch(item) for item in arches
    ):
        raise PromoteError("training_arches must be an array of bare gfx targets")
    if arch is None:
        if len(set(arches)) != 1:
            raise PromoteError(
                "pass --arch (or explicit --arch default): training_arches does not identify one target"
            )
        arch = arches[0]
    if arch != "default" and (not isinstance(arch, str) or not _ARCH.fullmatch(arch)):
        raise PromoteError(f"invalid target architecture {arch!r}")
    if arches and arch != "default" and arch not in arches:
        raise PromoteError(f"target {arch} is not in training_arches {arches}")
    if "trained_against" in manifest and validate_provenance(
        manifest["trained_against"]
    ) != validate_provenance(descriptor.get("trained_against")):
        raise PromoteError(
            "training manifest provenance differs from the UHD's trained_against"
        )
    recorded_role = manifest.get("role")
    if recorded_role is not None and recorded_role != role:
        raise PromoteError("incoming model was trained for another role")
    provenance = descriptor.get("trained_against", {})
    if descriptor.get("features_signature"):
        # The loader silently drops a model whose feature semantics differ from the build's.
        require_feature_semantics(
            provenance, evaluator_feature_semantics_revision(feature_evaluator)
        )
        # The loader recomputes this digest and drops the model on a mismatch; the
        # artifact agreeing with the descriptor does not prove the signature matches.
        computed = compute_features_hash(
            descriptor["features_signature"],
            descriptor.get("categorical_encoding"),
            feature_evaluator,
        )
        if computed != descriptor["features_hash"]:
            raise PromoteError(
                f"{descriptor_path}: features_hash {descriptor['features_hash']} is not the digest of "
                f"its features_signature and categorical_encoding ({computed}); the engine would "
                "refuse it. Retrain rather than editing the signature by hand"
            )
    if role == ROLE:
        validate_model(descriptor)
        if "ued" in provenance and "selector_revision" not in provenance:
            # The runtime requires a selector revision for engine-level models and reports
            # UNAVAILABLE without one.
            raise PromoteError(
                f"{descriptor_path}: an engine-level model for a descriptor-backed engine must "
                "record trained_against.selector_revision, which the runtime compares with the "
                "engine's current one; this model records none. Retrain from a corpus collected "
                "with uhd_gen generate"
            )
    artifact_path, artifact_key = _artifact_path(descriptor, descriptor_path, model_dir)
    digest = _verify_artifact(descriptor, artifact_path, role)
    installed_descriptor = copy.deepcopy(descriptor)
    if digest is not None:
        # RFC 0019 §7.2: the body carries the digest of the bytes it names.
        installed_descriptor[descriptor["adapter"]].setdefault("hash", digest)

    index = load_descriptor_tree(descriptor_tree)
    if any(identity in entries for entries in index.values()):
        raise PromoteError(
            f"incoming UHD identity {identity} conflicts with a dependency descriptor"
        )
    if "ued" not in provenance:
        # No descriptor set (e.g. AITER, MIOpen): bound by a UUID declared in provider
        # code, so there is no UED to select or role map to edit (RFC 0019 §4.1).
        opaque = _opaque_plan(
            descriptor_path,
            descriptor,
            installed_descriptor,
            identity,
            artifact_path,
            artifact_key,
            descriptor_tree,
            engine,
            role,
            arch,
            remove_knobs,
            uhd_ids,
            manifest,
        )
        opaque.warnings.extend(gate_warnings)
        return opaque
    ued_path, original_ued = select_engine(index, engine)
    engine_name = str(original_ued.get("name", ""))
    metric = _score_metric(descriptor)
    requested = _requested_identity(uhd_ids, metric)
    if requested is not None and requested != identity:
        raise PromoteError(
            f"model {identity} is not the id --uhd-id names for metric {metric} "
            f"({requested}); promote never renames a model"
        )
    # One directory per metric (RFC 0019 §3.1); a metric-less ranker uses the arch directory.
    destination_dir = ued_path.parent / "heuristics" / original_ued["id"] / role / arch
    if metric is not None:
        destination_dir /= metric
    ueds = list(index["ued"].values())
    references = [ref for path, doc in ueds for ref in _role_references(path, doc)]
    installed = []
    installed_ids = {}
    for path in sorted(descriptor_tree.rglob(f"*{UHD_SUFFIX}")):
        if path.name == UHD_SUFFIX:
            continue
        doc = _load_json(path, "installed UHD")
        current_id = descriptor_id(doc.get("id"), str(path))
        if current_id in installed_ids:
            raise PromoteError(
                f"duplicate installed UHD identity {current_id}: {installed_ids[current_id]} and {path}"
            )
        installed_ids[current_id] = path
        installed.append((path, doc, current_id))
    ued = copy.deepcopy(original_ued)
    bound, replaced = _rebind(ued, role, arch, identity, metric, installed_ids)
    old = replaced[0] if replaced else None
    destination_descriptor = destination_dir / descriptor_path.name
    _contained(destination_descriptor, descriptor_tree, "destination descriptor")
    plan = PromotePlan(
        descriptor_path,
        identity,
        artifact_path,
        ued_path,
        ued,
        engine_name,
        old,
        role,
        arch,
        destination_descriptor,
        installed_descriptor,
        metric=metric,
    )
    plan.warnings.extend(gate_warnings)
    if remove_knobs:
        exposed = ued.get("knobs", [])
        if not isinstance(exposed, list) or any(
            not isinstance(item, str) for item in exposed
        ):
            raise PromoteError("UED knobs must be an array of strings")
        unknown = set(remove_knobs) - set(exposed)
        if unknown:
            raise PromoteError(f"cannot remove unauthored knobs: {sorted(unknown)}")
        plan.dropped_knobs = sorted(set(remove_knobs))
        major, _ = revision(ued.get("revision", "1.0"), "UED")
        ued["revision"] = f"{major + 1}.0"
        ued["knobs"] = [knob for knob in exposed if knob not in plan.dropped_knobs]
    if descriptor.get("features_signature"):
        # RFC 0019 §6.3 check 2, against the UED after any --remove-knob.
        knobs = ued.get("knobs", [])
        if not isinstance(knobs, list) or any(
            not isinstance(item, str) for item in knobs
        ):
            raise PromoteError("UED knobs must be an array of strings")
        kmd = index["kmd"].get(
            descriptor_id(ued.get("metadata"), f"{engine_name} metadata")
        )
        if kmd is None:
            raise PromoteError(
                f"{engine_name} names KMD {ued.get('metadata')}, which the tree does not contain"
            )
        kmd_fields = [
            item.get("name")
            for item in kmd[1].get("fields", [])
            if isinstance(item, dict)
        ]
        removal = " once --remove-knob is applied" if remove_knobs else ""
        require_admissible_kernel_axes(
            descriptor["features_signature"],
            knobs,
            kmd_fields,
            f"model {identity}{removal}",
        )

    actual = provenance_for_engine(index, ued, arch)
    if actual is not None and "trained_against" in descriptor:
        try:
            compare_provenance(
                provenance,
                actual,
                foreign_matchers=foreign_matcher_ids(index, ued, arch),
            )
        except ProvenanceError as error:
            suffix = (
                "; retrain against the intended revised UED before removing knobs"
                if remove_knobs
                else ""
            )
            raise PromoteError(f"{error}{suffix}") from error
    elif remove_knobs:
        raise PromoteError(
            "explicit knob removal requires training provenance for the intended revised UED; retrain first"
        )

    target = (ued_path.resolve(), role, arch)
    # Every binding except the ones this promotion replaces, including other metrics here.
    others = [
        (path, slot, target_arch, ref)
        for path, slot, target_arch, ref in references
        if (path.resolve(), slot, target_arch) != target or ref not in replaced
    ]
    # A UUID shared by another role or arch is immutable here, even at the same filename.
    descriptor_changes = not _same_file(descriptor_path, destination_descriptor)
    destination_artifact = None
    if artifact_path is not None:
        destination_artifact = destination_dir / artifact_path.name
        _contained(destination_artifact, descriptor_tree, "destination artifact")
        installed_descriptor[descriptor["adapter"]][artifact_key] = artifact_path.name
        if destination_artifact.name.endswith(
            tuple(
                f".{kind}.json"
                for kind in ("ued", "umd", "kmd", "kdp", "ukd", "udd", "uhd")
            )
        ):
            raise PromoteError("artifact would overwrite or masquerade as a descriptor")
        if destination_artifact.exists() and not destination_artifact.is_file():
            raise PromoteError(
                f"destination artifact is not a regular file: {destination_artifact}"
            )
        if not _same_file(artifact_path, destination_artifact):
            plan.copies.append((artifact_path, destination_artifact))
    descriptor_changes = descriptor_changes or installed_descriptor != descriptor
    for path, doc, current_id in installed:
        if current_id == identity and not _same_file(path, destination_descriptor):
            # D2: the same model promoted for another arch key binds the installed copy.
            _require_same_model(
                path, doc, installed_descriptor, artifact_path, artifact_key
            )
            _reuse(plan, path)
            destination_descriptor, destination_artifact, descriptor_changes = (
                path,
                None,
                False,
            )
    for path, doc, current_id in installed:
        same_destination = _same_file(path, destination_descriptor)
        if not same_destination:
            continue
        holders = [
            (p, r, a) for p, r, a, ref in others if ref in (current_id, identity)
        ]
        if holders and (descriptor_changes or plan.copies):
            raise PromoteError(
                f"would clobber UHD {current_id} used by other role/arch entries: {holders}"
            )
        if current_id != identity and current_id != old:
            plan.warnings.append(f"OVERWRITING unreferenced UHD {current_id} at {path}")
    if any(ref == identity for _, _, _, ref in others) and (
        descriptor_changes or plan.copies
    ):
        raise PromoteError(
            f"incoming UHD id {identity} is owned by another role/arch entry"
        )
    if destination_artifact is not None and plan.copies:
        for path, doc, current_id in installed:
            if _same_file(path, destination_descriptor):
                continue
            adapter = doc.get("adapter")
            body = doc.get(adapter) if isinstance(adapter, str) else None
            key = "library" if adapter == "custom_library" else "artifact"
            payload = body.get(key) if isinstance(body, dict) else None
            if isinstance(payload, str) and _same_file(
                path.parent / payload, destination_artifact
            ):
                raise PromoteError(
                    f"artifact collision: {destination_artifact} belongs to {path} ({current_id})"
                )
        if destination_artifact.exists() and not destination_descriptor.exists():
            raise PromoteError(
                f"unowned destination artifact already exists: {destination_artifact}"
            )
    if remove_knobs:
        for path, other_role, other_arch, ref in others:
            if path.resolve() != ued_path.resolve():
                continue
            model_path = installed_ids.get(ref)
            if model_path is None:
                raise PromoteError(
                    f"cannot prove knob removal safe: missing model {ref} for {other_role}/{other_arch}"
                )
            model = _load_json(model_path, "other role UHD")
            if model.get("features_signature"):
                try:
                    compare_provenance(
                        model.get("trained_against", {}),
                        provenance_for_engine(index, ued, other_arch),
                        foreign_matchers=foreign_matcher_ids(index, ued, other_arch),
                    )
                except ProvenanceError as error:
                    raise PromoteError(
                        f"knob removal would invalidate {other_role}/{other_arch}: {error}"
                    ) from error
            if set(plan.dropped_knobs) & kernel_axes(
                model.get("features_signature", [])
            ):
                raise PromoteError(
                    f"knob removal would affect {other_role}/{other_arch}"
                )
    ued.setdefault(role, {})[arch] = bound
    plan.write_descriptor = descriptor_changes
    return plan


def _score_metric(document: dict) -> str | None:
    """The ranking metric a UHD's score estimates; None for a metric-less ranker."""
    score = document.get("score")
    return score.get("metric") if isinstance(score, dict) else None


def _role_value(value, role: str, where: str) -> list[str]:
    """The UHD ids one arch key binds: one id, or for a metric-keyed role a list of them."""
    if role in _METRIC_ROLES and isinstance(value, list):
        if not value:
            raise PromoteError(
                f"{where}: an empty list binds nothing; omit the architecture instead"
            )
        identities = [descriptor_id(item, where) for item in value]
        if len(set(identities)) != len(identities):
            raise PromoteError(f"{where}: lists the same UHD twice")
        return identities
    return [descriptor_id(value, where)]


def _rebind(
    ued: dict,
    role: str,
    arch: str,
    identity: str,
    metric: str | None,
    installed_ids: dict[str, Path],
) -> tuple:
    """The arch key's value once `identity` is bound, and the ids it displaces.

    A metric-keyed role holds one UHD per metric, and the metric lives only in each UHD, so
    every bound UHD must be installed in this tree to be classified.
    """
    where = f"{role}.{arch}"
    value = ued.get(role, {}).get(arch)
    current = [] if value is None else _role_value(value, role, where)
    if role not in _METRIC_ROLES:
        return identity, current

    def same_metric(ref: str) -> bool:
        if ref == identity:
            return True
        path = installed_ids.get(ref)
        if path is None:
            raise PromoteError(
                f"{where} binds UHD {ref}, which is not installed in this descriptor "
                "tree; its score.metric decides whether this promotion replaces it"
            )
        return _score_metric(_load_json(path, "bound UHD")) == metric

    replaced = [ref for ref in current if same_metric(ref)]
    if len(replaced) > 1:
        raise PromoteError(
            f"{where} already binds {len(replaced)} UHDs for metric "
            f"{metric or '(none)'} ({', '.join(replaced)}); at most one is allowed"
        )
    # In place, so a replacement keeps its position; a new metric is appended.
    bound = [identity if ref in replaced else ref for ref in current]
    if not replaced:
        bound.append(identity)
    return bound, replaced


def _role_references(path: Path, document: dict):
    if "heuristic" in document:
        raise PromoteError(
            f"{path}: a top-level 'heuristic' field is not supported; use role/arch maps"
        )
    for role in ROLES:
        if role not in document:
            continue
        entries = document[role]
        if not isinstance(entries, dict) or not entries:
            raise PromoteError(f"{path}.{role} must be a nonempty arch-to-UUID map")
        for arch, value in entries.items():
            if arch != "default" and not _ARCH.fullmatch(arch):
                raise PromoteError(f"{path}.{role}: invalid architecture {arch!r}")
            for identity in _role_value(value, role, f"{path}.{role}.{arch}"):
                yield path, role, arch, identity


def _validate_descriptor(document: dict, path: Path) -> None:
    descriptor_id(document.get("id"), str(path))
    known = {
        "version",
        "id",
        "name",
        "adapter",
        "objective",
        "score",
        "features_signature",
        "features_hash",
        "categorical_encoding",
        "trained_against",
        # Free-form authoring notes, which the loader admits at the root and never reads.
        "provenance",
        *_ADAPTERS,
    }
    if set(document) - known:
        raise PromoteError(
            f"{path}: unknown UHD fields {sorted(set(document) - known)}"
        )
    if document.get("version") != "1.0":
        raise PromoteError(
            f"{path}: unsupported UHD file-format version {document.get('version')!r}"
        )
    if not isinstance(document.get("name"), str) or not document["name"]:
        raise PromoteError(f"{path}: missing required name")
    adapter = document.get("adapter")
    if adapter not in _ADAPTERS:
        raise PromoteError(f"{path}: unsupported adapter {adapter!r}")
    bodies = [key for key in _ADAPTERS if key in document]
    if bodies != [adapter] or not isinstance(document[adapter], dict):
        raise PromoteError(
            f"{path}: requires exactly one body matching adapter {adapter}"
        )
    if (adapter != "static_order" or "objective" in document) and document.get(
        "objective"
    ) not in ("min", "max"):
        raise PromoteError(f"{path}: scorer requires objective min or max")
    signature = document.get("features_signature")
    if "features_signature" in document or adapter in ("tree_data", "table", "onnx"):
        if not isinstance(signature, list) or not signature:
            raise PromoteError(f"{path}: requires nonempty features_signature")
        for entry in signature:
            if not (
                (isinstance(entry, str) and entry.startswith("$") and len(entry) > 1)
                or (isinstance(entry, dict) and len(entry) == 1)
            ):
                raise PromoteError(
                    f"{path}: features_signature requires bare references or inline expressions"
                )
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
            not isinstance(codes, dict)
            or not codes
            or any(type(code) is not int for code in codes.values())
            for codes in encoding.values()
        ):
            raise PromoteError(
                f"{path}: categorical_encoding requires value-to-integer maps"
            )
    if "score" in document:
        score = document["score"]
        if not isinstance(score, dict) or set(score) - {
            "metric",
            "calibrated",
            "transform",
        }:
            raise PromoteError(f"{path}: invalid score header")
        if "transform" in score and (
            not isinstance(score["transform"], str) or not score["transform"]
        ):
            raise PromoteError(f"{path}: score.transform must be a nonempty string")
        if "calibrated" in score and not isinstance(score["calibrated"], bool):
            raise PromoteError(f"{path}: score.calibrated must be a boolean")
        # RFC 0019 §4.1: a registered metric fixes the objective direction.
        if "metric" in score:
            if score["metric"] not in RANKING_METRICS:
                raise PromoteError(
                    f"{path}: score.metric {score['metric']!r} is not a registered "
                    f"ranking metric ({', '.join(RANKING_METRICS)})"
                )
            expected = RANKING_METRICS[score["metric"]].objective
            if document.get("objective") != expected:
                raise PromoteError(
                    f"{path}: score.metric {score['metric']} ranks {expected}; "
                    f"objective {document.get('objective')!r} contradicts it"
                )
        elif score.get("calibrated") is True:
            raise PromoteError(f"{path}: a calibrated score must name its metric")
    body = document[adapter]
    # static_order takes no criteria; refused here as UhdParser refuses them.
    if adapter == "static_order" and "order" in body:
        raise PromoteError(
            f"{path}: static_order.order is not supported: declared ordering criteria are "
            "not implemented; static_order ranks by priority, then descriptor id"
        )
    allowed = (
        set()
        if adapter == "static_order"
        else (
            {"symbol"}
            if adapter == "native"
            else (
                {"library", "symbol", "hash", "config"}
                if adapter == "custom_library"
                else {"artifact", "hash"}
            )
        )
    )
    if set(body) - allowed:
        raise PromoteError(f"{path}: unknown {adapter} body fields")
    if adapter not in ("static_order", "native"):
        key = "library" if adapter == "custom_library" else "artifact"
        payload = body.get(key)
        if not isinstance(payload, str) or not payload:
            raise PromoteError(f"{path}: missing {adapter}.{key}")
        if not is_contained_relative_path(payload):
            raise PromoteError(
                f"{path}: {adapter}.{key} {payload!r} must be a relative path inside "
                "the descriptor's directory"
            )
        # A library is loaded as code, so it must declare the digest it is verified against.
        if adapter == "custom_library" and "hash" not in body:
            raise PromoteError(f"{path}: custom_library.hash is required")
        if "hash" in body and (
            not isinstance(body["hash"], str) or not _SHA256.fullmatch(body["hash"])
        ):
            raise PromoteError(
                f"{path}: {adapter}.hash must be the SHA-256 of the {key} as 64 "
                f"lowercase hexadecimal digits, got {body['hash']!r}"
            )
    # UhdParser admits only an omitted or empty config.
    if adapter == "custom_library" and "config" in body and body["config"] != {}:
        raise PromoteError(
            f"{path}: custom_library configuration is not supported; omit config or "
            "leave it an empty object"
        )


def _artifact_path(descriptor, descriptor_path, model_dir):
    adapter = descriptor["adapter"]
    if adapter in ("static_order", "native"):
        return None, None
    key = "library" if adapter == "custom_library" else "artifact"
    resolved = (descriptor_path.parent / descriptor[adapter][key]).resolve()
    _contained(resolved, model_dir, "model artifact")
    if not resolved.is_file():
        raise PromoteError(f"model artifact does not exist: {resolved}")
    return resolved, key


def _contained(path, root, what):
    if not path.resolve().is_relative_to(root.resolve()):
        raise PromoteError(f"{what} escapes root {root}: {path}")


def _find_descriptor(model_dir):
    if not model_dir.is_dir():
        raise PromoteError(f"--model-dir {model_dir} is not a directory")
    paths = sorted(
        path for path in model_dir.glob(f"*{UHD_SUFFIX}") if path.name != UHD_SUFFIX
    )
    if len(paths) != 1:
        raise PromoteError(
            f"{model_dir} must contain exactly one *{UHD_SUFFIX}; found {len(paths)}"
        )
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
    return left.resolve() == right.resolve() or (
        left.exists() and right.exists() and left.samefile(right)
    )


def _write_json(path, document):
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _apply(plan: PromotePlan) -> None:
    """Execute only the operations approved by build_plan; never restamp provenance."""
    plan.destination_descriptor.parent.mkdir(parents=True, exist_ok=True)
    for source, destination in plan.copies:
        shutil.copy2(source, destination)
    if plan.write_descriptor:
        _write_json(plan.destination_descriptor, plan.descriptor_document)
    if plan.ued_path is not None:
        _write_json(plan.ued_path, plan.ued_document)


def _report(plan: PromotePlan, dry_run: bool) -> None:
    print(
        "UHD promotion plan (dry run, nothing written)" if dry_run else "UHD promoted"
    )
    binding = (
        plan.ued_path if plan.ued_path is not None else "declared in provider code"
    )
    print(
        f"  engine: {plan.engine_name}\n  binding: {binding}\n  role/arch: {plan.role}/{plan.arch}"
    )
    print(f"  metric: {plan.metric or '(none: metric-less ranker)'}")
    print(
        f"  heuristic was: {plan.old_heuristic or '(none)'}\n  heuristic now: {plan.descriptor_id}"
    )
    for source, destination in plan.copies:
        print(f"  {'would copy' if dry_run else 'copy'}: {source} -> {destination}")
    if plan.write_descriptor:
        print(f"  descriptor: {plan.destination_descriptor}")
    elif not plan.copies:
        print(
            f"  model already installed (bound, not copied): {plan.destination_descriptor}"
        )
    for knob in plan.dropped_knobs:
        print(f"  removed knob: {knob}; UED revision {plan.ued_document['revision']}")
