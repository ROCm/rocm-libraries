#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Install a trained UHD into a descriptor tree and point an engine at it.

`train` writes a descriptor/artifact pair into an output directory and prints the id
it minted. Until something copies that pair next to the engine's UED and writes the id
into the UED's `heuristic` field, the engine keeps whatever heuristic it named before
-- or none, in which case `KernelIngestorEngine` ranks by priority then descriptor id.
Either way the model is inert and nothing reports an error, so the only symptom is
that the numbers did not move. That hand-edit is what this subcommand replaces.

Everything is validated before anything is written. A promote that copies the artifact
and then fails, or writes a `heuristic` id no descriptor defines, leaves a tree whose
engine the loader drops entirely -- strictly worse than the stale-model state it was
called to fix.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: Suffixes DescriptorLoader discovers by (DescriptorLoader.hpp SUFFIX_UHD/SUFFIX_UED).
UHD_SUFFIX = ".uhd.json"
UED_SUFFIX = ".ued.json"


class PromoteError(Exception):
    """A refusal raised while planning, i.e. before the tree has been touched."""


@dataclass
class PromotePlan:
    """Every file operation promote intends, decided before the first one runs."""

    descriptor_path: Path
    descriptor_id: str
    artifact_path: Path
    ued_path: Path
    ued_document: dict
    engine_name: str
    old_heuristic: str | None
    #: (source, destination) pairs; a file already in place is not listed.
    copies: list[tuple[Path, Path]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def add_promote_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the `promote` flags."""
    parser.add_argument(
        "--model-dir",
        required=True,
        dest="model_dir",
        help="Directory `uhd_gen train --output-dir` wrote: one <stem>.uhd.json plus "
        "the artifact its adapter body names.",
    )
    parser.add_argument(
        "--descriptor-tree",
        required=True,
        dest="descriptor_tree",
        help="Descriptor tree holding the engine's <name>.ued.json. Searched "
        "recursively; the pair is installed beside the UED that is updated.",
    )
    parser.add_argument(
        "--engine",
        default=None,
        help="UED `name` to update (e.g. hipkernel:pointwise_model). Required when the "
        "tree holds more than one UED -- promote never picks for you.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Report the plan, including every check it had to pass, without writing.",
    )


def run_promote(args: argparse.Namespace) -> int:
    """Plan, then either report or execute. Returns a process exit code."""
    try:
        plan = build_plan(
            Path(args.model_dir),
            Path(args.descriptor_tree),
            args.engine,
        )
    except PromoteError as error:
        logger.error("%s", error)
        return 1

    for warning in plan.warnings:
        logger.warning("%s", warning)

    if not args.dry_run:
        _apply(plan)
    _report(plan, dry_run=args.dry_run)
    return 0


def build_plan(model_dir: Path, descriptor_tree: Path, engine: str | None) -> PromotePlan:
    """Resolve and check everything. Raises PromoteError; never writes."""
    descriptor_path = _find_descriptor(model_dir)
    descriptor = _load_json(descriptor_path, "UHD descriptor")
    descriptor_id = _descriptor_id(descriptor, descriptor_path)
    artifact_path = _artifact_path(descriptor, descriptor_path)

    ueds = _load_ueds(descriptor_tree)
    ued_path, ued_document = _select_ued(ueds, engine, descriptor_tree)

    destination = ued_path.parent
    plan = PromotePlan(
        descriptor_path=descriptor_path,
        descriptor_id=descriptor_id,
        artifact_path=artifact_path,
        ued_path=ued_path,
        ued_document=ued_document,
        engine_name=str(ued_document.get("name", "")),
        old_heuristic=_optional_str(ued_document.get("heuristic")),
    )

    destination_descriptor = destination / descriptor_path.name
    destination_artifact = destination / artifact_path.name
    _check_descriptor_collision(plan, destination_descriptor, ueds)
    _check_artifact_collision(plan, destination_descriptor, destination_artifact)

    for source, target in (
        (artifact_path, destination_artifact),
        (descriptor_path, destination_descriptor),
    ):
        # Promoting a pair that already lives beside the UED (a retrain straight into
        # the tree) is a legitimate no-copy case; shutil.copy2 would raise SameFileError.
        if not _same_file(source, target):
            plan.copies.append((source, target))

    return plan


def _find_descriptor(model_dir: Path) -> Path:
    if not model_dir.is_dir():
        raise PromoteError(f"--model-dir {model_dir} is not a directory")
    # A bare `.uhd.json` has an empty stem, which DescriptorLoader::findFileType rejects;
    # treating one as the model here would install a file the runtime then ignores.
    candidates = sorted(
        path for path in model_dir.glob(f"*{UHD_SUFFIX}") if path.name != UHD_SUFFIX
    )
    if not candidates:
        raise PromoteError(
            f"no *{UHD_SUFFIX} in {model_dir}; that directory is not a `uhd_gen train "
            "--output-dir` result"
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise PromoteError(
            f"{model_dir} holds {len(candidates)} descriptors ({names}); promote "
            "installs exactly one pair and will not choose between them"
        )
    return candidates[0]


def _descriptor_id(descriptor: dict, descriptor_path: Path) -> str:
    """The id the UED will name. Must be a UUID: the loader indexes descriptors by it,
    and an id it cannot parse resolves to nothing, dropping the engine."""
    raw = descriptor.get("id")
    if not isinstance(raw, str) or not raw:
        raise PromoteError(f"{descriptor_path} has no `id`; nothing for a UED to reference")
    try:
        uuid.UUID(raw)
    except ValueError as error:
        raise PromoteError(
            f"{descriptor_path} has id {raw!r}, which is not a UUID ({error}); a UED "
            "naming it would resolve to nothing and the engine would load without a "
            "heuristic"
        ) from error
    return raw


def _artifact_path(descriptor: dict, descriptor_path: Path) -> Path:
    """The model file the descriptor's adapter body names, resolved as the runtime does:
    relative to the descriptor itself."""
    adapter = descriptor.get("adapter")
    if not isinstance(adapter, str) or not adapter:
        raise PromoteError(f"{descriptor_path} has no `adapter`; cannot locate its body")
    body = descriptor.get(adapter)
    if not isinstance(body, dict):
        raise PromoteError(
            f"{descriptor_path} declares adapter {adapter!r} but has no {adapter!r} body"
        )
    artifact = body.get("artifact")
    if not isinstance(artifact, str) or not artifact:
        raise PromoteError(f"{descriptor_path} has no `{adapter}.artifact`")
    # The pair is installed flat next to the UED, so a nested or absolute artifact would
    # arrive under a name the copied descriptor no longer points at.
    if Path(artifact).name != artifact or artifact in (".", ".."):
        raise PromoteError(
            f"{descriptor_path} names artifact {artifact!r}; promote installs the pair "
            "side by side and only handles an artifact in the descriptor's own directory"
        )
    resolved = descriptor_path.parent / artifact
    if not resolved.is_file():
        raise PromoteError(
            f"{descriptor_path} names artifact {artifact!r}, which does not exist at "
            f"{resolved}; installing the descriptor without it would make the engine "
            "fail to build its heuristic"
        )
    return resolved


def _load_ueds(descriptor_tree: Path) -> list[tuple[Path, dict]]:
    if not descriptor_tree.is_dir():
        raise PromoteError(f"--descriptor-tree {descriptor_tree} is not a directory")
    paths = sorted(
        path for path in descriptor_tree.rglob(f"*{UED_SUFFIX}") if path.name != UED_SUFFIX
    )
    if not paths:
        raise PromoteError(f"no *{UED_SUFFIX} under {descriptor_tree}; nothing to promote into")
    return [(path, _load_json(path, "UED")) for path in paths]


def _select_ued(
    ueds: list[tuple[Path, dict]], engine: str | None, descriptor_tree: Path
) -> tuple[Path, dict]:
    """Pick the UED to update, or refuse.

    Guessing is the one thing this must never do. Promoting into the wrong engine is a
    double failure: the engine you meant to retrain keeps its old model, and one you
    never touched starts ranking with a model trained on a different kernel set --
    both of which load cleanly and report nothing.
    """
    if engine is None:
        if len(ueds) == 1:
            return ueds[0]
        listing = "\n".join(f"  {_engine_label(document)}  ({path})" for path, document in ueds)
        raise PromoteError(
            f"{descriptor_tree} holds {len(ueds)} UEDs; pass --engine NAME to say which "
            f"one this model ranks for:\n{listing}"
        )

    matches = [(path, document) for path, document in ueds if document.get("name") == engine]
    if not matches:
        listing = "\n".join(f"  {_engine_label(document)}  ({path})" for path, document in ueds)
        raise PromoteError(f"no UED named {engine!r} under {descriptor_tree}; found:\n{listing}")
    if len(matches) > 1:
        listing = "\n".join(f"  {path}" for path, _ in matches)
        raise PromoteError(
            f"{len(matches)} UEDs under {descriptor_tree} are named {engine!r}; the tree "
            f"is ambiguous and promote will not choose:\n{listing}"
        )
    return matches[0]


def _check_descriptor_collision(
    plan: PromotePlan, destination_descriptor: Path, ueds: list[tuple[Path, dict]]
) -> None:
    """Refuse or warn when the destination filename already holds a different UHD."""
    if _same_file(plan.descriptor_path, destination_descriptor):
        return
    if not destination_descriptor.is_file():
        return
    existing = _load_json(destination_descriptor, "installed UHD descriptor")
    existing_id = _optional_str(existing.get("id"))
    if existing_id == plan.descriptor_id:
        # A newer build of the same heuristic. The ordinary retrain.
        return
    if existing_id is not None and existing_id == plan.old_heuristic:
        # The descriptor this very engine currently ranks by, being replaced by the
        # model trained to succeed it. Also ordinary -- and the report prints both ids,
        # so warning here would only teach readers to ignore the warning that matters
        # below.
        return

    # Some *other* engine ranks by the descriptor about to be overwritten. Replacing it
    # silently repoints that engine's `heuristic` at nothing, and the loader drops it.
    holders = [
        path
        for path, document in ueds
        if path != plan.ued_path and _optional_str(document.get("heuristic")) == existing_id
    ]
    if holders:
        listing = "\n".join(f"  {path}" for path in holders)
        raise PromoteError(
            f"{destination_descriptor} is a different UHD (id {existing_id}) and is "
            f"still referenced by:\n{listing}\nOverwriting it would strand those "
            "engines. Rename this model's descriptor with `train --descriptor-name` "
            "and promote again."
        )
    plan.warnings.append(
        f"OVERWRITING a different UHD: {destination_descriptor} currently has id "
        f"{existing_id}, the incoming descriptor has {plan.descriptor_id}. No UED "
        "references the old id, so nothing is stranded, but the file is not a newer "
        "build of the same heuristic -- it is a different one that happens to share a "
        "filename."
    )


def _check_artifact_collision(
    plan: PromotePlan, destination_descriptor: Path, destination_artifact: Path
) -> None:
    """Refuse when the artifact would clobber a model another installed UHD points at.

    `train` names every artifact `model.bin`, so two differently-stemmed descriptors in
    one directory collide by default. The victim descriptor keeps loading and keeps its
    features_hash, so the mismatch surfaces as a refused pair or, worse, as silently
    different rankings.
    """
    if not destination_artifact.exists() or _same_file(plan.artifact_path, destination_artifact):
        return
    for neighbour in sorted(destination_artifact.parent.glob(f"*{UHD_SUFFIX}")):
        if neighbour.name == UHD_SUFFIX or _same_file(neighbour, destination_descriptor):
            continue
        document = _load_json(neighbour, "installed UHD descriptor")
        adapter = document.get("adapter")
        body = document.get(adapter) if isinstance(adapter, str) else None
        artifact = body.get("artifact") if isinstance(body, dict) else None
        if not isinstance(artifact, str):
            continue
        if _same_file(neighbour.parent / artifact, destination_artifact):
            raise PromoteError(
                f"{destination_artifact} is the artifact of {neighbour.name} (id "
                f"{_optional_str(document.get('id'))}); installing this model there "
                "would replace that heuristic's trees while leaving its features_hash "
                "untouched. Give this model a distinct `train --descriptor-name`."
            )


def _apply(plan: PromotePlan) -> None:
    """Perform the planned writes. Only reached once build_plan has approved them all."""
    for source, destination in plan.copies:
        shutil.copy2(source, destination)

    plan.ued_document["heuristic"] = plan.descriptor_id
    with open(plan.ued_path, "w", encoding="utf-8") as handle:
        # indent=2 + trailing newline + ensure_ascii=False reproduces how every
        # descriptor in the tree is written, so the diff is the one changed line rather
        # than a whole-file reformat nobody can review.
        json.dump(plan.ued_document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _report(plan: PromotePlan, dry_run: bool) -> None:
    label = "would copy:" if dry_run else "copy:"
    print("\nUHD promotion plan (dry run, nothing written)" if dry_run else "\nUHD promoted")
    print(f"  {'engine:':<16}{plan.engine_name or '(unnamed)'}")
    print(f"  {'UED:':<16}{plan.ued_path}")
    print(f"  {'heuristic was:':<16}{plan.old_heuristic or '(none)'}")
    print(f"  {'heuristic now:':<16}{plan.descriptor_id}")
    if plan.copies:
        for source, destination in plan.copies:
            print(f"  {label:<16}{source} -> {destination}")
    else:
        print(f"  {label:<16}(descriptor and artifact already in place)")


def _load_json(path: Path, what: str) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise PromoteError(f"cannot read {what} {path}: {error}") from error
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise PromoteError(f"{what} {path} is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise PromoteError(f"{what} {path} is not a JSON object")
    return document


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _engine_label(document: dict) -> str:
    return _optional_str(document.get("name")) or "(unnamed)"


def _same_file(left: Path, right: Path) -> bool:
    """Same path on disk, tolerating one side not existing yet."""
    if left.exists() and right.exists():
        return left.samefile(right)
    return left.resolve() == right.resolve()
