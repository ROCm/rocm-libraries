# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Reads the ids a previous generation of a bundle left on disk.

The reader works on files only and knows nothing of `IngestorConfig`: it
resolves descriptors by UUID reference, never by file name (a shared KMD's file
name does not follow the packs that use it), and returns what it found. Which
prior kernel matches which configured kernel is decided in ``generator.py``.

A root plays one of two roles:

* **home**: it holds the engine's UED and with it the engine-level files (KMD,
  optional UHD, UDD and the kernel-scoped matcher UMD), beside its KDPs;
* **satellite**: it holds only a KDP whose ``engine`` field names the home's
  UED id.

Every failure is a ``ValueError`` naming the offending paths.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

#: Descriptor kinds the reader collects, by the ``<kind>.json`` file suffix.
_KINDS = ("ued", "kmd", "uhd", "udd", "umd", "kdp")


class NotOverridable(ValueError):
    """A refusal that naming another id source cannot cure: the tree read is
    itself unusable, or the config cannot take part in sibling mode."""


@dataclass(frozen=True)
class PriorKdp:
    path: Path
    id: str
    name: str  # "<namespace>:<kdp_stem>"
    dispatch: str  # the UDD id
    arch: list[str] | None  # KDP-level arch, None if absent
    operation_umd_id: str | None  # the graph-scope matcher; multi-pack only
    entries: list[dict]  # kernelDescriptors[] as parsed

    def effective_arch(self, entry: dict) -> list[str] | None:
        """The arch ``entry`` is emitted for: its own, else the KDP's."""
        return entry.get("arch") or self.arch


@dataclass(frozen=True)
class EngineDocs:
    """The engine-level documents, parsed."""

    ued: dict
    kmd: dict
    uhd: dict | None
    udd: dict
    kernel_match_umd: dict


@dataclass(frozen=True)
class PriorBundle:
    root: Path
    engine: EngineDocs | None  # None for a satellite
    kdps: list[PriorKdp]
    all_ids: frozenset[str]


class _Root:
    """Every descriptor of one root, parsed once and checked for id uniqueness."""

    def __init__(self, root: Path):
        if not root.is_dir():
            raise ValueError(f"prior-id root {str(root)!r} is not a directory")
        self.root = root
        self.docs: dict[str, list[tuple[Path, dict]]] = {k: [] for k in _KINDS}
        self.by_id: dict[str, tuple[Path, dict]] = {}
        self.kind_of: dict[str, str] = {}
        self.dvc = sorted(root.rglob("*.dvc"))
        owner: dict[str, Path] = {}

        def claim(identity, path: Path) -> None:
            if not isinstance(identity, str) or not identity:
                raise ValueError(f"{path}: a descriptor or kernel has no string 'id'")
            if identity in owner:
                raise NotOverridable(
                    f"id {identity} appears twice under {root}: in {owner[identity]} "
                    f"and in {path}. A prior tree with a repeated id cannot say which "
                    f"descriptor owns it: delete the stale file of the two (a removed "
                    f"or renamed pack's KDP is the usual one)."
                )
            owner[identity] = path

        for kind in _KINDS:
            for path in sorted(root.rglob(f"*.{kind}.json")):
                try:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as e:
                    raise ValueError(f"{path}: cannot read descriptor: {e}") from e
                if not isinstance(doc, dict):
                    raise ValueError(f"{path}: a {kind} descriptor must be an object")
                claim(doc.get("id"), path)
                self.docs[kind].append((path, doc))
                self.by_id[doc["id"]] = (path, doc)
                self.kind_of[doc["id"]] = kind
                if kind == "kdp":
                    for entry in doc.get("kernelDescriptors") or []:
                        if not isinstance(entry, dict):
                            raise ValueError(f"{path}: a kernel entry is not an object")
                        claim(entry.get("id"), path)
        self.all_ids = frozenset(owner)

    def resolve(self, kind: str, identity, role: str, referrer: Path) -> dict:
        found = self.by_id.get(identity) if isinstance(identity, str) else None
        if found is None or self.kind_of[identity] != kind:
            raise ValueError(
                f"{referrer}: {role} {identity!r} is not a {kind} descriptor under "
                f"{self.root}"
            )
        return found[1]

    def kdps_of(self, engine_id: str) -> list[tuple[Path, dict]]:
        return [(p, d) for p, d in self.docs["kdp"] if d.get("engine") == engine_id]

    def no_kdp_error(self, engine_id: str) -> ValueError:
        dvc = (
            f" The root holds DVC pointer(s) {[str(p) for p in self.dvc]}: the KDP is "
            f"an artifact that has not been fetched, so every kernel id would be "
            f"reminted. Fetch it first (dvc pull, or download the object)."
            if self.dvc
            else ""
        )
        return ValueError(
            f"no KDP under {self.root} names engine {engine_id}, so there are no "
            f"kernel ids to reuse.{dvc}"
        )


def _umd_kinds(root: _Root, kdp_path: Path, kdp: dict) -> dict[str, dict]:
    """``{scope: umd doc}`` for the KDP's matchers, resolved by UUID."""
    by_scope: dict[str, dict] = {}
    for matcher in kdp.get("matchers") or []:
        umd = root.resolve("umd", matcher, "matcher", kdp_path)
        scope = umd.get("scope")
        if scope in by_scope:
            raise ValueError(
                f"{kdp_path}: two {scope}-scope matchers; a KDP carries one "
                f"kernel-scope matcher and at most one graph-scope matcher"
            )
        by_scope[scope] = umd
    return by_scope


def _check_dispatch(kdps: list[tuple[Path, dict]], engine: str, root: Path) -> None:
    """Every KDP of one engine names the same UDD."""
    dispatch = {d.get("dispatch") for _, d in kdps}
    if len(dispatch) != 1 or not all(isinstance(d, str) for d in dispatch):
        raise ValueError(
            f"KDPs of engine {engine} under {root} disagree on dispatch: "
            f"{sorted(map(str, dispatch))}"
        )


def find_engine(root: Path, engine_name: str) -> EngineDocs | None:
    """The engine-level documents of the UED named ``engine_name`` under ``root``,
    or None when no UED there has that name.

    The UDD and the kernel-scope matcher are not referenced by the UED: they are
    resolved from the engine's KDPs, so a home root with no KDP is an error.
    """
    return _engine_docs(_Root(root), engine_name)


def _engine_docs(tree: _Root, engine_name: str) -> EngineDocs | None:
    root = tree.root
    ueds = [(p, d) for p, d in tree.docs["ued"] if d.get("name") == engine_name]
    if not ueds:
        return None
    if len(ueds) > 1:
        raise ValueError(
            f"ambiguous: {len(ueds)} UEDs named {engine_name!r} under {root}: "
            f"{[str(p) for p, _ in ueds]}"
        )
    ued_path, ued = ueds[0]
    kmd = tree.resolve("kmd", ued.get("metadata"), "metadata", ued_path)
    uhd = None
    catalog = ued.get("sort_kernel_catalog")
    if catalog is not None:
        if not isinstance(catalog, dict) or "default" not in catalog:
            raise ValueError(
                f"{ued_path}: 'sort_kernel_catalog' must be an object with a "
                f"'default' UHD reference"
            )
        uhd = tree.resolve(
            "uhd", catalog["default"], "sort_kernel_catalog.default", ued_path
        )
    kdps = tree.kdps_of(ued["id"])
    if not kdps:
        raise tree.no_kdp_error(ued["id"])
    _check_dispatch(kdps, repr(engine_name), root)
    kdp_path, kdp = kdps[0]
    udd = tree.resolve("udd", kdp.get("dispatch"), "dispatch", kdp_path)
    kernel_match = _umd_kinds(tree, kdp_path, kdp).get("kernel")
    if kernel_match is None:
        raise ValueError(f"{kdp_path}: no kernel-scope matcher among its matchers")
    return EngineDocs(ued=ued, kmd=kmd, uhd=uhd, udd=udd, kernel_match_umd=kernel_match)


def _is_arch_list(arch) -> bool:
    return isinstance(arch, list) and all(isinstance(a, str) for a in arch)


def _check_entry(path: Path, entry: dict) -> None:
    """The fields of a previous kernel that matching reads must have the shape the
    generator writes: ``metadata`` and ``arch`` make up the kernel's key, and the
    source (``kernel_source``, or ``sources`` for a multi-launch kernel) is
    compared launch by launch, each ``spec`` key by key."""
    name = entry.get("name")
    if not isinstance(entry.get("metadata") or {}, dict):
        raise ValueError(f"{path}: kernel {name!r} has a non-object 'metadata'")
    if entry.get("arch") is not None and not _is_arch_list(entry["arch"]):
        raise ValueError(
            f"{path}: kernel {name!r} has an 'arch' that is not a list of strings"
        )
    if "sources" in entry:
        sources = entry["sources"]
        if (
            "kernel_source" in entry
            or not isinstance(sources, dict)
            or not sources
            or not all(isinstance(launch, dict) for launch in sources.values())
        ):
            raise ValueError(
                f"{path}: kernel {name!r} needs 'sources' to be a non-empty object "
                f"of launch name to kernel_source object, with no 'kernel_source' "
                f"beside it"
            )
        launches = list(sources.values())
    elif isinstance(entry.get("kernel_source"), dict):
        launches = [entry["kernel_source"]]
    else:
        raise ValueError(f"{path}: kernel {name!r} has no 'kernel_source' object")
    for launch in launches:
        spec = launch.get("spec")
        if spec is not None and not isinstance(spec, dict):
            raise ValueError(
                f"{path}: kernel {name!r} has a 'spec' that is not an object"
            )


def read_prior_bundle(root: Path, engine_id: str, kernel_match_id: str) -> PriorBundle:
    """The KDPs under ``root`` that name ``engine_id``, plus the engine-level
    documents when ``root`` holds the engine's UED.

    A satellite root holds no UMD files, so its KDP may not carry a graph-scope
    matcher: sibling mode is single-pack only.
    """
    tree = _Root(root)
    kdps = tree.kdps_of(engine_id)
    if not kdps:
        raise tree.no_kdp_error(engine_id)
    _check_dispatch(kdps, engine_id, root)
    prior: list[PriorKdp] = []
    names: dict[str, Path] = {}
    for path, kdp in kdps:
        matchers = list(kdp.get("matchers") or [])
        if kernel_match_id not in matchers:
            raise ValueError(
                f"{path}: matchers {matchers} do not include the engine's "
                f"kernel-scope matcher {kernel_match_id}"
            )
        others = [m for m in matchers if m != kernel_match_id]
        if len(others) > 1:
            raise ValueError(
                f"{path}: {len(others)} matchers besides the kernel-scope one; a "
                f"KDP carries at most one graph-scope matcher"
            )
        operation = None
        if others:
            umd = tree.resolve("umd", others[0], "matcher", path)
            if umd.get("scope") != "graph":
                raise ValueError(f"{path}: matcher {others[0]} is not graph-scope")
            operation = others[0]
        name = kdp.get("name")
        if name in names:
            raise ValueError(
                f"two KDPs named {name!r} under {root}: {names[name]} and {path}"
            )
        names[name] = path
        arch = kdp.get("arch")
        if arch is not None and not _is_arch_list(arch):
            raise ValueError(f"{path}: 'arch' is not a list of strings")
        for entry in kdp.get("kernelDescriptors") or []:
            _check_entry(path, entry)
        prior.append(
            PriorKdp(
                path=path,
                id=kdp["id"],
                name=name,
                dispatch=kdp["dispatch"],
                arch=list(arch) if arch is not None else None,
                operation_umd_id=operation,
                entries=list(kdp.get("kernelDescriptors") or []),
            )
        )
    engine = None
    ueds = [d for _, d in tree.docs["ued"] if d.get("id") == engine_id]
    if ueds:
        engine = _engine_docs(tree, ueds[0]["name"])
    return PriorBundle(root=root, engine=engine, kdps=prior, all_ids=tree.all_ids)


def list_docs(root: Path, *kinds: str) -> list[Path]:
    """The descriptor files of the given kinds under ``root``; empty when
    ``root`` is not a directory."""
    if not root.is_dir():
        return []
    return sorted(p for kind in kinds for p in root.rglob(f"*.{kind}.json"))


def holds_docs(root: Path, *kinds: str) -> bool:
    """Whether ``root`` holds a descriptor of any of the given kinds."""
    return root.is_dir() and any(
        True for kind in kinds for _ in root.rglob(f"*.{kind}.json")
    )


def _dvc_outs(pointer: Path) -> list[dict]:
    """The ``outs`` entries of a ``*.dvc`` pointer that name a ``path``; empty
    when the pointer cannot be read or is not the mapping DVC writes."""
    try:
        doc = yaml.safe_load(pointer.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    outs = doc.get("outs") if isinstance(doc, dict) else None
    if not isinstance(outs, list):
        return []
    return [
        out
        for out in outs
        if isinstance(out, dict) and isinstance(out.get("path"), str)
    ]


def _dvc_md5(path: Path) -> str:
    """The md5 DVC records for a file: text is hashed with CRLF read as LF, so a
    Windows checkout hashes like the object in the remote."""
    data = path.read_bytes()
    if b"\0" not in data:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.md5(data).hexdigest()


def dvc_mismatches(root: Path) -> list[str]:
    """One message per ``*.dvc`` pointer under ``root`` whose tracked output is
    present but does not hash to what the pointer records.

    The reader takes whatever is on disk, so a stale or edited artifact would
    otherwise be reused without a word. Only warned about, since copying a new
    artifact over the old one before ``dvc add`` is the normal editing flow.
    """
    messages = []
    for pointer in sorted(root.rglob("*.dvc")):
        for out in _dvc_outs(pointer):
            recorded, rel = str(out.get("md5", "")), out.get("path")
            target = pointer.parent / rel if rel else None
            if not recorded or target is None or not target.exists():
                continue
            if recorded.endswith(".dir"):
                if not target.is_dir():
                    continue
                manifest = [
                    {
                        "md5": _dvc_md5(file),
                        "relpath": file.relative_to(target).as_posix(),
                    }
                    for file in sorted(target.rglob("*"))
                    if file.is_file()
                ]
                manifest.sort(key=lambda item: item["relpath"])
                actual = hashlib.md5(json.dumps(manifest).encode()).hexdigest() + ".dir"
            elif target.is_file():
                actual = _dvc_md5(target)
            else:
                continue
            if actual != recorded:
                messages.append(
                    f"{target} hashes to {actual}, but {pointer} records "
                    f"{recorded}: the local artifact is stale or edited (dvc pull "
                    f"to restore it, dvc add to record it)."
                )
    return messages


def holds_kdp_of(directory: Path, engine_id: str) -> bool:
    """Whether a KDP under ``directory`` names ``engine_id``: the only trace of
    the engine in a satellite, which holds no UED. A KDP that cannot be read, or
    is not a JSON object, names no engine."""
    for kdp in list_docs(directory, "kdp"):
        try:
            text = kdp.read_text(encoding="utf-8")
            if engine_id not in text:
                continue
            doc = json.loads(text)
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("engine") == engine_id:
            return True
    return False


def _ued_ids(directory: Path, engine_name: str) -> list[str]:
    ids = []
    for ued in sorted(directory.rglob("*.ued.json")):
        try:
            doc = json.loads(ued.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("name") == engine_name:
            ids.append(doc.get("id"))
    return ids


def unnamed_siblings(parent: Path, engine_id: str, named: Sequence[Path]) -> list[Path]:
    """The directories directly under ``parent`` holding a KDP of ``engine_id``
    that are not among ``named``: bundles of the same engine sitting beside this
    one that nobody told the generator about."""
    if not parent.is_dir():
        return []
    skip = {path.resolve() for path in named}
    return [
        child
        for child in sorted(parent.iterdir())
        if child.is_dir()
        and child.resolve() not in skip
        and holds_kdp_of(child, engine_id)
    ]


def discover_siblings(parent: Path, engine_name: str, own: Path) -> list[Path]:
    """The other bundles of engine ``engine_name`` among the directories directly
    under ``parent``, ``own`` excluded.

    A bundle counts when it holds the engine's UED or a KDP naming that UED's id.
    The UED is looked for in ``own`` as well, so a home bundle finds its
    satellites; a satellite whose own directory is not installed yet finds the
    home beside where it will be."""
    if not parent.is_dir():
        return []
    skip = own.resolve()
    children = [
        child
        for child in sorted(parent.iterdir())
        if child.is_dir() and child.resolve() != skip
    ]
    ids = set(_ued_ids(own, engine_name)) if own.is_dir() else set()
    homes = {child: _ued_ids(child, engine_name) for child in children}
    ids.update(identity for found in homes.values() for identity in found)
    if not ids:
        return []
    return [
        child
        for child in children
        if homes[child] or any(holds_kdp_of(child, identity) for identity in ids)
    ]


def unfetched_dvc(parent: Path, skip: Sequence[Path]) -> list[str]:
    """One message per ``*.dvc`` pointer in a directory beside this bundle whose
    tracked output is absent, for directories holding no UED and no KDP.

    A satellite holds only a KDP, so a satellite whose KDP is an unpulled DVC
    artifact is a directory with no descriptor for discovery to find. A directory
    that does hold a UED or a KDP is already identified as some engine's, and one
    whose artifact is missing is caught when it is read, so it is not repeated
    here for every unrelated engine."""
    if not parent.is_dir():
        return []
    skipped = {path.resolve() for path in skip}
    messages = []
    for child in sorted(parent.iterdir()):
        if not child.is_dir() or child.resolve() in skipped:
            continue
        if holds_docs(child, "ued", "kdp"):
            continue
        for pointer in sorted(child.rglob("*.dvc")):
            for out in _dvc_outs(pointer):
                rel = out.get("path")
                if rel and not (pointer.parent / rel).exists():
                    messages.append(
                        f"{pointer} tracks {rel}, which is not present: if that is a "
                        f"KDP of this engine, it is invisible here and its kernels "
                        f"are missing from the duplicate check (dvc pull)."
                    )
    return messages
