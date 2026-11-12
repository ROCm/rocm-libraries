# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Comparator for flavor-keyed LLVM-IR golden fixtures.

A golden stores one sub-document per distinct emission profile under
``flavors``. ``aliases`` maps a compiler lane to another lane with the same
profile, avoiding duplicate fixtures while still lowering and checking the
aliased lane. Each sub-document holds a ``cases`` map and optionally an
``expected_failures`` map. Lowering takes the lane as an argument, so any host
can verify every qualified lane, not just the one it autodetects.

A golden is either one JSON file or a directory of shard files, one per family:
``<dir>/<family>.json`` holds the cases whose ``case_id`` starts with
``<family>/``. Each shard is a complete document (``schema``, ``aliases`` and
``flavors``), and :func:`load_golden` merges them. Sharding keeps unrelated
changes in separate files, so they merge without conflicts.

:data:`GOLDEN_FLAVORS` is :data:`~rocke.core.lower_llvm.LLVM_FLAVORS` itself.
Adding a lane there makes every golden checked through :func:`check_golden`
fail until it receives a profile document or an explicit shared-profile alias,
with no edit to the golden test. This module is
shipped so ``library/`` goldens can share it (the dependency stays
``library -> platform``).

It also holds the writer side every representative-IR corpus shares: lowering
one case (:func:`lower_golden_case`), running a corpus at one flavor
(:func:`run_golden_cases`), assembling the profile-keyed document
(:func:`golden_document`), writing it as shards (:func:`write_golden_shards`)
and the ``--write`` / ``--check`` command line (:func:`golden_main`). A corpus
module (the platform harness, ``library/tests/test_conv_ir_golden.py``) keeps
only its cases and thin wrappers over these.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .llvm_profile import LLVM_EMISSION_ALIASES, LLVM_EMISSION_FLAVORS, LLVM_FLAVORS
from .lower_llvm import _lower_kernel_to_llvm_python

#: Lanes every golden must represent directly or through an alias. Same object as
#: :data:`LLVM_FLAVORS`, never a copy.
GOLDEN_FLAVORS = LLVM_FLAVORS
GOLDEN_PROFILE_FLAVORS = LLVM_EMISSION_FLAVORS
GOLDEN_ALIASES = LLVM_EMISSION_ALIASES
#: Schema tag every representative-IR golden document and shard carries.
GOLDEN_SCHEMA = "ck.dsl.ir_golden_sha256/v2"

__all__ = [
    "GOLDEN_ALIASES",
    "GOLDEN_FLAVORS",
    "GOLDEN_PROFILE_FLAVORS",
    "GOLDEN_SCHEMA",
    "check_golden",
    "check_golden_flavors",
    "compare",
    "golden_document",
    "golden_main",
    "golden_shard",
    "golden_subdocument",
    "ir_sha256",
    "load_golden",
    "lower_golden_case",
    "run_golden_cases",
    "write_golden_shards",
]

_SECTIONS = ("cases", "expected_failures")

# ``lower(case, flavor) -> (record, llvm_text)``, as :func:`lower_golden_case`.
_LowerCase = Callable[[Mapping[str, Any], str], tuple[dict[str, Any], str]]


def golden_shard(case_id: str) -> str:
    """Return the shard (family) a case is filed under: its first path segment."""
    return case_id.split("/", 1)[0]


def load_golden(path: Path) -> dict[str, Any]:
    """Read a golden file, or merge a directory of per-family shards.

    Raises ``ValueError`` if a shard is not valid JSON, if the shards disagree
    on ``schema`` or ``aliases``, or if a shard holds a case that belongs to
    another shard.
    """
    path = Path(path)
    if not path.is_dir():
        return json.loads(path.read_text())
    shards = sorted(path.glob("*.json"))
    if not shards:
        raise ValueError(f"golden directory {path} has no *.json shards")
    merged: dict[str, Any] = {}
    flavors: dict[str, dict[str, dict[str, Any]]] = {}
    for shard in shards:
        try:
            doc = json.loads(shard.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"golden shard {shard.name}: {e}") from e
        for key in ("schema", "aliases"):
            if shard is shards[0]:
                if key in doc:
                    merged[key] = doc[key]
            elif doc.get(key) != merged.get(key):
                raise ValueError(
                    f"golden shard {shard.name} has {key} {doc.get(key)!r}, "
                    f"{shards[0].name} has {merged.get(key)!r}; re-bless every shard"
                )
        for fl, sub in doc.get("flavors", {}).items():
            out = flavors.setdefault(fl, {s: {} for s in _SECTIONS})
            for section in _SECTIONS:
                for cid, rec in sub.get(section, {}).items():
                    if golden_shard(cid) != shard.stem:
                        raise ValueError(
                            f"golden shard {shard.name} holds {cid!r}, which "
                            f"belongs in {golden_shard(cid)}.json"
                        )
                    out[section][cid] = rec
    merged["flavors"] = flavors
    return merged


def golden_subdocument(doc: Mapping[str, Any], flavor: str) -> Mapping[str, Any] | None:
    """Return the baseline for ``flavor``, following one profile alias."""
    source = doc.get("aliases", {}).get(flavor, flavor)
    return doc.get("flavors", {}).get(source)


def compare(base: Mapping[str, Any], cur: Mapping[str, Any]) -> list[str]:
    """Diff one flavor sub-document against a fresh run. Empty list == OK."""
    errors = []
    for section in ("cases", "expected_failures"):
        bkeys = set(base.get(section, {}))
        ckeys = set(cur.get(section, {}))
        for missing in sorted(bkeys - ckeys):
            errors.append(f"{section}: missing current {missing}")
        for new in sorted(ckeys - bkeys):
            errors.append(f"{section}: new current {new}")
    for cid, brec in sorted(base.get("cases", {}).items()):
        crec = cur.get("cases", {}).get(cid)
        if not crec:
            continue
        if brec.get("sha256") != crec.get("sha256"):
            errors.append(f"{cid}: {brec.get('sha256')} -> {crec.get('sha256')}")
    for cid, brec in sorted(base.get("expected_failures", {}).items()):
        crec = cur.get("expected_failures", {}).get(cid)
        if not crec:
            continue
        if brec.get("type") != crec.get("type") or brec.get("message") != crec.get(
            "message"
        ):
            errors.append(f"{cid}: failure changed {brec} -> {crec}")
    return errors


def _flavor_set_errors(doc: Mapping[str, Any]) -> list[str]:
    """Check which flavor documents and aliases a loaded golden holds."""
    have = doc.get("flavors", {})
    aliases = doc.get("aliases", {})
    errors: list[str] = []
    if aliases != GOLDEN_ALIASES:
        errors.append(
            f"golden aliases {aliases} do not match emission-profile aliases "
            f"{GOLDEN_ALIASES}; re-bless it"
        )
    if set(have) != set(GOLDEN_PROFILE_FLAVORS):
        errors.append(
            f"golden profile documents {sorted(have)} do not match "
            f"{list(GOLDEN_PROFILE_FLAVORS)}; re-bless it"
        )
    overlap = set(have) & set(aliases)
    if overlap:
        errors.append(
            f"golden lanes are both direct and aliased: {sorted(overlap)}; "
            "re-bless it"
        )
    if errors:
        # Report the documents themselves first; the checks below only add
        # detail once those are right.
        return errors
    represented = set(have) | set(aliases)
    for fl in sorted(represented - set(GOLDEN_FLAVORS)):
        errors.append(
            f"golden has stale flavor {fl!r} not in LLVM_FLAVORS; re-bless it"
        )
    for fl, source in sorted(aliases.items()):
        if source not in have:
            errors.append(
                f"golden alias {fl!r} names missing flavor {source!r}; re-bless it"
            )
    return errors


def check_golden_flavors(golden_path: Path) -> list[str]:
    """Check that a golden holds exactly the flavor documents and aliases the
    current flavor set needs, without lowering anything.

    This is the part of :func:`check_golden` (with no ``flavor``) that looks
    only at the golden itself: a document for each emission profile, the
    expected aliases, and nothing for a flavor no longer in
    :data:`GOLDEN_FLAVORS`. A caller that checks the flavors one at a time,
    ``check_golden(path, run, flavor)`` for each entry of
    :data:`GOLDEN_FLAVORS`, runs this once as well to get the same coverage as
    one all-flavor call. Empty list == OK.
    """
    try:
        doc = load_golden(golden_path)
    except ValueError as e:
        return [str(e)]
    return _flavor_set_errors(doc)


def check_golden(
    golden_path: Path,
    run: Callable[[str], Mapping[str, Any]],
    flavor: str | None = None,
) -> list[str]:
    """Compare fresh runs against a golden's flavor sub-documents.

    ``run(flavor)`` lowers every case at ``flavor`` and returns a sub-document
    in the golden's shape. With no ``flavor``, the golden's flavor documents
    are checked first (see :func:`check_golden_flavors`) and then every entry
    of :data:`GOLDEN_FLAVORS` is compared. With a ``flavor``, only that
    flavor is compared. Drift strings are prefixed with the flavor when more
    than one is checked. ``golden_path`` may be a file or a shard directory
    (see :func:`load_golden`). Empty list == OK.
    """
    try:
        doc = load_golden(golden_path)
    except ValueError as e:
        return [str(e)]
    if flavor:
        wanted = [flavor]
    else:
        errors = _flavor_set_errors(doc)
        if errors:
            return errors
        wanted = list(GOLDEN_FLAVORS)
    have = doc.get("flavors", {})
    errors = []
    for fl in wanted:
        base = golden_subdocument(doc, fl)
        if base is None:
            errors.append(
                f"golden has no entry or valid alias for flavor {fl!r} "
                f"(have {sorted(have)}); re-bless it"
            )
            continue
        prefix = "" if len(wanted) == 1 else f"[{fl}] "
        errors.extend(prefix + e for e in compare(base, run(fl)))
    return errors


def ir_sha256(text: str) -> str:
    """Return the hex SHA-256 of ``text`` (UTF-8), the hash a golden records."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lower_golden_case(
    case: Mapping[str, Any], flavor: str
) -> tuple[dict[str, Any], str]:
    """Lower one corpus case at ``flavor``; return its golden record and the .ll.

    ``case`` carries ``family``, ``arch`` and a zero-argument ``build``. The
    NATIVE Python lowerer is pinned (not the backend-dispatched
    ``lower_kernel_to_llvm``, whose default ``cpp`` path silently falls back to
    Python) with an EXPLICIT llvm flavor, so the recorded sha is a function of
    the committed IR alone -- independent of whether ``rocke_engine`` is built
    or which LLVM version the loaded COMGR compiler reports.
    """
    kernel = case["build"]()
    llvm = _lower_kernel_to_llvm_python(kernel, arch=case["arch"], llvm_flavor=flavor)
    return {
        "status": "ok",
        "family": case["family"],
        "arch": case["arch"],
        "kernel_name": getattr(kernel, "name", "<unknown>"),
        "sha256": ir_sha256(llvm),
        "bytes": len(llvm.encode("utf-8")),
    }, llvm


def run_golden_cases(
    cases: Iterable[Mapping[str, Any]],
    flavor: str,
    *,
    lower: _LowerCase = lower_golden_case,
    shards: set[str] | None = None,
    ir_dir: Path | None = None,
    ir_name: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Lower every case (or only those in ``shards``) at ``flavor``.

    Returns one flavor's golden sub-document: ``cases`` maps each case id to
    its record, and a case that raises is recorded under
    ``expected_failures`` with its exception type and message. With
    ``ir_dir``, each lowered module is also written there as
    ``ir_name(case_id)``.
    """
    if ir_dir is not None and ir_name is None:
        raise ValueError("run_golden_cases: ir_dir needs ir_name")
    results: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    if ir_dir:
        ir_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        cid = case["case_id"]
        if shards is not None and golden_shard(cid) not in shards:
            continue
        try:
            rec, llvm = lower(case, flavor)
            results[cid] = rec
            if ir_dir:
                (ir_dir / ir_name(cid)).write_text(llvm)
        except Exception as e:  # noqa: BLE001 -- recorded as an expected failure
            failures[cid] = {"type": type(e).__name__, "message": str(e)}
    return {"cases": results, "expected_failures": failures}


def golden_document(run: Callable[[str], Mapping[str, Any]]) -> dict[str, Any]:
    """Return the profile-keyed golden document: ``run(flavor)`` for each of
    :data:`GOLDEN_PROFILE_FLAVORS`, with the schema and lane aliases."""
    return {
        "schema": GOLDEN_SCHEMA,
        "aliases": GOLDEN_ALIASES,
        "flavors": {fl: run(fl) for fl in GOLDEN_PROFILE_FLAVORS},
    }


def write_golden_shards(
    golden_dir: Path,
    doc: Mapping[str, Any],
    shards: set[str] | None = None,
) -> None:
    """Write ``doc`` to ``golden_dir`` as one ``<shard>.json`` per family.

    With ``shards``, only those files are rewritten (a shard left with no cases
    is deleted); the rest are untouched. Without it every shard is rewritten
    and shard files for families that no longer exist are deleted.
    """
    golden_dir = Path(golden_dir)
    by_shard: dict[str, dict] = {}
    for fl, sub in doc["flavors"].items():
        for section in _SECTIONS:
            for cid, rec in sub[section].items():
                shard = by_shard.setdefault(
                    golden_shard(cid),
                    {
                        "schema": doc["schema"],
                        "aliases": doc["aliases"],
                        "flavors": {
                            f: {"cases": {}, "expected_failures": {}}
                            for f in doc["flavors"]
                        },
                    },
                )
                shard["flavors"][fl][section][cid] = rec
    golden_dir.mkdir(parents=True, exist_ok=True)
    stale = {p.stem for p in golden_dir.glob("*.json")} - set(by_shard)
    if shards is not None:
        stale &= shards
    for name in sorted(stale):
        (golden_dir / f"{name}.json").unlink()
    for name, shard in sorted(by_shard.items()):
        (golden_dir / f"{name}.json").write_text(
            json.dumps(shard, indent=2, sort_keys=True) + "\n"
        )


def golden_main(
    *,
    cases: Callable[[], Iterable[Mapping[str, Any]]],
    write_golden: Callable[[Path, set[str] | None], Mapping[str, Any]],
    check_golden: Callable[[Path, str | None], list[str]],
    default_golden: Path | None = None,
    dump: Callable[[Path | None, str | None], Mapping[str, Any]] | None = None,
    label: str = "IR parity",
    argv: list[str] | None = None,
) -> None:
    """The ``--write`` / ``--check`` command line of a representative-IR golden.

    ``write_golden(dir, shards)`` blesses ``dir`` and returns the document it
    wrote; ``check_golden(path, flavor)`` returns the drift (empty == OK).

    With ``default_golden``, ``--write`` and ``--check`` take that directory
    when given no path, and a bare run checks it. Otherwise ``--write`` and
    ``--check`` need a path, and a bare run prints ``dump(ir_dir, flavor)``
    (``--ir-dir``, ``--flavor``; ``flavor`` is None when not given) as JSON.
    Exits 1 on drift.
    """
    if default_golden is None and dump is None:
        raise TypeError("golden_main needs default_golden or dump")
    ap = argparse.ArgumentParser()
    path_kw: dict[str, Any] = {"type": Path}
    if default_golden is not None:
        path_kw.update(nargs="?", const=default_golden)
    where = " (default: the checked-in fixture)" if default_golden is not None else ""
    ap.add_argument(
        "--write",
        help=f"(re-)bless the profile-keyed golden shard directory{where}: run "
        f"every case under each of {GOLDEN_PROFILE_FLAVORS} and write one "
        "<family>.json shard per family, with lane aliases. Run only from a "
        "verified-good tree; re-blessing should accompany a reviewed, expected "
        "output change.",
        **path_kw,
    )
    ap.add_argument(
        "--only",
        help="with --write, re-bless only these comma-separated families "
        "(shard names, the first case_id segment) and leave other shards alone.",
    )
    ap.add_argument(
        "--check",
        help="compare a fresh run against EVERY golden sub-doc (narrow it with "
        "--flavor); exit 1 on any drift. Accepts a golden file or shard "
        f"directory{where}.",
        **path_kw,
    )
    if dump is not None:
        ap.add_argument(
            "--flavor",
            choices=GOLDEN_FLAVORS,
            help="check (or dump) just this llvm flavor instead of all of them; "
            "for --ir-dir it overrides the autodetected flavor.",
        )
        ap.add_argument("--ir-dir", type=Path)
    else:
        ap.add_argument(
            "--flavor",
            choices=GOLDEN_FLAVORS,
            help="check just this llvm flavor instead of all of them.",
        )
    ns = ap.parse_args(argv)

    if ns.only is not None and not ns.write:
        ap.error("--only requires --write")
    if ns.write:
        if ns.write.suffix == ".json" or ns.write.is_file():
            ap.error(
                f"--write takes the golden shard directory, not a file: {ns.write}"
            )
        shards = None
        if ns.only is not None:
            shards = {s.strip() for s in ns.only.split(",") if s.strip()}
            if not shards:
                ap.error("--only: no families given")
            known = {golden_shard(c["case_id"]) for c in cases()}
            known |= {p.stem for p in ns.write.glob("*.json")}
            if shards - known:
                ap.error(f"--only: unknown families {sorted(shards - known)}")
        doc = write_golden(ns.write, shards)
        for fl, sub in doc["flavors"].items():
            print(
                f"wrote {fl}: {len(sub['cases'])} ok, "
                f"{len(sub['expected_failures'])} failures"
            )
        print(f"-> {ns.write}")
        return

    golden = ns.check or default_golden
    if golden is None:
        doc = dump(ns.ir_dir, ns.flavor)
        print(json.dumps(doc, indent=2, sort_keys=True))
        return

    checked = ns.flavor or ", ".join(GOLDEN_FLAVORS)
    errors = check_golden(golden, ns.flavor)
    if errors:
        for e in errors:
            print(f"IR DRIFT: {e}")
        raise SystemExit(1)
    print(f"{label} OK [{checked}] vs {golden}")
