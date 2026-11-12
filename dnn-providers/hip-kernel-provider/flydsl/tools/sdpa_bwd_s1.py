# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compose the SDPA backward catalogue ("S1") from a sweep ref and a selection policy, or rebuild a reference dir from the tree.

Tree mode, the one a regeneration from this repository alone uses (REGEN.md §2):

    python tools/sdpa_bwd_s1.py --from-content <content root> --arch <gfx950|gfx11-generic> --reference-dir <new dir>

It writes ``<hash>.hsaco`` into the new dir for every pin of the arch's reference CSV (``tests/data``) whose object the shipped content
holds: the pin's sha256 is the object's. The content roots are the arch folders of the ``sdpa_bwd_*`` ops (the objects and the
``companions`` of their manifests). The pins the content never holds (a combined object's main kernel, gfx950's standalone reduce
kernels; ``sdpa_bwd_pins.py``) are not written: their function digests are in ``tests/data/sdpa_bwd_pin_functions.csv``, and the tool
fails when one has none or any other pin has no object. The reference dir is the ``--reference-dir`` of the ``gen_sdpa_bwd_*`` generators.

Sweep mode, the maintainers' path that produces new catalogue data. It needs the AMD-internal sweep repository, which this
repository does not include and which is not publicly reproducible; a regeneration never needs it:

    python tools/sdpa_bwd_s1.py --sweep-repo <sweep git dir> --ref <git ref> \
        --selection selection.toml --out <S1 dir> \
        [--reference-dir DIR] [--tables] [--object-root DIR ...]

Inputs are the sweep repository at an immutable ref (manifests are read with
``git show``, never from the working tree), the selection policy (``selection.toml``)
and the object roots (every ``<sweep parent>/sweep*`` worktree's ``objects*/`` dirs
plus ``--object-root`` dirs, searched by hash). A rerun changes only inputs.

The sweep repository is the AMD-internal git repository that holds the kernel sweep's
per-run manifests and objects; the ref a snapshot was composed at is recorded in its
``SNAPSHOT.md``. ``selection.toml`` is the selection policy the sweep's maintainers supply
with it, not part of this tree, and nothing a regeneration needs is in it: what it selects is
installed here as the ``_instances.py`` tables, the reference CSVs, the pin function digests and
the pinned-exception list, which the tests hold to one another. A catalogue change arrives here
as new outputs.

``selection.toml`` is an ordered list of ``[[step]]`` tables applied to a
hash-keyed membership map; every step asserts its own count, the end state asserts
the per-kernel counts:

* ``base``: the ``ok`` rows of the listed manifests, deduplicated by hash (rows of
  one hash must agree), minus the hashes of ``skip``.
* ``remove``: the hashes of a list file. Each must be a member, its spec must equal
  the list's spec column, and every covering object of the list must survive all steps.
* ``repin``: replaces ``sha256`` and the resource row of a member. The hash and spec
  stay, the old sha256 must equal the list's ``old_sha256`` (when a list is given).
* ``add``: new members; a hash already present (collision) or added by two steps
  (double add) is rejected. Rows of one hash inside a step must be identical.

Source paths are ``git:<path>`` (glob, at the ref) or ``file:<path>`` (relative to the
evidence root, ``<sweep repo>/../..``). ``[[exclude]]`` hashes may not be added and
must be absent at the end.

Outputs in ``--out``:

* ``manifest.csv``: the resolved membership (hash, spec, kernel, group, sha256,
  old_sha256, resource row, source, reason, object_state, object_path, rmir_scan). A gfx950
  member's ``rmir_scan`` is ``exception`` for a row of ``sdpa_bwd_pinned_exceptions.csv``, else
  ``clean`` when the sweep's ``objs/<hash>/rmir.json`` beside a candidate object is a scan of the
  pin (``pinned_sha256``) whose dump build was the pinned object (``matched``) with no finding;
  a gfx950 member with a binary and neither fails the run;
* ``known-flaky.csv``: hash, spec, source of every known-nondeterministic pin;
* ``rebuild-required.csv``: members with no binary in any root (each must be rebuilt
  and equal its sha256, then supplied through ``--object-root``);
* ``SNAPSHOT.md``: ref, commit, input digests, counts.

``--reference-dir`` copies every member's object to ``<hash>.hsaco`` in a temporary
sibling, verifies every sha256, requires exactly the member set, then renames it into
place; it fails while any member lacks a binary. ``--tables`` rewrites the generated
``_instances.py`` block and ``tests/data`` CSVs through
``sdpa_bwd_tables_from_sweep.py``; with ``--reference-dir`` it also rewrites the arch's rows of
``sdpa_bwd_pin_functions.csv`` from the installed pins' objects. A member whose binaries all differ from its sha256
always fails the run; members with no binary leave exit status 3.
"""

from __future__ import annotations

import argparse
import ast
import csv
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generators._sdpa_bwd_rdna_build import RESOURCE_COLUMNS  # noqa: E402

PIN_COLUMNS = ("hash", "spec", "sha256", *RESOURCE_COLUMNS)
# gfx950 pins also record their R-MIR scan (``tests/data/sdpa_bwd_reference.csv``): ``clean`` is a scan whose dump build is the pinned object
# with no finding, ``exception`` a row of ``sdpa_bwd_pinned_exceptions.csv`` (a device screen stands in). The detector reads gfx950 MIR only.
RMIR_CLEAN = "clean"
RMIR_EXCEPTION = "exception"
RMIR_SCANS = (RMIR_CLEAN, RMIR_EXCEPTION)
RMIR_COLUMN = "rmir_scan"
MANIFEST_COLUMNS = (
    "hash",
    "spec",
    "kernel",
    "group",
    "sha256",
    "old_sha256",
    *RESOURCE_COLUMNS,
    "source",
    "reason",
    "object_state",
    "object_path",
    RMIR_COLUMN,
)
FLAKY_COLUMNS = ("hash", "spec", "source")
REBUILD_COLUMNS = ("hash", "spec", "kernel", "group", "sha256", "source")
HEX12 = re.compile(r"`([0-9a-f]{12})`")
HASH_RE = re.compile(r"[0-9a-f]{12}")
KINDS = ("base", "remove", "repin", "add")
EXIT_REBUILD_REQUIRED = 3


class S1Error(SystemExit):
    """A violated S1 invariant; the message is the diagnosis."""


def fail(message: str) -> S1Error:
    return S1Error(f"sdpa_bwd_s1: {message}")


def kernel_of(spec: str) -> str:
    return spec.split(",", 1)[0]


def is_gfx950(spec: str) -> bool:
    """Whether the Spec key names gfx950 (the arch is spelled only when it is another: ``;arch=gfx11-generic``)."""
    return ";arch=" not in spec


def pin_columns(arch: str) -> tuple[str, ...]:
    """The columns of the reference CSV of @p arch."""
    return (*PIN_COLUMNS, RMIR_COLUMN) if arch == "gfx950" else PIN_COLUMNS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=columns, lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_resolved(s1_dir: Path) -> tuple[list[dict], list[dict]]:
    """(manifest rows, known-flaky rows) of an S1 output dir."""
    with (s1_dir / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        manifest = list(csv.DictReader(fh))
    with (s1_dir / "known-flaky.csv").open(newline="", encoding="utf-8") as fh:
        flaky = list(csv.DictReader(fh))
    return manifest, flaky


# --------------------------------------------------------------------------- sources


class Sources:
    """``git:`` (at the ref) and ``file:`` (evidence root) inputs; records the digest of each."""

    def __init__(self, repo: Path, ref: str, evidence_root: Path):
        self.repo = repo
        self.evidence_root = evidence_root
        self.commit = self._git("rev-parse", "--verify", f"{ref}^{{commit}}").strip()
        self.tree = self._git("ls-tree", "-r", "--name-only", self.commit).splitlines()
        self.digests: dict[str, str] = {}

    def _git(self, *args: str, binary: bool = False):
        done = subprocess.run(
            ["git", "-C", str(self.repo), *args], capture_output=True, check=False
        )
        if done.returncode:
            raise fail(f"git {' '.join(args)}: {done.stderr.decode().strip()}")
        return done.stdout if binary else done.stdout.decode("utf-8")

    def expand(self, source: str) -> list[str]:
        """Source names a (possibly globbed) ``git:``/``file:`` spec stands for, sorted."""
        scheme, _, pattern = source.partition(":")
        if scheme == "git":
            regex = re.compile(
                "".join(
                    "[^/]*" if part == "*" else re.escape(part)
                    for part in re.split(r"(\*)", pattern)
                )
                + r"\Z"
            )
            names = sorted(p for p in self.tree if regex.match(p))
            if not names:
                raise fail(f"{source}: no such path at {self.commit[:7]}")
            return [f"git:{p}" for p in names]
        if scheme == "file":
            if not (self.evidence_root / pattern).is_file():
                raise fail(f"{source}: no such file under {self.evidence_root}")
            return [source]
        raise fail(f"{source!r}: expected a git: or file: source")

    def text(self, name: str) -> str:
        scheme, _, path = name.partition(":")
        if scheme == "git":
            raw = self._git("show", f"{self.commit}:{path}", binary=True)
        else:
            raw = (self.evidence_root / path).read_bytes()
        self.digests[name] = hashlib.sha256(raw).hexdigest()
        return raw.decode("utf-8")

    def rows(self, name: str) -> list[dict]:
        return list(csv.DictReader(io.StringIO(self.text(name))))


# ---------------------------------------------------------------------------- policy


@dataclass
class Member:
    row: dict  # hash, spec, sha256, resources
    group: str
    source: str
    reason: str
    old_sha256: str = ""


@dataclass
class Composition:
    members: dict[str, Member]
    steps: list[dict]  # name, kind, count, note
    removed: dict[str, str]  # hash -> step
    skipped: list[tuple[str, str, str]]  # (step, hash, reason)
    retried: set[str]  # pinned hashes with attempts > 1 in a base manifest row


def pin_row(row: dict, where: str) -> dict:
    missing = [c for c in PIN_COLUMNS if c not in row]
    if missing:
        raise fail(f"{where}: columns {missing} missing")
    if not HASH_RE.fullmatch(row["hash"]) or kernel_of(row["spec"]) == "":
        raise fail(f"{where}: malformed row {row['hash']!r}")
    return {c: row[c] for c in PIN_COLUMNS}


def read_step_rows(
    src: Sources, step: dict, skip: set[str]
) -> tuple[dict[str, tuple[dict, str]], set[str]]:
    """({hash: (pin row, source)} of the step's ok rows, retried hashes)."""
    found: dict[str, tuple[dict, str]] = {}
    retried: set[str] = set()
    for pattern in step["manifests"]:
        for name in src.expand(pattern):
            for raw in src.rows(name):
                if int(raw.get("attempts") or 1) > 1:
                    retried.add(raw["hash"])
                if raw.get("status", "ok") != "ok" or raw["hash"] in skip:
                    continue
                row = pin_row(raw, name)
                seen = found.setdefault(row["hash"], (row, name))
                for column in PIN_COLUMNS:
                    if seen[0][column] != row[column]:
                        raise fail(
                            f"step {step['name']}: {row['hash']} disagrees on "
                            f"{column} between {seen[1]} and {name}: "
                            f"{seen[0][column]!r} vs {row[column]!r}"
                        )
    return found, retried


def list_hashes(src: Sources, source: str) -> set[str]:
    out: set[str] = set()
    for name in src.expand(source):
        rows = src.rows(name)
        out |= {r["hash"] for r in rows}
    return out


def compose(selection: dict, src: Sources) -> Composition:
    members: dict[str, Member] = {}
    steps: list[dict] = []
    removed: dict[str, str] = {}
    skipped: list[tuple[str, str, str]] = []
    retried: set[str] = set()
    excluded: dict[str, str] = {}
    for entry in selection.get("exclude", []):
        for h in entry["hashes"]:
            excluded[h] = entry["reason"]
    added_by: dict[str, tuple[str, str]] = {}  # hash -> (step, kind)
    covering: list[tuple[str, str, str]] = []  # (step, removed hash, covering hash)
    names: set[str] = set()

    for step in selection["step"]:
        name, kind = step["name"], step["kind"]
        if kind not in KINDS or name in names:
            raise fail(f"step {name!r}: unknown kind {kind!r} or duplicate name")
        names.add(name)
        skip = set()
        for source in step.get("skip", []):
            skip |= list_hashes(src, source)
        reason = step.get("reason", name)

        if kind == "base":
            if members:
                raise fail(f"step {name}: base must come first")
            found, retried = read_step_rows(src, step, skip)
            for h, (row, source) in found.items():
                members[h] = Member(row, name, source, reason)
                added_by[h] = (name, kind)
            count = len(found)

        elif kind == "remove":
            listed = src.rows(src.expand(step["list"])[0])
            where = step.get("where", {})
            listed = [r for r in listed if all(r[k] == v for k, v in where.items())]
            if "crosscheck" in step:
                other = src.rows(src.expand(step["crosscheck"])[0])
                other = [
                    r
                    for r in other
                    if all(
                        r[k] == v for k, v in step.get("crosscheck_where", {}).items()
                    )
                ]
                a = {r[step["hashes_column"]] for r in listed}
                b = {r[step["hashes_column"]] for r in other}
                if a != b:
                    raise fail(
                        f"step {name}: list and crosscheck differ: "
                        f"{sorted(a ^ b)[:4]} ..."
                    )
            count = 0
            for r in listed:
                hashes = r[step["hashes_column"]].split(";")
                specs = [s.strip() for s in r[step["spec_column"]].split(" | ")]
                if len(hashes) != len(specs):
                    raise fail(f"step {name}: {hashes} do not pair with {specs}")
                for h, spec in zip(hashes, specs):
                    if h not in members:
                        raise fail(f"step {name}: removal {h} is not a member")
                    if members[h].row["spec"] != spec:
                        raise fail(
                            f"step {name}: removal {h} is "
                            f"{members[h].row['spec']!r}, not {spec!r}"
                        )
                    if h in removed:
                        raise fail(f"step {name}: {h} removed twice")
                    removed[h] = name
                    del members[h]
                    count += 1
                for c in filter(None, r[step["covering_column"]].split(";")):
                    covering.append((name, hashes[0], c))

        elif kind == "repin":
            found, _ = read_step_rows(src, step, skip)
            old_of: dict[str, str] = {}
            if "delta" in step:
                delta = src.rows(src.expand(step["delta"])[0])
                where = step.get("delta_where", {})
                delta = [r for r in delta if all(r[k] == v for k, v in where.items())]
                missing = {r["hash"] for r in delta} - found.keys()
                if missing:
                    raise fail(
                        f"step {name}: {sorted(missing)[:4]} listed but not in the source"
                    )
                old_of = {r["hash"]: r["old_sha256"] for r in delta}
                new_of = {r["hash"]: r["new_sha256"] for r in delta}
                found = {h: v for h, v in found.items() if h in old_of}
                for h, (row, source) in found.items():
                    if row["sha256"] != new_of[h]:
                        raise fail(
                            f"step {name}: {h} source sha256 differs from the list's new_sha256"
                        )
            count = 0
            for h, (row, source) in sorted(found.items()):
                if h in step.get("ignore", {}):
                    skipped.append((name, h, step["ignore"][h]))
                    continue
                member = members.get(h)
                if member is None:
                    raise fail(f"step {name}: {h} is not a member (no ignore entry)")
                if member.row["spec"] != row["spec"]:
                    raise fail(f"step {name}: {h} changes spec")
                if h in old_of and old_of[h] != member.row["sha256"]:
                    raise fail(
                        f"step {name}: {h} is {member.row['sha256']}, "
                        f"the list replaces {old_of[h]}"
                    )
                if member.row["sha256"] == row["sha256"]:
                    raise fail(f"step {name}: {h} already has sha256 {row['sha256']}")
                member.old_sha256 = member.old_sha256 or member.row["sha256"]
                member.row = row
                member.source = source
                member.reason = f"{member.reason}; {reason}"
                count += 1
            for h in step.get("ignore", {}):
                if h not in {t[1] for t in skipped}:
                    raise fail(f"step {name}: ignore entry {h} matches no source row")

        else:  # add
            found, _ = read_step_rows(src, step, skip)
            for h, (row, source) in found.items():
                if h in excluded:
                    raise fail(f"step {name}: {h} is excluded ({excluded[h]})")
                if h in members:
                    first, first_kind = added_by[h]
                    what = "double add" if first_kind == "add" else "collision"
                    raise fail(f"step {name}: {what}: {h} is a member (from {first})")
                if h in removed:
                    raise fail(f"step {name}: re-add of {h}, removed by {removed[h]}")
                members[h] = Member(row, name, source, reason)
                added_by[h] = (name, kind)
            count = len(found)

        if "expect" in step and step["expect"] != count:
            raise fail(f"step {name}: expected {step['expect']} rows, got {count}")
        steps.append({"name": name, "kind": kind, "count": count})

    for h, why in excluded.items():
        if h in members:
            raise fail(f"{h} is excluded ({why}) but a member")
    for step_name, removed_hash, c in covering:
        if c not in members:
            raise fail(
                f"step {step_name}: removal of {removed_hash} loses its covering object {c}"
            )
    return Composition(members, steps, removed, skipped, retried)


def check_expected(selection: dict, comp: Composition) -> dict[str, int]:
    counts = Counter(kernel_of(m.row["spec"]) for m in comp.members.values())
    expect = selection.get("expect", {})
    if "total" in expect and expect["total"] != len(comp.members):
        raise fail(f"expected {expect['total']} members, got {len(comp.members)}")
    wanted = expect.get("kernels", {})
    if wanted and dict(sorted(counts.items())) != dict(sorted(wanted.items())):
        raise fail(
            f"per-kernel counts {dict(sorted(counts.items()))} != "
            f"expected {dict(sorted(wanted.items()))}"
        )
    return dict(counts)


# ---------------------------------------------------------------------------- flaky


def external_hashes(text: str, which: str) -> set[str]:
    if which == "bimodal":
        return set(
            HEX12.findall(
                text.split("**Affected so far:**")[1].split("Each rebuilds")[0]
            )
        )
    (paragraph,) = [p for p in text.split("\n\n") if "needed retries" in p]
    return set(HEX12.findall(paragraph))


def flaky_sources(
    selection: dict, src: Sources, comp: Composition
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """({hash: [source tag, ...]} of the known-flaky members in policy order, {hash: removing step} of dropped ones)."""
    found: dict[str, list[str]] = {}

    def add(h: str, tag: str) -> None:
        tags = found.setdefault(h, [])
        if tag not in tags:
            tags.append(tag)

    for entry in selection.get("flaky", []):
        kind, tag = entry["kind"], entry.get("source", "")
        if kind == "csv":
            for r in src.rows(src.expand(entry["path"])[0]):
                for t in r["source"].split("+"):
                    add(r["hash"], t)
        elif kind == "hashes":
            for h in entry["hashes"]:
                add(h, tag)
        elif kind == "md-section":
            text = src.text(src.expand(entry["path"])[0])
            _, _, rest = text.partition(entry["start"])
            section = re.split(r"^## ", rest, maxsplit=1, flags=re.M)[0]
            hashes = HEX12.findall(section)
            if not hashes:
                raise fail(f"flaky {entry['path']}: no hash under {entry['start']!r}")
            for h in hashes:
                add(h, tag)
        elif kind == "py-literal":
            text = src.text(src.expand(entry["path"])[0])
            for node in ast.parse(text).body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == entry["name"]
                    for t in node.targets
                ):
                    for h in ast.literal_eval(node.value):
                        add(h, tag)
                    break
            else:
                raise fail(f"flaky {entry['path']}: no {entry['name']}")
        elif kind == "external-md":
            text = src.text(src.expand(entry["path"])[0])
            for h in sorted(external_hashes(text, entry["which"])):
                add(h, tag)
        elif kind == "attempts":
            for h in sorted(comp.retried & comp.members.keys()):
                add(h, tag)
        else:
            raise fail(f"flaky: unknown kind {kind!r}")

    out: dict[str, list[str]] = {}
    dropped: dict[str, str] = {}
    for h, tags in sorted(found.items()):
        if h in comp.members:
            out[h] = tags
        elif h in comp.removed:
            dropped[h] = comp.removed[h]
        else:
            raise fail(f"flaky {h} ({'+'.join(tags)}) is not a member")
    return out, dropped


# --------------------------------------------------------------------------- objects


class ObjectIndex:
    """hash -> candidate object files, in root order then path order."""

    def __init__(self, roots: list[tuple[Path, str]], base: Path):
        """``roots``: (dir, glob of the child dirs holding objects; ``*`` = the dir itself)."""
        self.base = base
        self.candidates: dict[str, list[Path]] = {}
        for root, object_dirs in roots:
            self._scan(root, object_dirs)

    def _add(self, h: str, path: Path) -> None:
        self.candidates.setdefault(h, []).append(path)

    def _scan(self, root: Path, object_dirs: str) -> None:
        """Binaries under ``objects*/``: ``objs/<hash>/{,load/}obj.hsaco`` and ``<hash>.hsaco``."""
        if not root.is_dir():
            return
        starts = [
            p for p in sorted(root.iterdir()) if fnmatch.fnmatch(p.name, object_dirs)
        ]
        if object_dirs == "*":
            starts = [root]
        flat: list[tuple[str, Path]] = []
        primary: list[tuple[str, Path]] = []
        loaded: list[tuple[str, Path]] = []
        for start in starts:
            if not start.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(start):
                dirnames[:] = sorted(d for d in dirnames if d != "cache")
                here = Path(dirpath)
                for name in sorted(filenames):
                    if name == "obj.hsaco":
                        if HASH_RE.fullmatch(here.name) and here.parent.name == "objs":
                            primary.append((here.name, here / name))
                        elif (
                            here.name == "load"
                            and HASH_RE.fullmatch(here.parent.name)
                            and here.parent.parent.name == "objs"
                        ):
                            loaded.append((here.parent.name, here / name))
                    elif name.endswith(".hsaco") and HASH_RE.fullmatch(name[:-6]):
                        flat.append((name[:-6], here / name))
        for h, path in flat + primary + loaded:
            self._add(h, path)

    def rel(self, path: Path) -> str:
        try:
            return path.relative_to(self.base).as_posix()
        except ValueError:
            return str(path)

    def resolve(self, h: str, sha256: str) -> tuple[Path | None, list[str]]:
        """(matching object or None, shas of the candidates that did not match)."""
        wrong: list[str] = []
        for path in self.candidates.get(h, []):
            digest = sha256_file(path)
            if digest == sha256:
                return path, wrong
            wrong.append(digest)
        return None, wrong

    def scan_is_clean(self, h: str, sha256: str) -> bool:
        """Whether a sweep scan of the pin ``sha256`` is on record and clean: ``objs/<hash>/rmir.json`` beside a candidate object, status ``ok``, its
        ``pinned_sha256`` the pin, ``matched`` (the dump build is the pinned object) and no spill finding.
        """
        for path in self.candidates.get(h, []):
            owner = path.parent.parent if path.parent.name == "load" else path.parent
            record = owner / "rmir.json"
            if owner.name != h or not record.is_file():
                continue
            scan = json.loads(record.read_text(encoding="utf-8"))
            if (
                scan.get("status") == "ok"
                and scan.get("pinned_sha256") == sha256
                and scan.get("matched") is True
                and scan.get("n_spill_findings") == 0
            ):
                return True
        return False


def object_roots(
    selection: dict, repo: Path, extra: list[Path]
) -> tuple[list[tuple[Path, str]], Path]:
    spec = selection.get("objects", {})
    base = repo.parent
    roots: list[Path] = []
    for pattern in spec.get("roots", ["sweep*"]):
        roots += sorted(p for p in base.glob(pattern) if p.is_dir())
    order = {name: i for i, name in enumerate(spec.get("prefer", []))}
    roots.sort(key=lambda p: (order.get(p.name, len(order)), p.name))
    dirs = spec.get("dirs", "objects*")
    return [*((r, dirs) for r in roots), *((r, "*") for r in extra)], base


# ---------------------------------------------------------------------------- outputs


def read_exception_hashes(data_dir: Path) -> set[str]:
    """The Spec hashes of ``sdpa_bwd_pinned_exceptions.csv``."""
    with (data_dir / "sdpa_bwd_pinned_exceptions.csv").open(
        newline="", encoding="utf-8"
    ) as fh:
        return {r["hash"] for r in csv.DictReader(fh)}


def resolve_objects(
    comp: Composition,
    index: ObjectIndex,
    exceptions: frozenset[str] | set[str] = frozenset(),
) -> tuple[list[dict], list[dict], list[str], list[str]]:
    """(manifest rows, rebuild-required rows, sha-mismatch diagnoses, gfx950 pins with no clean scan on record).

    A gfx950 member with a binary records ``rmir_scan``: ``exception`` for a row of @p exceptions, ``clean`` for a matched clean sweep scan of the
    pin (``ObjectIndex.scan_is_clean``), else it is listed as unscanned and its ``rmir_scan`` stays empty. A gfx11 member records none.
    """
    rows: list[dict] = []
    unscanned: list[str] = []
    rebuild: list[dict] = []
    mismatched: list[str] = []
    for h, m in sorted(comp.members.items(), key=lambda kv: kv[1].row["spec"]):
        path, wrong = index.resolve(h, m.row["sha256"])
        kernel = kernel_of(m.row["spec"])
        if path is not None:
            state, shown = "found", index.rel(path)
        elif wrong:
            mismatched.append(
                f"{h} {m.row['spec']}: pin {m.row['sha256']}, "
                f"{len(wrong)} binaries with other shas ({wrong[0]} ...)"
            )
            state, shown = "sha-mismatch", ""
        else:
            state, shown = "rebuild-required", ""
            rebuild.append(
                {
                    "hash": h,
                    "spec": m.row["spec"],
                    "kernel": kernel,
                    "group": m.group,
                    "sha256": m.row["sha256"],
                    "source": m.source,
                }
            )
        scan = ""
        if is_gfx950(m.row["spec"]) and state == "found":
            if h in exceptions:
                scan = RMIR_EXCEPTION
            elif index.scan_is_clean(h, m.row["sha256"]):
                scan = RMIR_CLEAN
            else:
                unscanned.append(f"{h} {m.row['spec']}")
        rows.append(
            {
                **m.row,
                RMIR_COLUMN: scan,
                "kernel": kernel,
                "group": m.group,
                "old_sha256": m.old_sha256,
                "source": m.source,
                "reason": m.reason,
                "object_state": state,
                "object_path": shown,
            }
        )
    return rows, rebuild, mismatched, unscanned


def build_reference_dir(rows: list[dict], index: ObjectIndex, target: Path) -> None:
    """Copy every member's object to ``<hash>.hsaco`` atomically: verified, exact, then renamed."""
    if target.exists():
        raise fail(f"{target} exists; refusing to replace a reference dir")
    missing = [r["hash"] for r in rows if r["object_state"] != "found"]
    if missing:
        raise fail(
            f"reference dir: {len(missing)} members have no verified binary "
            f"({', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''})"
        )
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        for r in rows:
            source = index.base / r["object_path"]
            if not source.is_file():
                source = Path(r["object_path"])
            dest = tmp / f"{r['hash']}.hsaco"
            shutil.copyfile(source, dest)
            if sha256_file(dest) != r["sha256"]:
                raise fail(f"{dest.name}: copied sha256 differs from the pin")
        have = {p.name for p in tmp.iterdir()}
        want = {f"{r['hash']}.hsaco" for r in rows}
        if have != want:
            raise fail(
                f"reference dir: missing {sorted(want - have)[:4]}, extra {sorted(have - want)[:4]}"
            )
        tmp.rename(target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def shipped_objects(content: Path, arch: str) -> dict[str, Path]:
    """{sha256: object file} of every ``sdpa_bwd_*`` object the content ships for @p arch: the instances and the companions of each manifest."""
    found: dict[str, Path] = {}
    manifests = sorted(content.glob(f"sdpa_bwd_*/{arch}/manifest.json"))
    if not manifests:
        raise fail(f"no sdpa_bwd_*/{arch}/manifest.json under {content}")
    for manifest in manifests:
        document = json.loads(manifest.read_text(encoding="utf-8"))
        for entry in (*document["instances"], *document.get("companions", ())):
            if entry.get("file") and entry.get("sha256"):
                found.setdefault(entry["sha256"], manifest.parent / entry["file"])
    return found


def reference_dir_from_content(
    content: Path, arch: str, target: Path, data_dir: Path | None = None
) -> tuple[int, int]:
    """Write the pins of @p arch the content ships to ``<hash>.hsaco`` under the new dir @p target; (objects written, pins named by a function digest).

    A gfx950 pin is taken only when its row records a scan (``rmir_scan``: ``clean`` or ``exception``); one without fails and nothing is written.
    Every reference row either has its object in the content (verified against the row's sha256) or is a pin the content never holds
    (``sdpa_bwd_pins.holds_no_shipped_object``) with a row in ``sdpa_bwd_pin_functions.csv``; anything else fails and nothing is written.
    """
    import sdpa_bwd_pins as pins
    import sdpa_bwd_tables_from_sweep as importer
    from generators._sdpa_bwd_spec import Spec

    data_dir = data_dir or pins.FLYDSL / "tests" / "data"
    if target.exists():
        raise fail(f"{target} exists; refusing to replace a reference dir")
    if arch not in importer.ARCH_OUTPUTS:
        raise fail(f"no reference data for arch {arch!r}")
    with (data_dir / importer.ARCH_OUTPUTS[arch].reference_csv).open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if arch == "gfx950":
        unscanned = [
            f"{r['hash']} ({r['spec']}): {r.get(RMIR_COLUMN)!r}"
            for r in rows
            if r.get(RMIR_COLUMN) not in RMIR_SCANS
        ]
        if unscanned:
            raise fail(
                f"{len(unscanned)} gfx950 pins record no clean R-MIR scan ({RMIR_COLUMN} is {'/'.join(RMIR_SCANS)}):\n  "
                + "\n  ".join(unscanned[:20])
            )
    functions = pins.read_pin_functions(data_dir / pins.PIN_FUNCTIONS_CSV.name)
    shipped = shipped_objects(content, arch)
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    written = named = 0
    try:
        for row in rows:
            source = shipped.get(row["sha256"])
            # A pin the content never holds is named by its function digest even when its sha256 is a shipped combined object
            # (a tree-built combined pin, ``sdpa_bwd_tables_from_sweep.py --tree-pins``): the object is not the one-kernel pin.
            if (
                pins.holds_no_shipped_object(Spec.parse(row["spec"]))
                and functions.get(row["hash"], {}).get("spec") == row["spec"]
            ):
                named += 1
            elif source is not None:
                dest = tmp / f"{row['hash']}.hsaco"
                shutil.copyfile(source, dest)
                if sha256_file(dest) != row["sha256"]:
                    raise fail(f"{source}: its sha256 is not the manifest's")
                written += 1
            else:
                raise fail(
                    f"{row['hash']} ({row['spec']}): no shipped object has the pin's sha256 "
                    f"and {pins.PIN_FUNCTIONS_CSV.name} names no function digest for it"
                )
        tmp.rename(target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return written, named


def write_pin_functions(
    arch: str, pins_by_hash: dict[str, dict], reference_dir: Path, data_dir: Path
) -> int:
    """Rewrite the rows of @p arch in ``sdpa_bwd_pin_functions.csv`` from the pins' objects under @p reference_dir; the number of rows."""
    import sdpa_bwd_pins as pins
    from generators._sdpa_bwd_spec import Spec

    path = data_dir / pins.PIN_FUNCTIONS_CSV.name
    kept = [
        row
        for row in pins.read_pin_functions(path).values()
        if Spec.parse(row["spec"]).arch != arch
    ]
    mine = [
        {
            "hash": row["hash"],
            "spec": row["spec"],
            "function_sha256": pins.function_sha256(
                (reference_dir / f"{row['hash']}.hsaco").read_bytes()
            ),
        }
        for row in pins_by_hash.values()
        if pins.holds_no_shipped_object(Spec.parse(row["spec"]))
    ]
    rows = sorted([*kept, *mine], key=lambda row: row["spec"])
    write_csv(path, pins.PIN_FUNCTION_COLUMNS, rows)
    return len(mine)


def snapshot_text(
    selection: dict,
    src: Sources,
    ref: str,
    comp: Composition,
    counts: dict[str, int],
    flaky: dict[str, list[str]],
    dropped: dict[str, str],
    rebuild: list[dict],
    roots: list[Path],
    selection_digest: str,
) -> str:
    lines = [
        f"# S1: sweep snapshot `{src.commit[:7]}` ({selection.get('name', 'unnamed')})",
        "",
        "Composed by `tools/sdpa_bwd_s1.py`; a rerun with the same inputs gives the same files.",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Sweep ref | `{ref}` = `{src.commit}` |",
        f"| Policy | `selection.toml` sha256 `{selection_digest}` |",
        f"| Members | {len(comp.members)} |",
        "| Per kernel | "
        + " · ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        + " |",
        f"| Known-flaky | {len(flaky)} |",
        f"| Rebuild required | {len(rebuild)} |",
        f"| Object roots | {len(roots)} |",
        "",
        "## Steps",
        "",
        "| Step | Kind | Rows |",
        "|---|---|---|",
        *[f"| {s['name']} | {s['kind']} | {s['count']} |" for s in comp.steps],
    ]
    if comp.skipped:
        lines += ["", "## Skipped", ""]
        lines += [f"* `{h}` ({step}): {why}" for step, h, why in comp.skipped]
    if dropped:
        lines += ["", "## Known-flaky hashes removed from the catalogue", ""]
        lines += [f"* `{h}` (step {step})" for h, step in sorted(dropped.items())]
    lines += ["", "## Inputs", "", "| Input | sha256 |", "|---|---|"]
    lines += [f"| `{n}` | `{d}` |" for n, d in sorted(src.digests.items())]
    return "\n".join(lines) + "\n"


def run(
    selection_path: Path,
    sweep_repo: Path,
    ref: str,
    out: Path,
    extra_roots: list[Path],
    reference_dir: Path | None,
    tables: bool,
    spec_check: bool,
    instances: Path | None,
    data_dir: Path | None,
    evidence_root: Path | None = None,
) -> int:
    raw = selection_path.read_bytes()
    selection = tomllib.loads(raw.decode("utf-8"))
    src = Sources(sweep_repo, ref, evidence_root or sweep_repo.parent.parent)
    expected = selection.get("sweep", {}).get("expect_commit")
    if expected and not src.commit.startswith(expected):
        raise fail(f"{ref} is {src.commit[:7]}, the policy is for {expected}")

    comp = compose(selection, src)
    counts = check_expected(selection, comp)
    if spec_check:
        from generators._sdpa_bwd_spec import Spec

        for h, m in comp.members.items():
            spec = Spec.parse(m.row["spec"])
            if spec.key() != m.row["spec"] or spec.hash() != h:
                raise fail(f"{h} is not the hash of {m.row['spec']!r}")
    flaky, dropped = flaky_sources(selection, src, comp)

    roots, base = object_roots(selection, sweep_repo, extra_roots)
    index = ObjectIndex(roots, base)
    data = data_dir or Path(__file__).resolve().parent.parent / "tests" / "data"
    rows, rebuild, mismatched, unscanned = resolve_objects(
        comp, index, read_exception_hashes(data)
    )

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "manifest.csv", MANIFEST_COLUMNS, rows)
    write_csv(
        out / "known-flaky.csv",
        FLAKY_COLUMNS,
        [
            {"hash": h, "spec": comp.members[h].row["spec"], "source": "+".join(t)}
            for h, t in flaky.items()
        ],
    )
    write_csv(out / "rebuild-required.csv", REBUILD_COLUMNS, rebuild)
    (out / "SNAPSHOT.md").write_text(
        snapshot_text(
            selection,
            src,
            ref,
            comp,
            counts,
            flaky,
            dropped,
            rebuild,
            roots,
            hashlib.sha256(raw).hexdigest(),
        ),
        encoding="utf-8",
    )
    print(
        f"{len(rows)} members at {src.commit[:7]}: "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    )
    print(f"{len(flaky)} known-flaky; {len(rebuild)} rebuild-required")
    if mismatched:
        raise fail(
            "binaries exist but none equals the pin sha256:\n  "
            + "\n  ".join(mismatched)
        )

    if unscanned:
        raise fail(
            f"{len(unscanned)} gfx950 pins have no matched clean R-MIR scan of the pinned object on record "
            "(objs/<hash>/rmir.json beside the object) and are no pinned exception:\n  "
            + "\n  ".join(unscanned[:20])
        )

    if tables:
        import sdpa_bwd_tables_from_sweep as importer

        kwargs = {}
        if instances is not None:
            kwargs["instances"] = instances
        if data_dir is not None:
            kwargs["data_dir"] = data_dir
        importer.install(out, **kwargs)
    if rebuild:
        print(
            f"rebuild required for {len(rebuild)} members (see {out / 'rebuild-required.csv'}); "
            "rebuild each, check it equals its pin, supply it with --object-root and rerun",
            file=sys.stderr,
        )
        if reference_dir is not None:
            print("reference dir not built", file=sys.stderr)
        return EXIT_REBUILD_REQUIRED
    if reference_dir is not None:
        build_reference_dir(rows, index, reference_dir)
        print(f"reference dir {reference_dir}: {len(rows)} objects verified")
        if tables:
            import sdpa_bwd_tables_from_sweep as importer

            arch, installed, _, _ = importer.read_pins(out)
            count = write_pin_functions(
                arch,
                installed,
                reference_dir,
                data,
            )
            print(f"{count} {arch} pin function digests written")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-content", type=Path)
    parser.add_argument("--arch")
    parser.add_argument(
        "--sweep-repo",
        type=Path,
        help="sweep mode only: the AMD-internal sweep repository (not part of this repository)",
    )
    parser.add_argument("--ref")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--object-root", type=Path, action="append", default=[])
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--tables", action="store_true")
    parser.add_argument("--instances", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument(
        "--no-spec-check",
        action="store_true",
        help="skip the Spec round trip (membership counts without the grammar)",
    )
    args = parser.parse_args(argv)
    if args.from_content is not None:
        if not (args.arch and args.reference_dir):
            parser.error("--from-content needs --arch and --reference-dir")
        written, named = reference_dir_from_content(
            args.from_content, args.arch, args.reference_dir, args.data_dir
        )
        print(
            f"reference dir {args.reference_dir}: {written} objects verified, "
            f"{named} pins named by their function digest"
        )
        return 0
    for flag in ("sweep_repo", "ref", "selection", "out"):
        if getattr(args, flag) is None:
            parser.error(
                f"--{flag.replace('_', '-')} is required without --from-content"
            )
    if args.tables and args.no_spec_check:
        parser.error("--tables needs the Spec round trip")
    return run(
        args.selection,
        args.sweep_repo,
        args.ref,
        args.out,
        args.object_root,
        args.reference_dir,
        args.tables,
        not args.no_spec_check,
        args.instances,
        args.data_dir,
        args.evidence_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
