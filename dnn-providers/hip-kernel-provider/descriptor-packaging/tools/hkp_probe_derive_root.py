#!/usr/bin/env python3
"""Derive a minimal descriptor root for a packaging probe.

Copies every file under `--from` to `--out`, except that each KDP shipping for
`--arch` has its inline `kernelDescriptors` reduced to the UKDs the probe packs,
among the UKDs that themselves ship for `--arch`.

The root is read with the packer's own loader (`hkp_pack.descriptors.load_flat_input`),
so derive sees exactly the descriptors the packer sees and rejects what the packer
rejects. Which KDPs ship for `--arch` is the packer's `kdp_survives`, and which of their
UKDs ship is the packer's `arch_matches`: an empty or absent `arch` list is a wildcard.

Default mode (no `--ukd`): UKDs are partitioned into compile groups, where the group
key is the kind plus the kernel_source fields hkp_probe_kinds.KINDS names for it:

    rocke (kind, source, builder), hip (kind, source, build), hsaco and
    embedded_source (kind: compile nothing, so one group per KDP)

Every field of the key must be present in kernel_source. Within a group:

  - a kind with no `sweep_field` keeps one UKD, the first by sorted UKD `name`;
  - a kind with a `sweep_field` (rocke: `spec`) keeps a value sweep. The universe is
    every (field, value) pair of the sweep block whose field takes more than one value
    within the group (values compared as canonical JSON; a field absent from some UKDs
    counts absence as one of its values). Greedily, the UKD covering the most
    uncovered pairs is kept, ties broken by sorted UKD `name`, until every pair is
    covered. A group in which no field varies keeps one UKD, the first by sorted name.

The sweep guarantees that each value of each varying field is packed at least once; it
does not guarantee any combination of values (e.g. fp16 together with causal). Kept
UKDs are written in authored order. Authors who want specific UKDs packed list them
with `--ukd`.

`--ukd <name>` mode (repeatable): each KDP keeps exactly the UKDs whose `name` is
listed. A KDP shipping for `--arch` that keeps nothing is omitted from the derived
root, so the packer cannot pack it in full. KDPs that keep nothing are not inspected
for standalone references or kinds.

In both modes KDPs that do not ship for `--arch` are copied unchanged; the packer
prunes them.

`expect.json` is written beside `--out` (`<out parent>/expect.json`): a list of
{"kdp": <KDP path relative to the root>, "name": <UKD name>, "kind": <authored
kernel_source.kind>}, one entry per kept UKD, for hkp_probe_assert.py. Entries of kinds
whose packed output is a pass-through also carry the authored "kernel_source".

Files are written only when their content differs and files absent from the
derived set are removed, so deriving twice leaves every mtime of the derived root
alone and the pack stamp stays fresh across a reconfigure.

`--list-arches --from <root>` prints, one per line and sorted, the union of the
explicit `arch` entries of every KDP and UKD (inline and standalone) under the root.
Descriptors with no `arch` (wildcards) contribute nothing.

Standalone UKDs (a `kernelDescriptors` entry that is a bare id) are out of probe
scope by policy: a KDP that would be trimmed and references one is refused.

Exit code 0 on success. Exit code 2, with a `hkp_probe_derive: ...` message on
stderr, when the root cannot be derived: the packer's loader rejects the root
(malformed or unreadable descriptor, dangling reference, UKD arch outside its KDP's),
a KDP that keeps UKDs (or, in default mode, any KDP shipping for `--arch`) references
a standalone UKD, a kept UKD has a kind not in hkp_probe_kinds.KINDS, a default-mode
candidate lacks a group key field, kept UKDs share a name, or a `--ukd` name is listed
twice or kept by no KDP.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

# Put the package dir ahead of everything, including this script's own dir:
# tools/ also holds a module literally named hkp_pack (hkp_pack.py), which would
# otherwise shadow the hkp_pack package when tools/ is sys.path[0].
_PKG_ROOT = str(Path(__file__).resolve().parent.parent / "python")
while _PKG_ROOT in sys.path:
    sys.path.remove(_PKG_ROOT)
sys.path.insert(0, _PKG_ROOT)

from hkp_pack.descriptors import (
    arch_matches,
    kdp_survives,
    load_flat_input,
)  # noqa: E402
from hkp_pack.errors import HkpPackError  # noqa: E402
from hkp_probe_kinds import KINDS  # noqa: E402

_PREFIX = "hkp_probe_derive:"


class DeriveError(Exception):
    pass


def _discard(*_args, **_kwargs):
    pass


def _load(src):
    """The packer's view of `src`; DeriveError when the packer rejects it."""
    try:
        return load_flat_input(src, log=_discard)
    except HkpPackError as exc:
        raise DeriveError(str(exc)) from exc


def _check_kind(ukd, where):
    """Return (kernel_source, kind); DeriveError when the kind is unregistered."""
    source = ukd.get("kernel_source")
    kind = source.get("kind") if isinstance(source, dict) else None
    if kind not in KINDS:
        raise DeriveError(
            f"{where} kind {kind!r} has no probe support. A kind the packer packs to "
            "kpack or ships as authored (pass-through) needs one entry in KINDS "
            "(hkp_probe_kinds.py). A kind the packer does not support yet needs "
            "packer work first; until then keep such UKDs in a KDP that does not "
            "ship for the probed arch"
        )
    return source, kind


def _group_key(ukd, where):
    source, kind = _check_kind(ukd, where)
    for f in KINDS[kind].group_by:
        if f not in source:
            raise DeriveError(
                f"{where} of kind {kind!r} has no kernel_source field {f!r}, which "
                "the compile group key of that kind requires"
            )
    return (kind,) + tuple(
        json.dumps(source[f], sort_keys=True) for f in KINDS[kind].group_by
    )


def _standalone_error(rel, entry):
    return DeriveError(
        f"{rel} references a standalone UKD ({entry!r}): standalone UKD "
        "references unsupported by probe derive"
    )


def _check_unique_names(rel, kept):
    names = [e["name"] for e in kept]
    if len(set(names)) != len(names):
        raise DeriveError(
            f"{rel} kept UKDs share a name, so expect.json could not tell them "
            f"apart: {sorted(names)}"
        )


# A field missing from a UKD's sweep block; distinct from every canonical JSON value.
_ABSENT = object()


def _sweep_values(ukd, field, where):
    """Map each field of `ukd`'s kernel_source[field] object to its canonical JSON."""
    block = ukd["kernel_source"].get(field)
    if not isinstance(block, dict):
        raise DeriveError(
            f"{where} kernel_source field {field!r} is not an object, so its values "
            "cannot be swept"
        )
    return {k: json.dumps(v, sort_keys=True) for k, v in block.items()}


def _select(members, field, rel):
    """Return the UKDs a compile group keeps; `members` is sorted by name."""
    if field is None:
        return [members[0]]
    values = [
        _sweep_values(u, field, f"{rel}: UKD '{u.get('id', '?')}'") for u in members
    ]
    keys = set().union(*values)
    pairs = [{(k, v.get(k, _ABSENT)) for k in keys} for v in values]
    varying = {k for k in keys if len({p.get(k, _ABSENT) for p in values}) > 1}
    uncovered = {(k, x) for p in pairs for (k, x) in p if k in varying}
    if not uncovered:
        return [members[0]]
    kept = []
    while uncovered:
        # max() keeps the first of equal gains, i.e. the first by sorted name.
        best = max(range(len(members)), key=lambda i: len(pairs[i] & uncovered))
        kept.append(members[best])
        uncovered -= pairs[best]
    return kept


def _derive_kdp(kdp, rel, arch):
    """Return the kept UKDs of a KDP shipping for `arch`, in authored order."""
    entries = kdp["kernelDescriptors"]
    groups = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise _standalone_error(rel, entry)
        where = f"{rel}: UKD '{entry.get('id', '?')}'"
        if not isinstance(entry.get("name"), str):
            raise DeriveError(f"{where} has no string 'name'")
        if arch_matches(entry, arch):
            groups.setdefault(_group_key(entry, where), []).append(entry)

    picked = []
    for key, members in groups.items():
        members.sort(key=lambda e: e["name"])
        picked += _select(members, KINDS[key[0]].sweep_field, rel)
    kept = [e for e in entries if any(e is p for p in picked)]
    _check_unique_names(rel, kept)
    return kept


def _derive_kdp_listed(kdp, rel, arch, wanted):
    """Return the UKDs of a KDP shipping for `arch` whose name is in `wanted`."""
    entries = kdp["kernelDescriptors"]
    kept = [
        e
        for e in entries
        if isinstance(e, dict) and e.get("name") in wanted and arch_matches(e, arch)
    ]
    if not kept:
        return kept
    for entry in entries:
        if not isinstance(entry, dict):
            raise _standalone_error(rel, entry)
    for entry in kept:
        _check_kind(entry, f"{rel}: UKD '{entry.get('id', '?')}'")
    _check_unique_names(rel, kept)
    return kept


def _plan(src, arch, ukds=None):
    """Return (files, expect): files maps relative posix path -> source Path or bytes.

    `ukds` is the list of UKD names to keep, or None for the default selection.
    """
    if ukds is not None:
        dupes = sorted({n for n in ukds if ukds.count(n) > 1})
        if dupes:
            raise DeriveError(f"--ukd lists {dupes} more than once")
        wanted = set(ukds)

    flat = _load(src)
    files = {}
    for p in sorted(src.rglob("*")):
        if p.is_file():
            files[p.relative_to(src).as_posix()] = p

    expect = []
    kdps = {d.path.relative_to(src).as_posix(): d.doc for d in flat.kdps()}
    for rel in sorted(kdps):
        kdp = kdps[rel]
        if not kdp_survives(kdp, flat, arch):
            continue
        if ukds is None:
            kept = _derive_kdp(kdp, rel, arch)
        else:
            kept = _derive_kdp_listed(kdp, rel, arch, wanted)
        if not kept:
            # Only listed UKDs are packed, so this KDP must not ship in full.
            del files[rel]
            continue
        kdp = dict(kdp, kernelDescriptors=kept)
        files[rel] = (json.dumps(kdp, indent=2) + "\n").encode("utf-8")
        for u in kept:
            entry = {"kdp": rel, "name": u["name"], "kind": u["kernel_source"]["kind"]}
            if KINDS[entry["kind"]].output == "passthrough":
                # The packer ships the block as authored; the assertion compares it.
                entry["kernel_source"] = u["kernel_source"]
            expect.append(entry)

    if ukds is not None:
        missing = sorted(wanted - {e["name"] for e in expect})
        if missing:
            raise DeriveError(
                f"--ukd {missing} matched no inline UKD shipping for {arch} in a KDP "
                f"shipping for {arch} (standalone UKD references are unsupported by "
                "probe derive)"
            )
    return files, expect


def list_arches(src: Path) -> list[str]:
    """Sorted union of the explicit `arch` entries of every KDP and UKD under `src`."""
    flat = _load(src)
    arches = set()
    for kdp in flat.kdps():
        arches.update(kdp.doc.get("arch") or [])
        for entry in kdp.doc["kernelDescriptors"]:
            if isinstance(entry, dict):
                arches.update(entry.get("arch") or [])
    for ukd in flat.ukds():
        arches.update(ukd.doc.get("arch") or [])
    return sorted(arches)


def _same(dest, content):
    if not dest.is_file() or dest.is_symlink():
        return False
    if isinstance(content, bytes):
        return dest.read_bytes() == content
    return filecmp.cmp(content, dest, shallow=False)


def _write(dest, content):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_dir() and not dest.is_symlink():
        shutil.rmtree(dest)
    elif dest.is_symlink():
        dest.unlink()
    if isinstance(content, bytes):
        dest.write_bytes(content)
    else:
        shutil.copyfile(content, dest)


def _sync(out, files):
    """Make `out` hold exactly `files`, touching only what differs."""
    out.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        dest = out / rel
        if not _same(dest, content):
            _write(dest, content)
    wanted = set(files)
    # Deepest first so emptied directories can go after their files.
    for p in sorted(out.rglob("*"), key=lambda q: len(q.parts), reverse=True):
        rel = p.relative_to(out).as_posix()
        if p.is_symlink() or p.is_file():
            if rel not in wanted:
                p.unlink()
        elif p.is_dir() and not any(w.startswith(rel + "/") for w in wanted):
            shutil.rmtree(p)


def derive_root(
    src: Path, arch: str, out: Path, ukds: list[str] | None = None
) -> str | None:
    """Write the derived root to `out`; return an error message, or None on success.

    `ukds` lists the UKD names to keep (`--ukd`); None selects per compile group.
    """
    if not src.is_dir():
        return f"{_PREFIX} {src} is not a directory"
    try:
        files, expect = _plan(src, arch, ukds)
    except DeriveError as exc:
        return f"{_PREFIX} {exc}"

    _sync(out, files)
    expect_text = json.dumps(expect, indent=2) + "\n"
    expect_path = out.parent / "expect.json"
    if (
        not expect_path.is_file()
        or expect_path.read_text(encoding="utf-8") != expect_text
    ):
        expect_path.write_text(expect_text, encoding="utf-8")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="src", type=Path, required=True)
    parser.add_argument(
        "--list-arches",
        action="store_true",
        help="print the explicit arches of the root's KDPs and UKDs, one per line",
    )
    parser.add_argument("--arch", help="architecture to probe, e.g. gfx950")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--ukd",
        dest="ukds",
        action="append",
        metavar="NAME",
        help="keep the UKD with this name (repeatable); default: per compile group, "
        "one UKD or a value sweep (see the module docstring)",
    )
    args = parser.parse_args(argv)

    if args.list_arches:
        if args.arch or args.out or args.ukds:
            parser.error("--list-arches takes only --from")
        if not args.src.is_dir():
            print(f"{_PREFIX} {args.src} is not a directory", file=sys.stderr)
            return 2
        try:
            arches = list_arches(args.src)
        except DeriveError as exc:
            print(f"{_PREFIX} {exc}", file=sys.stderr)
            return 2
        for a in arches:
            print(a)
        return 0

    if not args.arch or args.out is None:
        parser.error("--arch and --out are required unless --list-arches is given")
    error = derive_root(args.src, args.arch, args.out, args.ukds)
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
