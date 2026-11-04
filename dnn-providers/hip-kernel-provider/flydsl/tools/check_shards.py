# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Check that every packed shard carries exactly the FlyDSL objects checked in.

    python tools/check_shards.py --content-dir <descriptors>/FlyDSL \\
        --out-root <arch_content>/hip-kernel-provider --arches gfx1100 gfx1151 ...

Runs after the product pack. ``gen_descriptors.py --check`` proves the authored
tree is consistent before the pack; this proves what the pack *shipped*. For
every requested arch and every content directory whose objects run on it (a
generic target's members, or the concrete arch itself), the arch's shard must
hold one shipped UKD per manifest instance, and each must

* list that shard's arch and nothing else,
* name, in its provenance, the checked-in object under that content directory,
* carry the payload SHA256 the manifest records for that object -- the digest
  the packer computed from the bytes it put in the archive, and the one the
  runtime verifies at load.

A dQ or merged row is a multi-launch UKD: it lists its objects under ``sources`` and
carries, in its provenance, ``origin_kind`` ``hsaco`` and ``sources``: per launch the
packed ``file``, ``sha256`` and ``symbol``. The launches are the row's own (``dq`` or
``merged``), the manifest's ``dbias_reduce`` companion of the row's dtype when the row
names a reduce symbol and the manifest lists one (a gfx11 dQ row that stores dB), and the
manifest's ``tailzero`` companion of the row's dtype when it lists one (``gen_descriptors.py``
refuses a dQ or merged manifest without it; this tool checks what the pack shipped against the manifest it is given). Each must be the checked-in object
it names; a companion kept in another op's folder is resolved relative to the row's own.

So for a generic target, every member's shard is shown to ship the one shared
object set, byte for byte, and no member is left without it.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
# The packer's own reader for the provenance it moves out of each packed UKD.
sys.path.insert(0, str(_HERE.parent.parent / "descriptor-packaging" / "python"))

from generators import _arch_families as families  # noqa: E402
from hkp_pack import provenance_sidecar  # noqa: E402

MANIFEST_NAME = "manifest.json"


def _manifests(content_dir: Path) -> list[tuple[str, str, dict]]:
    """(op, content arch, manifest) for every checked-in object set."""
    found = []
    for path in sorted(content_dir.glob(f"*/gfx*/{MANIFEST_NAME}")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        found.append((path.parent.parent.name, path.parent.name, manifest))
    return found


# The launches of a multi-launch row, as ``gen_descriptors.py`` packs them: the op's own object, then its dBias reduce companion
# when the manifest lists one for the row (a gfx11 dQ row that stores dB; a gfx950 combined row holds the reduce in its own
# object and has none), then the tail-zero companion that every dQ and merged row launches.
MAIN_LAUNCH = {"sdpa_bwd_dq": "dq", "sdpa_bwd_merged": "merged"}
REDUCE_ROLE = "dbias_reduce"
TAILZERO_ROLE = "tailzero"


def _companions_of(companions: list[dict], role: str) -> list[dict]:
    """The manifest ``companions`` entries of @p role (the first field of their ``spec``)."""
    return [c for c in companions if c["spec"].split(";")[0].split(",")[0] == role]


def _companion_of(companions: list[dict], role: str, dtype: str) -> dict | None:
    """The companion of @p role for @p dtype (the second field of its ``spec``), or None."""
    return next(
        (
            c
            for c in _companions_of(companions, role)
            if c["spec"].split(";")[0].split(",")[1] == dtype
        ),
        None,
    )


def _launches_companions(companions: list[dict], record: dict) -> bool:
    """Whether the manifest gives the row @p record a companion launch (a reduce it names, or a tail-zero of its dtype)."""
    reduce_symbol = record["knobs"].get("reduce_symbol", "")
    return bool(
        reduce_symbol
        and _companions_of(companions, REDUCE_ROLE)
        or _companion_of(companions, TAILZERO_ROLE, record["dtype"])
    )


def _launch_problems(
    where, name, document, record, companions, want_dir, main_launch
) -> list[str]:
    """What is wrong with the ``sources`` UKD @p document of the manifest row @p record."""
    provenance = document.get("provenance", {})
    if provenance.get("origin_kind") != "hsaco":
        return [f"{where}: {name} was not packed from an hsaco"]
    wanted = {main_launch: (record["file"], record["sha256"], record["symbol"])}
    reduce_symbol = record["knobs"].get("reduce_symbol", "")
    if reduce_symbol and _companions_of(companions, REDUCE_ROLE):
        reduce = _companion_of(companions, REDUCE_ROLE, record["dtype"])
        if reduce is None or reduce["symbol"] != reduce_symbol:
            return [
                f"{where}: {name} has no {REDUCE_ROLE} companion for reduce symbol {reduce_symbol!r}"
            ]
        wanted[REDUCE_ROLE] = (reduce["file"], reduce["sha256"], reduce["symbol"])
    tailzero = _companion_of(companions, TAILZERO_ROLE, record["dtype"])
    if tailzero is not None:
        wanted[TAILZERO_ROLE] = (
            tailzero["file"],
            tailzero["sha256"],
            tailzero["symbol"],
        )
    problems = []
    packed = provenance.get("sources", {})
    if set(packed) != set(wanted) or set(document.get("sources", {})) != set(wanted):
        return [
            f"{where}: {name} ships launches {sorted(packed)}, not {sorted(wanted)}"
        ]
    for launch, (file, digest, symbol) in wanted.items():
        entry = packed[launch]
        shipped = document["sources"][launch]
        # A companion kept in another op's folder (the merged tail-zero) is named relative to this one.
        want_file = posixpath.normpath(f"{want_dir}/{file}")
        if entry.get("file") != want_file:
            problems.append(
                f"{where}: {name} launch {launch} was packed from {entry.get('file')!r}, "
                f"not {want_file!r}"
            )
        if entry.get("sha256") != digest or shipped.get("sha256") != digest:
            problems.append(
                f"{where}: {name} launch {launch} ships payload {entry.get('sha256')}, "
                f"but the checked-in object is {digest}"
            )
        if entry.get("symbol") != symbol:
            problems.append(
                f"{where}: {name} launch {launch} is symbol {entry.get('symbol')!r}, not {symbol!r}"
            )
    return problems


def check(content_dir: Path, out_root: Path, arches: list[str]) -> list[str]:
    problems: list[str] = []
    manifests = _manifests(content_dir)
    if not manifests:
        return [f"no */gfx*/{MANIFEST_NAME} under {content_dir}"]

    for arch in arches:
        served = 0
        for op, target, manifest in manifests:
            if not families.covers(target, arch):
                continue
            served += 1
            folder = out_root / arch / content_dir.name / op / target
            shipped = {}
            # Each object ships as an inline UKD of its target's KDP; a packed UKD's
            # provenance lives in the KDP's sidecar beside it.
            for path in sorted(folder.glob("*.kdp.json")):
                kdp = json.loads(path.read_text(encoding="utf-8"))
                provenance_sidecar.attach(path, kdp)
                for document in kdp.get("kernelDescriptors", []):
                    if isinstance(document, dict):
                        shipped[document["name"]] = document
            where = f"{arch}: {content_dir.name}/{op}/{target}"
            expected = {record["name"]: record for record in manifest["instances"]}
            for name in sorted(set(expected) - set(shipped)):
                problems.append(f"{where}: {name} is not in the shard")
            for name in sorted(set(shipped) - set(expected)):
                problems.append(f"{where}: {name} ships but no manifest names it")
            for name in sorted(set(expected) & set(shipped)):
                document, record = shipped[name], expected[name]
                provenance = document.get("provenance", {})
                source = document.get("kernel_source", {})
                want_dir = f"{content_dir.name}/{op}/{target}"
                want_file = f"{want_dir}/{record['file']}"
                if document.get("arch") != [arch]:
                    problems.append(
                        f"{where}: {name} lists arch {document.get('arch')}"
                    )
                if "sources" in document and op not in MAIN_LAUNCH:
                    problems.append(
                        f"{where}: {name} ships launches, but {op} rows have one"
                    )
                    continue
                if "sources" in document:
                    problems += _launch_problems(
                        where,
                        name,
                        document,
                        record,
                        manifest.get("companions", []),
                        want_dir,
                        MAIN_LAUNCH[op],
                    )
                    continue
                if _launches_companions(manifest.get("companions", []), record):
                    problems.append(
                        f"{where}: {name} launches companion objects "
                        "but ships as one launch"
                    )
                if provenance.get("origin_kind") != "hsaco":
                    problems.append(f"{where}: {name} was not packed from an hsaco")
                if provenance.get("file") != want_file:
                    problems.append(
                        f"{where}: {name} was packed from {provenance.get('file')!r}, "
                        f"not {want_file!r}"
                    )
                if source.get("sha256") != record["sha256"]:
                    problems.append(
                        f"{where}: {name} ships payload {source.get('sha256')}, but "
                        f"the checked-in object is {record['sha256']}"
                    )
        if not served:
            problems.append(f"{arch}: no checked-in FlyDSL object set runs on it")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--content-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--arches", nargs="+", required=True)
    args = parser.parse_args(argv)

    problems = check(args.content_dir, args.out_root, args.arches)
    for problem in problems:
        print(f"SHARD FAIL: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"FlyDSL shards verified for {', '.join(args.arches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
