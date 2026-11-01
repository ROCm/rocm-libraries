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

A multi-launch UKD (a dQ row whose dBias reduce is a standalone object) lists its
objects under ``sources`` and carries, in its provenance, ``origin_kind`` ``hsaco``
and ``sources``: per launch the packed ``file``, ``sha256`` and ``symbol``. Each
launch must be the checked-in object it names: the row's own for ``dq``, the manifest's
companion of the row's reduce symbol and dtype for ``dbias_reduce``.

So for a generic target, every member's shard is shown to ship the one shared
object set, byte for byte, and no member is left without it.
"""

from __future__ import annotations

import argparse
import json
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


# The launches of a multi-launch row: the row's own object, then its standalone reduce object.
OWN_LAUNCH = "dq"
REDUCE_LAUNCH = "dbias_reduce"


def _launch_problems(where, name, document, record, companions, want_dir) -> list[str]:
    """What is wrong with the ``sources`` UKD @p document of the manifest row @p record."""
    provenance = document.get("provenance", {})
    if provenance.get("origin_kind") != "hsaco":
        return [f"{where}: {name} was not packed from an hsaco"]
    reduce_symbol = record["knobs"].get("reduce_symbol", "")
    companion = next(
        (
            c
            for c in companions
            if c["symbol"] == reduce_symbol
            and c["spec"].split(",")[1] == record["dtype"]
        ),
        None,
    )
    if companion is None:
        return [f"{where}: {name} has no companion for reduce symbol {reduce_symbol!r}"]
    wanted = {
        OWN_LAUNCH: (record["file"], record["sha256"], record["symbol"]),
        REDUCE_LAUNCH: (companion["file"], companion["sha256"], companion["symbol"]),
    }
    problems = []
    packed = provenance.get("sources", {})
    if set(packed) != set(wanted) or set(document.get("sources", {})) != set(wanted):
        return [
            f"{where}: {name} ships launches {sorted(packed)}, not {sorted(wanted)}"
        ]
    for launch, (file, digest, symbol) in wanted.items():
        entry = packed[launch]
        shipped = document["sources"][launch]
        if entry.get("file") != f"{want_dir}/{file}":
            problems.append(
                f"{where}: {name} launch {launch} was packed from {entry.get('file')!r}, "
                f"not {want_dir + '/' + file!r}"
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
                if "sources" in document:
                    problems += _launch_problems(
                        where,
                        name,
                        document,
                        record,
                        manifest.get("companions", []),
                        want_dir,
                    )
                    continue
                if manifest.get("companions") and record["knobs"].get("reduce_symbol"):
                    problems.append(
                        f"{where}: {name} stores dB through a standalone reduce object "
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
