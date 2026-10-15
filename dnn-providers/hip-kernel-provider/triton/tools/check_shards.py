# Adapted from dnn-providers/hip-kernel-provider/flydsl/tools/check_shards.py at commit 94ce74f41ed (FlyDSL provider)
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Check that every packed shard carries exactly the Triton objects committed.

    python tools/check_shards.py --content-dir <descriptors>/Triton \\
        --out-root <arch_content>/hip-kernel-provider --arches gfx942 gfx950

Runs after the product pack. ``gen_descriptors.py --check`` proves the committed
tree is what the generator produces before the pack; this proves what the pack
*shipped*. For every requested arch and every op with a content directory for
that arch, the arch's shard must hold one packed UKD per manifest instance, and
each must

* list that shard's arch and nothing else,
* name, in its provenance, the committed object under that content directory,
* carry the payload SHA256 the manifest records for that object -- the digest
  the packer computed from the bytes it put in the archive, and the one the
  runtime verifies at load.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# The packer's own reader for the provenance it moves out of each packed UKD.
sys.path.insert(0, str(_HERE.parent.parent / "descriptor-packaging" / "python"))

from hkp_pack import provenance_sidecar  # noqa: E402
from hkp_pack.errors import HkpPackError  # noqa: E402

MANIFEST_NAME = "manifest.json"


def _manifests(content_dir: Path) -> list[tuple[str, str, dict]]:
    """(op, content arch, manifest) for every committed object set."""
    found = []
    for path in sorted(content_dir.glob(f"*/gfx*/{MANIFEST_NAME}")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        found.append((path.parent.parent.name, path.parent.name, manifest))
    return found


def check(content_dir: Path, out_root: Path, arches: list[str]) -> list[str]:
    problems: list[str] = []
    manifests = _manifests(content_dir)
    if not manifests:
        return [f"no */gfx*/{MANIFEST_NAME} under {content_dir}"]

    for arch in arches:
        served = 0
        for op, target, manifest in manifests:
            if target != arch:
                continue
            served += 1
            folder = out_root / arch / content_dir.name / op / target
            where = f"{arch}: {content_dir.name}/{op}/{target}"
            shipped = {}
            for path in sorted(folder.glob("*.ukd.json")):
                document = json.loads(path.read_text(encoding="utf-8"))
                # A packed UKD's provenance lives in a sidecar beside it.
                try:
                    provenance_sidecar.attach(path, document)
                except HkpPackError as exc:
                    problems.append(f"{where}: {path.name}: {exc}")
                    continue
                shipped[document["name"]] = document
            expected = {record["name"]: record for record in manifest["instances"]}
            for name in sorted(set(expected) - set(shipped)):
                problems.append(f"{where}: {name} is not in the shard")
            for name in sorted(set(shipped) - set(expected)):
                problems.append(f"{where}: {name} ships but no manifest names it")
            for name in sorted(set(expected) & set(shipped)):
                document, record = shipped[name], expected[name]
                provenance = document.get("provenance", {})
                source = document.get("kernel_source", {})
                want_file = f"{content_dir.name}/{op}/{target}/{record['file']}"
                if document.get("arch") != [arch]:
                    problems.append(
                        f"{where}: {name} lists arch {document.get('arch')}"
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
                        f"the committed object is {record['sha256']}"
                    )
        if not served:
            problems.append(f"{arch}: no committed Triton object set runs on it")
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
    print(f"Triton shards verified for {', '.join(args.arches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
