# Adapted from dnn-providers/hip-kernel-provider/flydsl/gen_descriptors.py @ 03911535bd0 (origin/users/brpepers/hipdnn-flydsl-provider)
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emit the authored descriptors for the Triton RMSNorm objects.

    <venv>/bin/python <hip-kernel-provider>/triton/gen_descriptors.py \
        [--in <content-root>] [--out <content-root>]

Does not import ``triton``; needs ``msgpack`` (any venv that has it, e.g. the
generator venv; see README.md). Emit-only: there is no ``--check`` mode.

Reads ``<in>/rmsnorm/<arch>/manifest.json`` and the objects it names (written by
``generators.gen_rmsnorm``) and writes, under ``<out>`` (default: the committed
``descriptors/Triton/``):

* ``rmsnorm/``: the op-level UED/UHD/UDD/KMD/UMD shared by every arch;
* ``rmsnorm/<arch>/``: the KDP and one ``kind: "hsaco"`` UKD per object.

The manifests, objects and ``<arch>.SOURCE.md`` are carried across into the emitted
tree, so ``<out>`` is a complete tree (``diff -r`` against the committed one is the
re-derivation check). The new tree is staged in a temporary sibling, loaded with
``hkp_pack.descriptors.load_flat_input``, and only then swapped in (op-level files
replaced, each selected ``<arch>/`` folder replaced), so a failure leaves ``<out>``
untouched and no stale descriptor survives.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from generators import _contract as C  # noqa: E402
from generators import _triton_env as env  # noqa: E402
from generators._codeobject import (  # noqa: E402
    GeneratorError,
    describe,
    hkp_pack_module,
    verify_abi,
    verify_arch,
)
from generators._manifest import (  # noqa: E402
    new_stage,
    read_manifest,
    replace_dir,
    sha256,
    source_md_path,
)

hkp_pack_module()  # puts descriptor-packaging/python on sys.path

from hkp_pack.descriptors import (  # noqa: E402
    HkpPackError,
    load_flat_input,
    type_from_filename,
)

DESCRIPTOR_VERSION = "1.0"


def _id(name: str) -> str:
    return str(C.make_id(name))


def op_documents() -> dict[str, dict[str, Any]]:
    """The one-per-op descriptors, shared by every arch's pack."""
    kmd_id, uhd_id, udd_id = _id(C.ID_NAME_KMD), _id(C.ID_NAME_UHD), _id(C.ID_NAME_UDD)
    ued_id, umd_id = _id(C.ID_NAME_UED), _id(C.ID_NAME_UMD)
    return {
        C.KMD_FILE: {
            "version": DESCRIPTOR_VERSION,
            "id": kmd_id,
            "name": "triton rmsnorm variant fields",
            # All required, no defaults.
            "fields": [{"name": n, "type": t} for n, t in C.KMD_FIELDS],
        },
        C.UHD_FILE: {
            "version": DESCRIPTOR_VERSION,
            "id": uhd_id,
            "name": "triton rmsnorm selector",
            "kind": "native",
            "payload": C.SYMBOL_SCORE,
        },
        C.UDD_FILE: {
            "version": DESCRIPTOR_VERSION,
            "id": udd_id,
            "name": "triton rmsnorm dispatch",
            "dispatch_symbol": C.SYMBOL_DISPATCH,
        },
        C.UED_FILE: {
            "version": DESCRIPTOR_VERSION,
            "id": ued_id,
            "name": C.ENGINE_NAME,
            "graph_match": {"native": C.SYMBOL_GRAPH_MATCH},
            "heuristic": uhd_id,
            "metadata": kmd_id,
            "knobs": list(C.UED_KNOBS),
        },
        C.UMD_FILE: {
            "version": DESCRIPTOR_VERSION,
            "id": umd_id,
            "name": "triton rmsnorm: kernel variant fits the graph",
            "scope": C.UMD_SCOPE,
            "match_symbol": C.SYMBOL_KERNEL_MATCH,
        },
    }


def specialization_provenance() -> dict[str, Any]:
    """Nothing is compiled from a binding: every KMD field only picks which
    pre-built object fits, so metadata_fields is empty and every field is
    matcher-only (the packer requires the two to partition the KMD exactly)."""
    return {
        "specialization_contract": {
            "schema_version": 1,
            "consumers": [
                {
                    "engine_id": _id(C.ID_NAME_UED),
                    "kmd_id": _id(C.ID_NAME_KMD),
                    "metadata_fields": [],
                    "matcher_only_fields": [name for name, _ in C.KMD_FIELDS],
                    "bindings": {},
                    "vocabulary": {},
                }
            ],
        }
    }


def collect(in_root: Path, arch: str) -> list[dict]:
    """The manifest's instance records for ``arch``, each checked against its object."""
    arch_dir = in_root / C.OP / arch
    manifest_path = arch_dir / C.MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} missing; run generators.gen_rmsnorm")
    manifest = read_manifest(manifest_path)
    if manifest.get("arch") != arch or manifest.get("op") != C.MANIFEST_OP:
        raise ValueError(
            f"{manifest_path}: records arch/op {manifest.get('arch')}/{manifest.get('op')}"
        )
    records = manifest["instances"]
    expected = {i.name: i for i in C.instances()}
    if {r["name"] for r in records} != set(expected):
        raise ValueError(f"{manifest_path}: instance names differ from _contract")
    for record in records:
        path = arch_dir / record["file"]
        data = path.read_bytes()
        if sha256(data) != record["sha256"]:
            raise ValueError(
                f"{path}: SHA256 differs from manifest; regenerate the object"
            )
        where = str(path)
        described = describe(data, where, record["symbol"])
        verify_arch(described, arch, where)
        verify_abi(described, record["name"], where)
        if record["priority"] != expected[record["name"]].priority:
            raise ValueError(f"{record['name']}: priority differs from _contract")
    return records


def arch_documents(arch: str, records: list[dict]) -> dict[str, dict[str, Any]]:
    provenance = specialization_provenance()
    documents: dict[str, dict[str, Any]] = {}
    ids: list[str] = []
    for record in records:
        ukd_id = _id(C.id_name_ukd(arch, record["name"]))
        ids.append(ukd_id)
        documents[C.ukd_file(record["name"])] = {
            "version": DESCRIPTOR_VERSION,
            "id": ukd_id,
            "name": record["name"],
            "arch": [arch],
            "kernel_source": {
                "kind": C.UKD_SOURCE_KIND,
                "file": record["file"],
                "symbol": record["symbol"],
            },
            "metadata": {
                "dtype": record["dtype"],
                "block_n": record["block_n"],
                "block_threads": record["block"][0],
                "shared_mem_bytes": record["shared"],
            },
            "priority": record["priority"],
            "provenance": provenance,
        }
    documents[C.KDP_FILE] = {
        "version": DESCRIPTOR_VERSION,
        "id": _id(C.id_name_kdp(arch)),
        "name": f"triton rmsnorm kernels ({arch})",
        "arch": [arch],
        "matchers": [_id(C.ID_NAME_UMD)],
        "engine": _id(C.ID_NAME_UED),
        "dispatch": _id(C.ID_NAME_UDD),
        "kernelDescriptors": ids,
    }
    return documents


def _serialize(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2) + "\n"


def is_descriptor(path: Path) -> bool:
    return type_from_filename(path) is not None


def _write(folder: Path, documents: dict[str, dict[str, Any]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, document in sorted(documents.items()):
        (folder / name).write_text(_serialize(document), encoding="utf-8")


def _stage_tree(
    in_root: Path, stage_root: Path, collected: dict[str, list[dict]]
) -> None:
    """Build ``<stage_root>/rmsnorm``: op descriptors plus each selected arch folder."""
    in_op, stage_op = in_root / C.OP, stage_root / C.OP
    _write(stage_op, op_documents())
    for arch, records in collected.items():
        arch_stage = stage_op / arch
        _write(arch_stage, arch_documents(arch, records))
        shutil.copy2(in_op / arch / C.MANIFEST_FILE, arch_stage / C.MANIFEST_FILE)
        for record in records:
            shutil.copy2(in_op / arch / record["file"], arch_stage / record["file"])
        source = source_md_path(in_op, arch)
        if not source.is_file():
            raise FileNotFoundError(f"{source} missing; run generators.gen_rmsnorm")
        shutil.copy2(source, source_md_path(stage_op, arch))


def _publish(stage_root: Path, out_root: Path, arches: list[str]) -> None:
    """Swap the staged tree into ``out_root``; removes op-level descriptors it no longer emits."""
    stage_op, op_dir = stage_root / C.OP, out_root / C.OP
    op_dir.mkdir(parents=True, exist_ok=True)
    emitted = {path.name for path in stage_op.iterdir() if path.is_file()}
    for name in sorted(emitted):
        os.replace(stage_op / name, op_dir / name)
    for stale in sorted(op_dir.glob("*.json")):
        if is_descriptor(stale) and stale.name not in emitted:
            print(f"  removing stale descriptor: {stale}")
            stale.unlink()
    for arch in arches:
        replace_dir(stage_op / arch, op_dir / arch)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arch", action="append", choices=C.ARCHES, help="default: every contract arch"
    )
    parser.add_argument(
        "--in",
        dest="in_root",
        type=Path,
        default=env.CONTENT_DIR,
        help=f"content root holding rmsnorm/<arch>/ (default: {env.CONTENT_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=env.CONTENT_DIR,
        help=f"content root to emit into (default: {env.CONTENT_DIR})",
    )
    args = parser.parse_args(argv)
    arches = list(dict.fromkeys(args.arch or C.ARCHES))

    try:
        collected = {arch: collect(args.in_root, arch) for arch in arches}
        in_root, out_root = args.in_root.resolve(), args.out.resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        stage_root = new_stage(out_root)
        try:
            _stage_tree(in_root, stage_root, collected)
            load_flat_input(stage_root, log=lambda *a, **k: None)
            _publish(stage_root, out_root, arches)
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)
        count = sum(len(records) for records in collected.values())
    except (FileNotFoundError, ValueError, KeyError, GeneratorError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except HkpPackError as exc:
        print(f"Error: descriptor validation failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Emitted and validated {count} kernel descriptor(s) for {', '.join(arches)} under {out_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
