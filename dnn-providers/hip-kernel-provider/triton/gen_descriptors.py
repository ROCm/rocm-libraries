# Adapted from dnn-providers/hip-kernel-provider/flydsl/gen_descriptors.py at commit 03911535bd0 (FlyDSL provider)
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emit the authored descriptors for the Triton RMSNorm objects.

Re-derives the descriptors from the committed manifests without compiling;
``generators.gen_rmsnorm`` calls ``emit`` for each arch it compiles. Reads
``<in>/rmsnorm/<arch>/manifest.json`` and the objects it lists, and writes a complete
tree under ``<out>``: the op-level UED/UHD/UDD/KMD/UMD in ``rmsnorm/``, and in each
``rmsnorm/<arch>/`` the KDP, one ``hsaco`` UKD per object, and copies of the manifest
and objects, plus ``<arch>.SOURCE.md``. The whole op tree (unselected arches copied
from ``<out>``) is staged in a hidden folder inside ``<out>``, validated with
``hkp_pack.descriptors.load_flat_input``, then swapped in, removing stale descriptors
and non-contract arches; an input or validation failure leaves ``<out>`` untouched.
Usage: README.md.
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

DESCRIPTOR_VERSION = "1.0"
SOURCE_MD_SUFFIX = ".SOURCE.md"


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


def _input_error(message: str) -> GeneratorError:
    return GeneratorError(f"{C.ERR_INPUT}: {message}")


def collect(in_root: Path, arch: str) -> list[dict]:
    """The manifest's instance records for ``arch``, each checked against its object
    and its ``_contract`` instance."""
    arch_dir = in_root / C.OP / arch
    manifest_path = arch_dir / C.MANIFEST_FILE
    if not manifest_path.is_file():
        raise _input_error(f"{manifest_path} missing; run generators.gen_rmsnorm")
    try:
        return _checked_records(arch_dir, manifest_path, arch)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise _input_error(
            f"{manifest_path}: malformed manifest ({type(exc).__name__}: {exc})"
        ) from exc


def _checked_records(arch_dir: Path, manifest_path: Path, arch: str) -> list[dict]:
    manifest = read_manifest(manifest_path)
    if manifest.get("arch") != arch or manifest.get("op") != C.MANIFEST_OP:
        raise _input_error(
            f"{manifest_path}: records arch/op {manifest.get('arch')}/{manifest.get('op')}, "
            f"expected {arch}/{C.MANIFEST_OP}"
        )
    records = manifest["instances"]
    expected = {i.name: i for i in C.instances()}
    if {r["name"] for r in records} != set(expected):
        raise _input_error(f"{manifest_path}: instance names differ from _contract")
    for record in records:
        name = record["name"]
        instance = expected[name]
        path = arch_dir / record["file"]
        if not path.is_file():
            raise _input_error(f"{path} missing; run generators.gen_rmsnorm")
        data = path.read_bytes()
        if sha256(data) != record["sha256"]:
            raise _input_error(
                f"{path}: SHA256 differs from manifest; regenerate the object"
            )
        where = str(path)
        described = describe(data, where, record["symbol"])
        verify_arch(described, arch, where)
        verify_abi(described, name, where)
        for key, want in (
            ("priority", instance.priority),
            ("dtype", instance.dtype),
            ("block_n", instance.block_n),
        ):
            if record[key] != want:
                raise _input_error(
                    f"{name}: manifest {key}={record[key]!r}, _contract {key}={want!r}"
                )
        block = record["block"][0]
        if block != instance.block_threads:
            raise _input_error(
                f"{name}: manifest block[0]={block}, "
                f"_contract block_threads={instance.block_threads}"
            )
        if block != described["max_flat_workgroup_size"]:
            raise _input_error(
                f"{name}: manifest block[0]={block}, object "
                f".max_flat_workgroup_size={described['max_flat_workgroup_size']}"
            )
        if record["shared"] <= 0:
            raise _input_error(
                f"{name}: manifest shared={record['shared']}, expected > 0"
            )
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


def _write(folder: Path, documents: dict[str, dict[str, Any]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, document in sorted(documents.items()):
        (folder / name).write_text(_serialize(document), encoding="utf-8")


def _stage_tree(
    in_root: Path, out_root: Path, stage_root: Path, collected: dict[str, list[dict]]
) -> None:
    """Build ``<stage_root>/rmsnorm``: op descriptors, each selected arch from
    ``in_root``, and every other contract arch copied unchanged from ``out_root``."""
    in_op, out_op, stage_op = in_root / C.OP, out_root / C.OP, stage_root / C.OP
    _write(stage_op, op_documents())
    for arch, records in collected.items():
        arch_stage = stage_op / arch
        _write(arch_stage, arch_documents(arch, records))
        shutil.copy2(in_op / arch / C.MANIFEST_FILE, arch_stage / C.MANIFEST_FILE)
        for record in records:
            shutil.copy2(in_op / arch / record["file"], arch_stage / record["file"])
        source = source_md_path(in_op, arch)
        if not source.is_file():
            raise _input_error(f"{source} missing; run generators.gen_rmsnorm")
        shutil.copy2(source, source_md_path(stage_op, arch))
    for arch in C.ARCHES:
        if arch in collected:
            continue
        if (out_op / arch).is_dir():
            shutil.copytree(out_op / arch, stage_op / arch)
        source = source_md_path(out_op, arch)
        if source.is_file():
            shutil.copy2(source, source_md_path(stage_op, arch))


def _non_contract_arch_paths(op_dir: Path) -> list[Path]:
    """Non-hidden folders, and ``<name>.SOURCE.md`` files, whose name is not a contract arch."""
    stale = []
    for path in sorted(op_dir.iterdir()):
        if path.name.startswith("."):
            continue
        if path.is_dir():
            if path.name not in C.ARCHES:
                stale.append(path)
        elif path.name.endswith(SOURCE_MD_SUFFIX):
            if path.name[: -len(SOURCE_MD_SUFFIX)] not in C.ARCHES:
                stale.append(path)
    return stale


def _publish(stage_root: Path, out_root: Path, arches: list[str], descriptors) -> None:
    """Swap the staged op-level files and selected arches into ``out_root``; removes
    op-level descriptors it no longer emits and every non-contract arch."""
    stage_op, op_dir = stage_root / C.OP, out_root / C.OP
    op_dir.mkdir(parents=True, exist_ok=True)
    unselected = {source_md_path(stage_op, a).name for a in C.ARCHES if a not in arches}
    emitted = {
        path.name
        for path in stage_op.iterdir()
        if path.is_file() and path.name not in unselected
    }
    for name in sorted(emitted):
        os.replace(stage_op / name, op_dir / name)
    for stale in sorted(op_dir.glob("*.json")):
        if (
            descriptors.type_from_filename(stale) is not None
            and stale.name not in emitted
        ):
            print(f"  removing stale descriptor: {stale}")
            stale.unlink()
    for arch in arches:
        replace_dir(stage_op / arch, op_dir / arch)
    for stale in _non_contract_arch_paths(op_dir):
        if stale.is_dir():
            print(f"  removing stale arch folder: {stale}")
            shutil.rmtree(stale)
        else:
            print(f"  removing stale arch file: {stale}")
            stale.unlink()


def _hkp_descriptors():
    """``hkp_pack.descriptors``, imported from the sibling packaging tree."""
    hkp_pack_module()  # puts descriptor-packaging/python on sys.path
    try:
        from hkp_pack import descriptors  # noqa: PLC0415
    except ImportError as exc:
        raise GeneratorError(
            f"{C.ERR_ENV}: hkp_pack.descriptors is not importable: {exc}"
        ) from exc
    return descriptors


def emit(in_root: Path, out_root: Path, arches: list[str]) -> int:
    """Emit, validate and publish the descriptors for ``arches``; returns UKD count."""
    descriptors = _hkp_descriptors()
    in_root, out_root = in_root.resolve(), out_root.resolve()
    collected = {arch: collect(in_root, arch) for arch in arches}
    out_root.mkdir(parents=True, exist_ok=True)
    stage_root = new_stage(out_root)
    try:
        try:
            _stage_tree(in_root, out_root, stage_root, collected)
            descriptors.load_flat_input(stage_root, log=lambda *a, **k: None)
        except descriptors.HkpPackError as exc:
            raise _input_error(f"descriptor validation failed: {exc}") from exc
        _publish(stage_root, out_root, arches, descriptors)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
    return sum(len(records) for records in collected.values())


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
        count = emit(args.in_root, args.out, arches)
    except GeneratorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: {C.ERR_INTERNAL}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        f"Emitted and validated {count} kernel descriptor(s) for {', '.join(arches)} "
        f"under {args.out.resolve()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
