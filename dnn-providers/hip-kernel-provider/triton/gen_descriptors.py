# Adapted from dnn-providers/hip-kernel-provider/flydsl/gen_descriptors.py at commit 94ce74f41ed (FlyDSL provider)
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emit, or check, the authored descriptors for the Triton AOT objects.

Re-derives the descriptors from the committed manifests without compiling;
``tritongen.gen`` calls ``emit`` for each arch it compiles. For every op in
``tritongen.ops.OPS``, reads ``<in>/<op>/<arch>/manifest.json`` and the objects it
lists, and writes a complete tree under ``<out>``: the op-level UED/UHD/UDD/KMD/UMD in
``<op>/``, and in each ``<op>/<arch>/`` the KDP, one ``hsaco`` UKD per object, and
copies of the manifest and objects, plus ``<arch>.SOURCE.md``. The whole op tree
(unselected arches copied from ``<out>``) is staged in a hidden folder inside
``<out>``, validated with ``hkp_pack.descriptors.load_flat_input``, then swapped in,
removing stale descriptors and non-contract arches; an input or validation failure
leaves ``<out>`` untouched.

``--check`` writes nothing: it verifies the tree under ``<in>`` is what the generator
produces from the sources in this checkout, printing one ``TRITONGEN-CHECK: <path>:
<reason>`` line per fault and exiting 1, or one summary line and exiting 0. It needs
``msgpack`` and ``hkp_pack``, never ``triton``. Usage: README.md.
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

from tritongen import _family as F  # noqa: E402
from tritongen._codeobject import (  # noqa: E402
    arg_records,
    check_scratch,
    describe,
    expected_abi,
    hkp_pack_module,
    verify_abi,
    verify_arch,
)
from tritongen._family import GeneratorError  # noqa: E402
from tritongen._manifest import (  # noqa: E402
    input_files,
    inputs_digest,
    new_stage,
    read_manifest,
    replace_dir,
    sha256,
    source_md_path,
)
from tritongen.ops import OPS  # noqa: E402


def _id(name: str) -> str:
    return str(F.make_id(name))


def _ids(op) -> dict[str, str]:
    return {
        kind: _id(F.id_name(kind, op.OP)) for kind in ("kmd", "uhd", "udd", "ued")
    } | {"umd": _id(F.id_name("umd", op.OP, op.UMD_SCOPE))}


def op_documents(op) -> dict[str, dict[str, Any]]:
    """The one-per-op descriptors, shared by every arch's pack."""
    ids = _ids(op)
    return {
        F.op_file(op.OP, "kmd"): {
            "version": F.DESCRIPTOR_VERSION,
            "id": ids["kmd"],
            "name": f"triton {op.OP} variant fields",
            # All required, no defaults.
            "fields": [{"name": n, "type": t} for n, t in op.KMD_FIELDS],
        },
        F.op_file(op.OP, "uhd"): {
            "version": F.DESCRIPTOR_VERSION,
            "id": ids["uhd"],
            "name": f"triton {op.OP} selector",
            "kind": "native",
            "payload": F.symbol(op.OP, "score"),
        },
        F.op_file(op.OP, "udd"): {
            "version": F.DESCRIPTOR_VERSION,
            "id": ids["udd"],
            "name": f"triton {op.OP} dispatch",
            "dispatch_symbol": F.symbol(op.OP, "dispatch"),
        },
        F.op_file(op.OP, "ued"): {
            "version": F.DESCRIPTOR_VERSION,
            "id": ids["ued"],
            "name": op.ENGINE_NAME,
            "sdk_version": op.UED_SDK_VERSION,
            "graph_match": {"native": F.symbol(op.OP, "graph_match")},
            "heuristic": ids["uhd"],
            "metadata": ids["kmd"],
            "knobs": list(op.UED_KNOBS),
        },
        F.op_file(op.OP, "umd"): {
            "version": F.DESCRIPTOR_VERSION,
            "id": ids["umd"],
            "name": f"triton {op.OP}: kernel variant fits the graph",
            "scope": op.UMD_SCOPE,
            "match_symbol": F.symbol(op.OP, "kernel_match"),
        },
    }


def specialization_provenance(op) -> dict[str, Any]:
    """Nothing is compiled from a binding: every KMD field only picks which
    pre-built object fits, so metadata_fields is empty and every field is
    matcher-only (the packer requires the two to partition the KMD exactly)."""
    ids = _ids(op)
    return {
        "specialization_contract": {
            "schema_version": 1,
            "consumers": [
                {
                    "engine_id": ids["ued"],
                    "kmd_id": ids["kmd"],
                    "metadata_fields": [],
                    "matcher_only_fields": [name for name, _ in op.KMD_FIELDS],
                    "bindings": {},
                    "vocabulary": {},
                }
            ],
        }
    }


def arch_documents(op, arch: str, records: list[dict]) -> dict[str, dict[str, Any]]:
    provenance = specialization_provenance(op)
    documents: dict[str, dict[str, Any]] = {}
    ukd_ids: list[str] = []
    for record in records:
        ukd_id = _id(F.id_name("ukd", op.OP, arch, record["name"]))
        ukd_ids.append(ukd_id)
        documents[F.ukd_file(record["name"])] = {
            "version": F.DESCRIPTOR_VERSION,
            "id": ukd_id,
            "name": record["name"],
            "arch": [arch],
            "kernel_source": {
                "kind": F.UKD_SOURCE_KIND,
                "file": record["file"],
                "symbol": record["symbol"],
            },
            "metadata": op.metadata(record),
            "priority": record["priority"],
            "provenance": provenance,
        }
    ids = _ids(op)
    documents[F.op_file(op.OP, "kdp")] = {
        "version": F.DESCRIPTOR_VERSION,
        "id": _id(F.id_name("kdp", op.OP, arch)),
        "name": f"triton {op.OP} kernels ({arch})",
        "arch": [arch],
        "matchers": [ids["umd"]],
        "engine": ids["ued"],
        "dispatch": ids["udd"],
        "kernelDescriptors": ukd_ids,
    }
    return documents


# --- Reading the committed tree ------------------------------------------------------

# A fault found in the tree: the file it concerns, its TRITONGEN code, and the reason.
Problem = tuple[Path, str, str]


def _regenerate_hint(op, arch: str) -> str:
    return f"regenerate with python -m tritongen.gen --op {op.OP} --arch {arch}"


def _from_error(path: Path, exc: GeneratorError) -> Problem:
    """A raised check as a problem about ``path`` (its own ``where`` prefix dropped)."""
    code, _, rest = str(exc).partition(": ")
    return path, code, rest.removeprefix(f"{path}: ")


def _manifest_problems(
    op, arch_dir: Path, arch: str
) -> tuple[list[dict], list[Problem]]:
    manifest_path = arch_dir / F.MANIFEST_FILE
    if not manifest_path.is_file():
        return [], [
            (manifest_path, F.ERR_INPUT, f"missing; {_regenerate_hint(op, arch)}")
        ]
    manifest = read_manifest(manifest_path)
    if manifest.get("arch") != arch or manifest.get("op") != op.OP:
        return [], [
            (
                manifest_path,
                F.ERR_INPUT,
                f"records arch/op {manifest.get('arch')}/{manifest.get('op')} but sits "
                f"in {op.OP}/{arch}/",
            )
        ]
    problems: list[Problem] = []
    recorded = manifest.get("toolchain", {}).get("inputs_sha256")
    current = inputs_digest(op)
    if recorded != current:
        inputs = ", ".join(
            path.relative_to(F.TRITON_DIR).as_posix() for path in input_files(op)
        )
        problems.append(
            (
                manifest_path,
                F.ERR_INPUT,
                f"inputs_sha256 {recorded} differs from {current}, the digest of this "
                f"tree's generator inputs ({inputs}); {_regenerate_hint(op, arch)}",
            )
        )
    records = manifest["instances"]
    named = {record["file"] for record in records}
    for stray in sorted(arch_dir.glob("*.hsaco")):
        if stray.name not in named:
            problems.append(
                (stray, F.ERR_INPUT, "object the manifest does not name; delete it")
            )
    expected = {i.name: i for i in op.instances()}
    if sorted(r["name"] for r in records) != sorted(expected):
        problems.append(
            (manifest_path, F.ERR_INPUT, f"instance names differ from ops/{op.OP}.py")
        )
        return records, problems
    for record in records:
        problems += _record_problems(
            op, arch_dir, manifest_path, record, expected, arch
        )
    return records, problems


def _record_problems(
    op, arch_dir: Path, manifest_path: Path, record: dict, expected: dict, arch: str
) -> list[Problem]:
    name = record["name"]
    instance = expected[name]
    path = arch_dir / record["file"]
    if not path.is_file():
        return [(path, F.ERR_INPUT, f"missing; {_regenerate_hint(op, arch)}")]
    data = path.read_bytes()
    if sha256(data) != record["sha256"]:
        return [
            (path, F.ERR_INPUT, "SHA256 differs from manifest; regenerate the object")
        ]
    where = str(path)
    abi = expected_abi(op, instance)
    try:
        described = describe(data, where, record["symbol"])
        verify_arch(described, arch, where)
        verify_abi(described, abi, name, where)
        check_scratch(
            name, record["global_scratch_size"], record["profile_scratch_size"]
        )
    except GeneratorError as exc:
        if str(exc).startswith(f"{F.ERR_ENV}: "):
            raise
        return [_from_error(path, exc)]

    problems: list[Problem] = []
    contract = f"ops/{op.OP}.py"
    for field, recorded, want in (
        ("priority", record["priority"], instance.priority),
        ("dtype", record["dtype"], instance.dtype),
        ("knobs", record["knobs"], instance.knobs),
        ("kernarg_segment_size", record["kernarg_segment_size"], abi.segment_size),
        ("args", record["args"], arg_records(abi)),
    ):
        if recorded != want:
            problems.append(
                (
                    manifest_path,
                    F.ERR_INPUT,
                    f"{name}: manifest {field}={recorded!r}, {contract} {field}={want!r}",
                )
            )
    object_max = described["max_flat_workgroup_size"]
    if record["knobs"]["block_threads"] != object_max:
        problems.append(
            (
                manifest_path,
                F.ERR_INPUT,
                f"{name}: manifest knobs.block_threads="
                f"{record['knobs']['block_threads']}, object "
                f".max_flat_workgroup_size={object_max}",
            )
        )
    problems += [
        (manifest_path, F.ERR_INPUT, f"{name}: {reason}")
        for reason in op.record_problems(record)
    ]
    return problems


def _arch_problems(op, in_root: Path, arch: str) -> tuple[list[dict], list[Problem]]:
    """The manifest records for ``op``/``arch`` and every fault in them, their objects,
    and the generator inputs they were built from."""
    arch_dir = in_root / op.OP / arch
    try:
        return _manifest_problems(op, arch_dir, arch)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        return [], [
            (
                arch_dir / F.MANIFEST_FILE,
                F.ERR_INPUT,
                f"malformed manifest ({type(exc).__name__}: {exc})",
            )
        ]


def collect(op, in_root: Path, arch: str) -> list[dict]:
    """The manifest's instance records for ``op``/``arch``, each checked against its
    object and the op contract; raises on the first fault."""
    records, problems = _arch_problems(op, in_root, arch)
    if problems:
        path, code, reason = problems[0]
        raise GeneratorError(f"{code}: {path}: {reason}")
    return records


# --- Writing ---------------------------------------------------------------------------


def _serialize(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2) + "\n"


def _write(folder: Path, documents: dict[str, dict[str, Any]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, document in sorted(documents.items()):
        (folder / name).write_text(_serialize(document), encoding="utf-8")


def _input_error(message: str) -> GeneratorError:
    return GeneratorError(f"{F.ERR_INPUT}: {message}")


def _stage_tree(
    op,
    in_root: Path,
    out_root: Path,
    stage_root: Path,
    collected: dict[str, list[dict]],
) -> None:
    """Build ``<stage_root>/<op>``: op descriptors, each selected arch from ``in_root``,
    and every other contract arch copied unchanged from ``out_root``."""
    in_op, out_op, stage_op = in_root / op.OP, out_root / op.OP, stage_root / op.OP
    _write(stage_op, op_documents(op))
    for arch, records in collected.items():
        arch_stage = stage_op / arch
        _write(arch_stage, arch_documents(op, arch, records))
        shutil.copy2(in_op / arch / F.MANIFEST_FILE, arch_stage / F.MANIFEST_FILE)
        for record in records:
            shutil.copy2(in_op / arch / record["file"], arch_stage / record["file"])
        source = source_md_path(in_op, arch)
        if not source.is_file():
            raise _input_error(f"{source} missing; {_regenerate_hint(op, arch)}")
        shutil.copy2(source, source_md_path(stage_op, arch))
    for arch in op.ARCHES:
        if arch in collected:
            continue
        if (out_op / arch).is_dir():
            shutil.copytree(out_op / arch, stage_op / arch)
        source = source_md_path(out_op, arch)
        if source.is_file():
            shutil.copy2(source, source_md_path(stage_op, arch))


def _non_contract_arch_paths(op, op_dir: Path) -> list[Path]:
    """Non-hidden folders, and ``<name>.SOURCE.md`` files, whose name is not a contract arch."""
    stale = []
    for path in sorted(op_dir.iterdir()):
        if path.name.startswith("."):
            continue
        if path.is_dir():
            if path.name not in op.ARCHES:
                stale.append(path)
        elif path.name.endswith(F.SOURCE_MD_SUFFIX):
            if path.name[: -len(F.SOURCE_MD_SUFFIX)] not in op.ARCHES:
                stale.append(path)
    return stale


def _publish(
    op, stage_root: Path, out_root: Path, arches: list[str], descriptors
) -> None:
    """Swap the staged op-level files and selected arches into ``out_root``; removes
    op-level descriptors it no longer emits and every non-contract arch."""
    stage_op, op_dir = stage_root / op.OP, out_root / op.OP
    op_dir.mkdir(parents=True, exist_ok=True)
    unselected = {
        source_md_path(stage_op, a).name for a in op.ARCHES if a not in arches
    }
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
    for stale in _non_contract_arch_paths(op, op_dir):
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
            f"{F.ERR_ENV}: hkp_pack.descriptors is not importable: {exc}"
        ) from exc
    return descriptors


def emit(op, in_root: Path, out_root: Path, arches: list[str]) -> int:
    """Emit, validate and publish ``op``'s descriptors for ``arches``; returns UKD count."""
    descriptors = _hkp_descriptors()
    in_root, out_root = in_root.resolve(), out_root.resolve()
    collected = {arch: collect(op, in_root, arch) for arch in arches}
    out_root.mkdir(parents=True, exist_ok=True)
    stage_root = new_stage(out_root)
    try:
        try:
            _stage_tree(op, in_root, out_root, stage_root, collected)
            descriptors.load_flat_input(stage_root, log=lambda *a, **k: None)
        except descriptors.HkpPackError as exc:
            raise _input_error(f"descriptor validation failed: {exc}") from exc
        _publish(op, stage_root, out_root, arches, descriptors)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
    return sum(len(records) for records in collected.values())


# --- Checking ----------------------------------------------------------------------------


def _descriptor_problems(
    root: Path, folders: dict[Path, dict[str, dict[str, Any]]], descriptors
) -> list[Problem]:
    """Every derived descriptor byte-equal to the checked-in file; none missing or extra."""
    problems: list[Problem] = []
    for rel, documents in sorted(folders.items()):
        folder = root / rel
        present = {
            path.name
            for path in folder.glob("*.json")
            if descriptors.type_from_filename(path) is not None
        }
        for name, document in sorted(documents.items()):
            path = folder / name
            if not path.is_file():
                problems.append((path, F.ERR_CHECK, "missing; regenerate"))
            elif path.read_text(encoding="utf-8") != _serialize(document):
                problems.append(
                    (
                        path,
                        F.ERR_CHECK,
                        "differs from what the manifest derives; regenerate",
                    )
                )
        for extra in sorted(present - set(documents)):
            problems.append(
                (
                    folder / extra,
                    F.ERR_CHECK,
                    "not derived from any manifest; delete it",
                )
            )
    return problems


def check(
    ops: list, arches: dict[str, list[str]], root: Path
) -> tuple[list[Problem], str]:
    """Every fault in the committed tree under ``root``, and the summary line if none."""
    descriptors = _hkp_descriptors()
    root = root.resolve()
    problems: list[Problem] = []
    folders: dict[Path, dict[str, dict[str, Any]]] = {}
    objects = 0
    for op in ops:
        folders[Path(op.OP)] = op_documents(op)
        op_dir = root / op.OP
        if op_dir.is_dir():
            problems += [
                (path, F.ERR_CHECK, f"not a contract arch of ops/{op.OP}.py")
                for path in _non_contract_arch_paths(op, op_dir)
            ]
        for arch in arches[op.OP]:
            records, found = _arch_problems(op, root, arch)
            problems += found
            objects += len(records)
            if not found:
                folders[Path(op.OP) / arch] = arch_documents(op, arch, records)
    problems += _descriptor_problems(root, folders, descriptors)
    if problems:
        return problems, ""
    try:
        descriptors.load_flat_input(root, log=lambda *a, **k: None)
    except descriptors.HkpPackError as exc:
        return [(root, F.ERR_CHECK, f"descriptor validation failed: {exc}")], ""
    count = sum(len(documents) for documents in folders.values())
    scope = "; ".join(f"{op.OP} ({', '.join(arches[op.OP])})" for op in ops)
    return [], (
        f"Checked {objects} object(s) and {count} descriptor(s) of {scope} under "
        f"{root}: up to date"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--op", action="append", choices=sorted(OPS), help="default: every op"
    )
    parser.add_argument(
        "--arch",
        action="append",
        choices=sorted({arch for op in OPS.values() for arch in op.ARCHES}),
        help="default: every arch of each op",
    )
    parser.add_argument(
        "--in",
        dest="in_root",
        type=Path,
        default=F.CONTENT_DIR,
        help=f"content root holding <op>/<arch>/ (default: {F.CONTENT_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=F.CONTENT_DIR,
        help=f"content root to emit into (default: {F.CONTENT_DIR})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the tree under --in against this checkout and exit; write nothing",
    )
    args = parser.parse_args(argv)
    ops = [OPS[name] for name in dict.fromkeys(args.op or sorted(OPS))]
    arches = {
        op.OP: [a for a in dict.fromkeys(args.arch or op.ARCHES) if a in op.ARCHES]
        for op in ops
    }
    if args.arch:
        unknown = set(args.arch) - {a for selected in arches.values() for a in selected}
        if unknown:
            parser.error(f"no selected op builds {', '.join(sorted(unknown))}")

    try:
        if args.check:
            problems, summary = check(ops, arches, args.in_root)
        else:
            count = sum(
                emit(op, args.in_root, args.out, arches[op.OP])
                for op in ops
                if arches[op.OP]
            )
    except GeneratorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: {F.ERR_INTERNAL}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.check:
        for path, _, reason in problems:
            print(f"{F.ERR_CHECK}: {path}: {reason}", file=sys.stderr)
        if problems:
            return 1
        print(summary)
        return 0

    selected = ", ".join(sorted({a for selected in arches.values() for a in selected}))
    print(
        f"Emitted and validated {count} kernel descriptor(s) for {selected} "
        f"under {args.out.resolve()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
