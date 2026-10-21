#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Index every TensileLite solution in an installed hipBLASLt library.

Reads the TensileLibrary_*.dat (msgpack) and *.dat.zlib (zlib-compressed msgpack) files under
<rocm>/lib/hipblaslt/library/<arch>/ and writes one JSON line per solution: its runtime index,
kernel name, problem type, tile parameters and the size predicates the test planner must honor.

The runtime index is what hipblaslt-test's algo_method 2 (solution_index) selects. It is stable
within one build but shifts when logic files are added or retuned, so results should be keyed on
the kernel name and architecture, with the index alongside.

    corpus_index.py --rocm-dir build/dist/rocm --arch gfx90a --out corpus.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import zlib
from pathlib import Path
from typing import Iterable, Iterator, Optional

# Predicates that only restate the problem type; the planner reads the problem type instead.
_PROBLEM_TYPE_PREDICATES = {
    "OperationIdentifierEqual",
    "TypesEqual",
    "HighPrecisionAccumulate",
    "Activation",
    "ActivationComputeType",
    "ActivationNoGuard",
    "UseGradient",
    "UseBias",
    "UseE",
    "UseGateResidual",
    "DataTypeE",
    "StridedBatched",
    "GroupedGemm",
    "UseScaleAB",
    "UseScaleCD",
    "UseScaleAlphaVec",
    "Sparse",
    "F32XdlMathOp",
    "SupportDeviceUserArguments",
    "SwizzleTensorA",
    "SwizzleTensorB",
    "FusedGemmA2A",
    "MXBlockA",
    "MXBlockB",
    "KernelLanguageCompatible",
}

_PROBLEM_FIELDS = (
    "operationIdentifier",
    "transA",
    "transB",
    "aType",
    "bType",
    "cType",
    "dType",
    "computeType",
    "computeInputTypeA",
    "computeInputTypeB",
    "f32XdlMathOp",
    "highPrecisionAccumulate",
    "useBeta",
    "useBias",
    "useScaleAB",
    "useScaleCD",
    "useScaleAlphaVec",
    "activationType",
    "useE",
    "useGradient",
    "useGateResidual",
    "groupedGemm",
    "stridedBatched",
    "sparse",
    "swizzleTensorA",
    "swizzleTensorB",
    "outputAmaxD",
    "mxBlockA",
    "mxBlockB",
    "fusedGemmA2A",
    "supportDeviceUserArguments",
)

_SIZE_FIELDS = (
    "macroTile",
    "depthU",
    "globalSplitU",
    "workGroupMapping",
    "workGroupMappingXCC",
    "workAssignment",
    "tileProcessingStrategy",
    "streamKAtomic",
    "globalAccumulation",
    "matrixInstruction",
    "LocalSplitU",
)


def default_rocm_dir() -> Optional[Path]:
    """The install tree TheRock's test jobs use: THEROCK_BIN_DIR/.., else ROCM_PATH."""
    bin_dir = os.environ.get("THEROCK_BIN_DIR")
    if bin_dir:
        return Path(bin_dir).resolve().parent
    rocm_path = os.environ.get("ROCM_PATH")
    return Path(rocm_path) if rocm_path else None


def library_dir(rocm_dir: Path, arch: str) -> Path:
    return rocm_dir / "lib" / "hipblaslt" / "library" / arch


def load_library_file(path: Path) -> object:
    import msgpack

    data = path.read_bytes()
    if path.name.endswith(".zlib"):
        data = zlib.decompress(data)
    return msgpack.unpackb(data, strict_map_key=False, raw=False)


def flatten_predicates(tree: object) -> Iterator[dict]:
    """Yields the leaf predicates of an And/Or/Not tree."""
    if not isinstance(tree, dict):
        return
    kind = tree.get("type")
    if kind in ("And", "Or", "Not"):
        value = tree.get("value")
        for child in value if isinstance(value, list) else [value]:
            yield from flatten_predicates(child)
    else:
        yield tree


def hardware_requirements(tree: object) -> dict:
    """The processor and CU count a solution's hardware predicate demands, if any."""
    found: dict = {}

    def walk(node: object) -> None:
        if not isinstance(node, dict):
            return
        kind = node.get("type")
        value = node.get("value")
        if kind == "Processor":
            found["processor"] = value
        elif kind == "CUCount":
            found["cu_count"] = value
        elif kind == "PciChipIdEqual" or kind == "PciChipId":
            found["pci_chip_id"] = value
        if isinstance(value, dict):
            walk(value)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(tree)
    return found


def size_predicates(problem_predicate: object) -> dict:
    """The problem predicates that constrain sizes, keyed by type (and index when it has one)."""
    out: dict = {}
    for pred in flatten_predicates(problem_predicate):
        kind = pred.get("type")
        if not kind or kind in _PROBLEM_TYPE_PREDICATES:
            continue
        key = kind if "index" not in pred else f"{kind}[{pred['index']}]"
        out[key] = pred.get("value")
    return out


def solution_record(solution: dict, source: str, arch: str) -> dict:
    problem = solution.get("problemType", {})
    sizes = solution.get("sizeMapping", {})
    return {
        "arch": arch,
        "index": solution["index"],
        "name": solution.get("name", ""),
        "kernel_name": solution.get("kernelName", ""),
        "source": source,
        "problem": {k: problem[k] for k in _PROBLEM_FIELDS if k in problem},
        "sizes": {k: sizes[k] for k in _SIZE_FIELDS if k in sizes},
        "predicates": size_predicates(solution.get("problemPredicate")),
        "hardware": hardware_requirements(solution.get("hardwarePredicate")),
    }


def library_files(lib_dir: Path) -> list[Path]:
    """TensileLibrary files that hold solutions; the lazy-loading index files do not."""
    files = []
    for path in sorted(lib_dir.iterdir()):
        name = path.name
        if not name.startswith("TensileLibrary_"):
            continue
        if not (name.endswith(".dat") or name.endswith(".dat.zlib")):
            continue
        files.append(path)
    return files


def build_corpus(lib_dir: Path, arch: str) -> list[dict]:
    records: dict[int, dict] = {}
    for path in library_files(lib_dir):
        data = load_library_file(path)
        if not isinstance(data, dict) or "solutions" not in data:
            continue
        for solution in data["solutions"]:
            record = solution_record(solution, path.name, arch)
            records.setdefault(record["index"], record)
    return [records[i] for i in sorted(records)]


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Shards of one run may index the same library at once; each writes its own file.
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with tmp.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def problem_type_key(record: dict) -> str:
    p = record["problem"]
    trans = ("T" if p.get("transA") else "N") + ("T" if p.get("transB") else "N")
    parts = [
        p.get("aType", "?"),
        p.get("bType", "?"),
        p.get("cType", "?"),
        p.get("dType", "?"),
        p.get("computeType", "?"),
        p.get("computeInputTypeA", "?"),
        trans,
    ]
    if p.get("groupedGemm"):
        parts.append("grouped")
    if p.get("useGradient"):
        parts.append("grad")
    if p.get("useE"):
        parts.append("E")
    if p.get("useScaleAB"):
        parts.append(f"scaleAB={p['useScaleAB']}")
    return "/".join(str(x) for x in parts)


def summarize(records: list[dict]) -> str:
    names = {r["name"] for r in records}
    kernels = {r["kernel_name"] for r in records}
    types = collections.Counter(problem_type_key(r) for r in records)
    cu = collections.Counter(r["hardware"].get("cu_count", "any") for r in records)
    lines = [
        f"{len(records)} solutions, {len(names)} distinct solution names, "
        f"{len(kernels)} distinct kernels, {len(types)} problem types",
        "CU-count requirement: " + ", ".join(f"{k}: {v}" for k, v in sorted(cu.items(), key=str)),
    ]
    for key, count in types.most_common():
        lines.append(f"  {count:7d}  {key}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--rocm-dir",
        type=Path,
        default=default_rocm_dir(),
        help="Install tree holding lib/hipblaslt/library (default: THEROCK_BIN_DIR/.. or "
        "ROCM_PATH)",
    )
    parser.add_argument(
        "--library-dir",
        type=Path,
        help="Directory of TensileLibrary files; overrides --rocm-dir and --arch",
    )
    parser.add_argument("--arch", help="gfx architecture directory, e.g. gfx90a")
    parser.add_argument("--out", type=Path, required=True, help="corpus JSONL to write")
    args = parser.parse_args(argv)

    if args.library_dir:
        lib_dir = args.library_dir
        arch = args.arch or lib_dir.name
    else:
        if not args.rocm_dir or not args.arch:
            parser.error("give --library-dir, or --rocm-dir and --arch")
        lib_dir = library_dir(args.rocm_dir, args.arch)
        arch = args.arch
    if not lib_dir.is_dir():
        print(f"error: no library directory {lib_dir}", file=sys.stderr)
        return 2
    records = build_corpus(lib_dir, arch)
    if not records:
        print(f"error: no solutions found under {lib_dir}", file=sys.stderr)
        return 2
    write_jsonl(records, args.out)
    print(summarize(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
