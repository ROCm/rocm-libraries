# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Extract JIT tuning knowledge from TensileLite logic files.

Usage: python -m Tensile.JitKnowledge <logic directory> <output file> --architecture <gfx>

Reads every Equality, GridBased and Range logic file under the directory through
LibraryIO, merges each solution with its file's defaults and then the global
ones as LibraryIO does, and writes one file for the JIT's tuning knowledge:

    "HJKN", schema (u32 little-endian), header length (u32 little-endian)
    header: MessagePack {generator, source_commit, content_hash, arch, library_arch,
                         branches: [{kind, cu_count, pci_ids}],
                         index: [{branch, problem_type: {name, features}, core_key,
                                  offset, length, rows, sets}]}
    one zlib-compressed MessagePack block per index entry, back to back; offsets
    count from the end of the header:
        {param_dictionary: [[name, value]...],
         sets: [{macro_tile, waves, instruction, depth_u, nt, policy: {strategy, assignment},
                 gsu, gsu_algorithm, params: [dictionary ids], asserts, source: {file, index}}],
         rows: [[M, N, batch, K, set, library]]}

Branches are the hardware rows of the shipped library in its order (PCI ID, then
CU count, then generic). An index entry is one branch and ProblemType: the core
key holds the fields a request must equal; the features are the epilogue it
supports. A set's params are the valid fork parameters that differ from the
global defaults, plus the instruction, depth, cache hints, execution policy and
GlobalSplitU, which every set names. Parameters the request or the backend owns
are left out, and alignment asserts are kept apart. Library is 0 for Equality,
1 for GridBased and 2 for Range, whose rows hold the middle of each range.
"""

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import msgpack

from . import LibraryIO
from .Common.Architectures import gfxToIsa
from .Common.GlobalParameters import defaultSolution
from .Common.ValidParameters import validParameters
from .ExecutionPolicy import normalize_execution_policy_with_defaults
from .Hardware import HardwarePredicate
from .SolutionStructs.Problem import ProblemType

MAGIC = b"HJKN"
SCHEMA = 1
GENERATOR = "tensilelite-logic-knowledge"
LIBRARIES = {"Equality": 0, "GridBased": 1, "Range": 2}

# The request, the descriptor or the backend decides these.
_OWNED = {"MXScaleFormat", "ActivationFused", "ActivationFuncCall", "ActivationAlt",
          "KernelLanguage", "NoReject", "DebugPersistentKernelLoopForever", "DebugStreamK",
          "WavefrontSize", "MaxLDS", "ThreadTile", "PreloadKernArgs", "MagicDivAlg"}
# Valid only for the sizes the solution was tuned for.
ASSERTS = ("AssertFree0ElementMultiple", "AssertFree1ElementMultiple",
           "AssertSummationElementMultiple", "AssertAIGreaterThanEqual", "AssertAILessThanEqual")
_NAMED = ("MatrixInstruction", "DepthU", "NonTemporalA", "NonTemporalB",
          "TileProcessingStrategy", "WorkAssignment", "GlobalSplitU", "GlobalSplitUAlgorithm")
TRANSPORTED = sorted((set(defaultSolution) & set(validParameters)) - _OWNED - set(ASSERTS))

CORE = ("DataTypeA", "DataTypeB", "MacDataTypeA", "MacDataTypeB", "DestDataType",
        "ComputeDataType", "F32XdlMathOp", "HighPrecisionAccumulate", "TransposeA",
        "TransposeB", "Sparse", "SwizzleTensorA", "SwizzleTensorB", "MXBlockA", "MXBlockB")
_FLAGS = ("UseScaleAlphaVec", "UseScaleCD", "UseE", "Gradient", "OutputAmaxD",
          "UseGateResidual", "GroupedGemm")


def _value(value):
    return getattr(value, "value", value)


def coreKey(fields):
    """name=value pairs in CORE order, then the MX scale types of MX operands."""
    names = list(CORE)
    names += [f"DataTypeMXS{t}" for t in "AB" if fields[f"MXBlock{t}"]]
    return ",".join(f"{name}={int(fields[name])}" for name in names)


def problemFeatures(problemType):
    """The epilogue a ProblemType supports: Bias lists its types, UseScaleAB its mode."""
    features = {}
    if problemType["UseBias"]:
        features["Bias"] = sorted(_value(t) for t in problemType["BiasDataTypeList"])
    if problemType["Activation"] and problemType["ActivationType"] != "none":
        features["Activation"] = True
    if problemType["UseScaleAB"]:
        features["UseScaleAB"] = problemType["UseScaleAB"]
    for name in _FLAGS:
        if problemType[name]:
            features[name] = True
    return features


def _problem(state, path):
    # LibraryIO.parseLibraryLogicData fills these before building the ProblemType.
    state = dict(state)
    state.setdefault("MacDataTypeA", LibraryIO.getRealDataTypeA(state["DataType"]))
    state.setdefault("MacDataTypeB", LibraryIO.getRealDataTypeB(state["DataType"]))
    state["DataTypeA"] = LibraryIO.getRealDataTypeA(state.get("DataTypeA", state["MacDataTypeA"]))
    state["DataTypeB"] = LibraryIO.getRealDataTypeB(state.get("DataTypeB", state["MacDataTypeB"]))
    problemType = ProblemType(state, False, srcFile=path, raiseOnTypeMismatch=False)
    fields = {name: _value(problemType[name]) for name in CORE}
    for t in "AB":
        fields[f"DataTypeMXS{t}"] = _value(problemType[f"DataTypeMXS{t}"])
    return coreKey(fields), problemFeatures(problemType)


def _instruction(state):
    mi = list(state.get("MatrixInstruction") or [])
    if len(mi) == 4 and state.get("MIWaveTile") and state.get("MIWaveGroup"):
        mi += [1, *state["MIWaveTile"], *state["MIWaveGroup"]]
    return mi if len(mi) == 9 else None


def _set(solution, fileDefaults, source):
    """The set for one solution, or None when JIT cannot generate it."""
    custom = solution.get("CustomKernel")
    if solution.get("CustomKernelName") or (
            isinstance(custom, dict) and custom.get("name") and not custom.get("generated")):
        return None
    try:
        state = normalize_execution_policy_with_defaults(solution, fileDefaults)
    except ValueError:
        return None
    for key, value in defaultSolution.items():
        state.setdefault(key, value)
    mi = _instruction(state)
    if mi is None or "MacroTile0" not in state or "MacroTile1" not in state:
        return None
    params = {}
    for name in TRANSPORTED:
        value = mi if name == "MatrixInstruction" else state[name]
        if name in _NAMED or value != defaultSolution[name]:
            params[name] = value
    return {
        "macro_tile": [state["MacroTile0"], state["MacroTile1"]],
        "waves": mi[7:9],
        "instruction": mi[:4],
        "depth_u": state["DepthU"],
        "nt": [state["NonTemporalA"], state["NonTemporalB"]],
        "policy": {"strategy": state["TileProcessingStrategy"],
                   "assignment": state["WorkAssignment"]},
        "gsu": state["GlobalSplitU"],
        "gsu_algorithm": state["GlobalSplitUAlgorithm"],
        "params": params,
        "asserts": {name: state[name] for name in ASSERTS
                    if name in state and state[name] != defaultSolution.get(name)},
        "source": source,
    }


def readLogic(path, root):
    """One logic file's branch, ProblemType, sets and rows, or None when it has no table."""
    data = LibraryIO.read(str(path))
    if isinstance(data, list):
        data = LibraryIO.parseLibraryLogicList(data, str(path))
        library = data["Library"]["distance"]
    else:
        library = data.get("LibraryType")
    if library not in LIBRARIES or not data.get("ExactLogic"):
        return None
    relative = Path(path).relative_to(root).as_posix()
    core, features = _problem(data["ProblemType"], str(path))
    fileDefaults = data.get("DefaultSolution") or {}
    sets = [_set(s, fileDefaults, {"file": relative, "index": s.get("SolutionIndex", i)})
            for i, s in enumerate(data["Solutions"])]
    rows = []
    for sizes, value in data["ExactLogic"]:
        if library == "Range" and len(sizes) >= 8:
            sizes = [(sizes[i] + sizes[i + 1]) // 2 for i in range(0, 8, 2)]
        position = int(value[0])
        if 0 <= position < len(sets) and sets[position] is not None:
            rows.append([*sizes[:4], position, LIBRARIES[library]])
    names = data.get("DeviceNames")
    return {
        "file": relative,
        "device": (data["ArchitectureName"], data.get("CUCount"),
                   tuple(names) if isinstance(names, list) else names),
        "name": Path(path).stem.split("_", 1)[-1],
        "core_key": core,
        "features": features,
        "sets": sets,
        "rows": rows,
        "skipped": sum(s is None for s in sets),
    }


def _branch(predicate):
    """kind, cu_count and pci_ids of a HardwarePredicate from FromHardware."""
    parts = predicate.value.value if predicate.value.tag == "And" else [predicate.value]
    cuCount, pciIds = None, []
    for part in parts:
        if part.tag == "CUCount":
            cuCount = part.value
        elif part.tag == "PciChipId":
            pciIds = [part.value]
        elif part.tag == "Or":
            pciIds = sorted(p.value for p in part.value if p.tag == "PciChipId")
    kind = "pci" if pciIds else "cu" if cuCount is not None else "generic"
    return {"kind": kind, "cu_count": cuCount, "pci_ids": pciIds}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def build(records):
    """Branches and (index entry, block) pairs from readLogic records, deterministically.
    A file whose rows all name skipped solutions adds no group."""
    records = [r for r in records if r["rows"]]
    predicates = {}
    for record in records:
        arch, cuCount, names = record["device"]
        if record["device"] not in predicates:
            predicates[record["device"]] = HardwarePredicate.FromHardware(
                gfxToIsa(arch), cuCount, list(names) if isinstance(names, tuple) else names,
                logicFile=record["file"])
    ordered, branchOf = [], {}
    for predicate in sorted(predicates.values()):
        fields = _branch(predicate)
        if fields not in ordered:
            ordered.append(fields)
    for device, predicate in predicates.items():
        branchOf[device] = ordered.index(_branch(predicate))

    groups = {}
    for record in sorted(records, key=lambda r: r["file"]):
        key = (branchOf[record["device"]], record["core_key"], _canonical(record["features"]))
        group = groups.setdefault(key, {"name": record["name"], "features": record["features"],
                                        "sets": [], "seen": {}, "rows": []})
        local = []
        for s in record["sets"]:
            if s is None:
                local.append(None)
                continue
            identity = _canonical({k: v for k, v in s.items() if k != "source"})
            if identity not in group["seen"]:
                group["seen"][identity] = len(group["sets"])
                group["sets"].append(s)
            local.append(group["seen"][identity])
        group["rows"] += [[*row[:4], local[row[4]], row[5]] for row in record["rows"]]

    entries = []
    for (branch, core, _), group in sorted(groups.items(), key=lambda item: item[0]):
        dictionary, ids, sets = [], {}, []
        for s in group["sets"]:
            params = []
            for name, value in s["params"].items():
                pair = _canonical([name, value])
                if pair not in ids:
                    ids[pair] = len(dictionary)
                    dictionary.append([name, value])
                params.append(ids[pair])
            sets.append({**s, "params": params})
        block = zlib.compress(msgpack.packb(
            {"param_dictionary": dictionary, "sets": sets, "rows": group["rows"]},
            use_bin_type=True), 6)
        entries.append(({"branch": branch,
                         "problem_type": {"name": group["name"], "features": group["features"]},
                         "core_key": core, "rows": len(group["rows"]), "sets": len(sets)}, block))
    return ordered, entries


def write(path, arch, libraryArch, branches, entries, sourceCommit=""):
    index, offset, digest = [], 0, hashlib.sha256()
    digest.update(msgpack.packb([arch, libraryArch, branches, [e for e, _ in entries]],
                                use_bin_type=True))
    for entry, block in entries:
        index.append({**entry, "offset": offset, "length": len(block)})
        offset += len(block)
        digest.update(block)
    header = msgpack.packb({
        "generator": GENERATOR, "source_commit": sourceCommit,
        "content_hash": digest.hexdigest()[:16], "arch": arch, "library_arch": libraryArch,
        "branches": branches, "index": index}, use_bin_type=True)
    temporary = Path(f"{path}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as stream:
        stream.write(MAGIC + struct.pack("<II", SCHEMA, len(header)))
        stream.write(header)
        for _, block in entries:
            stream.write(block)
    os.replace(temporary, path)


def readHeader(path):
    """(header, offset of the first block)."""
    with open(path, "rb") as stream:
        preamble = stream.read(12)
        if len(preamble) != 12 or preamble[:4] != MAGIC:
            raise ValueError(f"{path} is not a JIT knowledge file")
        schema, length = struct.unpack("<II", preamble[4:])
        if schema != SCHEMA:
            raise ValueError(f"{path} has schema {schema}, not {SCHEMA}")
        return msgpack.unpackb(stream.read(length), raw=False, strict_map_key=False), 12 + length


def readGroup(path, start, entry):
    with open(path, "rb") as stream:
        stream.seek(start + entry["offset"])
        return msgpack.unpackb(zlib.decompress(stream.read(entry["length"])), raw=False,
                               strict_map_key=False)


def _sourceCommit(directory):
    try:
        done = subprocess.run(["git", "-C", str(directory), "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, timeout=30, check=False)
        return done.stdout.strip() if done.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _read(arguments):
    return readLogic(*arguments)


def extract(logicDirectory, output, architecture, libraryArchitecture=None, jobs=None):
    root = Path(logicDirectory)
    files = sorted(root.rglob("*.yaml"), key=lambda p: (-p.stat().st_size, str(p)))
    with Pool(jobs or os.cpu_count()) as pool:
        records = [r for r in pool.imap_unordered(_read, [(f, root) for f in files], chunksize=1)
                   if r is not None]
    branches, entries = build(records)
    write(output, architecture, libraryArchitecture or architecture, branches, entries,
          _sourceCommit(root))
    return {"files": len(files), "logic": len(records), "groups": len(entries),
            "rows": sum(e["rows"] for e, _ in entries), "sets": sum(e["sets"] for e, _ in entries),
            "skipped_solutions": sum(r["skipped"] for r in records),
            "bytes": os.path.getsize(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("logic_directory")
    parser.add_argument("output")
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--library-architecture")
    parser.add_argument("--jobs", type=int)
    args = parser.parse_args(argv)
    summary = extract(args.logic_directory, args.output, args.architecture,
                      args.library_architecture, args.jobs)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
