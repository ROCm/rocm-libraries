# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate built fixture identities/paths and write the unchecked-launch index map."""
import argparse
from pathlib import Path
import re
import shutil
import zlib

import msgpack
import yaml


def check(build, table, output):
    mapping = yaml.safe_load(table.read_text())
    solutions = {}
    for path in (build / "library").rglob("TensileLibrary*"):
        if path.suffix == ".yaml":
            document = yaml.safe_load(path.read_text())
        elif path.name.endswith(".dat.zlib"):
            document = msgpack.unpackb(zlib.decompress(path.read_bytes()), raw=False,
                                      strict_map_key=False)
        elif path.suffix == ".dat":
            document = msgpack.unpackb(path.read_bytes(), raw=False, strict_map_key=False)
        else:
            continue
        for solution in document.get("solutions", []):
            solutions[solution["index"]] = solution
    assembly = {}
    for path in (build / "build_tmp").rglob("*.s"):
        source = path.read_text()
        for kernel in re.findall(r"^\s*\.amdhsa_kernel\s+(\S+)", source, re.M):
            assembly[kernel] = source
    named = {}
    for index, (logic, source_index) in mapping.items():
        name = Path(logic).stem
        assert source_index == 0, (name, source_index)
        assert name not in named, ("duplicate fixture", name)
        assert index in solutions, ("missing indexed solution", name, index)
        named[name] = (index, solutions[index])
    kernels = [solution["kernelName"] for _, solution in named.values()]
    assert len(set(kernels)) == len(kernels), "general/bespoke fixture kernels collided"
    for name, (_, solution) in named.items():
        assert solution["kernelName"] in assembly, ("missing emitted assembly", name)
        # The host launch guard covers only solutions marked bespokeNarrowing.
        assert solution.get("bespokeNarrowing", False) is name.endswith("_bespoke"), \
            (name, "launch-guard marker does not match the fixture kind")
        if name.startswith("bias_grouped_"):
            assert solution["problemType"]["groupedGemm"] is True, \
                (name, "device-args positive control requires a grouped solution")
        if name == "beta_off_streamk_bespoke":
            assert solution["problemType"]["useBeta"] is False, "Stream-K discarded UseBeta=false"
        if name == "single_gelu_bespoke":
            assert solution["problemType"]["activationType"] == "Gelu", "single GELU was not retained"
        if name == "sigmoid_client_bespoke":
            assert solution["problemType"]["activationType"] == "Sigmoid", "single sigmoid was not retained"
        if name in ("exact_size160_bespoke", "edge_removal_bespoke"):
            source = assembly[solution["kernelName"]]
            edges = re.findall(r"\bs_cbranch_scc1\s+(label_GW_\w+(?:Then|Else))\b", source)
            assert not edges, (name, "full tiles still emit edge dispatch", edges)
            assert not re.search(r"^label_GW_\w+_Then(?:_\w+)?:", source, re.M), (name, "edge store body remains")
            general_name = "streamk_general" if name == "edge_removal_bespoke" else "exact_size160_general"
            general = named[general_name][1]
            assert re.search(r"\bs_cbranch_scc1\s+label_GW_\w+(?:Then|Else)\b",
                             assembly[general["kernelName"]]), (name, "general edge positive control absent")
        if name == "loop_exits_bespoke":
            source = assembly[solution["kernelName"]]
            operands = re.findall(r"\bs_sub_u32\s+s\[sgprLoopCounterL\],\s*s\[sgprLoopCounterL\],\s*(\S+)", source)
            assert len(operands) == 2 and all(x == "1" for x in operands), (name, "incorrect loop decrements", operands)
    # A stock carrier deliberately shares the bespoke code object and kernel
    # name, but not its host-only guard marker.
    carrier = build / "carrier-library"
    shutil.copytree(build / "library", carrier, dirs_exist_ok=True)
    carrier_index = named["bias_present_bespoke"][0]
    changed = 0
    for path in carrier.rglob("TensileLibrary*"):
        if path.suffix == ".yaml":
            document = yaml.safe_load(path.read_text())
        elif path.suffix == ".dat" or path.name.endswith(".dat.zlib"):
            data = path.read_bytes()
            document = msgpack.unpackb(zlib.decompress(data) if path.suffix == ".zlib" else data,
                                      raw=False, strict_map_key=False)
        else:
            continue
        matched = False
        for solution in document.get("solutions", []):
            if solution["index"] == carrier_index:
                assert solution.pop("bespokeNarrowing") is True
                changed += 1
                matched = True
        if not matched:
            continue
        if path.suffix == ".yaml":
            path.write_text(yaml.safe_dump(document, sort_keys=False))
        else:
            data = msgpack.packb(document, use_bin_type=True)
            path.write_bytes(zlib.compress(data) if path.suffix == ".zlib" else data)
    assert changed == 1, ("stock carrier did not identify exactly one bespoke solution", changed)
    output.write_text("".join(f"{name} {index} {solution['kernelName']}\n"
                              for name, (index, solution) in sorted(named.items())))
    print(f"fixture identities and generated paths verified: {len(named)} solutions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("table", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    check(args.build, args.table, args.output)
