# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Exact sizes: a generated CustomKernel stamp that already carries
a narrowing token must not merge a general recipe with its exact-size twin.

Feature: the narrowing token (BespokeNarrowing) follows the stamp from the
current facts and the stamp itself omits it. Trusting a token already in the
stamp (suffix check) gives both recipes the stamp's name and merges their
compile groups again.
"""
import copy
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
DATA = Path(__file__).parent / "exact_size_data"


def write_logic(tmp_path, data, name):
    path = tmp_path / (name + ".yaml")
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def edge_dispatch(source):
    return re.findall(r"\bs_cbranch_scc1\s+(label_GW_\w+(?:Then|Else))\b", source)


@pytest.mark.parametrize("order", [(False, True), (True, False)], ids=["general-first", "exact-first"])
def test_pretokenized_stamp_keeps_general_and_exact_identities(tmp_path, order):
    from codegen_harness import emit_kernels_from_logic, solutions_from_logic
    from tensilelite.CustomKernels import getCustomKernelConfig
    from tensilelite.SolutionStructs.Naming import getKernelFileBase, getKernelNameMin
    from tensilelite.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions, groupKernelRecords

    data = yaml.safe_load((DATA / "gfx950_mt160.yaml").read_text())
    recipe = data["Solutions"][0]
    exact = copy.deepcopy(data)
    exact["Solutions"][0].update(AssertSizeEqual={0: 320, 1: 64}, BespokeNarrowing=True)
    # Positive control and stamp source: the exact-size kernel emitted normally,
    # edge-free, under a symbol carrying its narrowing token. The stamp keeps
    # that token but must be short enough for a file name (the token is the
    # compact BN form, not the long assertion spelling).
    parentPath = write_logic(tmp_path, exact, "exact_parent")
    emitted = emit_kernels_from_logic(parentPath, canonical=False)
    assert len(emitted) == 1 and emitted[0][2] == 0 and not edge_dispatch(emitted[0][1])
    source = emitted[0][1]
    symbol = getKernelNameMin(solutions_from_logic(parentPath)[0], False)
    assert "_BN320x64x1" in symbol and re.search(r"\.amdhsa_kernel\s+%s\b" % re.escape(symbol), source)
    stamp = "exact_stamped_BN320x64x1"
    (tmp_path / (stamp + ".s")).write_text(source)
    recipe["CustomKernel"] = getCustomKernelConfig(stamp, {}, str(tmp_path))["CustomKernel"]
    recipe["CustomKernel"]["generated"] = True

    data["Solutions"] = []
    for index, isExact in enumerate(order):
        solution = copy.deepcopy(recipe)
        solution["SolutionIndex"] = index
        if isExact:
            solution.update(AssertSizeEqual={0: 320, 1: 64}, BespokeNarrowing=True)
        data["Solutions"].append(solution)
    path = write_logic(tmp_path, data, "pretokenized_pair")
    solutions = solutions_from_logic(path)
    assert len(solutions) == 2 and all(s["Valid"] for s in solutions)
    groups = groupKernelRecords(generateKernelObjectsFromSolutions(solutions), False)
    assert len(groups) == 2, "pre-tokenized stamp merged general and exact-size kernels"
    assert len({getKernelNameMin(g.representative, False) for g in groups}) == 2
    assert len({getKernelFileBase(False, g.representative) for g in groups}) == 2
    pair = emit_kernels_from_logic(path, canonical=False)
    assert len(pair) == 2 and all(error == 0 for _, _, error in pair)
    assert sorted(bool(edge_dispatch(source)) for _, source, _ in pair) == [False, True]
