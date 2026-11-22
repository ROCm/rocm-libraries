# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Real conversion-helper derivation/emission, shared by the retained
`test_conversion_usebeta_false.py` and an external matched-library acceptance test.

Self-contained (no external tools, workstation paths or fixed core lists); the
matched-library and launch-harness preparation lives with that external test.
"""
from pathlib import Path

import yaml

import codegen_harness as H

DATA = Path(__file__).parent / "data/test_data/_logic/conversion_beta"


def conversion_helpers(path):
    from tensilelite.KernelWriterConversion import KernelWriterConversion
    from tensilelite.TensileCreateLibrary.Run import (
        generateKernelObjectsFromSolutions, generateKernelHelperObjects,
    )
    with H._isolated_globals():
        solutions = H._solutions_from_logic_unguarded(path)
        assert len(solutions) == 1 and solutions[0]["Valid"], "normal derivation refused stock recipe"
        solution = solutions[0]
        assert solution["GlobalSplitUAlgorithm"] == "MultipleBuffer"
        assert solution["_GlobalAccumulation"] == "MultipleBuffer"
        kernels = generateKernelObjectsFromSolutions(solutions)
        helpers = generateKernelHelperObjects(kernels, str(H.get_assembler().path), H.get_isa_info_map())
        out = {}
        for helper in helpers:
            if isinstance(helper, KernelWriterConversion):
                err, source = helper.getSourceFileString()
                assert err == 0, "conversion helper emission failed"
                out[helper.getKernelName()] = source
        assert out, "derived MultipleBuffer recipe did not produce conversion helpers"
        return solution, out


def beta_variant(path, destination, use_beta, *, bespoke):
    """The recipe with UseBeta set; bespoke=True marks every solution BespokeNarrowing
    (only those get the beta-free helpers, keyless ones keep develop's helpers)."""
    logic = yaml.safe_load(Path(path).read_text())
    logic["ProblemType"]["UseBeta"] = use_beta
    if bespoke:
        for solution in logic["Solutions"]:
            solution["BespokeNarrowing"] = True
    destination.write_text(yaml.safe_dump(logic, sort_keys=False))
    return destination
