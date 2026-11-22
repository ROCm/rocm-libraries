# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Beta-off Stream-K: real derivation and assembly, not a forced writer-only fixture.

Feature: bespoke (BespokeNarrowing) beta-off SK3/4/5 assembles when the
partial-index alias uses its live beta-padding register; reverting that alias
breaks assembly again.
Feature: beta-off Stream-K without BespokeNarrowing (which did not assemble before the
beta-padding alias, and which the host does not guard) is refused at generation;
accepting it breaks this test.
Preservation: beta-on SK3/4/5, keyless beta-off without Stream-K and stock
grouped requests remain accepted; aliasing beta-on to an undefined padding
register / refusing keyless beta-off for every strategy / refusing stock
grouped requests breaks these controls (scratch mutations recorded in the
report).
"""
from pathlib import Path

import pytest
import yaml

import codegen_harness as H

pytestmark = pytest.mark.unit
DATA = Path(__file__).parent / "data/test_data/_logic/gfx950"


def logic_variant(tmp_path, seed, *, use_beta, user_args, stream_k=None, bespoke=False):
    logic = yaml.safe_load((DATA / seed).read_text())
    logic["ProblemType"].update(UseBeta=use_beta, SupportUserArgs=user_args)
    if stream_k is not None:
        logic["Solutions"][0]["StreamK"] = stream_k
    if bespoke:
        logic["Solutions"][0]["BespokeNarrowing"] = True
    # LibraryIO resets derived flags itself. No ForceGenerate/Valid overrides.
    assert logic["Solutions"][0]["GlobalSplitU"] == 1
    assert logic.get("DefaultSolution", {}).get("AssertSummationElementMultiple", 1) <= 256
    path = tmp_path / "request.yaml"
    path.write_text(yaml.safe_dump(logic, sort_keys=False))
    return path


def assemble(path, tmp_path):
    (solution,) = H.solutions_from_logic(path)
    assert solution["Valid"], "normal derivation refused a positive request"
    (name, source, err), = H.emit_kernels_from_logic(path, canonical=False)
    assert err == 0, "kernel emission failed"
    src, obj = tmp_path / (name + ".s"), tmp_path / (name + ".o")
    src.write_text(source)
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import ToolchainDefaults, validateToolchain

    assembler = makeAssemblyToolchain(validateToolchain("amdclang++"),
                                     validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER), "4").assembler
    assembler("gfx950", 64, str(src), str(obj))
    assert obj.stat().st_size > 0
    return solution, source


@pytest.mark.parametrize("user_args", [False, True], ids=["noUserArgs", "UserArgs"])
@pytest.mark.parametrize("stream_k", [3, 4, 5], ids=["SK3", "SK4", "SK5"])
def test_usebeta_false_streamk_emits_and_assembles(tmp_path, stream_k, user_args):
    path = logic_variant(tmp_path, "BSS_MT32x64_beta_streamk.yaml",
                         use_beta=False, user_args=user_args, stream_k=stream_k, bespoke=True)
    solution, source = assemble(path, tmp_path)
    assert solution["TileProcessingStrategy"] == "StreamK"
    assert solution["WorkAssignment"] == {3: "StaticGrid", 4: "DynamicWorkQueue", 5: "Hybrid"}[stream_k]
    assert solution["ProblemType"]["UseBeta"] is False
    assert solution["ProblemType"]["SupportUserArgs"] is user_args
    assert "label_GW_B1_" not in source  # executable beta != 0 store path omitted


@pytest.mark.parametrize("user_args", [False, True], ids=["noUserArgs", "UserArgs"])
@pytest.mark.parametrize("stream_k", [3, 4, 5], ids=["SK3", "SK4", "SK5"])
def test_generation_refuses_keyless_usebeta_false_streamk(tmp_path, stream_k, user_args):
    path = logic_variant(tmp_path, "BSS_MT32x64_beta_streamk.yaml",
                         use_beta=False, user_args=user_args, stream_k=stream_k)
    with pytest.raises(RuntimeError):
        H.solutions_from_logic(path)


@pytest.mark.parametrize("user_args", [False, True], ids=["noUserArgs", "UserArgs"])
def test_keyless_usebeta_false_without_streamk_still_assembles(tmp_path, user_args):
    path = logic_variant(tmp_path, "BSS_MT32x64_beta_streamk.yaml",
                         use_beta=False, user_args=user_args, stream_k=0)
    solution, source = assemble(path, tmp_path)
    assert solution["TileProcessingStrategy"] != "StreamK"
    assert solution["ProblemType"]["UseBeta"] is False
    assert "label_GW_B1_" not in source


@pytest.mark.parametrize("user_args", [False, True], ids=["noUserArgs", "UserArgs"])
@pytest.mark.parametrize("stream_k", [3, 4, 5], ids=["SK3", "SK4", "SK5"])
def test_usebeta_true_streamk_still_assembles(tmp_path, stream_k, user_args):
    path = logic_variant(tmp_path, "BSS_MT32x64_beta_streamk.yaml",
                         use_beta=True, user_args=user_args, stream_k=stream_k)
    solution, source = assemble(path, tmp_path)
    assert solution["TileProcessingStrategy"] == "StreamK"
    assert solution["WorkAssignment"] == {3: "StaticGrid", 4: "DynamicWorkQueue", 5: "Hybrid"}[stream_k]
    assert solution["ProblemType"]["UseBeta"] is True
    assert "label_GW_B1_" in source


@pytest.mark.parametrize("user_args", [False, True], ids=["grouped", "groupedDeviceArgs"])
def test_generation_refuses_beta_off_grouped(tmp_path, user_args):
    path = logic_variant(tmp_path, "HHS_grouped_beta.yaml",
                         use_beta=False, user_args=user_args)
    # A parser exception OR a derived invalid solution is an explicit refusal.
    # No assertion depends on rejection wording.
    try:
        solutions = H.solutions_from_logic(path)
    except (RuntimeError, ValueError):
        return
    assert not solutions or all(not s["Valid"] for s in solutions), \
        "generation accepted UseBeta:false grouped/device-argument request"


@pytest.mark.parametrize("user_args", [False, True], ids=["grouped", "groupedDeviceArgs"])
def test_generation_preserves_stock_grouped_beta_on(tmp_path, user_args):
    path = logic_variant(tmp_path, "HHS_grouped_beta.yaml",
                         use_beta=True, user_args=user_args)
    (solution,) = H.solutions_from_logic(path)
    assert solution["Valid"], "stock grouped beta-on request must remain valid"
    assert solution["ProblemType"]["GroupedGemm"] is True
    assert solution["ProblemType"]["UseBeta"] is True
