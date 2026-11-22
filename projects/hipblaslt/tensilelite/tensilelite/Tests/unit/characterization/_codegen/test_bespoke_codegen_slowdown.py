# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compiled-ISA regressions for beta-free conversion and dead runtime staggering.

Feature F1: make the beta-free W descriptor wave-uniform in the conversion
emitter; allowing a divergent descriptor again restores the load waterfalls.
The broadcast must reproduce the 64-bit limit exactly (readfirstlane is int).
Grouped beta-free variants already have scalar descriptors and are preservation
cases: introducing lane-varying descriptor limits makes them waterfall too.
Feature F2: disable runtime StaggerU emission only for BespokeNarrowing,
StaggerU=0, SupportCustomStaggerU=false (Stream-K and non-persistent GSU);
dropping that gate restores dead code. Bespoke kernels failing either
conjunct keep their stagger code (preservation; each conjunct mutation-proven).
Preservation: stock helper/kernel machine code and compiled assembly stay
byte-identical to the pinned base. Changing the shared buffer descriptor or
stock stagger eligibility breaks these pins (scratch mutation evidence).

The four one-solution fixtures are unchanged recipes from the slowdown report.
The five epilogues cover every distinct beta-free family in its helper scan,
including grouped siblings, PostGSU 2/4/8/16 and vector widths 1/2/4.
Feature C4: every beta-kept or beta-free conversion helper has no more W-load
waterfalls than stock helpers with the same type/GG/PostGSU/VW role. Extending
the wave-uniform descriptor limit to beta-kept bespoke bodies makes this pass;
gating it on betaFree again restores the beta-kept waterfalls. PostGSU1 uses
the normal SingleBuffer helper factory; PostGSU2/4/8/16 use MultipleBuffer.
The stock comparator assemblies are byte-pinned, including PostGSU1.
Beta-kept bespoke helpers have a distinct _Bespoke token in the _Beta0 slot;
without it a mixed library deduplicates different helper bodies by name.
No GPU is opened; this suite uses the production HIP compiler and assembler.
"""
import json
import re
import subprocess

import pytest
import yaml

from bespoke_codegen_common import (
    DATA, assembly_digest, compile_helpers, compile_main, helper_source, variant,
    stagger_instructions, workspace_waterfalls,
)

pytestmark = pytest.mark.unit
STOCK_RECIPES = ("bf16_streamk.yaml", "bf16_bias_streamk.yaml",
                 "f16_streamk.yaml", "f32_streamk.yaml")
# Literal identities/settings from helper_scan.txt; expectations are not derived
# from the emitter under test. GG is emitted by the normal helper-generation path.
FAMILIES = (
    ("bf16_streamk.yaml", "SB", "none", 0, [], 0),
    ("bf16_bias_streamk.yaml", "SB_BiasB", "none", 1, [7], 0),
    ("f16_streamk.yaml", "SH_SIGMOIDS_ScaleAlphaVec", "sigmoid", 0, [], 1),
    ("f32_streamk.yaml", "SS", "none", 0, [], 0),
    ("f32_streamk.yaml", "SS_SIGMOIDS", "sigmoid", 0, [], 0),
)
SINGLE_BUFFER_RECIPE = {"StreamK": 0, "GlobalSplitU": 2,
                        "GlobalSplitUAlgorithm": "SingleBuffer",
                        "WorkGroupMappingXCC": 1, "StoreRemapVectorWidth": 0}


@pytest.fixture(scope="module")
def general_conversion_helpers(tmp_path_factory):
    # Compile each unchanged comparator once, shared by feature and stock-pin tests.
    compiled = {}

    def get(recipe, algorithm):
        key = recipe, algorithm
        if key not in compiled:
            out = tmp_path_factory.mktemp(f"stock_{algorithm}_{recipe[:-5]}")
            path = (variant(recipe, out / "stock.yaml", solution=SINGLE_BUFFER_RECIPE)
                    if algorithm == "SingleBuffer" else DATA / recipe)
            with pytest.MonkeyPatch.context() as patch:
                patch.setenv("TENSILE_DISABLE_HELPER_CACHE", "1")
                compiled[key] = compile_helpers(path, out / "compiled")
        return compiled[key]

    return get


@pytest.fixture(scope="module", params=[(family, use_beta) for family in FAMILIES
                                       for use_beta in (True, False)],
                ids=[f"{family[1]}_beta_{'kept' if use_beta else 'free'}"
                     for family in FAMILIES for use_beta in (True, False)])
def paired_conversion_helpers(request, tmp_path_factory, general_conversion_helpers):
    (recipe, family, activation, bias, bias_types, sav), use_beta = request.param
    out = tmp_path_factory.mktemp(f"{family}_{use_beta}")
    general, bespoke = {}, {}
    for algorithm in ("MultipleBuffer", "SingleBuffer"):
        stock, _ = general_conversion_helpers(recipe, algorithm)
        general.update(stock)
        path = variant(recipe, out / f"{algorithm}.yaml",
                       problem={"UseBeta": use_beta, "ActivationType": activation,
                                "UseBias": bias, "BiasDataTypeList": bias_types,
                                "UseScaleAlphaVec": sav},
                       solution=dict(SINGLE_BUFFER_RECIPE if algorithm == "SingleBuffer" else {},
                                     BespokeNarrowing=True))
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("TENSILE_DISABLE_HELPER_CACHE", "1")
            helpers, _ = compile_helpers(path, out / algorithm)
        bespoke.update(helpers)
    beta = "_Bespoke" if use_beta else "_Beta0"
    return family, beta, general, bespoke


@pytest.mark.parametrize("gsu,loads", [(1, 1), (2, 2), (4, 4), (8, 8), (16, 152)])
@pytest.mark.parametrize("vw", [1, 2, 4])
@pytest.mark.parametrize("grouped", [False, True], ids=["strided", "grouped"])
def test_bespoke_conversion_waterfalls_do_not_exceed_general(paired_conversion_helpers, gsu, loads, vw, grouped):
    family, beta, general, bespoke = paired_conversion_helpers
    first, _, rest = family.partition("_")
    expected = {f"Cijk_{first}{'_GG' if gg else ''}{'_' + rest if rest else ''}"
                f"{beta}_PostGSU{g}_VW{v}"
                for gg in (False, True) for g in (1, 2, 4, 8, 16) for v in (1, 2, 4)}
    assert set(bespoke) == expected, f"{family}: compiled bespoke helper identities do not match the isolated namespace"
    name = (f"Cijk_{first}{'_GG' if grouped else ''}{'_' + rest if rest else ''}"
            f"{beta}_PostGSU{gsu}_VW{vw}")
    instructions = bespoke[name]
    workspace_loads = [code for _, code in instructions if code.startswith("buffer_load_")]
    assert len(workspace_loads) == loads, f"{name}: W reduction loads lost or duplicated"
    role = re.compile(rf"^Cijk_{first}{'_GG' if grouped else ''}(?:_(?!GG_)[A-Za-z0-9]+)*"
                      rf"_PostGSU{gsu}_VW{vw}$")
    comparators = {symbol: code for symbol, code in general.items() if role.fullmatch(symbol)}
    assert comparators, f"{name}: no compiled general helper of the same role"
    assert all(sum(code.startswith("buffer_load_") for _, code in stream) == loads
               for stream in comparators.values()), f"{name}: general W-load positive control changed"
    general_counts = {symbol: len(workspace_waterfalls(code)) for symbol, code in comparators.items()}
    allowed = max(general_counts.values())
    waterfalls = workspace_waterfalls(instructions)
    assert len(waterfalls) <= allowed, \
        (f"{name}: bespoke={len(waterfalls)} W-load waterfalls > general={allowed}; "
         f"loads={loads}, comparators={general_counts}")


@pytest.fixture(scope="module", params=FAMILIES, ids=[family[1] for family in FAMILIES])
def beta_free_helpers(request, tmp_path_factory):
    recipe, family, activation, bias, bias_types, sav = request.param
    out = tmp_path_factory.mktemp(family)
    path = variant(recipe, out / "beta_free.yaml",
                   problem={"UseBeta": False, "ActivationType": activation,
                            "UseBias": bias, "BiasDataTypeList": bias_types,
                            "UseScaleAlphaVec": sav},
                   solution={"BespokeNarrowing": True})
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TENSILE_DISABLE_HELPER_CACHE", "1")
        helpers, _ = compile_helpers(path, out / "compiled")
    first, _, rest = family.partition("_")
    expected = {f"Cijk_{first}{'_GG' if grouped else ''}{'_' + rest if rest else ''}"
                f"_Beta0_PostGSU{gsu}_VW{vw}"
                for grouped in (False, True) for gsu in (2, 4, 8, 16) for vw in (1, 2, 4)}
    assert set(helpers) == expected, "compiled helper variant inventory changed"
    return family, helpers


@pytest.mark.parametrize("gsu,loads", [(2, 2), (4, 4), (8, 8), (16, 152)])
@pytest.mark.parametrize("vw", [1, 2, 4])
@pytest.mark.parametrize("grouped", [False, True], ids=["strided", "grouped"])
def test_beta_free_workspace_loads_have_scalar_descriptors(beta_free_helpers, gsu, loads, vw, grouped):
    family, helpers = beta_free_helpers
    first, _, rest = family.partition("_")
    name = (f"Cijk_{first}{'_GG' if grouped else ''}{'_' + rest if rest else ''}"
            f"_Beta0_PostGSU{gsu}_VW{vw}")
    instructions = helpers[name]
    workspace_loads = [code for _, code in instructions if code.startswith("buffer_load_")]
    assert len(workspace_loads) == loads, f"{name}: W reduction loads lost or duplicated"
    assert all(re.search(r", s\[\d+:\d+\],", code) for code in workspace_loads), \
        f"{name}: W loads do not use an SGPR buffer descriptor"
    waterfalls = workspace_waterfalls(instructions)
    saveexec = sum(code.startswith("s_and_saveexec_") for _, code in instructions)
    readfirstlane = sum(code.startswith("v_readfirstlane_") for _, code in instructions)
    assert not waterfalls, (f"{name}: {len(waterfalls)}/{loads} W loads in descriptor waterfalls; "
                            f"saveexec={saveexec}, readfirstlane={readfirstlane}")


def test_bespoke_host_zero_stagger_emits_no_runtime_stagger(tmp_path):
    path = variant("bf16_streamk.yaml", tmp_path / "exact_k.yaml",
                   solution={"BespokeNarrowing": True,
                             "AssertSizeEqual": {3: 196608},
                             "AssertSummationElementMultiple": 256})
    solution, source, _, _ = compile_main(path, tmp_path / "compiled")
    assert solution["StaggerU"] == 0
    assert solution["InternalSupportParams"]["SupportCustomStaggerU"] is False
    assert solution["NoTailLoop"] is True
    assert solution["TailloopInNll"] is False
    dead = stagger_instructions(source)
    assert not dead, f"host-zero bespoke kernel emits {len(dead)} runtime stagger instructions: {dead[:3]}"


EXACT_K = {"BespokeNarrowing": True, "AssertSizeEqual": {3: 196608},
           "AssertSummationElementMultiple": 256}


def with_custom_stagger(recipe):
    params = yaml.safe_load((DATA / recipe).read_text())["Solutions"][0]["InternalSupportParams"]
    return dict(params, SupportCustomStaggerU=True)


def test_bespoke_gsu_host_zero_stagger_emits_no_runtime_stagger(tmp_path):
    # Non-persistent GSU (MultipleBuffer) sibling of the Stream-K feature case.
    from tensilelite.ExecutionPolicy import isPersistent

    path = variant("bf16_streamk.yaml", tmp_path / "gsu_exact_k.yaml",
                   solution=dict(EXACT_K, StreamK=0, GlobalSplitU=2, WorkGroupMappingXCC=1))
    solution, source, _, _ = compile_main(path, tmp_path / "compiled")
    assert solution["GlobalSplitU"] == 2 and solution["_GlobalAccumulation"] == "MultipleBuffer"
    assert not isPersistent(solution)
    assert solution["StaggerU"] == 0
    assert solution["InternalSupportParams"]["SupportCustomStaggerU"] is False
    assert solution["NoTailLoop"] is True and solution["TailloopInNll"] is False
    dead = stagger_instructions(source)
    assert not dead, f"host-zero GSU bespoke kernel emits {len(dead)} runtime stagger instructions: {dead[:3]}"


@pytest.mark.parametrize("change,custom,stagger", [
    ({"InternalSupportParams": with_custom_stagger("bf16_streamk.yaml")}, True, 0),
    ({"StaggerU": 32, "StaggerUStride": 256}, False, 32),
], ids=["host_custom_stagger", "nonzero_stagger"])
def test_bespoke_stagger_outside_the_dead_gate_is_kept(tmp_path, change, custom, stagger):
    # Preservation: F2 removes runtime staggering only when StaggerU == 0 and the host
    # cannot pack one; a bespoke kernel failing either conjunct keeps its stagger code.
    path = variant("bf16_streamk.yaml", tmp_path / "exact_k.yaml", solution=dict(EXACT_K, **change))
    solution, source, _, _ = compile_main(path, tmp_path / "compiled")
    assert solution["StaggerU"] == stagger
    assert solution["InternalSupportParams"]["SupportCustomStaggerU"] is custom
    assert solution["NoTailLoop"] is True and solution["TailloopInNll"] is False
    kept = stagger_instructions(source)
    assert len(kept) == 50, f"bespoke kernel outside the dead-stagger gate lost its runtime stagger ({len(kept)})"


def test_beta_free_descriptor_limit_broadcast_preserves_the_value(tmp_path):
    # The emitted wave-uniform reassignment must reproduce strideWLimit exactly.
    # readfirstlane is declared int(int); model it as that identity on the host.
    from tensilelite.Toolchain.Validators import validateToolchain

    path = variant("bf16_streamk.yaml", tmp_path / "beta_free.yaml",
                   problem={"UseBeta": False, "ActivationType": "none", "UseBias": 0,
                            "BiasDataTypeList": [], "UseScaleAlphaVec": 0},
                   solution={"BespokeNarrowing": True})
    source = helper_source(path, tmp_path / "helpers")
    expressions = set(re.findall(r"^\s*strideWLimit = (.+);\s*$", source, flags=re.M))
    assert len(expressions) == 1, f"expected one beta-free limit broadcast, found {expressions}"
    program = tmp_path / "limit.cpp"
    program.write_text(
        "#include <cstdint>\n#include <cstdio>\n"
        "static int rfl(int v) { return v; }\n#define __builtin_amdgcn_readfirstlane rfl\n"
        "int main() {\n"
        "  const uint64_t cases[] = {0x7fffffffull, 0x80000000ull, 0xffffffffull,\n"
        "                            0x100000000ull, 0x8000000080000000ull, 0x123456789abcdef0ull};\n"
        "  for (uint64_t c : cases) {\n"
        "    uint64_t strideWLimit = c;\n"
        f"    strideWLimit = {expressions.pop()};\n"
        "    if (strideWLimit != c) { std::printf(\"%llx -> %llx\\n\", (unsigned long long)c,"
        " (unsigned long long)strideWLimit); return 1; }\n"
        "  }\n  return 0;\n}\n")
    binary = tmp_path / "limit"
    subprocess.run([validateToolchain("amdclang++"), "-x", "c++", "-std=c++17", "-O0",
                    str(program), "-o", str(binary)], check=True)
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, f"beta-free limit broadcast changes the value: {run.stdout.strip()}"


@pytest.mark.parametrize("recipe", STOCK_RECIPES)
def test_stock_post_gsu_compiled_assembly_is_byte_identical(recipe, general_conversion_helpers):
    pins = json.loads((DATA / "stock_assembly_digests.json").read_text())[recipe]["helpers"]
    helpers, text_sha = general_conversion_helpers(recipe, "MultipleBuffer")
    assert text_sha == pins["text_sha256"], f"{recipe}: stock helper .text bytes changed"
    assert {name: assembly_digest(code) for name, code in helpers.items()} == pins["assembly_sha256"], \
        f"{recipe}: stock compiled helper assembly changed"
    assert not any(workspace_waterfalls(code) for code in helpers.values()), \
        f"{recipe}: stock positive control waterfalls W loads"


@pytest.mark.parametrize("recipe", STOCK_RECIPES)
def test_stock_single_buffer_compiled_assembly_is_byte_identical(recipe, general_conversion_helpers):
    pins = json.loads((DATA / "stock_single_buffer_digests.json").read_text())[recipe]
    helpers, text_sha = general_conversion_helpers(recipe, "SingleBuffer")
    assert text_sha == pins["text_sha256"], f"{recipe}: stock SingleBuffer helper .text bytes changed"
    assert {name: assembly_digest(code) for name, code in helpers.items()} == pins["assembly_sha256"], \
        f"{recipe}: stock SingleBuffer compiled helper assembly changed"
    assert not any(workspace_waterfalls(code) for code in helpers.values()), \
        f"{recipe}: stock SingleBuffer positive control waterfalls W loads"


@pytest.mark.parametrize("recipe", STOCK_RECIPES)
def test_stock_main_compiled_assembly_is_byte_identical(tmp_path, recipe):
    pins = json.loads((DATA / "stock_assembly_digests.json").read_text())[recipe]["main"]
    _, _, instructions, text_sha = compile_main(DATA / recipe, tmp_path / "compiled")
    assert text_sha == pins["text_sha256"], f"{recipe}: stock kernel .text bytes changed"
    assert assembly_digest(instructions) == pins["assembly_sha256"], \
        f"{recipe}: stock compiled kernel assembly changed"
