# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only assembly regressions for bespoke custom main-loop schedules.

Keeping the stagger sequence for CMS kernels makes the feature cases build;
applying the non-CMS dead-stagger gate to CMS again breaks their GRInc schedule.
The stock compilation in each case is a positive control for the unchanged recipe.
The MX-file sample is an f32 CMS recipe with MXBlockA=MXBlockB=0, not a
persistent mixed-precision kernel; it has the same scheduled GRInc contract.
The six sampled recipes are frozen, self-contained stock records in cms_samples;
unrelated production tuning changes must not change these scheduling inputs.
"""
from copy import deepcopy
import re

import pytest
import yaml

from bespoke_codegen_common import DATA, compile_main, stagger_instructions, variant

pytestmark = pytest.mark.unit
CMS_RECIPE = "bf16_cms_streamk.yaml"
# Five distinct non-MX-file macro tiles, plus one f32 recipe from an MX logic file.
SAMPLES = ((False, (96, 256, 64)), (False, (128, 192, 64)),
           (False, (192, 256, 64)), (False, (256, 224, 64)),
           (False, (320, 192, 64)), (True, (128, 128, 64)))


def compiled_name(source):
    names = re.findall(r"^\s*\.amdhsa_kernel\s+(\S+)", source, flags=re.M)
    assert len(names) == 1, "assembled main kernel has no unique entry point"
    return names[0]


@pytest.fixture(scope="module")
def stock_cms(tmp_path_factory):
    from tensilelite.ExecutionPolicy import hasHybridAssignment, isStreamK

    out = tmp_path_factory.mktemp("stock_cms")
    solution, source, _, _ = compile_main(DATA / CMS_RECIPE, out / "compiled")
    assert (solution["MacroTile0"], solution["MacroTile1"], solution["DepthU"]) == (256, 160, 64)
    assert isStreamK(solution) and hasHybridAssignment(solution)
    assert solution["PrefetchGlobalRead"] == 2
    assert solution["UseCustomMainLoopSchedule"] == 1
    assert "_CMS_" in compiled_name(source)
    assert stagger_instructions(source), "stock CMS positive control lost stagger instructions"
    return solution, source


@pytest.mark.parametrize("use_beta", [True, False], ids=["beta_kept", "beta_free"])
def test_bespoke_cms_host_zero_stagger_builds_with_stagger(tmp_path, stock_cms, use_beta):
    path = variant(CMS_RECIPE, tmp_path / "bespoke.yaml",
                   problem={"UseBeta": use_beta}, solution={"BespokeNarrowing": True})
    solution, source, _, _ = compile_main(path, tmp_path / "compiled")
    assert solution["BespokeNarrowing"] is True
    assert solution["ProblemType"]["UseBeta"] is use_beta
    assert solution["StaggerU"] == 0
    assert solution["InternalSupportParams"]["SupportCustomStaggerU"] is False
    assert solution["UseCustomMainLoopSchedule"] == 1
    assert "_CMS_" in compiled_name(source)
    assert stagger_instructions(source), "bespoke CMS kernel lost its scheduled stagger instructions"


@pytest.fixture(scope="module")
def vendored_cms_samples():
    root = DATA / "cms_samples"
    files = sorted(root.rglob("*.yaml"))
    assert files, "vendored CMS sample fixtures are missing"
    samples = {}
    eligible = 0
    for path in files:
        with path.open() as stream:
            doc = yaml.load(stream, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
        defaults = doc.get("DefaultSolution") or {}
        for raw in doc["Solutions"]:
            merged = dict(defaults, **raw)
            support = dict(defaults.get("InternalSupportParams") or {},
                           **(raw.get("InternalSupportParams") or {}))
            cms = merged.get("UseCustomMainLoopSchedule", -1)
            # A stale _CMS_ name on an explicitly disabled recipe is not CMS.
            if not (cms == 1 or (cms == -1 and "_CMS_" in merged.get("SolutionNameMin", ""))):
                continue
            if merged.get("StaggerU") != 0 or support.get("SupportCustomStaggerU") is not False:
                continue
            eligible += 1
            tile = tuple(merged[key] for key in ("MacroTile0", "MacroTile1", "DepthU"))
            key = "MX" in path.name, tile
            if key in SAMPLES and key not in samples:
                isolated = dict(doc, Solutions=[deepcopy(raw)],
                                ProblemType=deepcopy(doc["ProblemType"]),
                                DefaultSolution=deepcopy(defaults))
                isolated["ProblemType"].update(raw.get("ProblemType") or {})
                isolated.update(IndexOrder=None, ExactLogic=None, RangeLogic=None,
                                TileSelectionIndices=None, LibraryType="Prediction")
                samples[key] = path.relative_to(root), isolated
    assert set(samples) == set(SAMPLES), "vendored CMS sample fixtures do not cover all sampled macro tiles"
    print(f"vendored CMS samples: {len(files)} files, {eligible} eligible stock solutions")
    return samples


@pytest.mark.parametrize("sample", SAMPLES,
                         ids=[f"{'mx' if mx else 'cms'}_{m}x{n}x{k}" for mx, (m, n, k) in SAMPLES])
def test_sampled_gfx950_cms_stock_solutions_compile_as_bespoke(tmp_path, vendored_cms_samples, sample):
    mx_file, tile = sample
    relative, stock = vendored_cms_samples[sample]
    path = tmp_path / "stock.yaml"
    path.write_text(yaml.safe_dump(stock, sort_keys=False))
    general, stock_source, _, _ = compile_main(path, tmp_path / "stock")
    assert general["UseCustomMainLoopSchedule"] == 1, str(relative)
    assert "_CMS_" in compiled_name(stock_source), str(relative)
    assert stagger_instructions(stock_source), str(relative)
    if mx_file:
        assert general["ProblemType"]["MXBlockA"] == general["ProblemType"]["MXBlockB"] == 0
    bespoke = deepcopy(stock)
    bespoke["ProblemType"]["UseBeta"] = False
    bespoke["Solutions"][0]["BespokeNarrowing"] = True
    path = tmp_path / "bespoke.yaml"
    path.write_text(yaml.safe_dump(bespoke, sort_keys=False))
    solution, source, _, _ = compile_main(path, tmp_path / "compiled")
    assert solution["BespokeNarrowing"] is True
    assert solution["StaggerU"] == 0
    assert solution["InternalSupportParams"]["SupportCustomStaggerU"] is False
    assert (solution["MacroTile0"], solution["MacroTile1"], solution["DepthU"]) == tile
    assert solution["UseCustomMainLoopSchedule"] == 1, str(relative)
    assert "_CMS_" in compiled_name(source), str(relative)
    assert stagger_instructions(source), str(relative)
