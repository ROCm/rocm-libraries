# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Store-trim matched-library fixtures shared by the retained codegen suite and an
external store-trim GEMM acceptance test.

The synthetic subtile recipe is a test-only clone of its stock parent.
Normal derivation applies; no forced acceptance is used.
This module is self-contained (no external tools, workstation paths or fixed
core lists); the matched-library and launch-harness preparation lives with that
external test.
"""
import copy
import hashlib
import json
from pathlib import Path

import yaml

LOGIC_DIR = Path("projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full")
# Fixed parent recipes; no strategy/tuning control differs across a matched pair.
PARENTS = {
    "classic": ("gfx950/gfx950/Equality/gfx950_Cijk_Alik_Bljk_BSS_BH_BiasS_HAS_SAV_UserArgs.yaml", 0, 32, 64, 128, False),
    "synthetic_subtile": ("gfx950/gfx950/Equality/gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml", 11, 128, 64, 128, True),
    "stock_subtile_m": ("gfx950/gfx950/Origami/gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml", 259, 128, 448, 64, True),
    "stock_subtile_n": ("gfx950/gfx950/Origami/gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml", 277, 208, 256, 64, True),
}
CASES = [(parent, mode) for parent in ("classic", "synthetic_subtile") for mode in ("m", "n", "both")]
CASES += [("stock_subtile_m", "m"), ("stock_subtile_n", "n")]

# Review-boundary recipes: only these named mechanism knobs differ from each
# recorded stock parent; general/bespoke differ only in AF0EM/AF1EM.
PARENTS.update({
    "classic_remap_off": ("gfx950/gfx950/Equality/gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml", 610, 16, 16, 128, False),
    "classic_remap_on": ("gfx950/gfx950/Equality/gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml", 610, 16, 16, 128, False),
    "subtile_remap_off": ("gfx950/gfx950/Equality/gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml", 610, 16, 16, 128, True),
    "subtile_remap_on": ("gfx950/gfx950/Equality/gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml", 610, 16, 16, 128, True),
    "classic_remainder": ("gfx950/gfx950/Equality/gfx950_Cijk_Alik_Bljk_S_B_BiasS_HAS_SAV_UserArgs.yaml", 0, 64, 64, 64, False),
    "subtile_remainder": PARENTS["stock_subtile_m"],
})
STORE_KNOBS = {
    "classic_remap_off": {"StoreRemapVectorWidth": 0},
    "classic_remap_on": {"StoreRemapVectorWidth": 2},
    "subtile_remap_off": {"StoreRemapVectorWidth": 0, "1LDSBuffer": 0},
    "subtile_remap_on": {"StoreRemapVectorWidth": 2},
    "classic_remainder": {"NumElementsPerBatchStore": 3},
    "subtile_remainder": {"NumElementsPerBatchStore": 16}
}
STORE_BOUNDARY_CASES = [(parent, mode) for parent in
                        ("classic_remap_off", "classic_remap_on", "subtile_remap_off", "subtile_remap_on")
                        for mode in ("m", "n", "both")]
STORE_BOUNDARY_CASES += [("classic_remainder", "m"), ("classic_remainder", "both"),
                         ("subtile_remainder", "m")]
# Subtile+remap is a pre-existing defect that stock logic cannot reach.
# Keep its codegen observations, but never build it as a numerical control.
NUMERICAL_STORE_BOUNDARY_CASES = [(parent, mode) for parent, mode in STORE_BOUNDARY_CASES
                                  if parent != "subtile_remap_on"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_request(out, source_root, parent, mode):
    rel, index, mt0, mt1, du, subtile = PARENTS[parent]
    source = source_root / LOGIC_DIR / rel
    stock = yaml.safe_load(source.read_text())
    solution = copy.deepcopy(stock["Solutions"][index])
    assert (solution["MacroTile0"], solution["MacroTile1"], solution["DepthU"]) == (mt0, mt1, du)
    synthetic_subtile = parent in ("synthetic_subtile", "subtile_remap_off", "subtile_remap_on")
    recipe_changes = dict(STORE_KNOBS.get(parent, {}))
    if synthetic_subtile:
        recipe_changes["UseSubtileImpl"] = True
    else:
        assert bool(solution.get("UseSubtileImpl", stock.get("DefaultSolution", {}).get("UseSubtileImpl", False))) == subtile
    solution.update(recipe_changes)
    m = 3 * mt0 if mode in ("m", "both") else 2 * mt0 + 3
    n = 3 * mt1 if mode in ("n", "both") else 2 * mt1 + 3
    pid = "store_trim_" + parent + "_" + mode
    call = {"transA": "T", "transB": "N", "M": m, "N": n, "K": 257, "batch": 2,
            "lda": 257, "ldb": 257, "ldc": m, "ldd": m,
            "a_type": "f32" if parent == "classic_remainder" else "bf16",
            "b_type": "f32" if parent == "classic_remainder" else "bf16",
            "c_type": "f32" if parent in ("classic", "classic_remainder") else "bf16",
            "d_type": "f32" if parent in ("classic", "classic_remainder") else "bf16",
            "compute_type": "f32", "alpha": 1, "beta": 1, "pointer_mode": "host",
            "bias": "none", "bias_type": "", "activation": "none", "act_args": [], "sav": 0}
    base = {key: copy.deepcopy(stock[key]) for key in ("MinimumRequiredVersion", "ScheduleName",
            "ArchitectureName", "CUCount", "DeviceNames", "ProblemType", "DefaultSolution") if key in stock}
    base.update(Solutions=[solution], IndexOrder=[2, 3, 0, 1], LibraryType="Equality",
                ExactLogic=[[[m, n, 2, 257], [0, 0.0]]])
    facts = {"schema": 1, "problem_id": pid, "arch": "gfx950", "call": call,
             "stock": {"logic_file": str(source), "sha256": sha(source), "solution_index": index},
             "solution": {"MacroTile0": mt0, "MacroTile1": mt1, "DepthU": du, "UseSubtileImpl": subtile,
                          "StoreRemapVectorWidth": solution.get("StoreRemapVectorWidth", 0),
                          "NumElementsPerBatchStore": solution.get("NumElementsPerBatchStore", 0)},
             "fixture": {"class": "test-only synthetic-mechanism" if recipe_changes else "test-only synthetic-boundary",
                         "parent_recipe": rel, "parent_index": index, "recipe_changes": recipe_changes},
             "fingerprint": {"fixture_author": str(Path(__file__)), "sha256": sha(__file__)}}
    for variant in ("general", "bespoke"):
        target = out / variant
        target.mkdir(parents=True)
        data = copy.deepcopy(base)
        if variant == "bespoke":
            if mode in ("m", "both"):
                data["Solutions"][0]["AssertFree0ElementMultiple"] = mt0
            if mode in ("n", "both"):
                data["Solutions"][0]["AssertFree1ElementMultiple"] = mt1
        (target / (pid + "_" + variant + ".yaml")).write_text(yaml.safe_dump(data, sort_keys=False))
    (out / "facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    return out
