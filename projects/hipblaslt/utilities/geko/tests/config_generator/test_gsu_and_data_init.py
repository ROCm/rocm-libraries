################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""The GSU_PER_MI and DATA_INIT input-config keys."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from geko.config_generator.config_sections_generator import ConfigSectionGenerator
from geko.config_generator.fork_param_generator import generate_fork_params
from geko.config_generator.fork_params import get_optimization_params, get_post_processor
from geko.config_generator.load_input_config import (
    apply_input_config_defaults,
    get_gemm_problem,
    validate_input_config,
)
from geko.config_generator.mi_designer import MIDesign

# MAB on one 96-WGP partition: MT16x1024 leaves 48 of the 96 CUs a tile, so MIDesign gives it GSU 2.
MAB_SIZE = (16, 49152, 1, 32768)


def _config(**keys) -> dict:
    cfg = {
        "TRANSA": "T", "TRANSB": "N", "DataType": "B", "DestDataType": "B", "ComputeDataType": "S",
        "ARCH": "gfx1250_96cu", "search_space": "generic", "backend": "ductile",
        "LIBRARY_TYPE": "Equality", "EPILOGUES": False, "Sizes": [list(MAB_SIZE)], **keys,
    }
    validate_input_config(cfg)
    apply_input_config_defaults(cfg)
    get_gemm_problem(cfg)
    cfg["GemmProblem"] = cfg["GemmProblems"][0]
    return cfg


def _mi_entries(tmp_path: Path, **keys):
    cfg = _config(**keys)
    mi_log = tmp_path / "MI_finder_log"
    mi_log.mkdir()
    fork_params, _, _ = generate_fork_params(
        MIDesign(str(mi_log), copy.deepcopy(cfg)),
        get_optimization_params(cfg),
        cfg,
        MAB_SIZE,
        post_processor=get_post_processor(cfg),
    )
    return fork_params, fork_params["Groups"].values[0]


@pytest.mark.cg_components
def test_default_gsu_is_one_runtime_value(
    hipblaslt_path: str | None, tensilelite_sys_path: None, tmp_path: Path
) -> None:
    fork_params, entries = _mi_entries(tmp_path)
    assert fork_params["GlobalSplitU"].values == [-1]
    assert not any("GlobalSplitU" in entry for entry in entries)


@pytest.mark.cg_components
@pytest.mark.parametrize("streamk", [True, False])
def test_gsu_per_mi_keeps_the_midesign_gsu(
    streamk: bool, hipblaslt_path: str | None, tensilelite_sys_path: None, tmp_path: Path
) -> None:
    fork_params, entries = _mi_entries(tmp_path, GSU_PER_MI=True, StreamK=streamk)
    assert "GlobalSplitU" not in fork_params
    gsu_by_mt1 = {
        entry["MatrixInstruction"].values[1] * entry["MatrixInstruction"].values[6]
        * entry["MatrixInstruction"].values[8]: entry["GlobalSplitU"].values
        for entry in entries if "GlobalSplitU" in entry
    }
    assert gsu_by_mt1.get(1024) == [2]


def test_data_init_overrides_the_emitted_global_parameters() -> None:
    params = ConfigSectionGenerator(
        _config(DATA_INIT={"DataInitTypeC": 3, "DataInitTypeD": 2, "DataInitSeed": 0})
    )._global_params_base
    assert (params["DataInitTypeC"], params["DataInitTypeD"], params["DataInitSeed"]) == (3, 2, 0)
    assert (params["DataInitTypeA"], params["DataInitTypeB"]) == (12, 13)


def test_data_init_default_keeps_the_generator_pattern() -> None:
    params = ConfigSectionGenerator(_config())._global_params_base
    assert (params["DataInitTypeC"], params["DataInitTypeD"], params["DataInitSeed"]) == (12, 12, 1)


@pytest.mark.parametrize(
    "keys",
    [
        {"DATA_INIT": {"NumWarmups": 10}},
        {"DATA_INIT": {"DataInitTypeD": "2"}},
        {"DATA_INIT": {"DataInitTypeD": True}},
        {"DATA_INIT": [2]},
        {"GSU_PER_MI": "yes"},
    ],
)
def test_bad_gsu_or_data_init_values_are_rejected(keys: dict) -> None:
    with pytest.raises(ValueError):
        _config(**keys)
