# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Subtile beta epilogue must mask the lanes past N around every non-edge C load.

Every C load (a buffer load through sgprSrdC) is classified by the nearest
preceding write-batch header comment, which carries the writer's edge flag. Non-edge
loads must follow an exec mask and be restored right after; edge loads must not.

Negatives, each produced by a dedicated emission:
  - edge-path loads: present in every positive kernel, never masked.
  - non-subtile kernel: UseSubtileImpl False on non-MX fp8.
  - MultipleBufferSingleKernel store path: loads C without a mask.
  - MultipleBuffer store path and fused full-tile store: emitted beta-free, no C loads.
  - UseBeta False: no mask at all.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

GSU_SEED = (Path(__file__).parent
            / "characterization/_codegen/data/test_data/_designed/gfx950/mx_bias_act_gsu.yaml")
PARTIAL_N_SEED = (Path(__file__).parents[1]
                  / "common/gemm/gfx950/subtile_mx_beta_partial_n.yaml")

_GSU_FORKS = {
    "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
    "GlobalSplitU": 1,
    "TileProcessingStrategy": "None",
}
_PLSIN_FORKS = {"PrefetchGlobalRead": 2, "PrefetchLocalRead": 1}
_NO_MX = {"MXBlockA": 0, "MXBlockB": 0}

_C_LOAD = re.compile(r"^buffer_load_\w+ .*s\[sgprSrdC:sgprSrdC\+3\]")
_BATCH_HEADER = re.compile(r"/\* edge=(\d)")
_MULTI_BUFFER_BEGIN = re.compile(r"^label_GW_B\d+_(MB|MBSK):")
_MULTI_BUFFER_END = re.compile(r"^label_GW_End(_\d+)?:")
_FUSED_BEGIN = "PostLoopStoreInNll 4d-3a"
_FUSED_END = "PostLoopStoreInNll: FUSED done"
_MASK_COMMENT = "mask C load lanes"
_VECTOR_PREFIXES = ("v_", "ds_", "buffer_", "global_", "flat_")


def _emit(tmp_path, seed, problem_index=0, forks=None, problem=None, added_forks=None):
    config = yaml.safe_load(seed.read_text())
    config["BenchmarkProblems"] = [config["BenchmarkProblems"][problem_index]]
    problem_type, size_group = config["BenchmarkProblems"][0]
    fork_entries = size_group["ForkParameters"]
    for key, value in (problem or {}).items():
        assert key in problem_type, f"{key} missing from the {seed.name} problem type"
        problem_type[key] = value
    for key, value in (forks or {}).items():
        matches = [entry for entry in fork_entries if key in entry]
        assert len(matches) == 1, f"{key} not found in the {seed.name} forks"
        matches[0][key] = [value]
    for key, value in (added_forks or {}).items():
        assert not any(key in entry for entry in fork_entries), f"{key} already forked"
        fork_entries.append({key: [value]})
    config["GlobalParameters"]["CpuThreads"] = 1
    path = tmp_path / "c_load.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(path, limit=1, arch="gfx950", expected_fork_count=1)
    assert len(results) == 1, f"expected one kernel, got {len(results)}"
    name, assembly, error = results[0]
    assert error == 0, f"emission of {name} failed"
    return assembly.splitlines()


@dataclass(frozen=True)
class CLoad:
    index: int
    edge: bool
    multi_buffer: bool
    fused: bool
    masked: bool


def _c_loads(lines):
    loads = []
    edge = None
    multi_buffer = False
    fused = False
    last_exec = None
    for index, line in enumerate(lines):
        header = _BATCH_HEADER.search(line)
        if header:
            edge = header.group(1) == "1"
        if _MULTI_BUFFER_BEGIN.match(line):
            multi_buffer = True
        elif _MULTI_BUFFER_END.match(line):
            multi_buffer = False
        if _FUSED_BEGIN in line:
            fused = True
        elif _FUSED_END in line:
            fused = False
        if _C_LOAD.match(line):
            assert edge is not None, f"C load at line {index} precedes any write-batch header"
            masked = last_exec is not None and _MASK_COMMENT in last_exec
            loads.append(CLoad(index, edge, multi_buffer, fused, masked))
        elif "exec" in line:
            last_exec = line
    return loads


def _mask_sites(lines):
    return [index for index, line in enumerate(lines) if _MASK_COMMENT in line]


def _check_wrapped(lines, site):
    load = next((index for index in range(site + 1, len(lines)) if _C_LOAD.match(lines[index])),
                None)
    assert load is not None, f"no C load follows the mask at line {site}"
    # Only scalar row-advance code may sit between the mask and the load.
    for index in range(site + 1, load):
        assert not lines[index].startswith(_VECTOR_PREFIXES) and "exec" not in lines[index], \
            f"unexpected instruction between mask and load at line {index}"
    assert lines[load + 1].startswith("s_mov_b64 exec, -1"), \
        f"exec not restored after the C load at line {load}"


def _partial_n(problem_index, **kwargs):
    return dict(seed=PARTIAL_N_SEED, problem_index=problem_index, **kwargs)


@pytest.mark.parametrize("emit_args", [
    pytest.param(dict(seed=GSU_SEED, forks=_GSU_FORKS), id="fp8_fp32_dest"),
    pytest.param(dict(seed=GSU_SEED, forks=_GSU_FORKS, added_forks={"GroupLoadStore": True}),
                 id="fp8_fp32_dest_group_load_store"),
    pytest.param(dict(seed=GSU_SEED, forks={**_GSU_FORKS, "TileProcessingStrategy": "StreamK"}),
                 id="fp8_fp32_dest_streamk"),
    pytest.param(dict(seed=GSU_SEED,
                      forks={**_GSU_FORKS, "GlobalSplitUAlgorithm": "MultipleBufferSingleKernel"}),
                 id="fp8_fp32_dest_mbsk"),
    pytest.param(_partial_n(0), id="fp8_bf16_dest_streamk"),
    pytest.param(_partial_n(0, problem=_NO_MX), id="fp8_no_mx_streamk"),
    pytest.param(_partial_n(1), id="fp4_streamk"),
    pytest.param(_partial_n(1, forks=_PLSIN_FORKS),
                 id="fp4_streamk_plsin"),
])
def test_non_edge_c_loads_are_masked(emit_args, tmp_path):
    lines = _emit(tmp_path, **emit_args)
    loads = _c_loads(lines)
    non_edge = [load for load in loads if not load.edge and not load.multi_buffer]
    assert non_edge, "no non-edge C load found"
    for load in non_edge:
        assert load.masked, f"non-edge C load at line {load.index} is not masked"
    # Edge loads and multi-buffer loads must stay unmasked.
    edge = [load for load in loads if load.edge]
    assert edge, "no edge C load found"
    assert not any(load.masked for load in loads if load.edge or load.multi_buffer)
    sites = _mask_sites(lines)
    assert len(sites) == len(non_edge), \
        f"{len(sites)} mask sites for {len(non_edge)} non-edge C loads"
    for site in sites:
        _check_wrapped(lines, site)


def test_non_subtile_c_loads_are_not_masked(tmp_path):
    lines = _emit(tmp_path, **_partial_n(0, problem=_NO_MX, forks={"UseSubtileImpl": False}))
    assert any(not load.edge for load in _c_loads(lines)), "no non-edge C load found"
    assert not _mask_sites(lines)


@pytest.mark.parametrize("algorithm, loads_c", [
    ("MultipleBuffer", False),
    ("MultipleBufferSingleKernel", True),
])
def test_multi_buffer_c_loads_are_not_masked(algorithm, loads_c, tmp_path):
    lines = _emit(tmp_path, GSU_SEED, forks={**_GSU_FORKS, "GlobalSplitUAlgorithm": algorithm})
    assert any(_MULTI_BUFFER_BEGIN.match(line) for line in lines), \
        "no multi-buffer store section found"
    multi_buffer_loads = [load for load in _c_loads(lines) if load.multi_buffer]
    # The MultipleBuffer path is beta-free: the reduction kernel applies beta.
    assert bool(multi_buffer_loads) == loads_c, \
        f"{len(multi_buffer_loads)} multi-buffer C loads for {algorithm}"
    assert not any(load.masked for load in multi_buffer_loads)


def test_fused_full_tile_store_has_no_c_loads(tmp_path):
    lines = _emit(tmp_path, **_partial_n(1, forks=_PLSIN_FORKS))
    # The fused store is emitted beta-free, so its no-guards early return is defensive.
    assert any(_FUSED_BEGIN in line for line in lines), "no fused full-tile store found"
    assert not any(load.fused for load in _c_loads(lines))


def test_no_exec_mask_without_beta(tmp_path):
    lines = _emit(tmp_path, GSU_SEED, forks=_GSU_FORKS, problem={"UseBeta": False})
    assert not _mask_sites(lines)
