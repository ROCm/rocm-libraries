# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emit one-wave TDM persistent kernels, whose descriptors are not wave-separated."""

from pathlib import Path
import re

import pytest
import yaml

from config_harness import (assert_assembles, assert_cluster_barrier_balanced,
                            assert_real_gfx1250_kernels, emit_kernels_from_config)
from Tensile import __file__ as tensile_file

pytestmark = pytest.mark.unit

_K_OFFSET = "Apply StreamK K-offset to TDM tdm%sGroup0"


def _emit(tmp_path, name, **overrides):
    source = "Tests/common/streamk/gfx1250/data_parallel_cluster_multicast_boundaries.yaml"
    config = yaml.safe_load((Path(tensile_file).parent / source).read_text())
    config["BenchmarkProblems"] = config["BenchmarkProblems"][:1]
    group = config["BenchmarkProblems"][0][1]
    values = {key: value[:1] for entry in group["ForkParameters"]
              for key, value in entry.items()}
    values.update(MatrixInstruction=[[16, 16, 32, 1, 1, 1, 1, 1, 1]],
                  WorkAssignment=["StaticGrid"], PrefetchAcrossPersistent=[0],
                  PrefetchGlobalRead=[2], StaggerU=[0], GlobalSplitU=[0])
    values.update(overrides)
    group["ForkParameters"] = [{key: value} for key, value in values.items()]
    path = tmp_path / (name + ".yaml")
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(str(path), limit=1, arch="gfx1250", canonical=False)
    assert len(results) == 1
    assert_real_gfx1250_kernels(results)
    base, src, _ = results[0]
    assert_assembles(src, base)
    return base, src


def test_single_wave_streamk_offsets_each_descriptor(tmp_path):
    """A work item starting mid-tile must read K from its own start."""
    _, src = _emit(tmp_path, "streamk", TileProcessingStrategy=["StreamK"],
                   ClusterDim=[[1, 1]], StreamKFixupTreeReduction=[1])
    instructions = [line.split("//", 1)[0].strip() for line in src.splitlines()]
    for tc in ("A", "B"):
        increment = f"s[sgprStreamKLocalStart], s[sgprGlobalReadIncs{tc}+0]"
        products = []
        for opcode in ("s_mul_i32", "s_mul_hi_u32"):
            matches = [(i, re.fullmatch(rf"{opcode} (s\d+), {re.escape(increment)}", line))
                       for i, line in enumerate(instructions)]
            matches = [(i, match[1]) for i, match in matches if match]
            assert len(matches) == 1, f"{tc} needs both words of its own K offset"
            products.append(matches[0])
        assert products[0][1] != products[1][1]
        updates = []
        for word, (product, register) in enumerate(products):
            descriptor = f"s[sgprtdm{tc}Group0+{word + 2}]"
            opcode = "s_addc_u32" if word else "s_add_u32"
            update = instructions.index(f"{opcode} {descriptor}, {descriptor}, {register}", product + 1)
            # Scheduling may interleave A/B instructions, but the multiply's
            # result must reach this descriptor update without being overwritten.
            assert not any(re.match(rf"s_\w+ {register},", line)
                           for line in instructions[product + 1:update])
            updates.append(update)
        assert updates[0] < updates[1]
        # The low address addition's carry must reach the high address addition.
        assert all(line.startswith(("s_mul_i32 ", "s_mul_hi_u32 ", "s_delay_alu "))
                   for line in instructions[updates[0] + 1:updates[1]] if line)
        rebuild = (f"s_mov_b64 s[sgprtdm{tc}Group0+2:sgprtdm{tc}Group0+2+1], "
                   f"s[sgprAddress{tc}:sgprAddress{tc}+1]")
        assert instructions.index(rebuild) < products[0][0]


def test_single_wave_data_parallel_has_no_k_offset(tmp_path):
    _, src = _emit(tmp_path, "data_parallel", TileProcessingStrategy=["DataParallel"],
                   ClusterDim=[[1, 1]])
    assert _K_OFFSET % "A" not in src
    assert "StreamKLocalStart" not in src


def test_single_wave_cluster_handshake_is_paired(tmp_path):
    """One wave still needs the workgroup barriers the cluster handshake hangs on."""
    base, src = _emit(tmp_path, "cluster", TileProcessingStrategy=["DataParallel"],
                      ClusterDim=[[2, 2]])
    assert_cluster_barrier_balanced(src, base)
    loop = src[src.index("label_LoopBeginL:"):src.index("label_LoopEndL:")]
    assert "s_barrier_signal -3" in loop and "s_barrier_wait -3" in loop
    # The tail refill is waited on only after this workgroup has signalled.
    tail = src[src.index("Tail Loop"):]
    tail = tail[:tail.index("tensor_load_to_lds")]
    assert tail.index("s_barrier_signal -3") < tail.index("s_barrier_wait -3")
