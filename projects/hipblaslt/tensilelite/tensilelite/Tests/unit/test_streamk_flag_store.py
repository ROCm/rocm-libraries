# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check the real StreamK flag emitter, including its active lanes on a GPU."""

import re
import struct
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from rocisa import rocIsa
from rocisa.code import Module, SrdUpperValue
from rocisa.container import ContinuousRegister, EXEC, sgpr, vgpr
from rocisa.enum import RegisterType
from rocisa.instruction import (
    BufferStoreB32,
    SAndSaveExecB32,
    SAndSaveExecB64,
    SLoadB32,
    SLoadB64,
    SMovB32,
    SMovB64,
    SStoreB32,
    SWaitCnt,
    SWaitXCnt,
    VMovB32,
)
from rocisa.register import RegisterPool

# Import the writer first to populate the component registry.
from tensilelite.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401
from tensilelite.Component import Component
from tensilelite.Components.StreamK import StreamKTwoTileDPFirst
from tensilelite.Toolchain.Validators import ToolchainDefaults, validateToolchain

pytestmark = pytest.mark.unit


class _FlagWriter:
    def __init__(self, isa, wave):
        compiler = validateToolchain(ToolchainDefaults.CXX_COMPILER)
        ri = rocIsa.getInstance()
        ri.init(isa, compiler)
        ri.setKernel(isa, wave)
        if not ri.getAsmCaps()["SupportedISA"]:
            pytest.skip(f"{compiler} cannot assemble {isa}")
        self.states = SimpleNamespace(
            version=isa,
            laneSGPRCount=wave // 32,
            asmCaps=ri.getAsmCaps(),
            archCaps=ri.getArchCaps(),
        )
        self.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False)
        self.sgprPool.add(0, 64, "flag test")
        self.sgprPool.checkOut(16, "arguments and trace registers")
        self.vgprPool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False)
        self.vgprPool.add(0, 8, "flag test")
        self.vgprPool.checkOut(1, "lane id")
        self.compiler = compiler

    @contextmanager
    def allocTmpSgpr(self, size, alignment=1, tag=""):
        index = self.sgprPool.checkOutAligned(size, alignment or 1, tag)
        try:
            yield ContinuousRegister(index, size)
        finally:
            self.sgprPool.checkIn(index)


@pytest.mark.parametrize(
    "isa,wave,scalar",
    [
        ((12, 0, 1), 32, False),
        ((12, 0, 1), 64, False),
        ((11, 0, 0), 32, False),
        ((11, 0, 0), 64, False),
        ((10, 1, 0), 32, True),
        ((10, 1, 0), 64, True),
        ((9, 5, 0), 64, False),
        ((12, 5, 0), 32, False),
        ((9, 4, 2), 64, True),
    ],
)
def test_flag_store_preserves_fences_and_registers(isa, wave, scalar):
    writer = _FlagWriter(isa, wave)
    items = StreamKTwoTileDPFirst().emitFlagStore(writer, sgpr(12), sgpr(13)).flatitems()
    stores = [i for i, item in enumerate(items) if isinstance(item, (BufferStoreB32, SStoreB32))]
    assert len(stores) == 1
    if scalar:
        assert isinstance(items[stores[0]], SStoreB32)
        assert not any(isinstance(item, (SAndSaveExecB32, SAndSaveExecB64)) for item in items)
    else:
        save = SAndSaveExecB32 if wave == 32 else SAndSaveExecB64
        save_index = next(i for i, item in enumerate(items) if isinstance(item, save))
        restore = items[-1]
        assert (
            str(restore).split(",")[0].strip()
            == f"s_mov_b{wave} {'exec_lo' if wave == 32 else 'exec'}"
        )
        assert save_index < stores[0] < len(items) - 1
        order = Component.StreamKMemoryOrdering.find(writer)
        fence = str(order.releaseFence(writer)).strip()
        emitted = "".join(str(item) for item in items)
        assert fence in emitted
        assert emitted.index(fence) > emitted.index(str(items[stores[0]]))
        assert emitted.index(fence) < emitted.index(str(restore))
        if (
            writer.states.archCaps["RequiresXCntForVolatileVMEM"]
            or writer.states.archCaps["EnableXnackReplay"]
        ):
            drain_index = next(i for i, item in enumerate(items) if isinstance(item, SWaitXCnt))
            assert drain_index < save_index
    # The probe reserves only s[0:15] and v0; every helper temporary must be returned.
    assert writer.sgprPool.checkOutAligned(48, 4, "balance check") == 16
    writer.sgprPool.checkIn(16)
    assert writer.vgprPool.checkOut(7, "balance check") == 1
    writer.vgprPool.checkIn(1)


def _flag_store_indices(lines):
    # Follow descriptors copied from AddressFlags, so a store bypassing the
    # shared emitter is still checked even if it has no "set/reset flag" comment.
    bases = set()
    for index, line in enumerate(lines):
        move = re.match(r"s_mov_b64 s\[(\d+):(\d+)\], (.*)", line)
        if move:
            base = int(move[1])
            bases.discard(base)
            if "sgprAddressFlags" in move[3]:
                bases.add(base)
        move = re.match(r"s_mov_b32 s(\d+),", line)
        if move:
            register = int(move[1])
            bases.difference_update((register, register - 1))
        store = re.match(r"buffer_store_\w+ \S+, \S+, s\[(\d+):(\d+)\]", line)
        if store and int(store[1]) in bases:
            yield index


@pytest.mark.parametrize("arch,isa,wave", [("gfx950", (9, 5, 0), 64), ("gfx1250", (12, 5, 0), 32)])
@pytest.mark.parametrize(
    "assignment,tree",
    [("StaticGrid", 0), ("StaticGrid", 1), ("DynamicWorkQueue", 0), ("Hybrid", 0)],
)
def test_generated_flag_stores_select_one_wave_and_lane(
    tmp_path, monkeypatch, arch, isa, wave, assignment, tree
):
    from config_harness import emit_kernels_from_config
    import tensilelite.BenchmarkProblems as benchmark_problems

    _FlagWriter(isa, wave)
    fixture = (
        Path(__file__).parent
        / "characterization/_codegen/data/test_data/_designed/gfx942/streamk.yaml"
    )
    config = yaml.safe_load(fixture.read_text())
    group = config["BenchmarkProblems"][0][1]
    forks = {key: value for fork in group["ForkParameters"] for key, value in fork.items()}
    forks.update(
        WorkAssignment=[assignment],
        StreamKFixupTreeReduction=[tree],
        StreamKAtomic=[0],
        WavefrontSize=[wave],
    )
    if wave == 32:
        # Two waves leave room for the gfx1250 hybrid kernel's persistent state.
        forks["MatrixInstruction"][0][-2:] = [1, 2]
    group["ForkParameters"] = [{key: value} for key, value in forks.items()]
    path = tmp_path / "flags.yaml"
    path.write_text(yaml.safe_dump(config))

    def serial_map(function, objects, *_args, **_kwargs):
        return [function(*args) for args in objects]

    monkeypatch.setattr(benchmark_problems, "ParallelMap2", serial_map)
    results = emit_kernels_from_config(
        path, limit=1, arch=arch, canonical=False, expected_fork_count=1
    )
    assert len(results) == 1
    _, source, error = results[0]
    assert error == 0 and "overflowed resources" not in source
    lines = [line.split("//")[0].strip() for line in source.splitlines()]
    stores = list(_flag_store_indices(lines))
    assert len(stores) >= 2  # Publication and reset must both be exercised.
    assert sum("// set flag" in line for line in source.splitlines()) == 1
    assert sum("// reset flag" in line for line in source.splitlines()) == len(stores) - 1
    for store in stores:
        # Check the caller's wave selection as well as the shared lane election.
        guard = max(
            index
            for index in range(store)
            if lines[index].startswith("s_cbranch_scc0 label_SK_SkipFlag")
        )
        target = lines[guard].split()[1] + ":"
        end = lines.index(target, guard + 1)
        assert guard < store < end
        previous = [line for line in lines[:guard] if line and not line.startswith("s_wait")]
        compare = re.fullmatch(r"s_cmp_eq_u32 (s\d+), 0", previous[-1])
        assert compare is not None
        assert re.fullmatch(
            r"v_readfirstlane_b32 " + compare[1] + r", v\[vgprSerial(?:-256)?\]", previous[-2]
        )
        block = lines[guard + 1 : end]
        save = next(line for line in block if line.startswith(f"s_and_saveexec_b{wave}"))
        registers = save.split(" ", 1)[1].split(", ")
        assert registers[0] == registers[1]
        saved = registers[0]
        low = saved if wave == 32 else "s" + re.fullmatch(r"s\[(\d+):(\d+)\]", saved)[1]
        assert f"s_sub_u32 {low}, 0, exec_lo" in block[: block.index(save)]
        if wave == 64:
            high = "s" + re.fullmatch(r"s\[(\d+):(\d+)\]", saved)[2]
            assert f"s_subb_u32 {high}, 0, exec_hi" in block[: block.index(save)]
        restore = f"s_mov_b{wave} {'exec_lo' if wave == 32 else 'exec'}, {saved}"
        assert block.index(save) < block.index(lines[store]) < block.index(restore)


def _trace_kernel(writer, arch, wave):
    body = Module("flag store trace")
    body.add(SLoadB64(dst=sgpr(4, 2), base=sgpr(0, 2), soffset=0))
    body.add(SLoadB64(dst=sgpr(6, 2), base=sgpr(0, 2), soffset=8))
    body.add(SLoadB32(dst=sgpr(12), base=sgpr(0, 2), soffset=16))
    body.add(SWaitCnt(kmcnt=0))
    body.add(SMovB32(dst=sgpr(13), src=0))
    mov_exec = SMovB32 if wave == 32 else SMovB64
    body.add(mov_exec(dst=EXEC(), src=sgpr(6, wave // 32)))
    flag = StreamKTwoTileDPFirst().emitFlagStore(writer, sgpr(12), sgpr(13))
    for item in flag.flatitems():
        if isinstance(item, BufferStoreB32):
            # Capture EXEC at the actual store, without replacing the store or its fences.
            body.add(SMovB64(dst=sgpr(8, 2), src="exec"))
        body.add(item)
    body.add(SMovB64(dst=sgpr(10, 2), src="exec"))
    body.add(mov_exec(dst=EXEC(), src=1))
    order = Component.StreamKMemoryOrdering.find(writer)
    for offset, register in zip((4, 8, 12, 16), (8, 9, 10, 11)):
        body.add(VMovB32(dst=vgpr(3), src=sgpr(register)))
        body.add(SMovB32(dst=sgpr(13), src=offset))
        body.add(
            BufferStoreB32(
                src=vgpr(3),
                vaddr=vgpr(0),
                saddr=sgpr(16, 4),
                soffset=sgpr(13),
                mubuf=order.flagBufferMubuf(),
            )
        )
    body.add(order.releaseFence(writer))
    wave_directive = (
        f"  .amdhsa_wavefront_size32 {int(wave == 32)}\n"
        if writer.states.archCaps["HasWavefrontSize32Directive"]
        else ""
    )
    accum_directive = "  .amdhsa_accum_offset 8\n" if arch == "gfx950" else ""
    return f""".amdgcn_target "amdgcn-amd-amdhsa--{arch}"
.set sgprAddressFlags, 4
.set BufferOOB, 0xffffffff
.set Srd127_96, {SrdUpperValue(writer.states.version).getValue()}
.text
.globl test_kernel
.protected test_kernel
.p2align 8
.type test_kernel,@function
test_kernel:
{body}
  s_endpgm
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel test_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 64
  .amdhsa_system_vgpr_workitem_id 0
{wave_directive}{accum_directive}.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 1]
amdhsa.kernels:
  - .name: test_kernel
    .symbol: test_kernel.kd
    .kernarg_segment_size: 24
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: {wave}
    .sgpr_count: 64
    .vgpr_count: 8
    .max_flat_workgroup_size: {wave}
    .args:
      - {{.name: output, .size: 8, .offset: 0, .value_kind: global_buffer}}
      - {{.name: mask, .size: 8, .offset: 8, .value_kind: by_value}}
      - {{.name: value, .size: 4, .offset: 16, .value_kind: by_value}}
...
.end_amdgpu_metadata
"""


@pytest.mark.gpu
@pytest.mark.parametrize(
    "arch,isa,wave",
    [
        ("gfx1201", (12, 0, 1), 32),
        ("gfx1201", (12, 0, 1), 64),
        ("gfx950", (9, 5, 0), 64),
        ("gfx1250", (12, 5, 0), 32),
    ],
)
def test_flag_store_has_one_active_writer_and_restores_exec(tmp_path, arch, isa, wave):
    from gpu_test_helpers import GFX_TARGET, hip, run_on_gpu

    if hip is None or GFX_TARGET != arch:
        pytest.skip(f"requires HIP and {arch}, found {GFX_TARGET}")
    writer = _FlagWriter(isa, wave)
    source = tmp_path / "flag.s"
    code = tmp_path / "flag.co"
    source.write_text(_trace_kernel(writer, arch, wave))
    command = [
        writer.compiler,
        "-x",
        "assembler",
        "--target=amdgcn-amd-amdhsa",
        f"-mcpu={arch}",
        "-mcode-object-version=5",
    ]
    if writer.states.archCaps["HasWavefrontSize32Directive"]:
        command.append("-mwavefrontsize64" if wave == 64 else "-mno-wavefrontsize64")
    subprocess.run(command + [str(source), "-o", str(code)], check=True, capture_output=True)
    full = (1 << wave) - 1
    masks = [0, full, full ^ 1, full & 0xAAAAAAAAAAAAAAAA, full & 0x5555555555555555]
    masks += [1 << bit for bit in range(wave)]
    if wave == 64:
        masks += [0xFFFFFFFF00000000, 0x8000000100000000, 0x8000000000000001]
    for mask in masks:
        for value in (0, 1):
            raw = run_on_gpu(
                str(code), 20, scalars=(mask & 0xFFFFFFFF, mask >> 32, value), num_threads=wave
            )
            flag, store_lo, store_hi, after_lo, after_hi = struct.unpack("<5I", raw)
            store_mask = store_lo | (store_hi << 32)
            assert store_mask.bit_count() == int(mask != 0), (hex(mask), hex(store_mask))
            assert store_mask & mask == store_mask
            if mask:
                assert store_mask.bit_length() - 1 == min(
                    bit for bit in range(wave) if mask & (1 << bit)
                )
            assert after_lo | (after_hi << 32) == mask
            assert flag == (value if mask else 0xFFFFFFFF)
