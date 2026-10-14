# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU coverage for the MLA decode kernels (latent + split-KV combine).

No GPU: spec validation, per-arch admission, launch geometry, and a full lower
of every emitted ``KernelDef``. The lowering checks pin the per-arch operand
choices -- gfx942 must not pick up gfx950's transpose read (it would lower and
then fault on the device), and gfx950 must actually use the wide-K atom and the
transpose read it claims to. A comgr-compiled resource guard checks the gate's
static half: no scratch, and the VGPR/LDS budget of two workgroups per CU.

Numeric correctness is the on-device lane, ``test_mla_decode_numeric.py``.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from dispatch.mla.decode import choose_num_seg
from kernels.mla.mla_decode import (
    MlaDecodeSpec,
    build_mla_decode,
    mla_decode_block,
    mla_decode_combine_spec,
    mla_decode_grid,
    mla_decode_signature,
    mla_decode_workspace_shapes,
    supports_mla_decode,
)
from kernels.mla.mla_decode_combine import (
    MlaDecodeCombineSpec,
    build_mla_decode_combine,
    mla_decode_combine_grid,
    mla_decode_combine_signature,
    supports_mla_decode_combine,
)
from kernels.mla.mla_decode_gfx950 import MlaDecodeGfx950Spec, build_mla_decode_gfx950
from kernels.mla.mla_prefill_gfx942 import MlaPrefillSpec, mla_prefill_fwd_signature
from rocke.core.lower_llvm import lower_kernel_to_llvm

SPECS = {"gfx942": MlaDecodeSpec, "gfx950": MlaDecodeGfx950Spec}


def _pool_bytes(ir: str, name: str) -> int:
    m = re.search(rf"@smem_pool\.{re.escape(name)} = .*?\[(\d+) x i8\]", ir)
    assert m, f"no LDS pool for {name}"
    return int(m.group(1))


class TestSpec:
    def test_defaults(self):
        spec = MlaDecodeSpec(num_heads=128)
        assert (spec.block_h, spec.qa_cols, spec.threads) == (64, 576, 256)
        assert not spec.split_kv
        assert spec.kernel_name().startswith("mla_decode_gfx942_h128")

    @pytest.mark.parametrize(
        "kw, match",
        [
            ({"dtype": "fp16"}, "bf16 only"),
            ({"num_heads": 0}, "num_heads must be positive"),
            ({"num_seg": 0}, "num_seg must be positive"),
            ({"num_warps": 3}, "num_warps must be one of"),
            ({"page_block_size": 32}, "page_block_size must equal block_k"),
        ],
    )
    def test_impossible_specs_raise(self, kw, match):
        kw.setdefault("num_heads", 128)
        with pytest.raises(ValueError, match=match):
            MlaDecodeSpec(**kw)

    def test_name_records_split_but_not_the_count(self):
        # num_seg is a launch parameter; only split vs single pass is IR.
        a = MlaDecodeSpec(num_heads=128, num_seg=2)
        b = MlaDecodeSpec(num_heads=128, num_seg=7)
        assert a.kernel_name() == b.kernel_name()
        assert a.kernel_name().endswith("_split_bf16")
        assert "_split" not in MlaDecodeSpec(num_heads=128).kernel_name()

    def test_arch_is_in_the_name(self):
        # Same fields, different IR: the names must differ, or a name-keyed
        # cache would hand one arch's code object to the other.
        a = MlaDecodeSpec(num_heads=64)
        b = MlaDecodeGfx950Spec(num_heads=64)
        assert a.kernel_name() != b.kernel_name()
        assert a != b

    def test_identity_is_the_binary_not_the_launch(self):
        # num_seg is a launch argument: specs that differ only in it share a
        # binary. The arch is a class attribute asdict cannot see, so it has to
        # be in the identity explicitly.
        a = MlaDecodeSpec(num_heads=128, num_seg=4)
        assert a.identity() == MlaDecodeSpec(num_heads=128, num_seg=16).identity()
        assert a.identity() != MlaDecodeSpec(num_heads=128).identity()
        assert a.identity() != MlaDecodeGfx950Spec(num_heads=128, num_seg=4).identity()

    def test_gfx950_pads_for_16_byte_reads(self):
        assert MlaDecodeSpec(num_heads=64).lds_pad == 4
        assert MlaDecodeGfx950Spec(num_heads=64).lds_pad == 8


class TestGate:
    @pytest.mark.parametrize("arch", sorted(SPECS))
    @pytest.mark.parametrize("heads", [64, 128])
    def test_bring_up_geometry_admitted(self, arch, heads):
        assert supports_mla_decode(SPECS[arch](num_heads=heads), arch=arch) == (
            True,
            "",
        )

    @pytest.mark.parametrize(
        "arch, other", [("gfx942", "gfx950"), ("gfx950", "gfx942")]
    )
    def test_spec_class_is_bound_to_its_arch(self, arch, other):
        ok, why = supports_mla_decode(SPECS[arch](num_heads=128), arch=other)
        assert not ok and arch in why

    def test_head_count_must_fill_whole_workgroups(self):
        ok, why = supports_mla_decode(MlaDecodeSpec(num_heads=96), arch="gfx942")
        assert not ok and "BLOCK_H" in why

    def test_heads_per_wave_is_the_atom_n(self):
        ok, why = supports_mla_decode(
            MlaDecodeSpec(num_heads=128, heads_per_wave=32, num_warps=2), arch="gfx942"
        )
        assert not ok and "heads_per_wave" in why

    def test_wide_atom_needs_d_rope_in_whole_k_steps(self):
        ok, why = supports_mla_decode(
            MlaDecodeGfx950Spec(num_heads=64, d_rope=48), arch="gfx950"
        )
        assert not ok and "d_rope" in why

    def test_build_refuses_a_mismatched_arch(self):
        with pytest.raises(NotImplementedError):
            build_mla_decode(MlaDecodeSpec(num_heads=128), arch="gfx950")

    def test_build_defaults_to_the_spec_class_arch(self):
        spec = MlaDecodeGfx950Spec(num_heads=64)
        assert build_mla_decode(spec).name == build_mla_decode_gfx950(spec).name


class TestLaunchGeometry:
    def test_grid_is_head_groups_by_sequences_by_segments(self):
        spec = MlaDecodeSpec(num_heads=128, num_seg=5)
        assert mla_decode_grid(spec, num_seqs=7) == (2, 7, 5)
        assert mla_decode_block(spec) == (256, 1, 1)
        with pytest.raises(ValueError):
            mla_decode_grid(spec, num_seqs=0)

    def test_single_pass_pack_is_the_prefill_absorbed_pack(self):
        # A caller holds one set of tensors whichever kernel dispatch picks.
        decode = [a["name"] for a in mla_decode_signature(MlaDecodeSpec(num_heads=128))]
        prefill = [
            a["name"]
            for a in mla_prefill_fwd_signature(
                MlaPrefillSpec(
                    num_heads=128, heads_per_wg=4, block_q=16, absorbed_query=True
                )
            )
        ]
        assert decode == prefill
        assert len(decode) == 12

    def test_split_pack_swaps_outputs_for_the_workspace(self):
        spec = MlaDecodeSpec(num_heads=128, num_seg=4)
        names = [a["name"] for a in mla_decode_signature(spec)]
        assert names[:2] == ["part_acc_ptr", "part_ml_ptr"]
        assert names[-1] == "num_seg" and len(names) == 13
        assert mla_decode_workspace_shapes(spec, total_q=3) == {
            "part_acc_ptr": (3, 4, 128, 512),
            "part_ml_ptr": (3, 4, 128, 2),
        }
        assert (
            mla_decode_workspace_shapes(MlaDecodeSpec(num_heads=128), total_q=3) == {}
        )


class TestChooseNumSeg:
    def test_fills_the_device(self):
        # 1 sequence x 2 head groups, 304 CUs x 2 workgroups -> 304 wanted,
        # capped by 8192 / 16 / 32 = 16 tiles-per-segment floor.
        kw = dict(block_k=16, num_cus=304, wgs_per_cu=2)
        assert choose_num_seg(num_seqs=1, head_groups=2, max_seqlen_k=8192, **kw) == 16

    def test_large_batches_do_not_split(self):
        kw = dict(block_k=16, num_cus=304, wgs_per_cu=2)
        assert choose_num_seg(num_seqs=512, head_groups=2, max_seqlen_k=8192, **kw) == 1

    def test_cap_boundary(self):
        kw = dict(num_seqs=1, head_groups=1, block_k=16, num_cus=304, wgs_per_cu=2)
        assert choose_num_seg(max_seqlen_k=1008, **kw) == 1  # 63 tiles
        assert choose_num_seg(max_seqlen_k=1009, **kw) == 2  # 64 tiles

    def test_unknown_cu_count_still_splits(self):
        assert (
            choose_num_seg(
                num_seqs=1,
                head_groups=1,
                max_seqlen_k=8192,
                block_k=16,
                num_cus=0,
                wgs_per_cu=2,
            )
            == 2
        )

    def test_short_keys_never_split_below_the_floor(self):
        kw = dict(block_k=16, num_cus=304, wgs_per_cu=2)
        assert choose_num_seg(num_seqs=1, head_groups=1, max_seqlen_k=511, **kw) == 1
        assert choose_num_seg(num_seqs=1, head_groups=1, max_seqlen_k=0, **kw) == 1


@pytest.fixture(scope="module", params=sorted(SPECS))
def lowered(request):
    arch = request.param
    out = {}
    for split in (False, True):
        spec = SPECS[arch](num_heads=128, num_seg=4 if split else 1)
        kernel = build_mla_decode(spec, arch=arch)
        out[split] = (kernel, lower_kernel_to_llvm(kernel, arch=arch))
    return arch, out


class TestLowering:
    def test_lowers_with_the_predicted_pool(self, lowered):
        from kernels.mla.mla_decode import _lds_bytes

        arch, kernels = lowered
        for split, (kernel, ir) in kernels.items():
            assert kernel.name in ir
            spec = SPECS[arch](num_heads=128, num_seg=4 if split else 1)
            # Under 32 KiB: two workgroups per CU on either arch.
            assert _pool_bytes(ir, kernel.name) == _lds_bytes(spec) <= 32768
            assert '"amdgpu-flat-work-group-size"="64,256"' in ir
            assert "scratch" not in ir

    def test_kv_pages_are_addressed_from_a_64_bit_base(self, lowered):
        # Each page gets its own descriptor off a 64-bit pointer; a single
        # cache-wide descriptor (0x7FFF0000 bytes) with an i32 byte offset
        # would wrap past 2 GiB of cache.
        _arch, kernels = lowered
        for _split, (_kernel, ir) in kernels.items():
            assert str(0x7FFF0000) not in ir
            assert "getelementptr inbounds i8, ptr addrspace(1)" in ir

    def test_per_arch_operands(self, lowered):
        arch, kernels = lowered
        for _split, (_kernel, ir) in kernels.items():
            if arch == "gfx942":
                # CDNA3 has neither the transpose read nor a wide-K bf16 atom.
                assert "ds.read.tr" not in ir
                assert "16x16x32" not in ir
                assert "mfma.f32.16x16x16bf16" in ir
            else:
                assert "ds.read.tr16.b64" in ir
                assert "16x16x32" in ir

    def test_split_count_is_not_ir(self):
        a = build_mla_decode(MlaDecodeSpec(num_heads=64, num_seg=2), arch="gfx942")
        b = build_mla_decode(MlaDecodeSpec(num_heads=64, num_seg=9), arch="gfx942")
        assert lower_kernel_to_llvm(a, arch="gfx942") == lower_kernel_to_llvm(
            b, arch="gfx942"
        )


class TestCombine:
    def test_spec_follows_the_decode_spec(self):
        spec = mla_decode_combine_spec(MlaDecodeGfx950Spec(num_heads=64, num_seg=3))
        assert spec == MlaDecodeCombineSpec(num_heads=64)
        assert spec.cols_per_lane == 8

    def test_geometry_and_pack(self):
        spec = MlaDecodeCombineSpec(num_heads=128)
        assert mla_decode_combine_grid(spec, total_q=5) == (32, 5, 1)
        names = [a["name"] for a in mla_decode_combine_signature(spec)]
        assert names == ["out_ptr", "lse_ptr", "part_acc_ptr", "part_ml_ptr", "num_seg"]

    def test_gate(self):
        assert supports_mla_decode_combine(
            MlaDecodeCombineSpec(num_heads=128), arch="gfx950"
        ) == (True, "")
        ok, why = supports_mla_decode_combine(
            MlaDecodeCombineSpec(num_heads=128), arch="gfx1201"
        )
        assert not ok and "wave64" in why
        ok, why = supports_mla_decode_combine(
            MlaDecodeCombineSpec(num_heads=128, r_kv=320), arch="gfx942"
        )
        assert not ok and "r_kv" in why

    @pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
    def test_lowers_arch_neutral(self, arch):
        kernel = build_mla_decode_combine(
            MlaDecodeCombineSpec(num_heads=128), arch=arch
        )
        ir = lower_kernel_to_llvm(kernel, arch=arch)
        assert "mfma" not in ir and "smem_pool" not in ir


class TestResources:
    """Compiled through comgr: the static half of the P3/P6 gate.

    Zero scratch, and at most 256 VGPRs and 32 KiB of LDS, which is what two
    workgroups of four waves need per CU on both arches. Skipped where comgr or
    llvm-objdump is missing.
    """

    @pytest.mark.parametrize("arch", sorted(SPECS))
    @pytest.mark.parametrize("num_seg", [1, 4])
    def test_no_spills_two_workgroups_per_cu(self, arch, num_seg):
        try:
            import torch  # noqa: F401  -- binds the torch-bundled comgr
        except ImportError:
            pass
        try:
            from rocke.analysis.isa import analyze_hsaco
            from rocke.helpers.compile import compile_kernel
        except ImportError as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"comgr / resource tools unavailable: {exc}")
        kernel = build_mla_decode(
            SPECS[arch](num_heads=128, num_seg=num_seg), arch=arch
        )
        try:
            art = compile_kernel(kernel, arch=arch, capture_ir_text=False)
        except ImportError as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"comgr toolchain unavailable: {exc}")
        except RuntimeError as exc:
            # A forced ROCKE_LLVM_FLAVOR the loaded comgr cannot take is an
            # environment mismatch; any other compile failure is real.
            if "vintage mismatch" not in str(exc):
                raise
            pytest.skip(f"comgr cannot take this IR flavor: {exc}")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mla_decode.hsaco"
            path.write_bytes(bytes(art.hsaco))
            try:
                res = analyze_hsaco(path).resources
            except (FileNotFoundError, RuntimeError) as exc:  # pragma: no cover
                pytest.skip(f"HSACO introspection unavailable: {exc}")
        assert res.scratch_bytes == 0
        assert res.vgpr_count <= 256
        assert res.lds_bytes <= 32768
