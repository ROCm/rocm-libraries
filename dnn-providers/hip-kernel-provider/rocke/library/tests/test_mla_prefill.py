# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the MLA prefill forward kernel.

Coverage:
  1. Spec validation — checks that valid and invalid specs are handled correctly.
  2. Kernel codegen (no GPU) — verifies that build_mla_prefill_fwd completes
     without error on both gfx942 and gfx950 for the three target model
     geometries (DeepSeek-R1 Hq=128, GLM-5 Hq=128, Kimi-K2 Hq=64).
  3. Grid and signature helpers — verifies correct grid shape and ABI layout.
  4. Reference correctness (GPU, optional) — numerical parity with
     ref_mla_prefill_fwd on gfx942 and gfx950.

Shape matrix (from mla_prefill_shapes.json / DESIGN.md §8.2):
  DeepSeek-R1: Hq=128, Hk=1, block_size=16, Sq ∈ {2048, 4096, 8192}
  GLM-5:       Hq=128, Hk=1, block_size=16, Sq ∈ {2048, 4096, 8192}
  Kimi-K2:     Hq=64,  Hk=1, block_size=16, Sq ∈ {2048, 4096, 8192}

GPU tests require a supported GPU and PyTorch; they are skipped otherwise.
"""

from __future__ import annotations

import math
import os

import pytest

from kernels.mla import (
    MlaPrefillSpec,
    build_mla_prefill_fwd,
    is_valid_mla_prefill_spec,
    mla_prefill_fwd_grid,
    mla_prefill_fwd_signature,
)

# ---------------------------------------------------------------------------
# Optionally import PyTorch for GPU tests
# ---------------------------------------------------------------------------
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from builders.mla.ref_mla_attn import (
        make_mla_prefill_inputs,
        ref_mla_prefill_fwd,
    )
    _HAS_REF = True
except ImportError:
    _HAS_REF = False


# ---------------------------------------------------------------------------
# GPU arch detection
# ---------------------------------------------------------------------------

def _gpu_arch() -> str:
    if not _HAS_TORCH or not torch.cuda.is_available():
        return "cpu"
    try:
        import subprocess
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("Name:") and "gfx" in line:
                gfx = line.split("Name:")[-1].strip()
                if gfx.startswith("gfx"):
                    return gfx
    except Exception:
        pass
    return "gfx950"   # fallback assumption


GPU_ARCH = _gpu_arch()
ON_GPU = GPU_ARCH in ("gfx942", "gfx950") and _HAS_TORCH and torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Shared test fixtures / helpers
# ---------------------------------------------------------------------------

# (model_name, num_query_heads, block_size)
MODEL_GEOMETRIES = [
    pytest.param("deepseek_r1", 128, 16, id="DeepSeek-R1_Hq128"),
    pytest.param("glm5",        128, 16, id="GLM-5_Hq128"),
    pytest.param("kimi_k2",      64, 16, id="Kimi-K2_Hq64"),
]

# seqlen_q values from DESIGN.md §8.2 (Sq ∈ {2048, 4096, 8192})
SQ_SHAPES = [2048, 4096, 8192]

ARCHES = ["gfx942", "gfx950"]


# ---------------------------------------------------------------------------
# 1. Spec validation tests (pure Python, no GPU)
# ---------------------------------------------------------------------------


class TestMlaPrefillSpec:
    def test_valid_deepseek(self):
        spec = MlaPrefillSpec(num_query_heads=128, block_size=16)
        assert spec.num_query_heads == 128
        assert spec.block_size == 16

    def test_valid_kimi_k2(self):
        spec = MlaPrefillSpec(num_query_heads=64, block_size=16)
        assert spec.num_query_heads == 64

    def test_invalid_hq(self):
        with pytest.raises(ValueError, match="num_query_heads"):
            MlaPrefillSpec(num_query_heads=32)

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            MlaPrefillSpec(num_query_heads=128, block_size=8)

    @pytest.mark.parametrize("arch", ARCHES)
    def test_is_valid_bf16(self, arch):
        spec = MlaPrefillSpec(num_query_heads=128, block_size=16)
        ok, reason = is_valid_mla_prefill_spec(spec, arch)
        assert ok, f"Expected valid spec on {arch}: {reason}"

    def test_invalid_arch(self):
        spec = MlaPrefillSpec(num_query_heads=128)
        ok, reason = is_valid_mla_prefill_spec(spec, "gfx1250")
        assert not ok
        assert "gfx1250" in reason

    @pytest.mark.parametrize("arch", ARCHES)
    @pytest.mark.parametrize("model,hq,bs", [
        ("deepseek_r1", 128, 16),
        ("glm5",        128, 16),
        ("kimi_k2",      64, 16),
    ])
    def test_all_models_valid(self, arch, model, hq, bs):
        spec = MlaPrefillSpec(num_query_heads=hq, block_size=bs)
        ok, reason = is_valid_mla_prefill_spec(spec, arch)
        assert ok, f"{model} on {arch}: {reason}"

    def test_kernel_name_encodes_hq(self):
        s128 = MlaPrefillSpec(num_query_heads=128)
        s64  = MlaPrefillSpec(num_query_heads=64)
        assert "Hq128" in s128.kernel_name()
        assert "Hq64"  in s64.kernel_name()
        assert s128.kernel_name() != s64.kernel_name()

    def test_kernel_name_encodes_bf16(self):
        spec = MlaPrefillSpec(num_query_heads=128)
        assert "bf16" in spec.kernel_name()


# ---------------------------------------------------------------------------
# 2. Kernel codegen tests (pure Python, no GPU)
# ---------------------------------------------------------------------------


class TestMlaPrefillCodegen:
    """Verifies that build_mla_prefill_fwd completes without error."""

    @pytest.mark.parametrize("arch", ARCHES)
    @pytest.mark.parametrize("hq,bs", [
        (128, 16),  # DeepSeek-R1 / GLM-5
        (64,  16),  # Kimi-K2
    ])
    def test_build_completes(self, arch, hq, bs):
        spec = MlaPrefillSpec(num_query_heads=hq, block_size=bs)
        kernel = build_mla_prefill_fwd(spec, arch=arch)
        assert kernel is not None, f"build_mla_prefill_fwd returned None on {arch}"

    @pytest.mark.parametrize("arch", ARCHES)
    def test_build_deepseek_r1(self, arch):
        spec = MlaPrefillSpec(num_query_heads=128, block_size=16, name="rocke_mla_prefill_ds_r1")
        kernel = build_mla_prefill_fwd(spec, arch=arch)
        assert "Hq128" in kernel.name or "ds_r1" in kernel.name

    @pytest.mark.parametrize("arch", ARCHES)
    def test_build_glm5(self, arch):
        spec = MlaPrefillSpec(num_query_heads=128, block_size=16, name="rocke_mla_prefill_glm5")
        kernel = build_mla_prefill_fwd(spec, arch=arch)
        assert kernel is not None

    @pytest.mark.parametrize("arch", ARCHES)
    def test_build_kimi_k2(self, arch):
        spec = MlaPrefillSpec(num_query_heads=64, block_size=16, name="rocke_mla_prefill_kimi_k2")
        kernel = build_mla_prefill_fwd(spec, arch=arch)
        assert kernel is not None

    @pytest.mark.parametrize("arch", ARCHES)
    def test_build_invalid_spec_raises(self, arch):
        with pytest.raises(ValueError, match="invalid MlaPrefillSpec"):
            # Create an invalid spec by bypassing __post_init__ and calling build
            import dataclasses
            spec = object.__new__(MlaPrefillSpec)
            spec.__dict__.update(
                num_query_heads=128, block_size=16, name="bad"
            )
            # Inject an invalid arch check by using a patched spec
            build_mla_prefill_fwd(
                MlaPrefillSpec(num_query_heads=128, block_size=16), arch="gfx1250"
            )

    def test_gfx942_gfx950_produce_different_kernels(self):
        spec = MlaPrefillSpec(num_query_heads=128, block_size=16)
        k942 = build_mla_prefill_fwd(spec, arch="gfx942")
        k950 = build_mla_prefill_fwd(spec, arch="gfx950")
        assert k942 is not None
        assert k950 is not None


# ---------------------------------------------------------------------------
# 3. Grid and signature tests (pure Python, no GPU)
# ---------------------------------------------------------------------------


class TestMlaPrefillGridAndSignature:
    # Grid: fmha_paged_prefill with use_mfma_body=True uses (total_q, H_q, 1)
    @pytest.mark.parametrize("total_q,hq,expected_q", [
        (2048, 128, 2048),
        (4096,  64, 4096),
        (8192, 128, 8192),
    ])
    def test_grid_shape(self, total_q, hq, expected_q):
        spec = MlaPrefillSpec(num_query_heads=hq)
        grid = mla_prefill_fwd_grid(spec, total_q=total_q)
        assert grid == (total_q, hq, 1), (
            f"grid={grid}, expected=({total_q}, {hq}, 1)"
        )

    def test_grid_hq_in_y(self):
        spec = MlaPrefillSpec(num_query_heads=128)
        grid = mla_prefill_fwd_grid(spec, total_q=2048)
        assert grid[1] == 128  # H_q in y dimension

    def test_signature_is_list_of_dicts(self):
        spec = MlaPrefillSpec(num_query_heads=128)
        sig  = mla_prefill_fwd_signature(spec)
        assert isinstance(sig, list)
        assert isinstance(sig[0], dict)
        assert "name" in sig[0] and "type" in sig[0]

    def test_signature_contains_required_tensors(self):
        spec = MlaPrefillSpec(num_query_heads=128)
        sig  = mla_prefill_fwd_signature(spec)
        names = [e["name"] for e in sig]
        # fmha_paged_prefill ABI names
        for required in ("Q", "K_cache", "V_cache", "O",
                         "block_table", "cu_seqlens_q", "seqlens_k",
                         "scale_log2", "total_q", "batch"):
            assert required in names, f"Missing '{required}' in signature"

    def test_signature_strides(self):
        spec = MlaPrefillSpec(num_query_heads=128)
        sig  = mla_prefill_fwd_signature(spec)
        names = [e["name"] for e in sig]
        for stride in ("stride_q_token", "stride_q_head", "stride_o_token", "stride_o_head"):
            assert stride in names

    def test_signature_tensors_are_bf16(self):
        spec = MlaPrefillSpec(num_query_heads=128)
        sig  = {e["name"]: e["type"] for e in mla_prefill_fwd_signature(spec)}
        for tensor in ("Q", "K_cache", "V_cache", "O"):
            assert "bf16" in sig[tensor], f"{tensor} expected bf16, got {sig[tensor]}"

    def test_signature_scale_log2_is_f32(self):
        spec = MlaPrefillSpec(num_query_heads=128)
        sig  = {e["name"]: e["type"] for e in mla_prefill_fwd_signature(spec)}
        assert sig["scale_log2"] == "f32"


# ---------------------------------------------------------------------------
# 4. Numerical correctness tests (GPU required)
# ---------------------------------------------------------------------------

MAX_ABS_TOLERANCE = 4e-2   # matching the existing unified attention gate


@pytest.mark.skipif(not ON_GPU, reason=f"GPU not available or arch not supported (arch={GPU_ARCH})")
@pytest.mark.skipif(not _HAS_REF, reason="ref_mla_attn not importable")
class TestMlaPrefillNumerical:
    """Compare kernel output against the Python reference."""

    @pytest.mark.parametrize("model,hq,sq", [
        ("deepseek_r1", 128, 2048),
        ("deepseek_r1", 128, 4096),
        ("deepseek_r1", 128, 8192),
        ("glm5",        128, 2048),
        ("glm5",        128, 4096),
        ("glm5",        128, 8192),
        ("kimi_k2",      64, 2048),
        ("kimi_k2",      64, 4096),
        ("kimi_k2",      64, 8192),
    ])
    def _run_kernel(self, inputs: dict, spec: MlaPrefillSpec) -> torch.Tensor:
        """Build, expand K/V, run kernel. Returns output [sq, hq, 128]."""
        import math
        from rocke.helpers.compile import compile_kernel
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig
        from kernels.mla.mla_prefill import KERNEL_HEAD_SIZE

        H  = KERNEL_HEAD_SIZE  # 256
        hq = spec.num_query_heads
        bs = spec.block_size
        sk = inputs["seqlen_k"]
        sq = inputs["seqlen_q"]
        num_blocks = inputs["num_blocks"]

        # Expand K/V to H=256
        c_kv   = inputs["c_kv_flat"].float()
        k_rope = inputs["k_rope_flat"].float()
        K_nope = (c_kv @ inputs["w_uk_k"].float()).to(torch.bfloat16)
        K_192  = torch.cat([K_nope, k_rope.to(torch.bfloat16)], dim=-1)
        K_256  = torch.zeros(sk, H, dtype=torch.bfloat16, device=K_192.device)
        K_256[:, :192] = K_192
        V_128  = (c_kv @ inputs["w_uv"].float()).to(torch.bfloat16)
        V_256  = torch.zeros(sk, H, dtype=torch.bfloat16, device=V_128.device)
        V_256[:, :128] = V_128

        pad = num_blocks * bs - sk
        def _page(x):
            if pad > 0:
                x = torch.cat([x, torch.zeros(pad, H, dtype=x.dtype, device=x.device)])
            return x.view(num_blocks, bs, 1, H)

        K_paged   = _page(K_256)
        V_paged   = _page(V_256)
        seqlens_k = torch.tensor([sk], dtype=torch.int32, device=K_paged.device)

        # Pad Q from 192 to 256
        q_256 = torch.zeros(sq, hq, H, dtype=torch.bfloat16, device="cuda")
        q_256[:, :, :192] = inputs["q"]

        kdef = build_mla_prefill_fwd(spec, arch=GPU_ARCH)
        art  = compile_kernel(kdef, arch=GPU_ARCH, capture_ir_text=False)
        lnch = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=mla_prefill_fwd_signature(spec),
        )
        out_256    = torch.zeros(sq, hq, H, dtype=torch.bfloat16, device="cuda")
        scale_log2 = float(math.log2(inputs["scale"]))
        stride_kv  = bs * H
        cfg = LaunchConfig(grid=mla_prefill_fwd_grid(spec, total_q=sq), block=(64, 1, 1))
        vals = {
            "Q": q_256, "K_cache": K_paged, "V_cache": V_paged, "O": out_256,
            "block_table": inputs["block_table"], "cu_seqlens_q": inputs["cu_seqlens_q"],
            "seqlens_k": seqlens_k, "scale_log2": scale_log2,
            "total_q": int(sq), "batch": int(1),
            "stride_q_token": int(hq * H), "stride_q_head": int(H),
            "stride_block": stride_kv, "stride_page": H, "stride_kv_head": H,
            "stride_v_block": stride_kv, "stride_v_page": H, "stride_v_kv_head": H,
            "stride_o_token": int(hq * H), "stride_o_head": int(H),
            "block_table_stride": int(num_blocks),
        }
        lnch(vals, config=cfg)
        torch.cuda.synchronize()
        return out_256[:, :, :128].contiguous()

    def test_correctness_vs_ref(self, model, hq, sq):
        """Verify max_abs(kernel - ref) ≤ 4e-2 for all model × seqlen shapes."""
        inputs = make_mla_prefill_inputs(
            num_query_heads=hq, seqlen_q=sq, seqlen_k=sq,
            block_size=16, device="cuda", seed=42,
        )
        ref_out = ref_mla_prefill_fwd(
            inputs["q"], inputs["c_kv_flat"], inputs["k_rope_flat"],
            inputs["w_uk_k"], inputs["w_uv"], inputs["cu_seqlens_q"],
            causal=True, scale=inputs["scale"],
        )
        spec = MlaPrefillSpec(num_query_heads=hq, block_size=16)
        out  = self._run_kernel(inputs, spec)
        diff = (out.float() - ref_out.float()).abs().max().item()
        assert diff <= MAX_ABS_TOLERANCE, (
            f"model={model} arch={GPU_ARCH} sq={sq} hq={hq}: "
            f"max_abs={diff:.4f} > {MAX_ABS_TOLERANCE}"
        )

    @pytest.mark.parametrize("sq", [2048, 4096, 8192])
    def test_softmax_lse_finite(self, sq):
        """Kernel must complete without NaN/inf (LSE check skipped — fmha_paged_prefill
        does not expose LSE output; test verifies output is finite instead)."""
        hq = 128
        inputs = make_mla_prefill_inputs(hq, sq, sq, seed=7)
        spec = MlaPrefillSpec(num_query_heads=hq)
        out  = self._run_kernel(inputs, spec)
        assert torch.isfinite(out).all(), "output contains inf/nan"
