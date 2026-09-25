# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for the torch-free fused KDA host pack and manifest adapter."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

from builders.gfx942.kda.hostpack import (
    bf16_bits_to_f32,
    f32_to_bf16_bits,
    make_inputs,
    pack_initial_state,
    pack_v_partitions,
    ref_token_serial,
    unpack_outputs,
)
from builders.gfx942.kda.manifest import (
    KIND,
    RUNNER_MODULE,
    make_kda_fused_manifest,
    run_kda_fused_manifest_problem,
)
from kernels.gfx942.kda_chunkwise import (
    KdaChunkFusedSpec,
    kda_chunk_fused_signature,
)
from rocke.run_manifest import registered_manifest_kinds, resolve_manifest_runner
from rocke.runtime.host_buffers import as_u8_buffer
from rocke.runtime.packing import pack_args


def test_hostpack_does_not_import_torch_or_dispatch():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    probe = (
        "import builders.gfx942.kda.hostpack, sys; "
        "leaked = [m for m in sys.modules if m == 'torch' or "
        "m.startswith('rocke.dispatch')]; "
        "print(leaked)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


def test_manifest_adapter_does_not_import_dispatch():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    probe = (
        "import builders.gfx942.kda.manifest, sys; "
        "leaked = sorted(m for m in sys.modules if m.startswith('rocke.dispatch')); "
        "print(leaked)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


def test_bf16_roundtrip_preserves_magnitude():
    x = np.array([1.0, -0.5, 0.0, 3.25], dtype=np.float32)
    y = bf16_bits_to_f32(f32_to_bf16_bits(x))
    assert np.max(np.abs(y - x)) < 1e-2


def test_oracle_identity_query_on_one_token():
    B, H, T, DK, DV = 1, 1, 1, 4, 8
    q = np.zeros((B, H, T, DK), dtype=np.float32)
    k = np.zeros((B, H, T, DK), dtype=np.float32)
    q[..., 0] = 1.0
    k[..., 0] = 1.0
    v = np.arange(DV, dtype=np.float32).reshape(1, 1, 1, DV)
    g = np.zeros((B, H, T, DK), dtype=np.float32)
    beta = np.ones((B, H, T), dtype=np.float32)
    scale = DK**-0.5
    o, S = ref_token_serial(q, k, v, g, beta, scale)
    # S = k u^T with u = v, so o = scale * (q·k) v = scale * v
    np.testing.assert_allclose(o[0, 0, 0], scale * v[0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(S[0, 0, 0], v[0, 0, 0], atol=1e-12)


def test_unpack_inverts_v_pack():
    B, H, T, DK, DV, C, head_v = 2, 2, 32, 8, 8, 16, 4
    rng = np.random.default_rng(1)
    q = rng.integers(0, 65535, size=(B, H, T, DK), dtype=np.uint16)
    k = rng.integers(0, 65535, size=(B, H, T, DK), dtype=np.uint16)
    v = rng.integers(0, 65535, size=(B, H, T, DV), dtype=np.uint16)
    g = rng.random((B, H, T, DK), dtype=np.float32)
    beta = rng.random((B, H, T), dtype=np.float32)
    packed = pack_v_partitions(C, head_v, q, k, v, g, beta)
    assert packed.parts == DV // head_v
    assert packed.q.shape == (B * H * packed.parts * packed.nc, C * DK)
    # Q is duplicated across V partitions.
    np.testing.assert_array_equal(packed.q[0], packed.q[packed.nc])
    ht = np.zeros(
        (packed.bh * packed.parts, packed.head_v, packed.head_k),
        dtype=np.float32,
    )
    got, _ = unpack_outputs(
        chunk=C,
        head_k=DK,
        head_v=head_v,
        o=packed.v,
        ht=ht,
        B=B,
        H=H,
        T=T,
        DV=DV,
        NC=packed.nc,
        parts=packed.parts,
    )
    np.testing.assert_array_equal(got, v)


@pytest.mark.parametrize("parts", (1, 2, 4))
def test_pack_returns_writable_owned_buffers(parts):
    """Every packed field must be uploadable, at any partition count.

    ``broadcast_to`` yields a read-only view and ``ascontiguousarray`` will not
    copy one that is already contiguous, so at one partition the pack used to
    hand back read-only arrays. ``ctypes.from_buffer`` -- how both the manifest
    runner and the benchmark upload -- rejects those, which made logical
    ``head_v == 64`` unrunnable through the torch-free lane.
    """
    head_v = 4
    B, H, T, DK, C = 2, 2, 32, 8, 16
    dense = dict(
        zip(("q", "k", "v", "g", "beta"), make_inputs(B, H, T, DK, head_v * parts))
    )
    packed = pack_v_partitions(C, head_v, **dense)
    assert packed.parts == parts
    for name, source in dense.items():
        field = getattr(packed, name)
        assert field.flags.writeable, f"{name} is read-only"
        assert not np.shares_memory(field, source), f"{name} aliases its input"
        as_u8_buffer(field)  # raises TypeError on a read-only buffer


def test_pack_rejects_ragged_sequence():
    q, k, v, g, beta = make_inputs(1, 1, 17, 8, 8)
    with pytest.raises(ValueError, match="divisible"):
        pack_v_partitions(16, 4, q, k, v, g, beta)


def test_initial_state_pack_roundtrip_shape():
    B, H, DK, DV, head_v = 2, 3, 8, 8, 4
    h0 = np.arange(B * H * DK * DV, dtype=np.float32).reshape(B, H, DK, DV)
    parts = DV // head_v
    packed = pack_initial_state(head_v, h0, parts)
    assert packed.shape == (B * H * parts, head_v, DK)


def test_kind_registers_on_import():
    assert KIND in registered_manifest_kinds()
    builder = resolve_manifest_runner({"kind": KIND, "runner_module": RUNNER_MODULE})
    assert builder is run_kda_fused_manifest_problem


def test_runner_geometry_without_gpu():
    spec = KdaChunkFusedSpec(head_k=8, head_v=4)

    # Duck-type a tiny spec for the emitter; kernel_name is unused here.
    class _Art:
        kernel_name = "kda_test"
        timings = {}
        hsaco_bytes = 1

        class kernel:
            name = "kda_test"

    man = make_kda_fused_manifest(
        artifact=_Art(),
        spec=spec,
        args_signature=kda_chunk_fused_signature(spec),
        logical_head_v=8,
        default_shape=(1, 1, 32),
    )
    assert man["kind"] == KIND
    assert man["runner_module"] == RUNNER_MODULE
    assert man["default_shape"] == [1, 1, 32]
    make_args, grid, block, flop, bytes_xfer, check = run_kda_fused_manifest_problem(
        man, None, False
    )
    # 1*1*(8/4) partitions
    assert grid == (2, 1, 1)
    assert block == (spec.tile.block_size, 1, 1)
    assert flop > 0 and bytes_xfer > 0
    packed = pack_args(
        man["args_signature"],
        {
            "q_ptr": 1,
            "k_ptr": 2,
            "g_ptr": 3,
            "beta_ptr": 4,
            "v_ptr": 5,
            "o_ptr": 6,
            "h0_ptr": 7,
            "ht_ptr": 8,
            "scale": 0.25,
            "nc": 2,
        },
    )
    # 8 pointers + f32 + i32, naturally aligned, 72 bytes.
    assert len(packed) == 72


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_numpy_pack_matches_torch_builder_layout():
    import torch

    from builders.gfx942.kda import kda_chunk_fused as fused
    from kernels.gfx942.kda_chunkwise import KdaChunkFusedSpec, KdaTileSpec

    B, H, T, DK, DV, C, head_v = 1, 2, 32, 8, 8, 16, 4
    spec = KdaChunkFusedSpec(
        head_k=DK,
        head_v=head_v,
        tile=KdaTileSpec(chunk=C, block_size=256),
    )
    rng = np.random.default_rng(3)
    q_f = rng.standard_normal((B, H, T, DK), dtype=np.float32)
    k_f = rng.standard_normal((B, H, T, DK), dtype=np.float32)
    v_f = rng.standard_normal((B, H, T, DV), dtype=np.float32)
    g = rng.random((B, H, T, DK), dtype=np.float32)
    beta = rng.random((B, H, T), dtype=np.float32)
    q_b, k_b, v_b = (f32_to_bf16_bits(x) for x in (q_f, k_f, v_f))
    packed = pack_v_partitions(C, head_v, q_b, k_b, v_b, g, beta)

    def as_bf16(bits):
        return torch.from_numpy(np.ascontiguousarray(bits)).view(torch.bfloat16)

    q_t, k_t, v_t = (as_bf16(x) for x in (q_b, k_b, v_b))
    g_t = torch.from_numpy(np.ascontiguousarray(g))
    beta_t = torch.from_numpy(np.ascontiguousarray(beta))
    qf, kf, vf, gf, bf, BH, NC, parts = fused._pack_v_partitions(
        spec, q_t, k_t, v_t, g_t, beta_t
    )
    assert (BH, NC, parts) == (packed.bh, packed.nc, packed.parts)
    np.testing.assert_array_equal(qf.view(torch.uint16).numpy(), packed.q)
    np.testing.assert_array_equal(kf.view(torch.uint16).numpy(), packed.k)
    np.testing.assert_array_equal(vf.view(torch.uint16).numpy(), packed.v)
    np.testing.assert_allclose(gf.numpy(), packed.g, atol=0, rtol=0)
    np.testing.assert_allclose(bf.numpy(), packed.beta, atol=0, rtol=0)
