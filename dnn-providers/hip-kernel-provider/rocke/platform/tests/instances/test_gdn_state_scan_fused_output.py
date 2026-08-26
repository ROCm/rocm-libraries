# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coverage for the gfx942 fused GDN state-scan output path."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from rocke.helpers import compile_kernel
from rocke.instances.gfx942.gdn_state_scan import (
    GdnStateScanSpec,
    build_gdn_state_scan,
    gdn_state_scan_grid,
    gdn_state_scan_signature,
)
from rocke.runtime.hip_module import Runtime, get_device_arch
from rocke.runtime.launcher import DeviceMem, KernelLauncher, LaunchConfig


def _spec(*, use_g: bool, fused: bool) -> GdnStateScanSpec:
    return GdnStateScanSpec(
        K=128,
        V=16,
        H=1,
        Hg=1,
        BV=16,
        USE_G=use_g,
        USE_GK=not use_g,
        STORE_H=not fused,
        SAVE_NEW_VALUE=not fused,
        COMPUTE_OUTPUT=fused,
        SCALE=0.125 if fused else None,
        IS_VARLEN=False,
    )


def _walk_ops(region):
    for op in region.ops:
        yield op
        for nested in op.regions:
            yield from _walk_ops(nested)


@pytest.mark.parametrize("use_g", [True, False], ids=["scalar-g", "gk"])
def test_fused_spec_signature_and_body(use_g):
    spec = _spec(use_g=use_g, fused=True)
    kernel = build_gdn_state_scan(spec)
    signature = gdn_state_scan_signature(spec)
    ops = list(_walk_ops(kernel.body))

    assert spec.kt_transposed is False
    assert spec.lds_a_elems == spec.BT * spec.BT
    assert spec.kernel_name().endswith(
        f"_o_{'g' if use_g else 'gk'}_K128_V16_bt64_bv16_w4"
    )
    assert [arg["name"] for arg in signature] == [
        "Kt",
        "Wt",
        "Ut",
        "Gate",
        "H0",
        "Vnew",
        "Hout",
        "Ht",
        "Qt",
        "O",
        "T_val",
        "NT_val",
        "N_val",
    ]
    assert [param.name for param in kernel.params] == [arg["name"] for arg in signature]
    assert any(
        op.name == "tile.smem_alloc" and op.results[0].name.startswith("%sA")
        for op in ops
    )
    # K5 has two GEMMs; fused K6 adds q@h, q@k^T, and A@v_new.
    assert sum(op.name == "tile.mma" for op in ops) > 2


def test_k5_only_signature_and_layout_stay_unfused():
    spec = _spec(use_g=False, fused=False)
    kernel = build_gdn_state_scan(spec)
    names = [arg["name"] for arg in gdn_state_scan_signature(spec)]

    assert spec.kt_transposed is True
    assert spec.lds_a_elems == 0
    assert names == [
        "Kt",
        "Wt",
        "Ut",
        "Gate",
        "H0",
        "Vnew",
        "Hout",
        "Ht",
        "T_val",
        "NT_val",
        "N_val",
    ]
    assert [param.name for param in kernel.params] == names
    assert not any(
        op.name == "tile.smem_alloc" and op.results[0].name.startswith("%sA")
        for op in _walk_ops(kernel.body)
    )


def _f32_to_bf16_bits(values, *, round_away=False):
    f32 = np.asarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    if round_away:
        bias = np.uint32(0x8000)
    else:
        bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) >> np.uint32(16)).astype(np.uint16)


def _bf16_bits_to_f32(values):
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def _upload(runtime, values):
    raw = np.ascontiguousarray(values)
    memory = DeviceMem(raw.nbytes)
    host = (ctypes.c_ubyte * raw.nbytes).from_buffer_copy(raw.tobytes())
    runtime.memcpy_h2d(memory.ptr(), host, raw.nbytes)
    return memory


def _download(runtime, memory, dtype, count):
    nbytes = np.dtype(dtype).itemsize * count
    host = (ctypes.c_ubyte * nbytes)()
    runtime.memcpy_d2h(host, memory.ptr(), nbytes)
    return np.frombuffer(bytes(host), dtype=dtype, count=count).copy()


@pytest.mark.parametrize("use_g", [True, False], ids=["scalar-g", "gk"])
def test_fused_output_numeric_gfx942(use_g):
    if get_device_arch(0) != "gfx942":
        pytest.skip("requires a local gfx942 device")

    spec = _spec(use_g=use_g, fused=True)
    rng = np.random.default_rng(7)
    T, H, K, V = spec.BT, spec.H, spec.K, spec.V
    q = _bf16_bits_to_f32(_f32_to_bf16_bits(rng.normal(0.0, 0.05, (T, K))))
    k = _bf16_bits_to_f32(_f32_to_bf16_bits(rng.normal(0.0, 0.05, (T, K))))
    w = _bf16_bits_to_f32(_f32_to_bf16_bits(rng.normal(0.0, 0.05, (T, K))))
    u = _bf16_bits_to_f32(_f32_to_bf16_bits(rng.normal(0.0, 0.05, (T, V))))
    h0 = rng.normal(0.0, 0.01, (V, K)).astype(np.float32)
    if use_g:
        gate = -np.linspace(0.001, 0.25, T, dtype=np.float32)
    else:
        gate = -np.linspace(0.001, 0.2, T, dtype=np.float32)[:, None]
        gate = np.broadcast_to(gate, (T, K)).copy()

    h_snapshot = _bf16_bits_to_f32(_f32_to_bf16_bits(h0, round_away=True))
    v_new = u - w @ h_snapshot.T
    if use_g:
        g_last = gate[-1]
        v_for_update = v_new * np.exp(g_last - gate)[:, None]
    else:
        v_for_update = v_new
    v_lds = _bf16_bits_to_f32(_f32_to_bf16_bits(v_for_update, round_away=True))
    attention = np.tril(q @ k.T)
    attention_lds = _bf16_bits_to_f32(_f32_to_bf16_bits(attention, round_away=True))
    inter = q @ h_snapshot.T
    intra = attention_lds @ v_lds
    if use_g:
        expected = spec.SCALE * (
            np.exp(gate)[:, None] * inter + np.exp(gate - g_last)[:, None] * intra
        )
    else:
        expected = spec.SCALE * (inter + intra)
    expected = _bf16_bits_to_f32(_f32_to_bf16_bits(expected, round_away=True))

    artifact = compile_kernel(
        build_gdn_state_scan(spec),
        arch="gfx942",
        backend="python",
        capture_ir_text=False,
    )
    runtime = Runtime()
    host_values = {
        "Kt": _f32_to_bf16_bits(k),
        "Wt": _f32_to_bf16_bits(w),
        "Ut": _f32_to_bf16_bits(u),
        "Gate": gate.astype(np.float32),
        "H0": h0,
        "Vnew": np.zeros(T * H * V, dtype=np.float32),
        "Hout": np.zeros(H * V * K, dtype=np.float32),
        "Ht": np.zeros(H * V * K, dtype=np.float32),
        "Qt": _f32_to_bf16_bits(q),
        "O": np.zeros(T * H * V, dtype=np.uint16),
    }
    device = {name: _upload(runtime, value) for name, value in host_values.items()}
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=gdn_state_scan_signature(spec),
    )
    values = {name: memory.ptr() for name, memory in device.items()}
    values.update(T_val=T, NT_val=1, N_val=1)
    launcher(
        values,
        config=LaunchConfig(
            grid=gdn_state_scan_grid(spec, 1),
            block=(spec.block_threads, 1, 1),
            fence=True,
        ),
    )
    got = _bf16_bits_to_f32(
        _download(runtime, device["O"], np.uint16, T * H * V)
    ).reshape(T, V)
    np.testing.assert_allclose(got, expected, atol=5e-2, rtol=5e-2)
