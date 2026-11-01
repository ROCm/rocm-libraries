# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the quantum-circuit tensor-network monokernel.

The phase planner, the fragment packing and the kernel's index math (through
the lane-exact NumPy emulation of ``circuit_kernel._emit_phase``) are checked
against an independent complex128 gate-by-gate reference; every knob builds.
"""

import numpy as np
import pytest
from kernels.gfx950.tensor_network_mono import circuit_kernel as K
from kernels.gfx950.tensor_network_mono import circuit_plan as P
from kernels.gfx950.tensor_network_mono.circuit_emulate import emulate


def _apply_ref(psi: np.ndarray, gates, n: int) -> np.ndarray:
    """complex128 reference: amplitude index bit q <-> qubit q, U index bit i <-> gq[i]."""
    t = psi.reshape([2] * n)  # axis a <-> qubit n - 1 - a

    def ax(q):
        return n - 1 - q

    for gq, U in gates:
        k = len(gq)
        Ur = U.reshape([2] * (2 * k))  # rows: gq[k-1] .. gq[0], then cols the same
        src = [ax(gq[k - 1 - i]) for i in range(k)]
        t = np.tensordot(Ur, t, axes=(list(range(k, 2 * k)), src))
        t = np.moveaxis(t, list(range(k)), src)
    return t.reshape(-1)


def _circuit(rows=3, cols=4, depth=3, seed=5):
    gates = P.sycamore_like(rows, cols, depth, seed=seed)
    n = rows * cols
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
    psi /= np.linalg.norm(psi)
    return gates, n, psi


def test_real_embedding_matches_complex_product():
    rng = np.random.default_rng(0)
    U = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    x = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    xr = np.empty(8)
    xr[0::2], xr[1::2] = x.real, x.imag
    y = P.real_embedding(U) @ xr
    np.testing.assert_allclose(y[0::2] + 1j * y[1::2], U @ x, rtol=1e-12, atol=1e-12)


def test_plan_phases_keep_every_gate_and_locality():
    gates, n, _ = _circuit()
    t, c = 12, 5
    phases = P.plan_phases(gates, n, t=t, c=c, kf=4)
    assert sum(bl.ngates for ph in phases for bl in ph.blocks) == len(gates)
    for ph in phases:
        local = set(range(c - 1)) | set(ph.extra)
        assert len(ph.extra) == t - c
        for bl in ph.blocks:
            assert set(bl.qubits) <= local and len(bl.qubits) in (4, 5)


@pytest.mark.parametrize("swizzle", ["x4", "none"])
def test_emulated_kernel_matches_reference(swizzle):
    gates, n, psi = _circuit()
    t = 12
    phases = P.plan_phases(gates, n, t=t, c=5, kf=4)
    spec, mats = K.spec_from_plan(n, phases, t=t, c=5, grid=2, swizzle=swizzle)
    st = np.empty(2 << n, np.float32)
    st[0::2], st[1::2] = psi.real, psi.imag
    out = emulate(spec, mats, st)
    got = out[0::2].astype(np.float64) + 1j * out[1::2]
    ref = _apply_ref(psi, gates, n)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    nblocks = sum(len(ph.blocks) for ph in phases)
    assert rel <= nblocks * 64 * 2.0**-24 * 4, rel


@pytest.mark.parametrize(
    "kv",
    [
        {},
        dict(lazy=False),
        dict(rhoist=True),
        dict(prefetch=True),
        dict(bar="last"),
        dict(bar="xcd"),
        dict(nt=True),
        dict(wpe=2),
        dict(mono=False),
    ],
)
def test_build_every_knob(kv):
    gates, n, _ = _circuit(depth=2)
    phases = P.plan_phases(gates, n, t=11, c=5, kf=4)
    args = dict(t=11, c=5, grid=4)
    args.update(kv)
    spec, _ = K.spec_from_plan(n, phases, **args)
    ks = K.build_circuit_kernels(spec, arch="gfx950")
    assert len(ks) == (1 if spec.mono else len(phases))
    assert [p.name for p in ks[0].params] == ["state", "mats", "bar", "epoch"]
    assert ks[0].params[1].attrs.get("readonly", False) == spec.rhoist


def test_spec_rejects_unsafe_and_invalid():
    gates, n, _ = _circuit(depth=1)
    phases = P.plan_phases(gates, n, t=11, c=5, kf=4)
    with pytest.raises(ValueError, match="bar"):
        K.spec_from_plan(n, phases, t=11, c=5, grid=4, bar="nowait")
    spec, _ = K.spec_from_plan(n, phases, t=11, c=5, grid=3)
    with pytest.raises(ValueError, match="divide"):
        K.build_circuit_kernels(spec, arch="gfx950")
    spec, _ = K.spec_from_plan(n, phases, t=11, c=5, grid=4)
    with pytest.raises(ValueError, match="gfx950"):
        K.build_circuit_kernels(spec, arch="gfx942")
