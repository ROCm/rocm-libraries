# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics of the tensor GETT epilogue when D spans more than 2 GiB.

The epilogue drops the stores of rows / columns past a ragged tile through the
buffer range check. Its row / column sentinel only lands past D while D spans
at most ``NARROW_D_BYTES``; a larger D (a ragged 38400 x 30240 output, 2.3 GB)
used to wrap the sentinel plus an in-range offset back inside D and overwrote
valid outputs. A wide D now masks the summed offset with one select.

Each case runs the kernel once on random data, NaN-poisons D first, and checks
every output against a float64 reference. ``test_narrow_sentinel_is_wrong_on_wide_d``
forces the narrow sentinel on the same problem and requires wrong outputs, so
the check is known to see the defect.

Skipped unless the visible device is a gfx950 (the GETT family is gfx950-only).
"""

from __future__ import annotations

import pytest

_ARCH = None
try:
    from rocke.runtime.hip_module import get_device_arch

    _ARCH = get_device_arch(0)
except Exception:  # noqa: BLE001 - no runtime / no device
    _ARCH = None

pytestmark = pytest.mark.skipif(
    _ARCH != "gfx950", reason=f"needs a gfx950 device; device is {_ARCH}"
)

# canonical contraction D[m, n] = A[m, k] B[n, k] with hipTensor column-major
# descriptors (M / N fastest), as row-major mode lists; N and M are not tile
# multiples and D is 2.32 GB
_EXT = {"M": 38400, "N": 30240, "K": 3072}
_MODES = (("K", "M"), ("K", "N"), ("N", "M"))


def _launcher(kd):
    from rocke.core.ir import PtrType
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher

    sig = []
    for prm in kd.params:
        nm = getattr(prm.type, "name", "") or ""
        if isinstance(prm.type, PtrType):
            sig.append(
                {"name": prm.name, "type": nm if nm.startswith("ptr<") else f"ptr<{nm}>"}
            )
        else:
            sig.append({"name": prm.name, "type": str(nm)})
    art = compile_kernel(kd, arch="gfx950", capture_ir_text=False, backend="python")
    return KernelLauncher(
        hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig
    )


def _run(dtype: str, *, bilinear=False, epi_pair=False):
    """(bad, worst, nonfinite) of one launch checked on every output."""
    import torch
    from rocke.instances.common.tensor import (
        TensorGettSpec,
        build_tensor_gett,
        is_valid_spec,
        launch_operands,
        tensor_gett_grid,
    )
    from rocke.instances.common.tensor.gett import GettKnobs
    from rocke.runtime.launcher import LaunchConfig

    ma, mb, md = _MODES
    spec = TensorGettSpec.make(
        ma,
        mb,
        md,
        _EXT,
        dtype=dtype,
        knobs=GettKnobs(bilinear=bilinear, epi_pair=epi_pair),
    )
    ok, why = is_valid_spec(spec)
    assert ok, why
    plan = spec.plan()
    kd = build_tensor_gett(spec)
    L = _launcher(kd)
    tdt = torch.float16 if dtype == "fp16" else torch.bfloat16
    g = torch.Generator(device="cuda")
    g.manual_seed(7)
    M, N, K = _EXT["M"], _EXT["N"], _EXT["K"]
    A = (torch.rand((K, M), generator=g, device="cuda") * 2 - 1).to(tdt)
    B = (torch.rand((K, N), generator=g, device="cuda") * 2 - 1).to(tdt)
    Cin = (
        (torch.rand((N, M), generator=g, device="cuda") * 2 - 1).to(tdt)
        if bilinear
        else None
    )
    alpha, beta = 0.75, -1.25
    D = torch.full((N, M), float("nan"), device="cuda", dtype=tdt)
    ops = {"A": A, "B": B}
    ga, gb = launch_operands(spec)
    vals = {
        "A": ops[ga],
        "B": ops[gb],
        "C": D,
        "M": plan.M,
        "N": plan.N,
        "K": plan.K,
        "alpha": alpha,
    }
    if bilinear:
        vals.update(Cin=Cin, beta=beta)
    L(
        {p.name: vals[p.name] for p in kd.params},
        config=LaunchConfig(grid=tensor_gett_grid(spec), block=(256, 1, 1), fence=False),
    )
    torch.cuda.synchronize()
    u = 2.0**-11 if dtype == "fp16" else 2.0**-8
    A64 = A.double()
    bad, worst, nonfinite = 0, 0.0, int((~torch.isfinite(D)).sum())
    for n0 in range(0, N, 2048):
        n1 = min(N, n0 + 2048)
        Bc = B[:, n0:n1].double()
        ref = alpha * (Bc.t() @ A64)
        absb = abs(alpha) * (Bc.abs().t() @ A64.abs())
        if bilinear:
            ref += beta * Cin[n0:n1].double()
            absb += abs(beta) * Cin[n0:n1].double().abs()
        tol = 2 * u * ref.abs() + 4 * 2.0**-24 * K * absb + 2.0**-24
        q = (D[n0:n1].double() - ref).abs() / tol
        q = torch.nan_to_num(q, nan=float("inf"))
        bad += int((q > 1).sum())
        worst = max(worst, float(q.max()))
    return bad, worst, nonfinite


@pytest.mark.parametrize(
    "dtype,bilinear,epi_pair",
    [
        ("fp16", False, False),
        ("bf16", False, False),
        ("fp16", True, False),
        ("fp16", False, True),
        ("bf16", True, True),
    ],
)
def test_wide_ragged_output_is_exact(dtype, bilinear, epi_pair):
    bad, worst, nonfinite = _run(dtype, bilinear=bilinear, epi_pair=epi_pair)
    assert nonfinite == 0, f"{nonfinite} outputs not written"
    assert bad == 0, f"{bad} wrong outputs (worst {worst:.3g} x tolerance)"


def test_narrow_sentinel_is_wrong_on_wide_d(monkeypatch):
    """The pre-fix epilogue (row / column sentinel) on the same problem must
    produce wrong outputs: the device check above can see the defect."""
    from rocke.instances.common.tensor import gett as G

    monkeypatch.setattr(G, "NARROW_D_BYTES", 1 << 40)
    bad, worst, nonfinite = _run("fp16")
    assert bad > 0 or nonfinite > 0, "narrow sentinel unexpectedly exact on wide D"
