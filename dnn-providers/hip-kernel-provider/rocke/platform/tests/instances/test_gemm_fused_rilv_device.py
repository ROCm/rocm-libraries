# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics of a ``helpers.fuse`` epilogue on the row-interleave
vector-epilogue bodies the RCR dispatcher picks on gfx950.

The dispatcher's big-tile picks store C through the row-interleave vector
epilogue (``rilv_vec_epilogue``: 8 columns per lane, 16-byte stores, and the
branch-free ragged form ``rve_pad`` 2). A fused epilogue (bias + ReLU here)
used to be refused on these bodies; it now runs on the vector epilogue's
values, and :func:`rocke.dispatch.gemm.common.fused_epilogue_spec` reroutes
the levers whose epilogue has no hook (the persistent tile loop becomes one
workgroup per tile). Each case takes the dispatcher's pick, attaches the
epilogue, NaN-poisons C, runs once and checks every output against a float64
reference of ``relu(A @ B^T + bias)``.

Skipped unless the visible device is a gfx950.
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

# (M, N, K): an aligned row-interleave ping-pong pick, the same body
# persistent (rerouted to one workgroup per tile), and the ragged-M / ragged-N
# padded picks (``rve_pad`` 2)
_SHAPES = [
    (4096, 4096, 512),
    (8192, 8192, 512),
    (4000, 4096, 1024),
    (4096, 4000, 1024),
]


def _launcher(kd):
    from rocke.core.ir import PtrType
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher

    sig = []
    for prm in kd.params:
        nm = getattr(prm.type, "name", "") or ""
        if isinstance(prm.type, PtrType):
            sig.append(
                {
                    "name": prm.name,
                    "type": nm if nm.startswith("ptr<") else f"ptr<{nm}>",
                }
            )
        else:
            sig.append({"name": prm.name, "type": str(nm)})
    art = compile_kernel(kd, arch="gfx950", capture_ir_text=False, backend="python")
    return KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("M,N,K", _SHAPES, ids=[f"{m}x{n}x{k}" for m, n, k in _SHAPES])
def test_fused_bias_relu_on_row_interleave_pick(M, N, K, dtype):
    import torch

    from rocke.core.ir import BF16, F16
    from rocke.dispatch.gemm import (
        GemmRequest,
        dispatch_gemm_bf16,
        dispatch_gemm_fp16,
    )
    from rocke.dispatch.gemm.common import fused_epilogue_spec
    from rocke.dispatch.gemm.fp16_rcr import _grid
    from rocke.helpers.fuse import BiasAdd, FusedEpilogue, ReLU
    from rocke.instances.common.gemm_universal import (
        build_universal_gemm,
        is_valid_spec,
    )
    from rocke.runtime.launcher import LaunchConfig

    disp = dispatch_gemm_bf16 if dtype == "bf16" else dispatch_gemm_fp16
    req = GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dtype)
    pick = disp(req).spec
    assert pick.trait.rilv_vec_epilogue, "the pick is not a row-interleave body"
    spec = fused_epilogue_spec(pick)
    assert spec is not None and spec.trait.rilv_vec_epilogue
    ir_t = BF16 if dtype == "bf16" else F16
    object.__setattr__(
        spec,
        "_fused_epilogue",
        FusedEpilogue(ops=(BiasAdd(dtype=ir_t), ReLU(dtype=ir_t)), dtype=ir_t),
    )
    ok, why = is_valid_spec(spec, arch="gfx950")
    assert ok, why
    kd = build_universal_gemm(spec, arch="gfx950")
    launch = _launcher(kd)

    tdt = torch.bfloat16 if dtype == "bf16" else torch.float16
    g = torch.Generator(device="cuda")
    g.manual_seed(11)
    A = (torch.rand(M, K, device="cuda", generator=g) * 2 - 1).to(tdt)
    B = (torch.rand(N, K, device="cuda", generator=g) * 2 - 1).to(tdt)
    bias = (torch.rand(N, device="cuda", generator=g) * 2 - 1).to(tdt)
    C = torch.full((M, N), float("nan"), dtype=tdt, device="cuda")
    vals = {"A": A, "B": B, "C": C, "M": M, "N": N, "K": K, "bias": bias}
    vals = {p.name: vals[p.name] for p in kd.params}
    grid = tuple(int(x) for x in _grid(spec, req))
    launch(vals, config=LaunchConfig(grid=grid, block=(spec.block_size, 1, 1)))
    torch.cuda.synchronize()

    u = 2.0**-7 if dtype == "bf16" else 2.0**-10
    acc = A.double() @ B.double().t()
    absb = A.double().abs() @ B.double().abs().t()
    ref = torch.relu(acc + bias.double())
    # the accumulator is rounded to the output type, the bias added there and
    # rounded again; plus the f32 accumulation error over K
    tol = (acc.abs() + bias.double().abs()) * (2 * u) + absb * (K * 2.0**-23) + 2.0**-24
    err = ((C.double() - ref).abs() / tol).nan_to_num(nan=float("inf"))
    worst = float(err.max())
    nbad = int((err > 1.0).sum())
    assert worst <= 1.0, f"{nbad} of {M * N} outputs wrong (worst {worst:.3g})"
