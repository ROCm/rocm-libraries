# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tensor pack kernel (``instances.common.tensor.pack``): form selection and
validity on the CPU; on a gfx950 device every form is checked bit-exactly
against ``torch.permute(...).contiguous()``, including ragged tiles."""

from __future__ import annotations

import pytest
from rocke.instances.common.tensor.pack import (
    TensorPackSpec,
    build_tensor_pack,
    is_valid_spec,
    tensor_pack_grid,
)

# P2-style operand: M0 (10, fastest) interleaved with K modes
_SRC = ("K2", "M2", "K1", "M1", "K0", "M0")
_EXT = {"K0": 6, "K1": 4, "K2": 4, "M0": 10, "M1": 8, "M2": 6}
_LOCAL = ("K2", "M2", "K1", "K0", "M1", "M0")  # free run M0 x M1 = 80
_KC = ("M2", "M1", "M0", "K2", "K1", "K0")  # K-contiguous rows
_KCR = ("M2", "M1", "M0", "K0", "K1", "K2")  # K-contiguous, K order reversed


def test_forms():
    assert TensorPackSpec.make(_SRC, _LOCAL, _EXT).resolved_form() == "pairs"
    t = TensorPackSpec.make(_SRC, _KC, _EXT)
    assert t.resolved_form() == "tiled"
    assert t.tiled_groups() == (("M2", "M1", "M0"), ("K2", "K1", "K0"))
    assert TensorPackSpec.make(_SRC, _KCR, _EXT).resolved_form() == "tiled"
    # K modes may sit in the outer group too
    t2 = TensorPackSpec.make(_SRC, ("M2", "M1", "K2", "M0", "K1", "K0"), _EXT)
    assert t2.tiled_groups() == (("M2", "M1", "K2", "M0"), ("K1", "K0"))
    # an odd unit-stride run: no 4-byte pairs -> elements
    e = TensorPackSpec.make(_SRC, _KC, dict(_EXT, M0=5))
    assert e.resolved_form() == "elements"


def test_validity():
    assert is_valid_spec(TensorPackSpec.make(_SRC, _KC, _EXT)) == (True, "ok")
    assert not is_valid_spec(TensorPackSpec.make(_SRC, _KC[:-1], _EXT))[0]
    assert not is_valid_spec(TensorPackSpec.make(_SRC, _KC, _EXT, form="pairs"))[0]
    assert not is_valid_spec(
        TensorPackSpec.make(_SRC, _LOCAL, _EXT, block=128, form="tiled")
    )[0]
    odd = {"K0": 3, "K1": 3, "K2": 3, "M0": 5, "M1": 7, "M2": 3}
    assert not is_valid_spec(TensorPackSpec.make(_SRC, _KC, odd))[0]  # numel % 8


def test_grid():
    t = TensorPackSpec.make(_SRC, _KC, _EXT)
    # O = 480 rows, I = 96 -> 2 x 8 tiles
    assert tensor_pack_grid(t) == (2, 8, 1)
    p = TensorPackSpec.make(_SRC, _LOCAL, _EXT)
    assert tensor_pack_grid(p) == (-(-p.numel() // (256 * 2 * 8)), 1, 1)


@pytest.mark.parametrize(
    "dst", [_LOCAL, _KC, _KCR], ids=["pairs", "tiled", "tiled_rev"]
)
def test_builds(dst):
    kd = build_tensor_pack(TensorPackSpec.make(_SRC, dst, _EXT))
    assert [p.name for p in kd.params] == ["X", "Y"]


# ------------------------------------------------------------------------------------------ device
def _arch():
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch(0)
    except Exception:  # noqa: BLE001 - no runtime / no device
        return None


@pytest.mark.skipif(_arch() != "gfx950", reason="needs a gfx950 device")
@pytest.mark.parametrize(
    "dst,form",
    [
        (_LOCAL, "auto"),
        (_KC, "auto"),
        (_KCR, "auto"),
        (_KC, "elements"),
        (("M2", "M1", "K2", "M0", "K1", "K0"), "auto"),
        (("M2", "M1", "K2", "M0", "K1", "K0"), "elements"),
    ],
)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_device_matches_torch_permute(dst, form, dtype):
    torch = pytest.importorskip("torch")
    from rocke.core.ir import PtrType
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    # ragged tiles on both tile axes of the tiled form (O = 1050, I = 104)
    ext = {"K0": 13, "K1": 2, "K2": 4, "M0": 10, "M1": 15, "M2": 7}
    sp = TensorPackSpec.make(_SRC, dst, ext, dtype=dtype, form=form)
    ok, why = is_valid_spec(sp)
    assert ok, why
    kd = build_tensor_pack(sp)
    art = compile_kernel(kd, arch="gfx950", capture_ir_text=False, backend="python")
    sig = []
    for prm in kd.params:
        nm = getattr(prm.type, "name", "") or ""
        sig.append(
            {"name": prm.name, "type": nm if nm.startswith("ptr<") else f"ptr<{nm}>"}
            if isinstance(prm.type, PtrType)
            else {"name": prm.name, "type": str(nm)}
        )
    L = KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)
    tdt = torch.float16 if dtype == "fp16" else torch.bfloat16
    X = torch.randn([ext[m] for m in _SRC], device="cuda").to(tdt)
    Y = torch.full([ext[m] for m in dst], float("nan"), device="cuda", dtype=tdt)
    from rocke.instances.common.tensor.pack import tensor_pack_block

    L(
        {"X": X, "Y": Y},
        config=LaunchConfig(grid=tensor_pack_grid(sp), block=tensor_pack_block(sp)),
    )
    torch.cuda.synchronize()
    ref = X.permute([_SRC.index(m) for m in dst]).contiguous()
    assert torch.equal(Y.view(torch.int16), ref.view(torch.int16))


def test_tiled_needs_4_byte_aligned_pairs():
    # a padded source: rows of 4 unit-stride elements at an odd row stride, so
    # element pairs are contiguous but not 4-byte aligned -> no tile
    sp = TensorPackSpec.make(
        ("K", "Mb", "Ma"),
        ("Mb", "Ma", "K"),
        {"K": 8, "Mb": 4, "Ma": 4},
        strides_src={"K": 40, "Mb": 5, "Ma": 1},
    )
    assert sp.tiled_groups() is None
    assert sp.resolved_form() == "elements"
    even = TensorPackSpec.make(
        ("K", "Mb", "Ma"),
        ("Mb", "Ma", "K"),
        {"K": 8, "Mb": 4, "Ma": 4},
        strides_src={"K": 40, "Mb": 6, "Ma": 1},
    )
    assert even.resolved_form() == "tiled"
