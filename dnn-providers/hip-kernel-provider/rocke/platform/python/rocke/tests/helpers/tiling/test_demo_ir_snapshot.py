# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""IR snapshot of the in-tree tiling demos.

Each tracked demo configuration is built (no GPU, no compile) and its printed IR, ``print_ir(kernel)``,
is hashed. The printed IR carries every op, operand, attribute and result type, so it also catches an
LDS allocation that changes shape (``str(kernel)`` does not show result types). The hash must match
the stored one, so any change to the IR a tiling demo emits -- through the demo itself or through the
tiling verbs, descriptors and layouts it is built from -- is caught here, even on a host with no GPU
to run the bit-exact demo tests.

Two kinds of configuration are tracked:

- global-only: data moves only between global memory and registers (the manual demo);
- LDS-staged: data passes through LDS (the LDS-staged, interleaved and CRC demos, and the
  bank-conflict probe kernels), one configuration per staging knob.

When a change is meant to move the IR, regenerate the hashes with ``python <this file>``, replace only
the entries of the configurations the change is meant to move, and say why in the commit message. An
entry that changes without a reason is a regression.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import pytest

pytest.importorskip(
    "rocke.core.ir", reason="rocke IR substrate required (build only; no GPU)"
)

from rocke.core.ir import F16
from rocke.core.ir_print import print_ir
from rocke.helpers.tiling import lds_conflict
from rocke.helpers.tiling.kernels.tiling_gemm_crc_demo.crc_interleaved_gemm import (
    _macro_coop_descs_crc,
    build_crc_gemm,
)
from rocke.helpers.tiling.kernels.tiling_gemm_interleaved_demo import (
    _transpose_desc,
    _wave_descs_interleaved,
    b32_swizzle,
    b64_swizzle,
    build_interleaved_gemm,
    build_lds_staged_gemm,
)
from rocke.helpers.tiling.kernels.tiling_gemm_manual_demo import (
    build_manual_gemm,
)

_ARCH = "gfx90a"
_SHAPE = (256, 256, 256)


def _interleaved(**kwargs):
    return build_interleaved_gemm(*_SHAPE, arch=_ARCH, **kwargs)[0]


def _crc(**kwargs):
    return build_crc_gemm(256, 256, 64, arch=_ARCH, **kwargs)[0]


def _probe(mode: str, *, lds_pad: int = 0, force_vw: int = 0):
    """A bank-conflict probe at the geometry the probe tests build: a 256 x 32 tile staged by 16 waves,
    each reading a 64-wide wave tile."""
    descs = lds_conflict.ProbeDescs.from_coop(
        _macro_coop_descs_crc(256, 32, 16),
        _wave_descs_interleaved(4, 4, 2)[0],
        transpose=_transpose_desc,
    )
    return lds_conflict.build_probe(
        descs,
        mode,
        tile_free=256,
        tile_k=32,
        n_waves=16,
        warp_free=64,
        dtype=F16,
        wave_size=lds_conflict.GFX90A.WAVE,
        lds_pad=lds_pad,
        lds_swizzle=False,
        force_vw=force_vw,
    )


# label -> builder of the configuration's KernelDef. Global-only configurations first.
_CONFIGS: dict[str, Callable[[], object]] = {
    "manual": lambda: build_manual_gemm(*_SHAPE, arch=_ARCH)[0],
    "lds_staged": lambda: build_lds_staged_gemm(*_SHAPE, arch=_ARCH)[0],
    "interleaved_1wave": lambda: _interleaved(),
    "interleaved_1wave_k32": lambda: _interleaved(tile_k=32),
    "interleaved_4waves": lambda: _interleaved(
        tile_m=128, tile_n=128, waves_m=2, waves_n=2
    ),
    "interleaved_canonical": lambda: _interleaved(
        prefetch="canonical", mma_layout="canonical"
    ),
    # Crossed prefetch / MMA layouts are built for single-atom wave tiles only.
    "interleaved_prefetch_canonical_mma_interleaved": lambda: _interleaved(
        tile_m=16, tile_n=16, prefetch="canonical"
    ),
    "interleaved_prefetch_interleaved_mma_canonical": lambda: _interleaved(
        tile_m=16, tile_n=16, mma_layout="canonical"
    ),
    "interleaved_single_buffer": lambda: _interleaved(single_buffer=True),
    "interleaved_reg_prefetch": lambda: _interleaved(
        tile_m=128, tile_n=128, waves_m=2, waves_n=2, reg_prefetch=True
    ),
    "interleaved_mac_prio": lambda: _interleaved(mac_prio=1),
    "interleaved_pad": lambda: _interleaved(lds_pad=8),
    "interleaved_canonical_pad": lambda: _interleaved(
        prefetch="canonical", mma_layout="canonical", lds_pad=8
    ),
    "interleaved_builtin_swizzle": lambda: _interleaved(lds_swizzle=True),
    "interleaved_b32_swizzle": lambda: _interleaved(lds_swizzle=b32_swizzle),
    "crc": lambda: _crc(),
    "crc_ab_swap": lambda: _crc(ab_swap=True),
    "crc_single_buffer": lambda: _crc(single_buffer=True),
    "crc_b64_swizzle": lambda: _crc(lds_swizzle=b64_swizzle),
    "probe_store": lambda: _probe("store"),
    "probe_read": lambda: _probe("read"),
    "probe_read_pad": lambda: _probe("read", lds_pad=8),
    "probe_store_narrow": lambda: _probe("store", force_vw=2),
}

_GOLDEN: dict[str, str] = {
    "manual": "094016f37977588064b7fa099032b4e9df5499c0b21bf74d66bf99c6712b094b",
    "lds_staged": "18a693845e286a2c8d2d02f16bb7ba3ad485be31b35a6b002ee0a10a59f1e286",
    "interleaved_1wave": "ac875b1d77f7d8f5d6289e82e44f1960a824a9a011ab72d911f9790f82a6079f",
    "interleaved_1wave_k32": "93479e930cf3d3ef990be9b2bc6d7d26abec1f062532f0026416d5f9e333b881",
    "interleaved_4waves": "70e282871e9507a8e1b5b1e4f13e4b81c5f7f8568e8ed8326b4a9e98effff919",
    "interleaved_canonical": "1bd4dc6f020539a30e2336ff2a2a09a3f4b447080cdc3488d8cff1f6d6ba0091",
    "interleaved_prefetch_canonical_mma_interleaved": "1e19e50a1e7856c80a6b4533888b00e71457046eea5eabe00e2c597b8920403f",
    "interleaved_prefetch_interleaved_mma_canonical": "808c75e6b27d1bde372c1aa45b41c421017139df628cda7584c07c163e086cbd",
    "interleaved_single_buffer": "264cd0004fc3c9ddb07cf66c24e8de0eecd87eb376d8677bbc296862aa538c20",
    "interleaved_reg_prefetch": "dd41e82ad4b9c5fc49f7a42960bcbf1da69eed42ac6dae33064842dcc4bfd125",
    "interleaved_mac_prio": "238eee6ea636068d8586f8248ff4ae3422d83e98f05f34d7856d6662691f65ff",
    "interleaved_pad": "48c95646e87554db7595169c7128cc3591fb449e61e39cb3a4ecaf406b0b7ca6",
    "interleaved_canonical_pad": "f5998b2f16aa74a4b8a147eb300d0acab5390b56aa3fcd8f9a66c71a5d5ee6fe",
    "interleaved_builtin_swizzle": "60b69901f63fcc2d004af98e1add7e74051f56859b2240d4f11f76cb53fb68d8",
    "interleaved_b32_swizzle": "c00a6273915ebf9e7d4489f65e7a3ceb0108d020ddd7d3aa4634f5963fa44a93",
    "crc": "b9c5d146801e7a48ad5d6470be755465f398268663d42843816c4f17e2a236dd",
    "crc_ab_swap": "9cf8e5bff6e93492c4d16ba191e8dc13353e56fff07f134f5856e4cf33d5779d",
    "crc_single_buffer": "2612f23500244c3252d2c41696a3fcce8e247739e260940ace90efb6ac5106a3",
    "crc_b64_swizzle": "c1e9ded2376e2ad449b4f8b814d11a9c4d24318773017078f2042e93268be18d",
    "probe_store": "f82ef22b284d43443ca7905f3183144486abc6c50cb94de05ff5de37a6d905cb",
    "probe_read": "8999a9482f4e0c29043762eef2cba93ec11a37968da9d8fd53853113c374c4da",
    "probe_read_pad": "a90a7c896f19e6a8defc2a2c42ccaf9d6947bbbb10ac91a7636909b2006a7147",
    "probe_store_narrow": "4c4279ac035ac18915439abc0710971edaced761e0aaef3afccc81aa03cde750",
}


def _ir_sha256(label: str) -> str:
    return hashlib.sha256(print_ir(_CONFIGS[label]()).encode()).hexdigest()


def test_every_tracked_configuration_has_a_stored_hash() -> None:
    assert sorted(_GOLDEN) == sorted(_CONFIGS), (
        "the stored hashes and the tracked configurations disagree: "
        f"no hash for {sorted(set(_CONFIGS) - set(_GOLDEN))}, "
        f"hash for no configuration {sorted(set(_GOLDEN) - set(_CONFIGS))}"
    )


@pytest.mark.parametrize("label", list(_CONFIGS))
def test_demo_ir_matches_its_snapshot(label: str) -> None:
    actual = _ir_sha256(label)
    assert actual == _GOLDEN.get(label), (
        f"the IR of tiling demo configuration {label!r} changed.\n"
        f"  stored sha256 {_GOLDEN.get(label)}\n"
        f"  actual sha256 {actual}\n"
        "If the change is intended, regenerate the hashes (`python <this file>`), replace this "
        "configuration's entry and give the reason in the commit message; otherwise find the change."
    )


if (
    __name__ == "__main__"
):  # regeneration helper: `python <this file>` prints fresh hashes
    print("_GOLDEN: dict[str, str] = {")
    for _label in _CONFIGS:
        print(f"    {_label!r}: {_ir_sha256(_label)!r},")
    print("}")
