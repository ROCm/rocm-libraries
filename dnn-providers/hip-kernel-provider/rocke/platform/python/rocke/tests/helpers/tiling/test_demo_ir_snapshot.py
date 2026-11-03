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
    "manual": "edcd6e0afda22fc8adab78bcc78c2bf89a850b7c0fe3dd684a23429a3e0056ce",
    "lds_staged": "56c94d2b977a32881d06486ecba5306f15d543c6701b701380f966aa13285890",
    "interleaved_1wave": "d1bbded563e193de6cb250b5aeebd1c76dc0dac610437ab676a088af23d20485",
    "interleaved_1wave_k32": "804b4bd4cac5f360d37f935d39ff8370013f75d5cdaa14ab43180b052b48fcbf",
    "interleaved_4waves": "d3ecb96f117ab4eb29b1cb4b40c5bb9d612a4604d1fa27fdf4c9b91063c39367",
    "interleaved_canonical": "bacb079071ea6e85255a2c9a028088b91cc44f1f183d216635be75e8958b5ab9",
    "interleaved_prefetch_canonical_mma_interleaved": "914516d7c3e03f30434df9de138ba44033183fd3832d85c10e741aab93ee1f83",
    "interleaved_prefetch_interleaved_mma_canonical": "fb558edc59d0f1df42ccbc997c8972bdbfc0a99cde1790aa057396456042dc4b",
    "interleaved_single_buffer": "d205979096c0195dbeb9f847a393c1549042f1d0037865a89b6a6e800aeaa499",
    "interleaved_reg_prefetch": "3ae050d4db30a75090bd4113e5854e01ed7ab58cf9ff8cc1d2bb43d3be6362b4",
    "interleaved_mac_prio": "cce99d334b0e21e94e84964a8fb49ad15fcbc176c04205857c45a5bfa5bbf253",
    "interleaved_pad": "ddea73168a9be1a76337c2a75d95fa4cdb4d733696a86fe11d0277b9c187fbfa",
    "interleaved_canonical_pad": "aa54b00139a46babc47120f4ad88ffeb4de926ff5dd9d91f01bc3b7963150360",
    "interleaved_builtin_swizzle": "6fa6cb2f4782d8b7be6843879b4f1d651421d1c88b175e94a6edb2c030eb904f",
    "interleaved_b32_swizzle": "30b5fa8153de040d801c9f40977a19baaeda4023b56ff6567b4c5fa067823501",
    "crc": "a3c86d7495268db821a9f8f415db0c57a2cbf6b8574903de0ab00fc3d6d3f504",
    "crc_ab_swap": "15cd57ecf4ab6992093d5b03db78fdaa1161ad7019f4f565dad0f2232ddafc91",
    "crc_single_buffer": "70c29b35206d6d4a7d402269ead1f9d878e35a6152b2525d1b8b440aa27fb127",
    "crc_b64_swizzle": "e214067d349a336d4f93aa86f7fb005d8916b4a0cc65442a0d024735aff89e0b",
    "probe_store": "523e5531da73a0285134c298339d163d0b6c4739a15a834dcbb70a23ed33f470",
    "probe_read": "12cb15e2273823ba0180a43c5779abfd867bcf5d4806e3f3ffd2d18eef7b44f9",
    "probe_read_pad": "8c5d306a7ca6474fff604e076d8128ed0e118a48114d83d38e92cf4ed9146c35",
    "probe_store_narrow": "153d632dac1149cd895cbb3f11c8ac9034230bebc93dfa9e8c8ac9b3aa6811e5",
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
