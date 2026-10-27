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
    "manual": "2710b4ff4685b0573b24b0f19d024d0682de286d735c1e985243922c13561945",
    "lds_staged": "9d61b9f9c6ac6c75f7f7bda2defcda7b7dd5861a10a372ca83d2c598f2659a07",
    "interleaved_1wave": "448bd23ce39ad6a9ca7bd7b0c43991ef41d985b92b0062f7de9274f25f493f33",
    "interleaved_1wave_k32": "64d6a0cacd7df3176ec2fa21f89926de813501bcbe1a7a8e804f5af7d865a983",
    "interleaved_4waves": "9b24437197675a837d27e33d2747cf1115a89a936d2cff32099f36281b1bec5c",
    "interleaved_canonical": "c97b279ed4cb02bdf5ec1fb37d02f4a823c052e2f51771b1e5d5931bc6e7df06",
    "interleaved_prefetch_canonical_mma_interleaved": "2252e0cff742ecb922b9675737186624a5109608d785c87b793ffaddedeab4fd",
    "interleaved_prefetch_interleaved_mma_canonical": "d19ee83efd28f351fc1bc9968d1086ac1c1bd9f14a92eff552a98cc09af913cc",
    "interleaved_single_buffer": "4d7dd975c7ad2eae742630a4dc42c375ba32a3098fb959f55d77961a17a7df9f",
    "interleaved_reg_prefetch": "b2524aa7e7689d8daf9f4c59215036d2952ffd91892c86e1a985dc1ce29ce91a",
    "interleaved_mac_prio": "8965fcc52d27dcafd881e9523e85fb92242f15acc5591f87648a8ec279b66e2d",
    "interleaved_pad": "cf6a4e0ff1048d73296fcce81054dc4347d64924816fa93f9588f1c6fd1d95a8",
    "interleaved_canonical_pad": "ec4786c60c9531c04dda2e86230c1a41355a03b26096adb1e1e1bf84882948b9",
    "interleaved_builtin_swizzle": "79d348d09146233a422e780acc513429d3aacf779be6820f77cf24da9ea8cdd2",
    "interleaved_b32_swizzle": "924910ba15ec0a8a2eb3d54c6f8295ff0710ebe321a1546c69607c7f99141c3f",
    "crc": "42de250461dc7895b96fa01137ef3e8b0cf96d8f5f6b568b1a5e31441e95b8df",
    "crc_ab_swap": "b787f3569095eda682cd57fdfe08d6bb7a86401b7530aa692fef0e47d3c4c72d",
    "crc_single_buffer": "22893fa0a99d66569d33845385c04d401d4e9edc2be4cff604a0ff0892f46466",
    "crc_b64_swizzle": "698c999e1eb4f57aaa04a2bec846671f8230600f5fda8efb4c6df5a9b25faa71",
    "probe_store": "41e9d612e0c0a3479cbcedd5457f2a4ccc6b16bc109c381324837e6c57bb1595",
    "probe_read": "1ab980dfd8a2e7860c9d912e0491ae5cfb2ae307ed121b8e5411ce131ef91707",
    "probe_read_pad": "47cc67c8cacb30f1558d5c79f44cbac0d3e60274f3a196219224529717614dcc",
    "probe_store_narrow": "207741387057f6ba2eca9863ffa7862c15964311b70606ab187f2ab068337c71",
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
