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
    "manual": "acbe9945a75a30e440fc98a0841d5682e575d5a097eeec5bc674c84244394eea",
    "lds_staged": "57522909eb2a6715abc26141e47d1d791ff39b092f0982eddecf4480ae912b59",
    "interleaved_1wave": "e31c9aaa13b708fa41c27064e70db8045f527d31d4079d6267cd6905050d69ef",
    "interleaved_1wave_k32": "6fcd62f2e69f7710bc449ec5cbe072d7cdbd8c2d98fe5f34cf164a75b9117440",
    "interleaved_4waves": "ea397339440801f2700f78aa295b007c8f3e647227fa0d8a0cd9a5bb3155c736",
    "interleaved_canonical": "448fbad87c7620388ba3d5488c9a36725e29b79c45e7e47de86dc2eb8ae47d7c",
    "interleaved_prefetch_canonical_mma_interleaved": "9e9fe81cb39e7b74061a3775aae0e6699764eb30343d6f296f35bb066d206f36",
    "interleaved_prefetch_interleaved_mma_canonical": "a4ba5fb94882eebc59ce7b36eb948d11d97949773b17abd259b2ed769d652aa4",
    "interleaved_single_buffer": "7090182c83407a86ebb999db8fbb1417730d446b98e0bf8658825dd308155a36",
    "interleaved_reg_prefetch": "94b684915826cae99d158e56dcfb94f23dffb79bfeeb01aafc602d6748e9bf00",
    "interleaved_mac_prio": "9abc68f5e1118f0f713fbd617a6e4f36b502fdcaea9e7f8310f956d855ebc383",
    "interleaved_pad": "057bdac0c602f4c55810202b5c1d6c2f07eef2dca38072e5101d07fa3f49a82b",
    "interleaved_canonical_pad": "664204f7c792ee065c8671929bafa2341693a810a2eb410d502d301a05981f0f",
    "interleaved_builtin_swizzle": "d17fb8da346ebb2a41a37828d3635566505c357dba96420d601c5783c7a7c923",
    "interleaved_b32_swizzle": "3c6ff8ef79442cfaa5392e8ddd7e100908bbd219d1e6eb30a5b2b3d768a37c4f",
    "crc": "0581e2ca741571db68fd50b446c3e7cc9cdda71dfed5ab0df98252b1362a263e",
    "crc_ab_swap": "0b83d86d3d4fee2d9a06daa1bd0e912f2f64c95975fbbd9c9dea1deedde30e2c",
    "crc_single_buffer": "e0c3a08f2cfdafe0382f14225222c2469760e1322d3e9896cd6ee940e475acc0",
    "crc_b64_swizzle": "a0c56e8c55a0dc1d2f7964a70a87517891f91078ab0dbe6283302945c2fd2c7f",
    "probe_store": "f6ce8b8af3083db6fde9f8bbbcf56b2b1a994c18b7b699eb31537622d81fa54a",
    "probe_read": "ebcb996e03fb6f49d124b1b1535907dc468b7627dd8906e9d79601003104cedc",
    "probe_read_pad": "d0379463a3efa3df9a6580c626bbc5ed8b8a406b545655bbc04e06972380be81",
    "probe_store_narrow": "48d73d4631e1f4a029ed80b326e9b91ed9b2dbab45791c91937063c974d16104",
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
