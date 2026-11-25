# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""PAP next-tile prefetch must use the WorkGroupMapping that graWorkGroup uses.

With PrefetchAcrossPersistent (PAP), the current tile's NoLoadLoop issues the
first K group of the next tile's A/B loads. The next iteration's
``graWorkGroup`` derives WorkGroup0/1 from the tile identity and then applies
the WorkGroupMapping (WGM) remap. It skips its own first K group because PAP
already loaded it. If PAP derives WorkGroup0/1 without the WGM remap, a WGM != 1
kernel loads the first K group from one tile and the remaining K groups from
another.

The static StreamK path applied the remap. The dynamic StreamK setup and the
dynamic arm of the hybrid StreamK setup did not. With dynamic queues a
workgroup only gets a second, PAP-primed tile when it finishes before another
workgroup of its queue pops. That depends on timing, so the gfx950
``sk_sgemm_{dynamic,hybrid}_pap`` tests failed intermittently with a few dozen
wrong tiles.

The model tests below describe the remap and show that WGM=8 moves most tiles
of the test problems. The codegen tests assert the order of the emitted PAP
next-tile setup: tile identity, then the WGM remap, then the PAP loads, with no
jump from the identity past the remap.
"""

from pathlib import Path
import re

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_GFX950 = Path(__file__).resolve().parents[3] / "common" / "streamk" / "gfx950"
_CONFIGS = ("sk_sgemm_dynamic_pap.yaml", "sk_sgemm_hybrid_pap.yaml")


# ---------------------------------------------------------------------------
# Reference model of DefaultWGM (Components/WorkGroupMappingAlgos.py)
# ---------------------------------------------------------------------------

def _default_wgm(wg0, wg1, nwg0, nwg1, wgm):
    """Return the (WorkGroup0, WorkGroup1) that DefaultWGM writes."""
    if -1 <= wgm <= 1:
        # WGM in [0, 1] branches past the remap; WGM == -1 remaps by 1.
        return wg0, wg1
    if wgm > 1:
        first, second, n_second, width = wg0, wg1, nwg1, wgm
    else:
        first, second, n_second, width = wg1, wg0, nwg0, -wgm
    block = second // width
    serial = (second % width) * (nwg0 if wgm > 1 else nwg1) + first
    full_blocks = n_second // width
    remainder = n_second - full_blocks * width or width
    divisor = remainder if block >= full_blocks else width
    new_first = serial // divisor
    new_second = serial % divisor + block * width
    return (new_first, new_second) if wgm > 1 else (new_second, new_first)


# Tile grids of the CI problems: 4096x2048 and 4104x2056 with MT 128x128.
_GRIDS = ((32, 16), (33, 17))


@pytest.mark.parametrize("nwg0,nwg1", _GRIDS)
@pytest.mark.parametrize("wgm", (1, 3, 8, -8))
def test_model_wgm_is_a_tile_permutation(nwg0, nwg1, wgm):
    tiles = [(a, b) for b in range(nwg1) for a in range(nwg0)]
    mapped = {_default_wgm(a, b, nwg0, nwg1, wgm) for a, b in tiles}
    assert mapped == set(tiles)


@pytest.mark.parametrize("nwg0,nwg1", _GRIDS)
def test_model_wgm1_needs_no_remap(nwg0, nwg1):
    # Without a remap, PAP prefetches the right tile only when WGM is 1.
    assert all(_default_wgm(a, b, nwg0, nwg1, 1) == (a, b)
               for b in range(nwg1) for a in range(nwg0))


@pytest.mark.parametrize("nwg0,nwg1", _GRIDS)
def test_model_wgm8_moves_most_tiles(nwg0, nwg1):
    # The configs fork WorkGroupMapping [1, 8]. With WGM=8, a prefetch that skips
    # the remap targets a different tile for most PAP-primed work items.
    moved = sum(_default_wgm(a, b, nwg0, nwg1, 8) != (a, b)
                for b in range(nwg1) for a in range(nwg0))
    assert moved > 0.9 * nwg0 * nwg1


# ---------------------------------------------------------------------------
# Codegen: the PAP next-tile identity is remapped before the PAP loads
# ---------------------------------------------------------------------------

_WGM = "/* WGM Calculation */"
_PAP_LOADS = "Begin setupPrefetchAcrossPersistentLoads"
_LABEL = re.compile(r"^(label_\w+):")
_JUMP = re.compile(r"^s_(?:c?branch)\w* (label_\w+)")
# The static arm of the hybrid setup starts at its dispatch label.
_STATIC_ARM = re.compile(r"^label_\w*StaticPapSetup\w*:")


def _first_after(lines, start, pred):
    return next((i for i in range(start, len(lines)) if pred(lines[i].strip())), None)


@pytest.fixture(scope="module", params=_CONFIGS)
def kernels(request):
    emitted = emit_kernels_from_config(_GFX950 / request.param, arch="gfx950", limit=None)
    assert emitted and all(err == 0 for _, _, err in emitted)
    return request.param, emitted


def test_pap_next_tile_identity_is_remapped_before_the_pap_loads(kernels):
    config, emitted = kernels
    for base, src, _ in emitted:
        lines = src.splitlines()
        labels = {m.group(1): i for i, line in enumerate(lines) if (m := _LABEL.match(line.strip()))}
        # Dynamic setup (and the dynamic arm of the hybrid setup): identity from
        # the popped work item. Hybrid static arm: identity from the iteration.
        dynamic = [i for i, line in enumerate(lines) if "PAP: next tile work item (already popped)" in line]
        static = [i for i, line in enumerate(lines) if _STATIC_ARM.match(line.strip())]
        assert dynamic, f"{config}:{base}: no PAP dynamic next-tile identity"
        for start in dynamic + static:
            where = f"{config}:{base}: PAP next-tile identity at line {start + 1}"
            wgm = _first_after(lines, start, lambda l: l.startswith(_WGM))
            loads = _first_after(lines, start, lambda l: _PAP_LOADS in l)
            assert loads is not None, f"{where}: no PAP loads follow"
            assert wgm is not None and wgm < loads, (
                f"{where}: reaches the PAP loads without the WorkGroupMapping remap graWorkGroup applies")
            # No jump between the identity and the remap may land past the remap.
            for i in range(start, wgm):
                if m := _JUMP.match(lines[i].strip()):
                    target = labels[m.group(1)]
                    assert not (wgm < target), (
                        f"{where}: line {i + 1} jumps to {m.group(1)} past the WorkGroupMapping remap")


def test_graworkgroup_remaps_the_primed_work_item(kernels):
    # The requirement above mirrors graWorkGroup: the primed work item that the
    # next iteration reuses goes through the remap before its loads.
    config, emitted = kernels
    for base, src, _ in emitted:
        lines = src.splitlines()
        primed = [i for i, line in enumerate(lines) if "PAP: reuse work item popped by prior NLL" in line]
        assert primed, f"{config}:{base}: no primed work-item reuse"
        end = next(i for i, line in enumerate(lines) if "Local Read Addresses" in line)
        for start in primed:
            assert any(lines[i].strip().startswith(_WGM) for i in range(start, end)), (
                f"{config}:{base}: graWorkGroup applies no remap after the primed item")
