# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Keep prepared next-tile coordinates live through the complete scheduled NGLL.

The real MX increment module is cached before PAP reserves its carrier. Its
WrapU scratch used to overwrite the five-word carrier's free coordinates after
the mapper but before either address generation or the PAP consumer read them.
"""

from pathlib import Path
import re

import pytest

from rocisa.register import RegisterPool
from Tensile.Components.PersistentLoop import PersistentKernelState
from rocisa.enum import RegisterType
from config_harness import assert_assembles, assert_real_gfx1250_kernels, emit_kernels_from_config

pytestmark = pytest.mark.unit
_CONFIG = Path(__file__).parent / "data/test_data/_designed/gfx1250/dataparallel_pap_address_pipeline.yaml"


def _scalar_destinations(line):
    # Named SGPRs are reserved separately from the physical carrier checkout.
    code = line.split("//", 1)[0].strip()
    if re.match(r"s_(?:cmp|bitcmp|cbranch|branch|wait|barrier|store|buffer_store|setpc|rfe)", code):
        return set()
    match = re.match(r"[sv]_\w+\s+s(?:(\d+)|\[(\d+)(?::(\d+))?\])(?:,|\s|$)", code)
    if match is None:
        return set()
    first = int(match[1] or match[2])
    return set(range(first, int(match[3] or first) + 1))


@pytest.mark.parametrize("line, expected", [
    ("v_readfirstlane_b32 s80, v12", {80}),
    ("v_readlane_b32 s[80], v12, 4", {80}),
    ("s_mov_b64 s[80:81], 0", {80, 81}),
    ("s_cmp_eq_u32 s80, 0", set()),
    ("s_bitcmp1_b32 s[80], 7", set()),
    ("s_store_dword s80, s[0:1], 0", set()),
    ("s_setpc_b64 s[80:81]", set()),
])
def test_scalar_destinations_distinguish_writes_from_sources(line, expected):
    assert _scalar_destinations(line) == expected


@pytest.mark.parametrize("carrier_mode", ["five", "seven", "fragmented_fallback"])
def test_complete_ngll_preserves_pap_carrier(monkeypatch, carrier_mode):
    if carrier_mode == "five":
        checkout = RegisterPool.checkOutAligned

        def only_five(pool, count, alignment, tag="", *args, **kwargs):
            if count == 7 and tag == "PAP address carrier preflight":
                raise RuntimeError("exercise the existing five-word fallback")
            return checkout(pool, count, alignment, tag, *args, **kwargs)

        monkeypatch.setattr(RegisterPool, "checkOutAligned", only_five)
    elif carrier_mode == "fragmented_fallback":
        planner = PersistentKernelState.papAddressCarrierPlan

        def fragmented(writer, *args, **kwargs):
            real_pool = writer.sgprPool
            restricted = RegisterPool(105, RegisterType.Sgpr, defaultPreventOverflow=False)
            for start, count in ((90, 5), (99, 4)):
                restricted.add(start, count, "fragmented PAP preflight")
            before = (restricted.size(), str(restricted))
            writer.sgprPool = restricted
            try:
                assert planner(writer, *args, **kwargs) is None
                assert (restricted.size(), str(restricted)) == before
                return None
            finally:
                writer.sgprPool = real_pool

        monkeypatch.setattr(PersistentKernelState, "papAddressCarrierPlan", fragmented)

    results = emit_kernels_from_config(_CONFIG, limit=1, arch="gfx1250")
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        if carrier_mode == "fragmented_fallback":
            assert "PAP mapper save current EXEC" not in src
            assert "PAP borrow prepared next coordinate" not in src
            assert "checkpoint WorkGroup0 for PAP restore" in src
            continue

        lines = src.splitlines()
        start = next(i for i, line in enumerate(lines) if "PAP mapper save current EXEC" in line)
        restored = next(i for i in range(start, len(lines)) if "PAP mapper restore exact current EXEC" in lines[i])
        assert not any(re.match(r"(?:v_wmma|ds_|tensor_)", line.strip()) for line in lines[start:restored])
        compute = next(i for i in range(restored, len(lines)) if lines[i].startswith("v_wmma"))
        consume = next(i for i in range(compute, len(lines)) if "PAP borrow prepared next coordinate" in lines[i])
        coordinates = {int(match[1]) for line in lines[consume:] if "PAP borrow prepared next coordinate" in line
                       for match in [re.search(r", s(\d+)\s+//", line)] if match}
        assert len(coordinates) == 3
        clobbers = [line for line in lines[compute:consume] if _scalar_destinations(line) & coordinates]
        assert clobbers == [], "NGLL overwrites live PAP coordinates: " + repr(clobbers)
        assert any("PAP add free-tile offset" in line for line in lines[compute:consume])
        assert max(int(value) for value in re.findall(r"\.amdhsa_next_free_sgpr\s+(\d+)", src)) <= 106
