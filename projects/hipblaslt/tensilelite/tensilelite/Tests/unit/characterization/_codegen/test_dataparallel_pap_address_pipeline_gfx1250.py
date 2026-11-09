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
from tensilelite.Components.PersistentLoop import PersistentKernelState
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


def test_streamk_packed_preparation_preserves_identity_and_masks(tmp_path):
    """The packed path carries next addresses without losing current completion identity."""
    import yaml

    config = yaml.safe_load(_CONFIG.read_text())
    for entry in config["BenchmarkProblems"][0][1]["ForkParameters"]:
        if "TileProcessingStrategy" in entry:
            entry["TileProcessingStrategy"] = ["StreamK"]
        if "AssertSummationElementMultiple" in entry:
            entry["AssertSummationElementMultiple"] = [256]
    path = tmp_path / "streamk_packed.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    results = emit_kernels_from_config(path, limit=1, arch="gfx1250")
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        (tmp_path / "prepared.s").write_text(src)
        # Ignore encoding selectors while preserving the physical payload
        # register, then compare every identity/address lane at both ends.
        lines = re.sub(r"v\[(\d+)-\d+\]", r"v[\1]", src).splitlines()

        def lanes(op, tag):
            result = []
            for line in lines:
                if tag not in line:
                    continue
                code, _, comment = line.partition("//")
                assert code.startswith(op)
                args = [item.strip() for item in code[len(op):].split(",")]
                payload, scalar = args[:2] if op == "v_writelane_b32" else args[1::-1]
                result.append((payload, scalar, int(args[2]), comment.split(tag, 1)[1].strip()))
            return result

        saved = lanes("v_writelane_b32", "PAP pack current ")
        restored = lanes("v_readlane_b32", "PAP unpack current ")
        identity = {"WorkGroup0", "WorkGroup1", "WorkGroup2", "StreamKLocalStart", "StreamKLocalEnd"}
        assert {field for _, _, _, field in saved} == identity
        assert sorted(saved) == sorted(restored)
        assert len({(payload, lane) for payload, _, lane, _ in saved}) == 5
        # Later pipeline stages also retain descriptor words after addresses
        # are consumed; they are not part of the next-range identity.
        following = [move for move in lanes("v_writelane_b32", "PAP retain next ")
                     if move[3] in identity]
        activated = lanes("v_readlane_b32", "PAP unpack next ")
        assert set(following) == set(activated)
        assert {field for _, _, _, field in following} == identity
        addresses = lanes("v_writelane_b32", "PAP retain prepared StreamK address")
        consumed = lanes("v_readlane_b32", "PAP unpack prepared address")
        address_slots = {(payload, lane) for payload, _, lane, _ in addresses}
        assert len(address_slots) == 4
        assert address_slots == {(payload, lane) for payload, _, lane, _ in consumed}
        occupied = {(payload, lane) for payload, _, lane, _ in saved + following}
        assert address_slots.isdisjoint(occupied)
        assert len(occupied) == 10
        assert len({payload for payload, _, _, _ in saved + following + addresses}) == 1

        # The mapper restores both incoming masks before returning to compute.
        restore = [i for i, line in enumerate(lines) if "PAP mapper restore exact mask" in line]
        assert len(restore) == 2
        assert {lines[i].split()[1].rstrip(",") for i in restore} == {"exec_lo", "vcc_lo"}
        mapping = next(i for i, line in enumerate(lines) if "PAP mapper scratch uses full EXEC" in line)
        assert not any(line.startswith("v_wmma") for line in lines[mapping:max(restore)])
        assert max(map(int, re.findall(r"\.amdhsa_next_free_sgpr\s+(\d+)", src))) <= 106
