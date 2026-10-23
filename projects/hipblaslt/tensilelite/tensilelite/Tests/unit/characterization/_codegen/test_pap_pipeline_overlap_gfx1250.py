# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only assembly regressions for PAP synchronization and SGPR allocation."""
from pathlib import Path
import re

import pytest
import yaml
from config_harness import assert_assembles, assert_real_gfx1250_kernels, emit_kernels_from_config
from Tensile.Components.PersistentLoop import PersistentKernelState
from rocisa.register import RegisterPool

pytestmark = pytest.mark.unit
_BASE_CONFIG = (Path(__file__).resolve().parent / "data/test_data/_designed/gfx1250"
                / "dataparallel_pap_address_pipeline.yaml")


def _config(tmp_path, **overrides):
    """Derive one code-generation case from the existing tracked PAP fixture."""
    config = yaml.safe_load(_BASE_CONFIG.read_text())
    forks = config["BenchmarkProblems"][0][1]["ForkParameters"]
    for entry in forks:
        for key in entry.keys() & overrides.keys():
            entry[key] = [overrides[key]]
    path = tmp_path / "pap_codegen.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


@pytest.mark.parametrize("strategy", ["DataParallel", "StreamK"])
def test_carried_plr_completion_does_not_drain_second_tensor_group(tmp_path, strategy):
    results = emit_kernels_from_config(
        _config(tmp_path, AssertSummationElementMultiple=256, TileProcessingStrategy=strategy), limit=1, arch='gfx1250')
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        assert 'PAP group1 positive DepthU increment' in src
        assert 'PAP skip redundant first local reads' in src
        lines = src.splitlines()
        begin = next(i for i, line in enumerate(lines) if line.startswith('label_PapFirstPlrReady:'))
        end = next(i for i in range(begin, len(lines)) if 'PAP complete all carried first PLR' in lines[i])
        assert any('s_wait_dscnt 0' in line for line in lines[begin:end])
        assert not any('s_wait_tensorcnt' in line for line in lines[begin:end]), '\n'.join(lines[begin:end+1])

        # Group0 still acquires the current final producer before issuing any
        # next-tile work. Removing this wait corrupts the last current chunk.
        release0 = next(i for i, line in enumerate(lines) if 'PGR2 group0: release' in line)
        assert any('s_wait_tensorcnt 0' in line for line in lines[release0-8:release0])
        issue0 = next(i for i in range(release0, len(lines)) if 'tensor_load_to_lds' in lines[i])
        publish0 = next(i for i in range(issue0, len(lines)) if 'PGR2 group0 actually issued' in lines[i])
        assert release0 < issue0 < publish0

        # Group1 is a reader release, so group0 may remain outstanding. Native
        # scheduling must not move either tensor write ahead of its rendezvous.
        advance1 = next(i for i, line in enumerate(lines) if 'PAP group1 positive DepthU increment' in line)
        release1 = next(i for i in range(advance1, len(lines)) if 'PGR2 group1: release' in lines[i])
        publish1 = next(i for i in range(release1, len(lines)) if 'PGR2 group1 actually issued' in lines[i])
        issues1 = [i for i in range(advance1, publish1) if 'tensor_load_to_lds' in lines[i]]
        assert len(issues1) == 2 and all(i > release1 for i in issues1)
        assert not any('s_wait_tensorcnt' in line for line in lines[advance1:publish1])
        # The first next-tile PLR reads survive store/setup until next entry's
        # consumer. An unconditional DS drain here loses that overlap.
        ready0 = next(i for i, line in enumerate(lines) if 'PAP PLR waits group0' in line)
        first_read = next(i for i in range(ready0, len(lines)) if 'ds_load_' in lines[i])
        kernel_end = next(i for i in range(first_read, len(lines)) if 's_endpgm' in lines[i])
        assert not any('s_wait_dscnt' in line for line in lines[first_read:kernel_end])
        persistent_entry = next(i for i, line in enumerate(lines) if line.startswith('label_PersistentLoopStart:'))
        assert not any('s_wait_dscnt' in line for line in lines[persistent_entry:begin])


@pytest.mark.parametrize("prepare", [False, True])
def test_static_streamk_halfplr_pap_fits_scalar_budget_and_returns_to_all_copies(tmp_path, prepare):
    results = emit_kernels_from_config(
        _config(tmp_path, TileProcessingStrategy='StreamK', HalfPLR=1, SuppressNoLoadLoop=True,
                AssertSummationElementMultiple=256 if prepare else 32),
        limit=1, arch='gfx1250')
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        assert max(map(int, re.findall(r'\.amdhsa_next_free_sgpr\s+(\d+)', src))) <= 106
        assert 'restore HalfPLR loop-copy selector after PAP' in src
        if prepare:
            assert 'PAP retain prepared StreamK address' in src
            assert 'PAP publish durable next coordinates for entry' in src
            assert 'PAP activate retained StreamK coordinate' in src
            assert 'PAP unpack current StreamKLocalEnd' in src
        else:
            assert 'checkpoint StreamKLocalStart for PAP restore' in src
            assert 'restore current StreamKLocalEnd after PAP' in src
        assert 'PAP group1 positive DepthU increment' not in src
        for copy in range(3):
            assert f'select branch-back label for HalfPLR loop copy {copy}' in src


@pytest.mark.parametrize("cluster", [[1, 1], [2, 4]])
def test_prepared_entry_coordinates_survive_epilogue(tmp_path, monkeypatch, cluster):
    reclaimed, protected = [], {}
    original = PersistentKernelState.papReclaimValuRegisters
    original_add = RegisterPool.add

    def check_range(pool, start, size):
        for writer, held in protected.items():
            if pool is writer.vgprPool:  # includes copied pools used by store branches
                overlap = held.intersection(range(start, start + size))
                assert not overlap, f"scratch exposes retained PAP registers: {sorted(overlap)}"

    def add(pool, start, size, tag=""):
        # Observe raw pool additions too: a caller bypassing the PAP helper
        # must not escape this test.
        check_range(pool, start, size)
        return original_add(pool, start, size, tag)

    def checked_checkout(operation):
        def checkout(pool, size, *args, **kwargs):
            start = operation(pool, size, *args, **kwargs)
            check_range(pool, start, size)
            return start
        return checkout

    def reclaim(writer, kernel, start, size, tag, **kwargs):
        ranges = writer.papLocalReadPayloadRanges(kernel)
        assert len(ranges) == 4  # A, B, MXSA, MXSB in this fixture
        payload = writer.states.papNextTileVgpr
        assert payload >= 0
        held = {reg for first, count in ranges for reg in range(first, first + count)} | {payload}
        protected[writer] = held
        original(writer, kernel, start, size, tag, **kwargs)
        slots = writer.vgprPool.getPool()
        assert all(slots[reg].status != RegisterPool.Status.Available for reg in held)
        reclaimed.append((tag, kwargs.get("add", True)))

    monkeypatch.setattr(RegisterPool, "add", add)
    for name in ("checkOut", "checkOutAligned"):
        monkeypatch.setattr(RegisterPool, name, checked_checkout(getattr(RegisterPool, name)))
    monkeypatch.setattr(PersistentKernelState, "papReclaimValuRegisters", reclaim)
    results = emit_kernels_from_config(
        _config(tmp_path, AssertSummationElementMultiple=256, ClusterDim=cluster),
        limit=1, arch="gfx1250")
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        for tag in ("ValuAB", "ValuMXSAB"):
            assert reclaimed.count((tag, True)) >= 2  # setup and store
            assert (tag, False) in reclaimed  # return scratch before compute
        lines = re.sub(r"-(?:256|512|768)(?=[:\]])", "", src).splitlines()
        saves = [i for i, line in enumerate(lines) if "PAP retain next coordinate through store" in line]
        assert len(saves) == 3
        payloads = {int(re.search(r"v_writelane_b32\s+v\[?(\d+)\]?", lines[i])[1]) for i in saves}
        assert len(payloads) == 1
        payload = payloads.pop()
        assert int(re.search(r"\.set vgprSerial, (\d+)", src)[1]) == payload + 1
        restores = [line for line in lines if "PAP activate retained WorkGroup" in line]
        assert len(restores) == 3
        assert all(re.search(rf"v\[?{payload}\]?,", line) for line in restores)
        assert "PAP skip repeated workgroup mapping" in src
        assert "PAP consume retained coordinate handoff" in src
        assert "PAP prepared coordinates belong to an issued tile?" in src
        # The reserved scalar-lane payload must not enter epilogue scratch or
        # any later compute write range before the persistent backedge.
        descriptor_saves = [i for i, line in enumerate(lines) if "PAP retain next K0 descriptor" in line]
        last_save = max(saves + descriptor_saves)
        guard = next(i for i in range(last_save, len(lines))
                     if "Next assignment data already issued?" in lines[i])
        assert re.search(r"s_bitcmp1_b32\s+s\[sgprPersistentPrefetchState\],\s*0", lines[guard])
        assert "s_cbranch_scc1 " in lines[guard + 1]
        target = lines[guard + 1].split("//")[0].split()[-1]
        skip = lines.index(target + ":")
        labels = {line[:-1]: i for i, line in enumerate(lines) if line.endswith(":")}
        # Snapshot publication leaves bit0 set. ORs preserve it; reserve's
        # conditional write cannot run when the state is already nonzero.
        for i in range(last_save + 1, guard):
            code = lines[i].split("//")[0].strip()
            if re.match(r"s_\w+\s+s\[sgprPersistentPrefetchState\],", code):
                if code.startswith(("s_cmp", "s_bitcmp")):
                    continue
                if code.startswith("s_cmov_b32"):
                    assert "s_cmp_eq_u32 s[sgprPersistentPrefetchState], 0 " in lines[i - 1]
                else:
                    assert code.startswith("s_or_b32") and ", s[sgprPersistentPrefetchState]," in code
            branch = re.match(r"s_(?:branch|cbranch_\w+)\s+(\w+)", code)
            if branch and branch[1] in labels:
                assert not guard + 1 < labels[branch[1]] < skip
        for i, line in enumerate(lines[last_save + 1:], last_save + 1):
            match = re.match(r"\w+\s+v(?:(\d+)|\[(\d+)(?::(\d+))?\]),", line.strip())
            if match:
                destinations = ({int(match[1])} if match[1] is not None
                                else set(range(int(match[2]), int(match[3] or match[2]) + 1)))
                if payload in destinations:
                    # Only the unissued fallback may reuse snapshot lanes for
                    # its temporary identity; the guard excludes primed tiles.
                    assert guard + 1 < i < skip and line.startswith("v_writelane_b32"), line
                    assert int(line.split("//")[0].split(",")[-1]) in (8, 9, 10, 11), line
        # These matched kernels used 98/92 SGPRs before address overlap.
        assert max(map(int, re.findall(r"\.amdhsa_next_free_sgpr\s+(\d+)", src))) <= (98 if cluster == [1, 1] else 92)
        # Local-address scratch reuses consumed X1 operands, not fresh VGPRs.
        assert max(map(int, re.findall(r"\.amdhsa_next_free_vgpr\s+(\d+)", src))) <= 680
        if cluster != [1, 1]:
            assert "PAP retain next phantom identity through store" in src
            assert "PAP activate retained PersistentPhantomTile" in src


def test_tail_entry_keeps_ordinary_tile_setup(tmp_path):
    results = emit_kernels_from_config(_config(tmp_path), limit=1, arch="gfx1250")
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        assert "PAP activate retained" not in src
        assert "PAP publish durable next coordinates" not in src


def test_prepared_entry_does_not_publish_when_carrier_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(PersistentKernelState, "papAddressCarrierPlan", lambda *args, **kwargs: None)
    results = emit_kernels_from_config(
        _config(tmp_path, AssertSummationElementMultiple=256), limit=1, arch="gfx1250")
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        assert "PAP retain next coordinate through store" not in src
        assert "PAP publish durable next coordinates" not in src
        assert "checkpoint WorkGroup0 for PAP restore" in src
