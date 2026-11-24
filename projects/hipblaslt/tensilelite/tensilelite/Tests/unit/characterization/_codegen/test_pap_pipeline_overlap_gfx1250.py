# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only assembly regressions for PAP synchronization and SGPR allocation."""
from pathlib import Path
import re

import pytest
import yaml
from config_harness import assert_assembles, assert_real_gfx1250_kernels, emit_kernels_from_config
from tensilelite.Components.PersistentLoop import PersistentKernelState
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


def _assert_carried_plr_reentry_handoff(lines, first_read):
    close = next(i for i, line in enumerate(lines) if line == 'label_PersistentLoopClose:')
    end = next(i for i in range(close, len(lines)) if lines[i].startswith('s_endpgm'))
    assert first_read < close
    assert not any('s_wait_dscnt' in line for line in lines[first_read:close])
    assert not any('s_wait_tensorcnt' in line for line in lines[close:end])

    # Develop's closing LDS fence follows the stores and only runs when there
    # is another tile. It completes carried PLR without draining tensor group1.
    def unique(prefix):
        matches = [i for i in range(close, end) if lines[i].startswith(prefix)]
        assert len(matches) == 1, (prefix, matches)
        return matches[0]

    compare = unique('s_cmp_ge_u32 ')
    exhausted = unique('s_cbranch_scc1 label_StaticGrid_CloseDone')
    drain = unique('s_wait_dscnt ')
    assert lines[drain].split('//', 1)[0].split() == ['s_wait_dscnt', '0']
    signal = unique('s_barrier_signal -1')
    wait = unique('s_barrier_wait -1')
    backedge = unique('s_setpc_b64 ')
    assert 'branch to label_PersistentLoopStart' in lines[backedge]
    done = unique('label_StaticGrid_CloseDone:')
    assert compare < exhausted < drain < signal < wait < backedge < done
    return close, drain, signal, wait


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
        # Retained reads overlap the stores; the required closing LDS fence
        # completes them immediately before reentry without draining group1.
        ready0 = next(i for i, line in enumerate(lines) if 'PAP PLR waits group0' in line)
        first_read = next(i for i in range(ready0, len(lines)) if 'ds_load_' in lines[i])
        close, drain, signal, wait = _assert_carried_plr_reentry_handoff(lines, first_read)
        # Corrupt real emitted instructions to prove each side of that boundary.
        corruptions = [
            lines[:first_read+1] + ['s_wait_dscnt 0'] + lines[first_read+1:],
            lines[:close+1] + ['s_wait_tensorcnt 0'] + lines[close+1:],
            lines[:drain] + lines[drain+1:],
            [line for i, line in enumerate(lines) if i not in (signal, wait)],
        ]
        for corrupted in corruptions:
            with pytest.raises(AssertionError):
                _assert_carried_plr_reentry_handoff(corrupted, first_read)
        persistent_entry = next(i for i, line in enumerate(lines) if line.startswith('label_PersistentLoopStart:'))
        assert not any('s_wait_dscnt' in line for line in lines[persistent_entry:begin])


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
    # The VGPR ceiling below assumes a fixed store batch: an uncapped batch
    # takes every register PAP leaves free.
    results = emit_kernels_from_config(
        _config(tmp_path, AssertSummationElementMultiple=256, ClusterDim=cluster,
                NumElementsPerBatchStore=8),
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


def _vgpr_evaluator(src):
    raw = {}
    for match in re.finditer(r"^\.set (\w+), (.+)$", src, re.M):
        raw.setdefault(match[1], match[2].strip())

    def evaluate(expr):
        total = 0
        for sign, token in re.findall(r"([+-]?)\s*(\w+)", expr):
            value = int(token, 0) if token[0].isdigit() else evaluate(raw[token])
            total += -value if sign == "-" else value
        return total
    return evaluate


def _vgpr_destinations(line, evaluate):
    # Drop the s_set_vgpr_msb bias so every operand names its physical VGPR.
    code = re.sub(r"-(?:256|512|768)(?=[:\]+])", "", line.split("//")[0]).strip()
    match = re.match(r"(v_\w+|ds_load\w*|global_load\w*|buffer_load\w*)\s+(v\[[^\]]+\]|v\d+)", code)
    if not match or match[1].startswith(("v_cmp", "v_readlane", "v_readfirstlane")):
        return set()
    operand = match[2]
    if operand.startswith("v["):
        bounds = operand[2:-1].split(":")
        return set(range(evaluate(bounds[0]), evaluate(bounds[-1]) + 1))
    return {int(operand[1:])}


@pytest.mark.parametrize("wave_tile", [[4, 8], [4, 4]], ids=["miwt4x8", "miwt4x4-mx-interleave"])
def test_carried_first_plr_operands_survive_epilogue(tmp_path, wave_tile):
    # MIWT4x4 shares one MX-scale region between MXSA and MXSB, so the MXSB
    # X0 block does not start at its startVgprValu.
    config = yaml.safe_load(_config(tmp_path, AssertSummationElementMultiple=256,
                                    VectorWidthB=wave_tile[1]).read_text())
    group = config["BenchmarkProblems"][0][1]["ForkParameters"][-1]["Groups"][0][0]
    group["MatrixInstruction"] = [16, 16, 128, 1, 1, *wave_tile, 2, 2]
    path = tmp_path / "pap_wave_tile.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    results = emit_kernels_from_config(path, limit=1, arch="gfx1250")
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        shared = ".set vgprValuMXSB_X0_I0_BASE, vgprMXSBase+0" in src
        assert shared == (wave_tile == [4, 4])
        evaluate = _vgpr_evaluator(src)
        lines = src.splitlines()
        issued = next(i for i, line in enumerate(lines) if "PAP record first PLR operands issued" in line)
        first = max(i for i in range(issued) if "PAP both next groups issued?" in lines[i])
        carried = set()
        for line in lines[first:issued]:
            if line.startswith("ds_load"):
                carried |= _vgpr_destinations(line, evaluate)
        assert carried
        backedge = next(i for i in range(issued, len(lines))
                        if "branch to label_PersistentLoopStart" in lines[i])
        clobbers = [line for line in lines[issued + 1:backedge]
                    if _vgpr_destinations(line, evaluate) & carried]
        assert not clobbers, "\n".join(clobbers[:8])


def test_tail_entry_keeps_ordinary_tile_setup(tmp_path):
    results = emit_kernels_from_config(_config(tmp_path), limit=1, arch="gfx1250")
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        assert "PAP activate retained" not in src
        assert "PAP publish durable next coordinates" not in src


def _assert_classic_pap_overlaps_nll(lines):
    marker = next(i for i, line in enumerate(lines) if 'PAP descriptor in high LDS bank' in line)
    issues = [i for i in range(marker) if lines[i].startswith('tensor_load_to_lds ')]
    issue = issues[-2]  # data and MX-scale producers
    # WAR: peers' reads of the refilled bank complete before PAP overwrites it.
    wait = max(i for i in range(issue) if lines[i].startswith('s_barrier_wait -1'))
    signal = max(i for i in range(wait) if lines[i].startswith('s_barrier_signal -1'))
    assert not any(line.startswith(('ds_', 'tensor_')) for line in lines[signal:issue])
    nll_end = next(i for i in range(marker, len(lines)) if lines[i].startswith('label_toPGR1end_OrdNLL:'))
    assert not any(line.startswith('s_wait_tensorcnt') for line in lines[issue:nll_end])
    entry = next(i for i, line in enumerate(lines) if line.startswith('label_PersistentLoopStart:'))
    first_read = next(i for i in range(entry, len(lines)) if lines[i].startswith('ds_load_'))
    assert any(line.startswith('s_wait_tensorcnt') for line in lines[entry:first_read])
    return issue, signal, wait, nll_end, entry, first_read


def test_classic_pap_load_is_not_drained_in_no_load_loop(tmp_path, monkeypatch):
    # The NLL reads one bank while PAP refills the other. A PAP load tagged with
    # the NLL's token makes the token planner drain it before the NLL's reads.
    # The runtime tail split would move this tail kernel to the pipeline path.
    monkeypatch.setattr(PersistentKernelState, "papPgr2RuntimeTailSplit", staticmethod(lambda kernel: False))
    results = emit_kernels_from_config(_config(tmp_path), limit=1, arch='gfx1250')
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        lines = src.splitlines()
        issue, signal, wait, nll_end, entry, first_read = _assert_classic_pap_overlaps_nll(lines)
        corruptions = [
            [line for i, line in enumerate(lines) if i not in (signal, wait)],
            lines[:nll_end] + ['s_wait_tensorcnt 0'] + lines[nll_end:],
            [line for i, line in enumerate(lines)
             if not (entry < i < first_read and line.startswith('s_wait_tensorcnt'))],
        ]
        for corrupted in corruptions:
            with pytest.raises(AssertionError):
                _assert_classic_pap_overlaps_nll(corrupted)


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


def _bf16_config(tmp_path, cluster):
    # Reuse the tracked non-MX fixture with one no-tail, retained-PLR solution.
    base = (Path(__file__).resolve().parents[3] / "common/streamk/gfx1250"
            / "data_parallel_cluster_multicast_boundaries.yaml")
    config = yaml.safe_load(base.read_text())
    problem, settings = config["BenchmarkProblems"][0][:2]
    config["BenchmarkProblems"] = [[problem, settings]]
    settings["ForkParameters"] = [{key: [value]} for key, value in {
        "ClusterDim": cluster, "PrefetchGlobalRead": 2, "PrefetchAcrossPersistent": 1,
        "AssertSummationElementMultiple": 64, "TDMIterateMode": 0, "HalfPLR": 0,
        "SuppressNoLoadLoop": False,
    }.items()]
    path = tmp_path / "pap_bf16_codegen.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


@pytest.mark.parametrize("overrides,expected", [
    ({"ClusterDim": [2, 2]}, [True]),
    ({"ClusterDim": [1, 1]}, [False]),
    ({"TileProcessingStrategy": "StreamK"}, [False]),
    ({"ClusterDim": [2, 2], "PrefetchGlobalRead": 1}, []),
    ({"PrefetchAcrossPersistent": 0}, []),
    ({"AssertSummationElementMultiple": 32}, []),
], ids=["dp-cluster", "dp-noncluster", "streamk", "pgr1", "pap-off", "tail"])
def test_retained_plr_completion_caller_eligibility(tmp_path, monkeypatch, overrides, expected):
    calls = []
    original = PersistentKernelState.papLocalReadCompletion

    def completion(writer, **kwargs):
        calls.append(kwargs.get("completeGroup0", False))
        return original(writer, **kwargs)

    monkeypatch.setattr(PersistentKernelState, "papLocalReadCompletion", completion)
    parameters = {"AssertSummationElementMultiple": 256, **overrides}
    results = emit_kernels_from_config(_config(tmp_path, **parameters), limit=1, arch="gfx1250")
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
    assert calls == expected


@pytest.mark.parametrize("cluster,paired_mx", [([2, 2], True), ([2, 4], True), ([2, 2], False)],
                         ids=["mx-2x2", "mx-2x4", "bf16-2x2"])
def test_clustered_retained_plr_preserves_main_loop_overlap(tmp_path, cluster, paired_mx):
    # ASem=DepthU activates the no-tail, two-group and retained-PLR paths.
    # Without group0 completion provenance, entry's carried DS reads make the
    # native pass drain every tensor producer before the next loop's refill.
    config = (_config(tmp_path, AssertSummationElementMultiple=256, ClusterDim=cluster)
              if paired_mx else _bf16_config(tmp_path, cluster))
    results = emit_kernels_from_config(config, limit=1, arch="gfx1250")
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        assert "PAP skip redundant first local reads" in src
        assert "PAP defer entry-local group1 publication" in src
        loop = src.split("\nlabel_LoopBeginL:\n", 1)[1].split("\nlabel_LoopEndL:\n", 1)[0]
        code = [line.split("//", 1)[0].strip() for line in loop.splitlines()]
        issues = [i for i, line in enumerate(code) if line.startswith("tensor_load_to_lds ")]
        waits = [(i, int(line.split()[1])) for i, line in enumerate(code)
                 if line.startswith("s_wait_tensorcnt ")]
        expected_count = 2 if paired_mx else 1
        assert len(issues) == expected_count  # one wave's data and optional MX-scale producers
        assert waits and all(count == expected_count for _, count in waits), "\n".join(code)
        assert all(i > issues[-1] for i, _ in waits)

        # Refill waits for peers, then gathers completed local readers.
        last_read = max(i for i in range(issues[0]) if code[i].startswith("ds_load_"))
        release = code[last_read + 1:issues[0]]
        assert release.index("s_barrier_wait -3") < release.index("s_wait_dscnt 0")
        assert release.index("s_wait_dscnt 0") < release.index("s_barrier_signal -1")
        assert release.index("s_barrier_signal -1") < release.index("s_barrier_wait -1")

        # The partial tensor wait publishes the older bank before its reads;
        # younger producers can remain in flight through the following WMMAs.
        ready = waits[-1][0]
        next_read = next(i for i in range(ready + 1, len(code))
                         if code[i].startswith("ds_load_"))
        acquisition = code[ready + 1:next_read]
        assert acquisition.index("s_barrier_signal -1") < acquisition.index("s_barrier_wait -1")
