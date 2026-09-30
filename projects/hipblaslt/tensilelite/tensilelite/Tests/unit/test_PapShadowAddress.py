# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compare pure PAP address preparation and delayed commit with ordinary refresh."""

from copy import deepcopy
import re
from types import SimpleNamespace

import pytest
from rocisa.code import Module
from rocisa.container import ContinuousRegister

from Tensile.Components.ClusterLoad import ClusterLoadTDM
from Tensile.Components.TensorDataMover import TensorDataMoverLoad
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from test_PapIncrementalTdm import _Writer, _full, _kernel, _machine

pytestmark = pytest.mark.unit

_NEXT = {0: 84, 1: 85, 2: 86}
_OUTPUT = {"A": 80, "MXSA": 82}
_SCRATCH = ContinuousRegister(100, 3)


@pytest.fixture(autouse=True)
def _components(monkeypatch, _isolate_rocisa_state):
    from gpu_test_helpers import init_rocisa
    init_rocisa(target="gfx1250", wavesize=32)
    monkeypatch.setattr(TensorDataMoverLoad, "find", lambda writer: TensorDataMoverLoad())
    monkeypatch.setattr(ClusterLoadTDM, "find", lambda writer: ClusterLoadTDM())


def _setup(dtype="F8", *, depth=256, iterate=True, tlu=False, scales=True, cluster=(1, 1)):
    kernel, tensors = _kernel(dtype, depth=depth, iterate=iterate, tlu=tlu,
                              scales=scales, cluster=cluster)
    kernel.update(ScheduleIterAlg=4, PrefetchAcrossPersistent=1, HalfPLR=False,
                  SuppressNoLoadLoop=False, UseCustomMainLoopSchedule=False,
                  SpaceFillingAlgo=[], WorkGroupMapping=1)
    writer = _Writer(kernel)
    writer.states.version = (12, 5, 0)
    return kernel, tensors, writer


def _run(machine, module):
    # This selector is encoding state for the low Serial VGPR, not arithmetic.
    # Reject every nonzero selector until the interpreter models high VGPRs.
    lines = []
    for line in str(module).splitlines():
        instruction = line.split("//", 1)[0].strip()
        if instruction.startswith("s_set_vgpr_msb"):
            assert instruction == "s_set_vgpr_msb 0"
        else:
            lines.append(line)
    # Initial-stride symbols are assembly constants whose value is one.
    assembly = re.sub(r"\bconstStride(?:MXSA|MXSB|A|B)[IJKL]\b", "1", "\n".join(lines))
    machine.run(assembly)


def _pairs(scales):
    return [("A", "B")] + ([("MXSA", "MXSB")] if scales else [])


def _code(writer, kernel, tensors, *, scales, data_only=False, reverse=False):
    prep, commit, ordinary = (Module(name) for name in ("prepare", "commit", "ordinary"))
    outputs = set()
    for a, b in _pairs(scales):
        pair = (tensors[b], tensors[a]) if reverse else (tensors[a], tensors[b])
        prepared = None if data_only and a == "MXSA" else _OUTPUT[a]
        if prepared is not None:
            prep.add(writer.papTdmPrepareAddressPair(kernel, *pair, _NEXT, prepared, _SCRATCH))
            outputs.update((prepared, prepared + 1))
        commit.add(writer.papTdmUpdateDescriptor(kernel, *pair, preparedGlobalAddr=prepared))
        ordinary.add(writer.papTdmUpdateDescriptor(kernel, *pair))
    return prep, commit, ordinary, outputs


def _assert_pure(prep):
    allowed = {"v_readfirstlane_b32", "s_set_vgpr_msb", "s_bitcmp1_b32",
               "s_mov_b32", "s_cselect_b32", "s_or_b32", "s_lshr_b32",
               "s_lshr_b64", "s_lshl_b64", "s_mul_i32", "s_mul_hi_u32",
               "s_add_u32", "s_addc_u32"}
    for line in str(prep).splitlines():
        instruction = line.split("//", 1)[0].strip()
        if not instruction or instruction.startswith(("/*", "*")):
            continue
        assert not instruction.endswith(":"), instruction
        assert instruction.split()[0] in allowed, instruction


@pytest.mark.parametrize("dtype,iterate,depth,tlu,scales", [
    ("F8", True, 256, False, True),
    ("F8", (True, False), 128, False, True),
    ("F4", True, 256, False, True),
    ("F4", True, 128, False, True),
    ("F6", False, 256, False, False),
    ("H", False, 256, True, False),
])
@pytest.mark.parametrize("cluster", [(1, 1), (2, 4)])
@pytest.mark.parametrize("bank", [0, 1])
def test_prepared_commit_matches_refresh_after_bank_changes(dtype, iterate, depth, tlu, scales, cluster, bank):
    kernel, tensors, writer = _setup(dtype, depth=depth, iterate=iterate, tlu=tlu,
                                     scales=scales, cluster=cluster)
    full = _full(writer, kernel, tensors)
    prep, commit, ordinary, outputs = _code(writer, kernel, tensors, scales=scales)
    _assert_pure(prep)
    assert "TDM wave separated calc start addr" not in str(commit)
    assert "TDM set global addr" not in str(commit)
    assert "commit prepared global address" in str(commit)
    allowed_writes = outputs | set(range(_SCRATCH.idx, _SCRATCH.idx + 3))
    invariants = (1, 4, 8, 9, 10, 11, 17, 20, 24, 25, 26, 27, 32, 33, 34)

    for wave in range(4):
        current = _machine(kernel, tensors, wave)
        current.set("WorkGroup0", 2)
        current.set("WorkGroup1", 2)
        _run(current, full)
        for a, b in _pairs(scales):
            current.set(f"tdm{a}{b}Incs", 0x900)
            if bank:
                _run(current, writer.tdmSwapLdsOffset(kernel, tensors[a]))
        # Edge -> interior, batch changes, and interior -> empty edge wave.
        for coordinates in ((0, 0, 1), (2, 2, 3), (1, 1, 2)):
            for index, value in enumerate(coordinates):
                current.set(_NEXT[index], value)
            before = deepcopy(current.words)
            current.writes.clear()
            _run(current, prep)
            assert set(current.writes) <= allowed_writes
            assert all(current.words.get(index, 0) == value
                       for index, value in before.items() if index not in allowed_writes)
            prepared = {a: current.get(_OUTPUT[a], 2) for a, _ in _pairs(scales)}
            # Current NGLL work continues after preparation: rotate the bank,
            # advance its descriptors, and null their enable word before commit.
            for a, b in _pairs(scales):
                old_bank = current.get(f"tdm{a}Group0+1")
                _run(current, writer.tdmSwapLdsOffset(kernel, tensors[a]))
                assert current.get(f"tdm{a}Group0+1") != old_bank
                advance = writer.tdmIncrementABWaveSperated(kernel, tensors[a], tensors[b])
                for _ in range(3):
                    _run(current, advance)
                current.set(f"tdm{a}Group0", 0)
            expected = deepcopy(current)
            for machine in (current, expected):
                for index, value in enumerate(coordinates):
                    machine.set(f"WorkGroup{index}", value)
            preserved = {index: current.words.get(index, 0) for index in invariants}
            current.writes.clear()
            _run(current, commit)
            _run(expected, ordinary)
            assert current.descriptor(scales) == expected.descriptor(scales), (wave, coordinates)
            assert not set(invariants).intersection(current.writes)
            assert all(current.words.get(index, 0) == value for index, value in preserved.items())
            for a, _ in _pairs(scales):
                assert current.get(f"tdm{a}Group0+2", 2) == prepared[a]
                assert current.get(f"tdm{a}Group0") == 1
                assert current.get(f"tdm{a}Group0+3") >> 30 == 2


@pytest.mark.parametrize("cluster", [(1, 1), (2, 4)])
def test_user_args_direct_mode_matches_refresh_with_packed_wave_bits(cluster):
    kernel, tensors, writer = _setup(cluster=cluster)
    kernel["ProblemType"]["SupportUserArgs"] = True
    writer.states.tdmParityPackedInArgType = True
    writer.states.tdmWaveIdBitsInArgType = 2
    assert writer.papTdmSupportsShadowAddressPair(kernel, tensors["A"], tensors["B"])
    full = _full(writer, kernel, tensors)
    prep, commit, ordinary, _ = _code(writer, kernel, tensors, scales=True)
    _assert_pure(prep)
    assert "load A matrix address" in str(ordinary)
    assert "load B matrix address" in str(ordinary)
    assert "load A matrix address" not in str(commit)
    assert "load B matrix address" not in str(commit)
    for wave in range(4):
        current = _machine(kernel, tensors, wave)
        current.set("ArgType", wave << 8)  # Runtime domain 0; high bits are wave id.
        _run(current, full)
        for index, value in enumerate((1, 2, 3)):
            current.set(_NEXT[index], value)
        _run(current, prep)
        # The physical bank and address may change between producer and consumer.
        for a, b in _pairs(True):
            _run(current, writer.tdmSwapLdsOffset(kernel, tensors[a]))
            _run(current, writer.tdmIncrementABWaveSperated(kernel, tensors[a], tensors[b]))
            current.set(f"tdm{a}Group0", 0)
        expected = deepcopy(current)
        for machine in (current, expected):
            for index, value in enumerate((1, 2, 3)):
                machine.set(f"WorkGroup{index}", value)
        _run(current, commit)
        _run(expected, ordinary)
        assert current.descriptor(True) == expected.descriptor(True)
        assert current.get("ArgType") == wave << 8
        for a, _ in _pairs(True):
            assert current.get(f"tdm{a}Group0+3") >> 30 == 2


@pytest.mark.parametrize("arg_type", [1, 2, 3])
def test_unsupported_runtime_argument_modes_can_speculate_without_dereference(arg_type):
    kernel, tensors, writer = _setup()
    kernel["ProblemType"]["SupportUserArgs"] = True
    prep, _, _, outputs = _code(writer, kernel, tensors, scales=True)
    _assert_pure(prep)
    allowed_writes = outputs | set(range(_SCRATCH.idx, _SCRATCH.idx + 3))
    for wave in range(4):
        current = _machine(kernel, tensors, wave)
        current.set("ArgType", (wave << 8) | arg_type)
        current.set("PersistentPrefetchState", 0x80000002)
        for index, value in enumerate((0xffffffff, 0xffffffff, 0xffffffff)):
            current.set(_NEXT[index], value)
        # Deliberately provide no scalar-memory backing for pointer-array modes.
        assert not current.memory
        before = deepcopy(current.words)
        current.writes.clear()
        _run(current, prep)
        assert set(current.writes) <= allowed_writes
        assert all(current.words.get(index, 0) == value for index, value in before.items()
                   if index not in allowed_writes)
        # This leaf never sets PREPARED. Pipeline tests own mode validity/discard.
        assert current.get("PersistentPrefetchState") == 0x80000002
        assert current.get("ArgType") == (wave << 8) | arg_type


@pytest.mark.parametrize("data_only", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_pair_api_supports_reversed_arguments_and_data_only_carriers(data_only, reverse):
    kernel, tensors, writer = _setup(iterate=False)
    # Different data orientations exercise both selected stride expressions.
    tensors["A"]["tlu"] = True
    full = _full(writer, kernel, tensors)
    prep, commit, ordinary, outputs = _code(writer, kernel, tensors, scales=True,
                                           data_only=data_only, reverse=reverse)
    _assert_pure(prep)
    assert len(outputs) == (2 if data_only else 4)
    for wave in range(4):
        current = _machine(kernel, tensors, wave)
        _run(current, full)
        for index, value in enumerate((1, 2, 0x10003)):
            current.set(_NEXT[index], value)
        # Force 64-bit batch products and low-word address carries.
        for tc in tensors:
            current.set(f"Stride{tc}K", 0xfff12345)
        _run(current, prep)
        expected = deepcopy(current)
        for machine in (current, expected):
            for index, value in enumerate((1, 2, 0x10003)):
                machine.set(f"WorkGroup{index}", value)
        _run(current, commit)
        _run(expected, ordinary)
        assert current.descriptor(True) == expected.descriptor(True)


@pytest.mark.parametrize("both_transposed", [False, True])
def test_preparation_handles_real_constant_stride_operands(both_transposed):
    kernel, tensors, writer = _setup(iterate=False, scales=False)
    for tc in ("A", "B"):
        tensors[tc]["tlu"] = tc == "A" or both_transposed
        idx = tensors[tc]["idx"]
        tensors[tc]["ia"] = [idx, 3, 2] if tensors[tc]["tlu"] else [3, idx, 2]
        kernel["ProblemType"][f"IndexAssignments{tc}"] = tensors[tc]["ia"]
    kernel["ProblemType"]["UseInitialStridesAB"] = False
    writer.strideRef = lambda tc, dim: KernelWriterAssembly.strideRef(writer, tc, dim)
    full = _full(writer, kernel, tensors)
    prep, commit, ordinary, _ = _code(writer, kernel, tensors, scales=False)
    assert "constStrideAI" in str(prep)
    _assert_pure(prep)
    for wave in range(4):
        current = _machine(kernel, tensors, wave)
        _run(current, full)
        for index, value in enumerate((2, 1, 7)):
            current.set(_NEXT[index], value)
        _run(current, prep)
        expected = deepcopy(current)
        for machine in (current, expected):
            for index, value in enumerate((2, 1, 7)):
                machine.set(f"WorkGroup{index}", value)
        _run(current, commit)
        _run(expected, ordinary)
        assert current.descriptor(False) == expected.descriptor(False)


def test_preparation_has_no_hidden_allocations_or_state_writes(monkeypatch):
    kernel, tensors, writer = _setup()

    def forbidden(*args, **kwargs):
        raise AssertionError("pure preparation called an allocator or label factory")

    monkeypatch.setattr(writer, "allocTmpSgpr", forbidden)
    writer.vgprPool = SimpleNamespace(checkOut=forbidden, checkOutAligned=forbidden)
    writer.labels = SimpleNamespace(getNameInc=forbidden)
    for a, b in _pairs(True):
        code = writer.papTdmPrepareAddressPair(kernel, tensors[a], tensors[b], _NEXT, _OUTPUT[a], _SCRATCH)
        _assert_pure(code)
        machine = _machine(kernel, tensors, 3)
        for index, value in enumerate((0xffffffff, 0xffffffff, 0xffffffff)):
            machine.set(_NEXT[index], value)
        machine.set("PersistentPrefetchState", 0x80000000)
        before = deepcopy(machine.words)
        machine.writes.clear()
        _run(machine, code)  # Exhausted coordinates are safe: no dereference.
        allowed = {_OUTPUT[a], _OUTPUT[a] + 1, 100, 101, 102}
        assert set(machine.writes) <= allowed
        assert all(machine.words.get(index, 0) == value for index, value in before.items()
                   if index not in allowed)


@pytest.mark.parametrize("target,overrides", [
    ("kernel", {"TileProcessingStrategy": "StreamK"}),
    ("kernel", {"WorkAssignment": "DynamicWorkQueue"}),
    ("kernel", {"ScheduleIterAlg": 0}), ("kernel", {"PrefetchGlobalRead": 1}),
    ("kernel", {"HalfPLR": True}), ("kernel", {"SuppressNoLoadLoop": True}),
    ("kernel", {"ReuseAcrossPersistent": True}), ("kernel", {"GlobalSplitU": 1}),
    ("kernel", {"TDMFuse": 1}), ("kernel", {"TDMSplit": True}),
    ("kernel", {"TDMPlusLdsBuf": 1}), ("kernel", {"UseSubtileImpl": True}),
    ("kernel", {"StaggerU": 1}), ("kernel", {"SpaceFillingAlgo": [1]}),
    ("kernel", {"UseCustomMainLoopSchedule": True}), ("kernel", {"WorkGroupMapping": 2}),
    ("problem", {"StridedBatched": False}),
    ("problem", {"Batched": False}), ("problem", {"GroupedGemm": True}),
    ("problem", {"Sparse": 1}), ("caps", {"HasSMulHi": False}),
    ("B", {"bpeGR": 2}), ("B", {"ia": [3, 1, 4]}),
])
def test_unsupported_pair_is_rejected_before_emission(target, overrides):
    kernel, tensors, writer = _setup()
    destination = {"kernel": kernel, "problem": kernel["ProblemType"],
                   "caps": writer.states.asmCaps, "B": tensors["B"]}[target]
    destination.update(overrides)
    assert not writer.papTdmSupportsShadowAddressPair(kernel, tensors["A"], tensors["B"])
    with pytest.raises(AssertionError):
        writer.papTdmPrepareAddressPair(kernel, tensors["A"], tensors["B"], _NEXT, 80, _SCRATCH)
    assert writer.max_tmp == 200 and writer._label == 0


@pytest.mark.parametrize("coordinates,dst,scratch", [
    (_NEXT, 81, ContinuousRegister(100, 3)),
    (_NEXT, 80, ContinuousRegister(101, 3)),
    (_NEXT, 80, ContinuousRegister(100, 2)),
    (_NEXT, 80, ContinuousRegister(80, 3)),
    ({0: 80, 1: 85, 2: 86}, 80, ContinuousRegister(100, 3)),
    ({0: 100, 1: 85, 2: 86}, 80, ContinuousRegister(100, 3)),
    ({0: 84, 1: 84, 2: 86}, 80, ContinuousRegister(100, 3)),
])
def test_preparation_rejects_misaligned_or_aliasing_registers(coordinates, dst, scratch):
    kernel, tensors, writer = _setup()
    with pytest.raises(AssertionError):
        writer.papTdmPrepareAddressPair(kernel, tensors["A"], tensors["B"], coordinates, dst, scratch)
    assert writer.max_tmp == 200 and writer._label == 0


def test_prepared_address_cannot_replace_current_tail_refresh():
    kernel, tensors, writer = _setup()
    with pytest.raises(AssertionError):
        writer.papTdmUpdateDescriptor(kernel, tensors["A"], tensors["B"],
                                      preservePapBank=False, preparedGlobalAddr=80)
    assert writer.max_tmp == 200 and writer._label == 0
