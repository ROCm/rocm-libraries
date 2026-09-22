#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for subtile SQTT marker instrumentation (gfx1250).
#
# Covers Components/Subtile/SqttMarkers.py: the marker word encoding, the id
# registry, the post-schedule splice, and the .sqtt_funcmap section. The emitted
# text *is* the ABI contract with rocprof-trace-decoder -- nothing downstream
# validates it, so it is easy to break silently.
#
# Usage:
#   pytest test_subtile_sqtt_markers.py -v
################################################################################

import os
import shutil
import sys

import pytest
from types import SimpleNamespace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

WAVESIZE_32 = 32
GFX1250_ISA = (12, 5, 0)


def _init_rocisa_gfx1250():
    from rocisa import rocIsa
    from Tensile.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx1250")
    asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
    ri.init(isa, asmpath)
    ri.setKernel(isa, WAVESIZE_32)


@pytest.fixture(autouse=True)
def _rocisa_gfx1250():
    _init_rocisa_gfx1250()


def _make_writer(kernel_name="TestKernel"):
    return SimpleNamespace(states=SimpleNamespace(kernelName=kernel_name))


def _kernel(enabled=True, isa=GFX1250_ISA, wavesize=WAVESIZE_32, mode=None):
    # `enabled` stays bool-shaped so the pre-mode tests read naturally; True is
    # mode 1 (int(True) == 1), which is exactly how existing YAML behaves.
    return {"SubtileSqttMarkers": mode if mode is not None else enabled,
            "ISA": list(isa), "WavefrontSize": wavesize,
            "LdsNumBytes": 0x44000}


def _module(*items):
    from rocisa.code import Module
    mod = Module("section")
    for it in items:
        mod.add(it)
    return mod


def _nop(comment="filler"):
    from rocisa.instruction import SNop
    return SNop(0, comment)


def _cb_barrier(comment):
    from rocisa.instruction import SBarrier
    return SBarrier(comment=comment)


def _markers(module):
    from rocisa.instruction import STtraceDataImm
    return [i for i in module.flatitems() if isinstance(i, STtraceDataImm)]


class TestEncoding:
    """The 32-bit marker word: (id << 2) | flags."""

    def test_point_has_no_flags(self):
        from Tensile.Components.Subtile.SqttMarkers import encodeMarker
        assert encodeMarker(7) == 0x1C

    def test_enter_sets_bit1(self):
        from Tensile.Components.Subtile.SqttMarkers import encodeMarker, FLAG_ENTER
        assert encodeMarker(7, FLAG_ENTER) == 0x1E

    def test_fused_exit_enter_sets_both(self):
        from Tensile.Components.Subtile.SqttMarkers import (
            encodeMarker, FLAG_ENTER, FLAG_EXIT_PREV)
        assert encodeMarker(7, FLAG_ENTER | FLAG_EXIT_PREV) == 0x1F

    def test_max_id_still_fits_the_8bit_capture_window(self):
        # The whole reason for MAX_MARKER_ID: hardware captures 8 payload bits.
        from Tensile.Components.Subtile.SqttMarkers import (
            encodeMarker, MAX_MARKER_ID, FLAG_ENTER, FLAG_EXIT_PREV)
        assert encodeMarker(MAX_MARKER_ID, FLAG_ENTER | FLAG_EXIT_PREV) == 0xFF

    def test_id_past_the_window_is_rejected(self):
        from Tensile.Components.Subtile.SqttMarkers import encodeMarker, MAX_MARKER_ID
        with pytest.raises(ValueError):
            encodeMarker(MAX_MARKER_ID + 1)


class TestRegistry:
    """Id allocation and funcmap rows."""

    def test_ids_start_at_one_and_are_stable_per_name(self):
        from Tensile.Components.Subtile.SqttMarkers import MarkerRegistry
        reg = MarkerRegistry()
        assert reg.idFor("a") == 1
        assert reg.idFor("b") == 2
        assert reg.idFor("a") == 1

    def test_rows_are_prefixed_and_ordered_by_id(self):
        from Tensile.Components.Subtile.SqttMarkers import (
            MarkerRegistry, KIND_SCOPE, KIND_POINT)
        reg = MarkerRegistry()
        reg.idFor("scope", KIND_SCOPE)
        reg.idFor("point", KIND_POINT)
        assert reg.rows() == ["U:1:scope", "P:2:point"]

    def test_budget_exhaustion_is_loud(self):
        from Tensile.Components.Subtile.SqttMarkers import MarkerRegistry, MAX_MARKER_ID
        reg = MarkerRegistry()
        for i in range(MAX_MARKER_ID):
            reg.idFor(f"m{i}")
        with pytest.raises(ValueError):
            reg.idFor("one_too_many")


class TestEnableGating:
    def test_off_by_default(self):
        from Tensile.Components.Subtile.SqttMarkers import sqttMarkersEnabled
        assert not sqttMarkersEnabled(_make_writer(), _kernel(enabled=False))

    def test_rejected_off_gfx1250(self):
        # gfx9 has no s_ttracedata_imm at all; emitting it would not assemble.
        from Tensile.Components.Subtile.SqttMarkers import sqttMarkersEnabled
        assert not sqttMarkersEnabled(_make_writer(), _kernel(isa=(9, 4, 2)))

    def test_enabled_on_gfx1250(self):
        from Tensile.Components.Subtile.SqttMarkers import sqttMarkersEnabled
        assert sqttMarkersEnabled(_make_writer(), _kernel())


class TestInsertSqttMarkers:
    """The post-schedule splice."""

    def test_disabled_is_a_no_op(self):
        from Tensile.Components.Subtile.SqttMarkers import insertSqttMarkers
        mod = _module(_nop())
        out = insertSqttMarkers(mod, _make_writer(), _kernel(enabled=False), "MAINLOOP")
        assert _markers(out) == []

    def test_mainloop_gets_exactly_one_marker_at_the_head(self):
        from Tensile.Components.Subtile.SqttMarkers import insertSqttMarkers
        from rocisa.instruction import STtraceDataImm
        out = insertSqttMarkers(_module(_nop(), _nop()), _make_writer(),
                                _kernel(), "MAINLOOP")
        items = out.flatitems()
        assert isinstance(items[0], STtraceDataImm)
        assert len(_markers(out)) == 1

    def test_non_mainloop_sections_get_no_iteration_marker(self):
        from Tensile.Components.Subtile.SqttMarkers import insertSqttMarkers
        out = insertSqttMarkers(_module(_nop()), _make_writer(), _kernel(), "PRELOOP")
        assert _markers(out) == []

    def test_cluster_barrier_halves_are_marked(self):
        from Tensile.Components.Subtile.ClusterBarrier import (
            CB_SIGNAL_COMMENT, CB_WAIT_COMMENT)
        from Tensile.Components.Subtile.SqttMarkers import insertSqttMarkers
        mod = _module(_cb_barrier(CB_SIGNAL_COMMENT), _nop(), _cb_barrier(CB_WAIT_COMMENT))
        out = insertSqttMarkers(mod, _make_writer(), _kernel(), "PRELOOP")
        assert len(_markers(out)) == 2

    def test_marker_follows_the_barrier_it_timestamps(self):
        from Tensile.Components.Subtile.ClusterBarrier import CB_SIGNAL_COMMENT
        from Tensile.Components.Subtile.SqttMarkers import insertSqttMarkers
        from rocisa.instruction import STtraceDataImm, SBarrier
        out = insertSqttMarkers(_module(_cb_barrier(CB_SIGNAL_COMMENT)),
                                _make_writer(), _kernel(), "PRELOOP")
        items = out.flatitems()
        assert isinstance(items[0], SBarrier)
        assert isinstance(items[1], STtraceDataImm)

    def test_input_module_is_not_mutated(self):
        from Tensile.Components.Subtile.SqttMarkers import insertSqttMarkers
        mod = _module(_nop())
        insertSqttMarkers(mod, _make_writer(), _kernel(), "MAINLOOP")
        assert _markers(mod) == []

    def test_repeated_sections_reuse_one_id(self):
        # Every MAINLOOP body shares the one iteration marker id.
        from Tensile.Components.Subtile.SqttMarkers import insertSqttMarkers, registryFor
        writer, kernel = _make_writer(), _kernel()
        for _ in range(3):
            insertSqttMarkers(_module(_nop()), writer, kernel, "MAINLOOP")
        assert len(registryFor(writer).rows()) == 1


class TestFuncmapSection:

    def _funcmap_text(self, writer, kernel):
        from Tensile.Components.Subtile.SqttMarkers import sqttFuncmapSection
        return str(sqttFuncmapSection(writer, kernel))

    def test_empty_when_no_markers_were_placed(self):
        assert self._funcmap_text(_make_writer(), _kernel()).strip() == ""

    def test_disabled_emits_nothing(self):
        from Tensile.Components.Subtile.SqttMarkers import insertSqttMarkers
        writer = _make_writer()
        insertSqttMarkers(_module(_nop()), writer, _kernel(), "MAINLOOP")
        assert self._funcmap_text(writer, _kernel(enabled=False)).strip() == ""

    def test_contains_wavesize_kernel_and_marker_rows(self):
        # The mainloop row must be U: (scope), not P: -- the decoder takes the
        # marker kind from the funcmap, and a P: row would make consumers treat
        # a real scope as a zero-width instant.
        from Tensile.Components.Subtile.SqttMarkers import (
            insertSqttMarkers, MARKER_MAINLOOP_ITER)
        writer, kernel = _make_writer("MyKernel"), _kernel()
        insertSqttMarkers(_module(_nop()), writer, kernel, "MAINLOOP")
        text = self._funcmap_text(writer, kernel)
        assert '.section .sqtt_funcmap,"S",@progbits' in text
        assert "W:32" in text
        assert "K:MyKernel" in text
        assert f"U:1:{MARKER_MAINLOOP_ITER}" in text

    def test_cluster_barrier_rows_stay_points(self):
        from Tensile.Components.Subtile.ClusterBarrier import CB_SIGNAL_COMMENT
        from Tensile.Components.Subtile.SqttMarkers import (
            insertSqttMarkers, MARKER_CB_SIGNAL)
        writer, kernel = _make_writer(), _kernel()
        insertSqttMarkers(_module(_cb_barrier(CB_SIGNAL_COMMENT)), writer,
                          kernel, "PRELOOP")
        assert f"P:1:{MARKER_CB_SIGNAL}" in self._funcmap_text(writer, kernel)

    def test_no_clock_layout_row(self):
        # An M: row advertises shader-clock packing, which the immediate marker
        # form cannot carry. Emitting one would make the decoder misparse ids.
        from Tensile.Components.Subtile.SqttMarkers import insertSqttMarkers
        writer, kernel = _make_writer(), _kernel()
        insertSqttMarkers(_module(_nop()), writer, kernel, "MAINLOOP")
        assert "M:shader_clock_bits" not in self._funcmap_text(writer, kernel)

    def test_rows_are_escaped_for_asciz(self):
        # A raw newline inside .asciz would not assemble.
        from Tensile.Components.Subtile.SqttMarkers import insertSqttMarkers
        writer, kernel = _make_writer(), _kernel()
        insertSqttMarkers(_module(_nop()), writer, kernel, "MAINLOOP")
        asciz = [l for l in self._funcmap_text(writer, kernel).splitlines()
                 if l.startswith(".asciz")]
        assert len(asciz) == 1
        assert "\\n" in asciz[0]


class TestEmittedAssembly:
    """The marker instruction itself."""

    def test_point_marker_renders_as_ttracedata_imm(self):
        from Tensile.Components.Subtile.SqttMarkers import markerPoint
        text = str(markerPoint(_make_writer(), "probe"))
        assert "s_ttracedata_imm 0x4" in text  # id 1, no flags

    def test_transition_sets_both_flags(self):
        # (id << 2) | 0x3 -- pop the open scope and push the next one.
        from Tensile.Components.Subtile.SqttMarkers import markerTransition
        assert "s_ttracedata_imm 0x7" in str(markerTransition(_make_writer(), "probe"))

    def test_exit_marker_is_the_bare_flag(self):
        from Tensile.Components.Subtile.SqttMarkers import markerExit
        assert "s_ttracedata_imm 0x1" in str(markerExit(_make_writer()))

    def test_enter_marker_sets_the_enter_flag(self):
        from Tensile.Components.Subtile.SqttMarkers import markerEnter
        assert "s_ttracedata_imm 0x6" in str(markerEnter(_make_writer(), "probe"))


class TestParameterPlumbing:
    """Defaults, gating, and -- most importantly -- kernel naming."""

    def test_defaults_off(self):
        from Tensile.Common.GlobalParameters import defaultBenchmarkCommonParameters
        default = next(d["SubtileSqttMarkers"] for d in defaultBenchmarkCommonParameters
                       if "SubtileSqttMarkers" in d)
        assert default == [False]

    def test_in_the_min_name_roster(self):
        # Without this, an instrumented kernel hashes to the same name as the
        # production one and one of the two is silently dropped as a duplicate.
        from Tensile.Common.RequiredParameters import getRequiredParametersMin
        assert "SubtileSqttMarkers" in getRequiredParametersMin()

    def test_name_abbreviation_is_unique(self):
        import collections
        from Tensile.Common.RequiredParameters import getRequiredParametersMin
        from Tensile.SolutionStructs.Naming import getParameterNameAbbreviation
        abbr = collections.defaultdict(list)
        for key in getRequiredParametersMin():
            abbr[getParameterNameAbbreviation(key)].append(key)
        assert abbr["SSM"] == ["SubtileSqttMarkers"]

    def test_naming_is_gated_so_disabled_kernels_keep_their_names(self):
        """The regression this guards is a repo-wide rename.

        Every shipped kernel is uninstrumented. If the parameter were named
        unconditionally, adding it would change the name of every kernel in
        every logic file -- so the off case must contribute nothing, exactly as
        TDMFuse=0 does.
        """
        import re
        src = open(os.path.join(TENSILE_ROOT, "Tensile",
                                "SolutionStructs", "Naming.py")).read()
        # The discard branch is what keeps existing names stable.
        assert re.search(r'requiredParametersTemp\.discard\("SubtileSqttMarkers"\)', src)
        assert re.search(r'requiredParametersTemp\.add\("SubtileSqttMarkers"\)', src)

    def test_solution_gates_on_gfx1250_and_subtile(self):
        """assignDerivedParameters is the single authority on the flag.

        The emitter trusts kernel["SubtileSqttMarkers"] without rechecking the
        ISA, so a request on a target without s_ttracedata_imm has to be cleared
        here or the kernel will not assemble.
        """
        src = open(os.path.join(TENSILE_ROOT, "Tensile",
                                "SolutionStructs", "Solution.py")).read()
        marker = 'state["SubtileSqttMarkers"] = '
        assert marker in src, "SubtileSqttMarkers is never derived in Solution.py"
        expr = src[src.index(marker) + len(marker):].split("\n\n")[0]
        assert "isgfx1250" in expr, expr
        assert 'state["UseSubtileImpl"]' in expr, expr


class TestBackendTolerance:
    """SqttMarkers loads for every subtile kernel, so a backend without the
    immediate opcode must not take the subtile path down with it."""

    def test_missing_opcode_is_inert_while_disabled(self, monkeypatch):
        import Tensile.Components.Subtile.SqttMarkers as sm
        monkeypatch.setattr(sm, "STtraceDataImm", None)
        assert not sm.sqttMarkersEnabled(_make_writer(), _kernel(enabled=False))

    def test_missing_opcode_is_loud_when_requested(self, monkeypatch):
        import Tensile.Components.Subtile.SqttMarkers as sm
        monkeypatch.setattr(sm, "STtraceDataImm", None)
        with pytest.raises(RuntimeError, match="s_ttracedata_imm"):
            sm.sqttMarkersEnabled(_make_writer(), _kernel())


class TestMainloopScope:
    """The mainloop iteration must be a scope a consumer can attribute time to.

    RCV's marker walker gives a point enter_time == exit_time and never pushes
    it on the stack, so a point contributes zero width to the marker flamegraph.
    These pin the scope shape that makes the markers measurable.
    """

    def test_mainloop_marker_is_a_fused_transition(self):
        from Tensile.Components.Subtile.SqttMarkers import (
            insertSqttMarkers, FLAG_TRANSITION, encodeMarker)
        out = insertSqttMarkers(_module(_nop()), _make_writer(), _kernel(), "MAINLOOP")
        marker = _markers(out)[0]
        assert f"{encodeMarker(1, FLAG_TRANSITION):#x}" in str(marker)

    def test_mainloop_marker_is_registered_as_a_scope(self):
        from Tensile.Components.Subtile.SqttMarkers import (
            insertSqttMarkers, registryFor, MARKER_MAINLOOP_ITER, KIND_SCOPE)
        writer = _make_writer()
        insertSqttMarkers(_module(_nop()), writer, _kernel(), "MAINLOOP")
        assert registryFor(writer).rows() == [f"{KIND_SCOPE}:1:{MARKER_MAINLOOP_ITER}"]

    def test_scope_open_primes_with_a_plain_enter(self):
        # Without the prime, every wave's first transition pops an empty stack.
        from Tensile.Components.Subtile.SqttMarkers import (
            sqttMainloopScopeOpen, FLAG_ENTER, encodeMarker)
        text = str(sqttMainloopScopeOpen(_make_writer(), _kernel()))
        assert f"{encodeMarker(1, FLAG_ENTER):#x}" in text

    def test_scope_close_is_a_bare_exit(self):
        from Tensile.Components.Subtile.SqttMarkers import sqttMainloopScopeClose
        assert "s_ttracedata_imm 0x1" in str(sqttMainloopScopeClose(_make_writer(), _kernel()))

    def test_open_and_close_share_the_iteration_id(self):
        # Prime, transitions and exit must all refer to one scope.
        from Tensile.Components.Subtile.SqttMarkers import (
            sqttMainloopScopeOpen, insertSqttMarkers, registryFor)
        writer, kernel = _make_writer(), _kernel()
        sqttMainloopScopeOpen(writer, kernel)
        insertSqttMarkers(_module(_nop()), writer, kernel, "MAINLOOP")
        assert len(registryFor(writer).rows()) == 1

    def test_scope_helpers_are_noops_when_disabled(self):
        from Tensile.Components.Subtile.SqttMarkers import (
            sqttMainloopScopeOpen, sqttMainloopScopeClose)
        off = _kernel(enabled=False)
        assert str(sqttMainloopScopeOpen(_make_writer(), off)).strip() == ""
        assert str(sqttMainloopScopeClose(_make_writer(), off)).strip() == ""


def _group(pi, k):
    from rocisa.code import Module
    m = Module("g")
    m.addComment0(f"partition={pi} subIterK={k}")
    return m.flatitems()[0]


class TestNestedPhaseScopes:
    """One scope per (partition, subIterK) group, nested in the iteration scope.

    The group is the only contiguous unit: instructionSchedule interleaves
    gr/lr/mfma between the MFMAs, so an opType breakdown would not nest.
    """

    def _out(self, writer, kernel, groups, label="MAINLOOP_C0"):
        from Tensile.Components.Subtile.SqttMarkers import insertSqttMarkers
        items = []
        for pi, k in groups:
            items += [_group(pi, k), _nop()]
        return insertSqttMarkers(_module(*items), writer, kernel, label)

    def test_first_group_enters_and_rest_transition(self):
        from Tensile.Components.Subtile.SqttMarkers import (
            FLAG_ENTER, FLAG_TRANSITION, encodeMarker)
        out = self._out(_make_writer(), _kernel(), [(0, 0), (0, 1), (0, 2)])
        text = str(out)
        # id 1 is the iteration scope; phases start at 2.
        assert f"{encodeMarker(2, FLAG_ENTER):#x}" in text
        assert f"{encodeMarker(3, FLAG_TRANSITION):#x}" in text
        assert f"{encodeMarker(4, FLAG_TRANSITION):#x}" in text

    def test_same_subiterk_across_partitions_shares_one_id(self):
        # Partitions aggregate: k=0 in partition 1 must reuse k=0's id.
        from Tensile.Components.Subtile.SqttMarkers import registryFor
        w = _make_writer()
        self._out(w, _kernel(), [(0, 0), (0, 1), (1, 0), (1, 1)])
        rows = registryFor(w).rows()
        assert len(rows) == 3  # iteration + subiterk_0 + subiterk_1

    def test_phases_are_closed_before_the_body_ends(self):
        """Without the trailing exit the body leaves the loop one level deep.

        Checks the *last* marker is a bare exit, by decoded value -- a substring
        test for "0x1" would also match 0x13, 0x1a and friends.
        """
        from Tensile.Components.Subtile.SqttMarkers import FLAG_EXIT_PREV
        out = self._out(_make_writer(), _kernel(), [(0, 0), (0, 1)])
        last = str(_markers(out)[-1]).split()[1]
        assert int(last, 16) == FLAG_EXIT_PREV

    def test_marker_count_is_n_plus_one_not_two_n(self):
        # Fused chaining: N groups + 1 closing exit, plus the iteration marker.
        groups = [(0, k) for k in range(4)]
        out = self._out(_make_writer(), _kernel(), groups)
        assert len(_markers(out)) == 1 + len(groups) + 1

    def test_no_phases_outside_the_mainloop(self):
        # No iteration scope is open there, so a phase would sit at depth 0.
        out = self._out(_make_writer(), _kernel(), [(0, 0), (0, 1)], label="NLL_C0")
        assert _markers(out) == []

    def test_phase_rows_are_scopes_in_the_funcmap(self):
        from Tensile.Components.Subtile.SqttMarkers import sqttFuncmapSection
        w = _make_writer()
        self._out(w, _kernel(), [(0, 0), (0, 1)])
        text = str(sqttFuncmapSection(w, _kernel()))
        assert "U:2:subtile_subiterk_0" in text
        assert "U:3:subtile_subiterk_1" in text

    def test_disabled_emits_no_phases(self):
        out = self._out(_make_writer(), _kernel(enabled=False), [(0, 0), (0, 1)])
        assert _markers(out) == []


class TestMarkerForm:
    """Every marker is one s_ttracedata_imm and touches no register.

    The alternative -- an m0-sourced s_ttracedata carrying a packed gfx12 shader
    clock -- is deliberately not offered. rocprofv3 flattens .sqtt_funcmap into
    code.json as (id, kind, name) rows, which the M: row announcing such a
    layout is not, and RCV's marker walker decodes id = value >> 2 with no clock
    mask; a packed word therefore resolves as Unknown. See the module docstring.
    """

    def _text(self, mode=1, **kw):
        from Tensile.Components.Subtile.SqttMarkers import insertSqttMarkers
        return str(insertSqttMarkers(_module(_nop()), _make_writer(),
                                     _kernel(mode=mode, **kw), "MAINLOOP"))

    def test_uses_the_immediate_form(self):
        t = self._text()
        assert "s_ttracedata_imm" in t
        assert "m0" not in t

    def test_touches_no_register(self):
        # No scratch SGPR and no m0 save/restore: these kernels sit near the
        # SGPR ceiling, and m0 may hold a live DirectToLds base address.
        import re
        assert not re.search(r"\bs\[?\d", self._text())

    def test_costs_one_instruction_per_marker(self):
        from Tensile.Components.Subtile.SqttMarkers import markerExit
        w = _make_writer()
        assert len([l for l in str(markerExit(w)).splitlines() if l.strip()]) == 1

    def test_mode_2_is_not_a_valid_parameter_value(self):
        from Tensile.Common.ValidParameters import validParameters
        assert validParameters["SubtileSqttMarkers"] == [0, 1]

    def test_mode_0_is_off(self):
        from Tensile.Components.Subtile.SqttMarkers import sqttMarkersEnabled
        assert not sqttMarkersEnabled(_make_writer(), _kernel(mode=0))
