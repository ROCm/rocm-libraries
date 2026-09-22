# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SQTT marker instrumentation for the subtile mainloop (gfx1250).

Emits ``s_ttracedata_imm`` markers whose payload follows the rocprof-trace-decoder
marker ABI, together with the ``.sqtt_funcmap`` code-object section that names
them, so a ``rocprofv3 --att`` capture decodes into per-wave timelines with no
matching LLVM instrumentation pass.

Wire format -- one 32-bit marker word::

    bit    0   exit_prev   pop the top scope
    bit    1   enter       push scope `id`
    bits 31:2  id

so ``encoded = (id << 2) | flags``.  The marker *kind* is never encoded; the
decoder resolves it from the funcmap.

Why the immediate form
----------------------
``s_ttracedata`` is SOPP with no operand field, so hardware sources its 32-bit
payload from ``m0``.  The subtile mainloop keeps ``m0`` live for DirectToLds base
addressing, so that form would need a save/restore around every marker.
``s_ttracedata_imm`` carries the payload in SIMM16 and clobbers nothing: one
instruction, no hazard.  Two costs follow, and both shape this module:

* hardware captures only the low 8 bits of SIMM16, which after the two flag bits
  leaves ids 1..63 (``MAX_MARKER_ID``);
* it cannot carry gfx12 shader-clock packing, which needs the high bits, so no
  ``M:`` row is emitted and timestamps carry the usual SQTT drain skew.  A
  packed word would not survive the trip anyway: ``rocprofv3`` flattens
  ``.sqtt_funcmap`` into ``code.json`` as ``(id, kind, name)`` rows, which an
  ``M:`` row is not, and RCV's marker walker decodes ``id = value >> 2`` with no
  clock mask -- so the clock bits would land in the id and every marker would
  resolve as Unknown.

``s_ttracedata_imm`` does not exist on gfx9, which is why this is gfx1250-only.

Scopes via the fused transition
-------------------------------
The mainloop iteration is a *scope*, not a point, because the consumer that
attributes time -- ROCprof Compute Viewer's marker flamegraph -- works on
scopes.  A point gets ``enter_time == exit_time`` in RCV's marker walker and is
never pushed on the stack, so it contributes zero width and shows up only as an
instant on the timeline.  Markers that cannot be attributed time are close to
useless for answering "is my mainloop uniform across waves".

The scope is opened and closed by the *fused* exit+enter word (both flags set)
at the head of the loop body.  One instruction, in one place that every
iteration passes through, which:

* costs exactly what the point cost -- one ``s_ttracedata_imm``;
* pops iteration N and pushes iteration N+1 at the same stack depth, so each
  span is precisely one iteration;
* cannot be skipped by a branch out of the body, because it precedes the body.

A naive enter-at-top/exit-at-bottom pair would have that last problem, which is
what makes the fused form the right shape here rather than a convenience.  The
scope is primed by a plain enter before the loop and closed by a plain exit at
the post-loop join point, both placed by ``LogicalScheduler``; without the prime
the first transition of every wave would trip the decoder's "transition marker
with empty stack" warning.

Nested phase scopes
-------------------
Inside the iteration scope sits one scope per ``(partition, subIterK)`` group,
giving the flamegraph a second level: not just "an iteration costs N cycles" but
which K-step that went to.

The group is the *only* honest nesting unit here.  A breakdown by opType --
global read, local read, WMMA -- would be wrong, because ``instructionSchedule``
deliberately interleaves those between the MFMAs to hide latency, so they are
not contiguous intervals and cannot nest.  ``(partition, subIterK)`` groups are
emitted sequentially and do not overlap.

Groups are chained with fused transitions exactly like iterations, so N groups
cost N+1 instructions, not 2N.  They are named by ``subIterK`` alone: partitions
are a spatial tiling rather than a temporal phase, so aggregating them is what
answers "which K-step costs what", and it caps the id budget at ``numSubIterK``
instead of ``numPartitions * numSubIterK``.  Split them per partition by putting
``pi`` in :data:`MARKER_PHASE_FMT` if you need it -- the budget is 63.

The cluster-barrier markers stay points: they are genuinely instantaneous, and
as points they now nest *inside* the iteration scope at depth 1, which is where
they belong.

Using it
--------
Build the kernel with ``SubtileSqttMarkers: [1]`` in the solution YAML.  The
parameter is part of the kernel name (tag ``SSM``), so an instrumented kernel can
never dedup against, or be cached as, a production one.  Then::

    rocprofv3 --att -d trace_out -- ./hipblaslt-bench <args>

Then open the resulting ``ui_output_agent_<id>_dispatch_<id>/`` directory in
ROCprof Compute Viewer.  RCV reads markers natively: ``rocprofv3`` already
extracts ``.sqtt_funcmap`` into ``code.json``, and RCV's marker walker turns the
records into spans.  No post-processing step is needed.  The iteration scopes
appear in RCV's marker flamegraph; the cluster-barrier points appear as instants
nested inside them.

Note this is also why the mainloop marker must be a scope.  RCV's marker view is
a flamegraph weighted by latency, so a zero-width point contributes nothing to
it -- see the scope section above.

Two caveats when reading a capture:

* every wave emits, with no CU/wave gating -- gating would need a branch inside
  the mainloop, which would itself distort what is being measured.  Restrict the
  capture with rocprofv3's own SE/CU selection instead;
* ids are allocated per kernel, so linking several instrumented kernels into one
  code object collides them.  Instrument one kernel at a time.
"""

from __future__ import annotations

import re

from rocisa.code import Module, TextBlock

from .ClusterBarrier import CB_SIGNAL_COMMENT, CB_WAIT_COMMENT

# The alternative stinkytofu backend (ROCISA_BACKEND=stinkytofu) carries
# s_ttracedata but not the immediate form, so the import must not be fatal:
# this module is loaded for every subtile kernel, markers on or off, and a hard
# import would take the whole subtile path down on that backend. Absence is
# instead reported at the point a marker is actually requested.
try:
    from rocisa.instruction import STtraceDataImm
except ImportError:  # pragma: no cover - backend-dependent
    STtraceDataImm = None


# The SubtileSqttMarkers solution parameter: 0 off, 1 on (s_ttracedata_imm).
MODE_OFF = 0

# 8 captured payload bits minus the 2 flag bits.
MAX_MARKER_ID = 63

FLAG_EXIT_PREV = 0x1
FLAG_ENTER = 0x2
# Both flags: pop the previous scope and push a new one in one record.
FLAG_TRANSITION = FLAG_EXIT_PREV | FLAG_ENTER

# Funcmap row prefixes (see the marker SPEC): 'U' is a named scope, 'P' a point.
KIND_SCOPE = "U"
KIND_POINT = "P"

# Marker names.  These are what show up in the Perfetto / flamegraph output.
MARKER_MAINLOOP_ITER = "subtile_mainloop_iter"
MARKER_CB_SIGNAL = "subtile_cluster_barrier_signal"
MARKER_CB_WAIT = "subtile_cluster_barrier_wait"

# Nested phase scopes, one per (partition, subIterK) group inside an iteration.
# Named by subIterK only: partitions are a spatial tiling, not a temporal phase,
# so aggregating them is what answers "which K-step costs what", and it bounds
# the id budget at numSubIterK instead of numPartitions * numSubIterK.
MARKER_PHASE_FMT = "subtile_subiterk_{k}"

# The group header LogicalScheduler already emits: "/* partition=0 subIterK=2 */".
# Anchoring on it keeps every marker insertion post-schedule and in one place,
# the same way the cluster-barrier markers anchor on their comments.
_GROUP_ANCHOR = re.compile(r"partition=(\d+)\s+subIterK=(\d+)")


def encodeMarker(markerId: int, flags: int = 0) -> int:
    """``(id << 2) | flags``, validated against the immediate form's 8-bit window."""
    if not 0 <= markerId <= MAX_MARKER_ID:
        raise ValueError(
            f"SQTT marker id {markerId} outside 0..{MAX_MARKER_ID}; "
            "s_ttracedata_imm captures only 8 payload bits")
    return (markerId << 2) | flags


class MarkerRegistry:
    """Per-kernel marker id allocator and funcmap row builder.

    Ids come from a single counter starting at 1, as the SPEC requires.  The
    registry is per *kernel*; ids are only unique within one code object, so a
    build that links several instrumented kernels into one ``.co`` will collide.
    That is acceptable while ``SubtileSqttMarkers`` is an opt-in debug build, and
    is the first thing to revisit if that changes.
    """

    def __init__(self):
        self._ids = {}     # name -> id
        self._kinds = {}   # name -> KIND_*
        self._next = 1

    def idFor(self, name: str, kind: str = KIND_POINT) -> int:
        if name not in self._ids:
            if self._next > MAX_MARKER_ID:
                raise ValueError(
                    f"SQTT marker budget exhausted (>{MAX_MARKER_ID} distinct markers); "
                    "s_ttracedata_imm cannot address more")
            self._ids[name] = self._next
            self._kinds[name] = kind
            self._next += 1
        return self._ids[name]

    def isEmpty(self) -> bool:
        return not self._ids

    def rows(self):
        """Funcmap rows for every registered marker, in id order."""
        return [f"{self._kinds[n]}:{i}:{n}"
                for n, i in sorted(self._ids.items(), key=lambda kv: kv[1])]


def registryFor(writer) -> MarkerRegistry:
    """The registry for the kernel currently being written, created on demand."""
    reg = getattr(writer, "_sqttMarkerRegistry", None)
    if reg is None:
        reg = MarkerRegistry()
        writer._sqttMarkerRegistry = reg
    return reg


def resetRegistry(writer) -> MarkerRegistry:
    """Start a fresh registry.  Called once per kernel, before the body is built."""
    writer._sqttMarkerRegistry = MarkerRegistry()
    return writer._sqttMarkerRegistry


def sqttMarkersEnabled(writer, kernel) -> bool:
    """True when this kernel should carry markers.

    ``Solution.py`` already restricts ``SubtileSqttMarkers`` to gfx1250 subtile
    kernels; the ISA check here is a backstop so a hand-written solution YAML
    cannot emit ``s_ttracedata_imm`` onto a target that lacks the opcode.
    """
    mode = int(kernel.get("SubtileSqttMarkers", MODE_OFF) or MODE_OFF)
    if mode == MODE_OFF:
        return False
    if tuple(kernel["ISA"])[:2] != (12, 5):
        return False
    if STtraceDataImm is None:
        raise RuntimeError(
            "SubtileSqttMarkers requires s_ttracedata_imm, which the active "
            "rocisa backend does not provide (the stinkytofu backend has only "
            "the m0-sourced s_ttracedata). Build with the default backend.")
    return True


def _marker(writer, name: str, flags: int, kind: str, note: str) -> Module:
    """One ``s_ttracedata_imm``: the payload rides in SIMM16, no register is
    touched, and there is no hazard to pad against."""
    markerId = registryFor(writer).idFor(name, kind)
    mod = Module(f"sqtt_{name}")
    mod.add(STtraceDataImm(encodeMarker(markerId, flags),
                           f"sqtt {note} {name} (id={markerId})"))
    return mod


def markerPoint(writer, name: str) -> Module:
    """A zero-width event: no flags, so the decoder's scope stack is untouched."""
    return _marker(writer, name, 0, KIND_POINT, "point")


def markerEnter(writer, name: str) -> Module:
    """Push a scope.  The caller owns balancing this against :func:`markerExit`."""
    return _marker(writer, name, FLAG_ENTER, KIND_SCOPE, "enter")


def markerTransition(writer, name: str) -> Module:
    """Close the open scope and open a fresh one, in a single record.

    The loop-head form: span N ends and span N+1 begins at the same instant, at
    the same stack depth, for the cost of one instruction.
    """
    return _marker(writer, name, FLAG_TRANSITION, KIND_SCOPE, "transition")


def markerExit(writer) -> Module:
    """Pop the top scope.  The id field is ignored on exit, so the word is 0x1."""
    mod = Module("sqtt_exit")
    mod.add(STtraceDataImm(FLAG_EXIT_PREV, "sqtt exit"))
    return mod


def _commentOf(item) -> str:
    return getattr(item, "comment", "") or ""


def insertSqttMarkers(module, writer, kernel, label: str):
    """Splice markers into the post-schedule instruction order.

    No-op unless ``SubtileSqttMarkers`` is on.  Runs after ``insertClusterBarrier``
    so the barrier halves it anchors to are already in place.

    Placed markers:

    * one point at the head of each MAINLOOP body, giving one timestamp per
      iteration per wave -- the per-iteration period falls straight out of the
      Perfetto timeline;
    * one point after the cluster-barrier signal and one after the wait, which
      together measure whether the handshake's cross-CU latency really does hide
      behind the WMMAs issued in between.

    Returns a rebuilt Module; the input is left untouched.
    """
    if not sqttMarkersEnabled(writer, kernel):
        return module

    items = module.flatitems()
    out = Module(module.name)

    isMainloop = label.startswith("MAINLOOP")
    if isMainloop:
        for m in markerTransition(writer, MARKER_MAINLOOP_ITER).flatitems():
            out.add(m)

    phaseOpen = False
    for item in items:
        # Phase scopes nest one level inside the iteration scope. Chained with
        # fused transitions, so N groups cost N+1 instructions rather than 2N.
        if isMainloop and isinstance(item, TextBlock):
            anchor = _GROUP_ANCHOR.search(str(item))
            if anchor:
                name = MARKER_PHASE_FMT.format(k=int(anchor.group(2)))
                phase = (markerTransition(writer, name) if phaseOpen
                         else markerEnter(writer, name))
                out.add(item)
                for m in phase.flatitems():
                    out.add(m)
                phaseOpen = True
                continue
        out.add(item)
        # Anchor on the comments ClusterBarrier authors; the rendered mnemonic
        # varies with arch caps, the comment does not.
        comment = _commentOf(item)
        if comment == CB_SIGNAL_COMMENT:
            for m in markerPoint(writer, MARKER_CB_SIGNAL).flatitems():
                out.add(m)
        elif comment == CB_WAIT_COMMENT:
            for m in markerPoint(writer, MARKER_CB_WAIT).flatitems():
                out.add(m)

    # Close the last phase so the body ends back at the iteration scope's depth.
    # This lands before the loop-control setup and back-edge branch, which the
    # caller appends after this module.
    if phaseOpen:
        for m in markerExit(writer).flatitems():
            out.add(m)

    return out


def sqttMainloopScopeOpen(writer, kernel) -> Module:
    """Prime the iteration scope, immediately before the mainloop label.

    Without this the first fused transition in every wave pops an empty stack,
    which the decoder reports as "transition marker with empty stack" -- once
    per wave, which drowns the diagnostics pane.
    """
    mod = Module("sqtt_mainloop_scope_open")
    if not sqttMarkersEnabled(writer, kernel):
        return mod
    for m in markerEnter(writer, MARKER_MAINLOOP_ITER).flatitems():
        mod.add(m)
    return mod


def sqttMainloopScopeClose(writer, kernel) -> Module:
    """Close the iteration scope at the post-mainloop join point.

    Every loop exit converges here, so one exit closes the scope on all paths.
    Left open, the final span would carry exit_time = INT64_MAX and render as a
    bar running to the end of the trace.
    """
    mod = Module("sqtt_mainloop_scope_close")
    if not sqttMarkersEnabled(writer, kernel):
        return mod
    for m in markerExit(writer).flatitems():
        mod.add(m)
    return mod


def sqttFuncmapSection(writer, kernel) -> Module:
    """The ``.sqtt_funcmap`` section naming every marker this kernel emitted.

    A non-alloc ``@progbits`` blob of newline-separated rows.  The TensileLite
    link step passes no ``--gc-sections``, so it survives into the linked
    ``.co``, where ``rocprofv3`` (or ``llvm-objcopy --dump-section``) reads it
    back.

    No ``M:`` row: that advertises shader-clock packing, which the immediate
    marker form cannot carry.  Emitted after the body so every marker the body
    registered is included.
    """
    mod = Module("sqtt_funcmap")
    if not sqttMarkersEnabled(writer, kernel):
        return mod

    reg = registryFor(writer)
    if reg.isEmpty():
        return mod

    rows = [f"W:{kernel['WavefrontSize']}",
            f"K:{writer.states.kernelName}"]
    rows += reg.rows()

    # '\n' must reach the assembler as the two-character escape inside .asciz.
    payload = "".join(r + "\\n" for r in rows)
    mod.add(TextBlock(
        "\n// SQTT marker map -- consumed by rocprof-trace-decoder\n"
        ".section .sqtt_funcmap,\"S\",@progbits\n"
        f".asciz \"{payload}\"\n"
        ".text\n"))
    return mod
