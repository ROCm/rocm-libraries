# WaveScope Integration

WaveScope is an ATT trace viewer. This page is about the seam between it and
rocke: what each side owns, what flows across, and how to use the result to make
a kernel faster.

The tools live in
[`../optimization/utilities/tools/wavescope/`](../optimization/utilities/tools/wavescope/README.md),
which is the operational guide — installing the extension, capturing, reading a
folder. This page is the design behind it.

## The gap it closes

`rocprofv3 --att` records what every wave on one CU executed, cycle by cycle.
Decoded, it tells you exactly which instruction stalled and for how long. That is
ground truth, and for hand-written assembly it is enough.

It is not enough for rocke. A rocke kernel is Python that *builds* IR, so the
instruction that stalled has no obvious author: by the time the compiler sees
anything, the builder call that emitted it has returned and its stack is gone. The
question you actually have — "which part of my tiling scheme causes this
`s_waitcnt`" — is one level above what the trace can answer.

Closing that gap needs the authoring context recorded during the build, carried
through compilation as debug info, and rejoined to the trace afterwards.

## Pipeline

```
your bench script                    ROCKE_DEBUG_LOC=1 set on this process
  │
  ▼  core/ir.py           IRBuilder._emit() walks the Python stack
op.loc = "file:line:col:func;..."    innermost frame first
  │
  ▼  core/lower_llvm.py   DICompileUnit / DISubprogram / DILocation chain, !dbg
.ll with debug metadata
  │
  ▼  comgr                normal compile, no extra flags
.hsaco carrying DWARF
  │
  ▼  rocprofv3 --att      decode; also dumps the code object and copies sources in
ui_output_*_dispatch_*/   code.json (innermost frame only) + source_* snapshots
  │
  ▼  emit_inline_frames.py   llvm-dwarfdump inline tree, joined to code.json
                             by (Codeobj, Vaddr)
inline_frames.json        the full call stack per instruction
  │
  ▼  WaveScope extension
Source tab: self / + inlined
```

Each stage is separately observable, which matters because the failure mode is
silence: with the environment variable unset every stage still "succeeds" and you
simply get an empty Source tab at the end. `capture_wavescope_trace.py` exists to
run the stages that are easy to forget as one command.

### Build: capturing the author

`IRBuilder._emit()` is the single point every op passes through on its way into a
region, so the capture hooks there rather than in `_op()`. That placement is
load-bearing: `scf.for`, `scf_for_iter` and `scf_if` construct their `Op` directly
and would otherwise be unlabeled, which is exactly the control flow you most want
attributed.

The stack is filtered to drop stdlib frames and stored on `Op.loc` as a
`file:line:col:func` chain. Packing the chain into the existing `loc` string keeps
the IR schema unchanged, so serialization and the C++ engine seam are untouched.

### Lower: chain, not leaf

`lower_llvm.py` turns that chain into a linked list of `!DILocation` nodes joined
by `inlinedAt`, which is precisely how LLVM represents an inlined C++ call stack.
That representation is the reason the frames survive: `-O3` inlines the helper
away, but the inlining metadata is what optimization passes are obliged to
maintain, so the chain arrives intact in the final object.

Two details are easy to get wrong. The module needs the `Debug Info Version` flag
or LLVM silently drops every `!dbg` attachment. And file paths can contain colons,
so the frame parser reads right-to-left — digits first for column and line — rather
than splitting on the first separator.

### Post-process: why a sidecar

The decoder flattens each instruction's DWARF to its innermost frame. On a kernel
assembled from helpers that is nearly useless — on the GEMM this was built
against, one line of a masking helper owned about 94% of the stall cycles, which
says nothing about which phase issued the loads.

Rather than change the decoder, which is a separate upstream component,
`emit_inline_frames.py` reads the `DW_TAG_inlined_subroutine` tree out of the code
object and joins its PC ranges to `code.json`'s `Codeobj` and `Vaddr` columns. The
result is purely additive: a viewer without the sidecar behaves exactly as before,
and a sidecar that does not fit the trace is warned about rather than fatal — the
viewer compares how many entries found an instruction against how many the sidecar
carries, so a rebuild that moved half the addresses is reported too, not just one
that moved all of them. That property is what lets the
feature ship without coupling to a decoder release.

## Invariants

Debug capture is off by default. It costs a Python stack walk per op, which is
material on sweeps that build thousands of kernels, and populating `Op.loc`
**changes the emitted `.ll` bytes** — so the byte-identity gate between the Python
and C++ engines, and the IR goldens, run with it off.

What it does *not* change is the generated ISA: the same kernel built with and
without debug disassembles to the same 268 instructions. A trace captured with
capture enabled therefore measures the kernel you actually ship, which is the
whole reason the feature is usable for optimization rather than just for reading.

## Data contracts

| Artifact | Shape | Source of truth |
| --- | --- | --- |
| `Op.loc` | `file:line:col:func` frames, `;`-separated, innermost first | `core/ir.py` |
| debug metadata | `DILocation` chain via `inlinedAt`, one `DISubprogram` per Python function | `core/lower_llvm.py` |
| `inline_frames.json` | `{version: 2, functions, files, stacks: {"codeobj:addr": [[func, call_file, call_line, call_col], ...]}}`, outermost frame first, indices into the interned tables | `emit_inline_frames.py` |
| `code.json` | per-instruction rows; `Codeobj` and `Vaddr` together are the join key | rocprofv3 |

Virtual addresses are per code object, so a trace that loaded more than one has the
same address standing for different instructions. Both columns are therefore in the
key, and the producer skips rows belonging to any object other than the one the
DWARF came from. `version` is checked by the viewer, which refuses a layout it does
not know rather than reading it on the assumption that it resembles a known one.

Function and file names are interned in the sidecar because the same handful
repeat across hundreds of instructions and the file crosses a network hop to the
viewer on a remote workspace. Each frame records the **call site** — where that
frame was entered — so the innermost frame's own line remains in `code.json`'s
Source column and the two sources of truth do not contradict each other.

## Using it for optimization

Capture, then work top-down. The temptation is to open the Source tab first; the
wave-state breakdown is the better starting point because it tells you what kind
of problem you have before you go looking for a line to blame.

1. **Capture.** `capture_wavescope_trace.py -- python3 bench.py`. The default
   iteration range traces dispatches 2–3, skipping warmup, so you are not
   measuring first-call compilation.
2. **Read the state mix first.** The per-wave timeline is authoritative for where
   time went, independent of the per-instruction columns. A profile dominated by
   `WAIT` is a memory or dependency problem; one dominated by `STALL` is issue
   contention. These want different fixes, and the instruction listing cannot
   distinguish them.
3. **Find the hot instructions**, sorted by stall in the Trace tab.
4. **Attribute them.** In the Source tab, `self` shows the lines the compiler
   credits directly. When that lands on a one-line helper — which it usually does —
   switch to `+ inlined`. Call sites light up, files that contain nothing but
   calls appear as tabs, and the cost is charged to the phase that asked for the
   work rather than the utility that performed it.
5. **Walk the stack.** Selecting an instruction shows the frames it came from,
   innermost first, each clickable. This is the step that answers "who asked for
   this": on the GEMM, it separates the A-tile and B-tile loads that `self` mode
   collapses onto the same helper line. Source coverage went from 26 lines to 51
   on `gemm_universal.py` once the chain was available.
6. **Change one thing, recapture, compare.** Regenerate the sidecar if you rebuilt
   the kernel — a stale one silently attributes to the previous layout.

An agent-assisted variant of this loop exists: the viewer reads `annotations.json`
from the trace folder on open, and writes `notes.json` back into it, so an analysis
pass can mark up a trace and a human can reply in place.

### Pitfalls

- `code.json` columns 7 and 8 are hit-weighted **totals**, not per-execution
  averages. Divide by `Hit`. Multiplying instead produces stall figures larger
  than the kernel's wall-clock, which is the usual sign of this mistake.
- ATT traces **one CU**. It is the right instrument for instruction-level
  behavior and the wrong one for whole-GPU throughput or occupancy-limited
  effects; use the stage1/stage5 benchmark tooling for those.
- The Source tab reads the `source_*` snapshots rocprofv3 copied into the folder,
  not your working tree. A trace stays readable after you edit the kernel — and
  keeps showing the old source, which is a feature when comparing two captures and
  a trap when you forget.

## Limits

One dispatch per decoded folder and one CU per trace. The sidecar depends on
`llvm-dwarfdump`, which ships with ROCm. The viewer is not on the marketplace, so
the extension is built from source; the folder README covers that, including the
remote-SSH case.
