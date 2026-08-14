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

## Vocabulary

Two debug-info vocabularies meet in this pipeline, and they are similar enough to
be confusing: LLVM's `DI*` metadata nodes, which rocke *writes*, and DWARF's DIEs
and tags, which those nodes become once compiled and which the sidecar *reads*.
Everything defined here is used somewhere below or in the tools.

### What rocke writes: LLVM debug metadata

`DI` is simply LLVM's prefix for "debug info". These nodes live in the emitted
`.ll` as `!`-prefixed metadata, and building them is all `core/lower_llvm.py` and
`cpp/core/lower_llvm/debug.cpp` do.

| Node | What it is |
| --- | --- |
| `!DIFile` | one source file, as a filename plus a directory |
| `!DICompileUnit` | the root of a module's debug info: which file, which producer (`rocke`), and how much detail to emit |
| `!DISubprogram` | a function. rocke emits one for the kernel, plus one for each Python function that contributed ops to it |
| `!DILocation` | a single point in source — line, column, and enclosing scope. Attached to an instruction as `!dbg !N` |
| `!DILexicalBlockFile` | a scope whose file differs from its function's, used for a location that arrived without a call stack |

Two fields matter more than the rest. **`inlinedAt`** on a `!DILocation` points at
the location it was inlined into, so a chain of them *is* a call stack — that
chain is the entire mechanism by which a rocke kernel's Python authorship
survives to the profiler. And **`emissionKind`** says how much to emit;
`LineTablesOnly` means enough to map addresses to lines and inline scopes, but not
the variable and type description a source debugger would want.

The `Debug Info Version` module flag is not optional: without it LLVM silently
discards every `!dbg` attachment, and nothing downstream can distinguish that from
a build that never captured locations.

### What the sidecar reads: DWARF

**DWARF** is the debug format that ends up *inside* the compiled code object.
Everything downstream — debuggers, `rocprofv3`, `emit_inline_frames.py` — reads
DWARF, never the `.ll`.

DWARF is a tree of **DIEs** (Debugging Information Entries). Each DIE has a **tag**
naming what it describes and **attributes** carrying the details.
`llvm-dwarfdump --debug-info` prints one DIE per line, indented by depth, which is
how the sidecar recovers which frame encloses which.

| Tag | What it describes |
| --- | --- |
| `DW_TAG_compile_unit` | the root, from `!DICompileUnit` |
| `DW_TAG_subprogram` | a real function, from `!DISubprogram` |
| `DW_TAG_inlined_subroutine` | **one inlined call** — a function body pasted into its caller. One per call site rather than one per function, and the entry the sidecar is built from |

| Attribute | What it carries |
| --- | --- |
| `DW_AT_name` | the name |
| `DW_AT_abstract_origin` | on an inlined subroutine, a pointer to the DIE holding the name |
| `DW_AT_call_file`, `DW_AT_call_line`, `DW_AT_call_column` | where the call was made *from*. Recorded on the callee, which is why the sidecar's frames carry call sites rather than declaration sites |
| `DW_AT_low_pc` / `DW_AT_high_pc`, or a list of `[lo, hi)` ranges | the instruction addresses this DIE covers |

A **PC** is a program counter — the address of one instruction. PC ranges are the
hinge of the whole design: the trace measures per address and DWARF describes per
address range, so the two can be joined afterwards without either side having been
built to know about the other.

### The trace side

| Term | Meaning |
| --- | --- |
| **ATT** | Advanced Thread Trace, the `rocprofv3` mode that records what each wave executed, cycle by cycle |
| **dispatch** | one launch of one kernel — the kernel is the compiled code, a dispatch is one execution of it. Profiling is per dispatch, and each decoded dispatch gets its own folder |
| **code object** | the compiled GPU binary that gets loaded (`.hsaco`), and the thing carrying the DWARF |
| **`Vaddr`, `Codeobj`** | `code.json` columns: an instruction's address, and which code object it belongs to. Addresses restart per object, so both are needed to name an instruction |
| **wave / wavefront** | lanes executing in lockstep, 64 of them on the CDNA parts. ATT reports per wave |
| **CU** | compute unit. ATT traces exactly one |
| **comgr** | the ROCm compiler library rocke hands the `.ll` to |
| **sidecar** | a file written beside a trace that the viewer loads if present and ignores if absent — here, `inline_frames.json` |

## Pipeline

```
your bench script                    ROCKE_DEBUG_LOC=1 set on this process
  │
  ▼  core/ir.py           IRBuilder._emit() walks the Python stack
op.loc = "file:line:col:func;..."    innermost frame first
  │
  ▼  core/lower_llvm.py   DICompileUnit / DISubprogram / DILocation chain, !dbg
.ll with debug metadata      (the C++ engine emits the same bytes)
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

What survives the filter is the point: `helpers/` and `instances/` are where a
shipped kernel is actually written, so those frames are kept, and they are kept
whether rocke is imported from a checkout or from site-packages. Classifying an
installed rocke as the harness that launched the build — which a plain
`site-packages` test did — ended the walk at the first helper and left every op
with no location at all, from a run that otherwise looked like it worked.

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

Both engines do this, not just the Python one. The C++ engine receives `debug_info`
and every `@loc` through the serialized IR, and `cpp/core/lower_llvm/debug.cpp`
mirrors `_DebugInfo` node for node — same id allocation order, same cache keys — so
the two emit the same bytes and `ROCKE_BACKEND=both` still passes with capture on.
It has to: `cpp` is the default backend wherever `rocke_engine` is installed, so an
engine that ignored the locations would mean the documented one-command capture
produced an object with no DWARF in it on exactly the installations most people
have.

`emissionKind` stays `LineTablesOnly`. That is enough for the inline tree —
`DW_TAG_subprogram` with `DW_AT_name`, `DW_TAG_inlined_subroutine` with
`DW_AT_abstract_origin` / `DW_AT_call_file` / `DW_AT_call_line` — and
`TestObjectRoundTrip` in `tests/core/test_debug_info.py` proves it by assembling
the emitted IR into a real AMDGPU object and dumping the DWARF back out, which is
the only check that would notice the metadata being dropped downstream of the `.ll`.

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

Nothing in the file can prove it belongs to the trace it sits beside: the keys are
bare addresses. So the producer does not try to validate what it finds, it
replaces it — every sidecar under the trace is removed as soon as the dispatch
folders are known, ahead of code-object discovery, of locating `llvm-dwarfdump`,
and of the loop, all of which can end the run early. Each write then goes to a
temporary that is renamed over the destination and deleted if anything raises.
A dispatch is therefore left holding a complete sidecar from this run or none,
where none degrades to innermost-frame attribution and the previous run's answer
would silently attribute to a layout that no longer exists. `--no-source` follows
the same rule: it writes no sidecar, so it removes any it finds rather than
leaving one beside a trace it does not describe.

## Invariants

Debug capture is off by default. It costs a Python stack walk per op, which is
material on sweeps that build thousands of kernels, and populating `Op.loc`
**changes the emitted `.ll` bytes** — so the byte-identity gate between the Python
and C++ engines, and the IR goldens, run with it off.

The gate runs with it off for a second reason: its two sides build the same kernel
independently, once in Python and once in C, and the C emitters have no Python
stack to record. Comparing them with capture on would report drift for two kernels
that lowered identically, so `run_diff.py` drops the variable from the reference
side. The engines' debug output is compared where the comparison means something —
one kernel, both lowerers — by `tests/core/test_debug_info.py` under
`ROCKE_BACKEND=both`.

What it does *not* change is the generated ISA: the same kernel built with and
without debug disassembles to the same 268 instructions. A trace captured with
capture enabled therefore measures the kernel you actually ship, which is the
whole reason the feature is usable for optimization rather than just for reading.

## Data contracts

| Artifact | Shape | Source of truth |
| --- | --- | --- |
| `Op.loc` | `file:line:col:func` frames, `;`-separated, innermost first | `core/ir.py` |
| debug metadata | `DILocation` chain via `inlinedAt`, one `DISubprogram` per Python function | `core/lower_llvm.py`, mirrored by `cpp/core/lower_llvm/debug.cpp` |
| `inline_frames.json` | `{version: 2, functions, files, stacks: {"codeobj:addr": [[func, call_file, call_line, call_col], ...]}}`, outermost frame first, indices into the interned tables | `emit_inline_frames.py` |
| `code.json` | per-instruction rows; `Codeobj` and `Vaddr` together are the join key | rocprofv3 |

Virtual addresses are per code object, so a trace that loaded more than one has the
same address standing for different instructions. Both columns are therefore in the
key, and the producer skips rows belonging to any object other than the one the
DWARF came from. `version` is checked by the viewer, which refuses a layout it does
not know rather than reading it on the assumption that it resembles a known one.

Which object that is gets decided per dispatch, by matching the id rocprofv3 named
the dump with against the ids that dispatch's own rows carry — not once for the
whole trace. A wrong pick does not fail: the addresses overlap, so the join
succeeds and reports another kernel's source. So a dispatch whose object cannot be
identified is reported and skipped, and `--code-object` is how you settle it.

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
