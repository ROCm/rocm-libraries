# WaveScope

**WaveScope** is the viewer that turns an ATT trace into a per-wave timeline, an
ISA listing, and a Source tab that maps instructions back to the Python that
authored them. The scripts here produce trace, source-attribution, and PMC artifacts.

All scripts use the Python standard library and run from any working directory.
The PMC utility selects the adjacent `platform/python` perf package; the ATT and
inline-frame utilities import no rocKE. From `platform/`, these tools live in
`dsl_docs/optimization/utilities/tools/wavescope/`. Each takes `--help`.

- `capture_wavescope_trace.py` — capture a source-correlated trace end to end.
- `emit_inline_frames.py` — write the inline-frames sidecar for a trace you
  already decoded.
- `capture_wavescope_pmc.py` — capture hardware-counter CSVs for WaveScope and
  versioned measurement JSON, through the perf primitives.

## What WaveScope shows

`rocprofv3 --att` records what every wave executed, cycle by cycle, on one CU.
Decoded, that becomes a folder of JSON the viewer reads. Opened in the editor you
get the wave timeline (where the stalls are), the ISA with per-instruction hit and
stall totals, and — if the kernel was built with source locations — a Source tab
that highlights the Python lines those instructions came from, clickable in both
directions.

WaveScope lives in its own repo, <https://github.com/ROCm/WaveScope>. It is
not published to the marketplace, so installing means building the `.vsix` once.

## Install

You need Node. The viewer's `package.json` pins the supported versions
(`^22.22.2 || ^24.15.0 || >=26.0.0` at the time of writing), and VS Code / Cursor
must be 1.75 or newer. Then, from a clone of WaveScope:

```bash
cd WaveScope/vscode
npm install
npx --yes @vscode/vsce package --allow-missing-repository --skip-license
```

That builds the React viewer, copies it into `media/`, bundles the extension host,
and writes `wavescope-<version>.vsix` (~114 KB). There is also an `npm run
package` script, but it calls `vsce` directly and fails with `vsce: not found`
unless you have it installed globally; the `npx` line above needs nothing
preinstalled.

Install the result:

```bash
code --install-extension /path/to/wavescope-0.4.0.vsix --force
```

Then reload the window (**Developer: Reload Window**). `--force` is what lets a
rebuilt `.vsix` replace an already-installed copy of the same version, which is
the normal case when you are iterating on the viewer.

### If you work over SSH

The extension has to be installed on the **remote** host, because that is where
the trace files are and where the extension host runs. Running `code
--install-extension` from a terminal inside the remote window does the right thing
— the `code` on `PATH` there is the remote CLI. Confirm with:

```bash
code --list-extensions --show-versions
```

which prints a header naming the host it is reporting on, e.g. `Extensions
installed on SSH: my-box:`. If you see your laptop's extensions instead, you are
in a local terminal and the install went to the wrong side.

## Capture a trace

```bash
python3 capture_wavescope_trace.py -- python3 bench.py
```

Three separate things have to line up before the Source tab works, and each fails
quietly on its own, so prefer the script over assembling the `rocprofv3` command
by hand. It sets `ROCKE_DEBUG_LOC=1` on the process that *builds* the kernel
(source correlation is not a compiler flag — see
[`env_flags.md`](../../../../reference/env_flags.md)), runs the capture via
`../stage2_capture/capture_att_trace.py`, generates the inline-frames sidecar, and
prints the folder to open. Unrecognized flags are forwarded to the capture script.

## Capture PMC evidence (CSV and JSON)

This is separate from ATT tracing. The utility delegates to the existing rocKE
perf CLI; it does not reimplement profiling or require WaveScope to be installed.
From `platform/`:

```bash
python3 dsl_docs/optimization/utilities/tools/wavescope/capture_wavescope_pmc.py \
  --output-dir /tmp/gemm-pmc-before \
  --arch gfx950 --op gemm --shape '{"M":512,"N":512,"K":512}' \
  --kernel-name my_gemm --match-kernel my_gemm --repeats 3 --warmup 5 \
  --per-dispatch -- python3 /path/to/bench.py
```

To build one WaveScope-ready folder after ATT capture, set `TRACE_DIR` to the
reported `ui_output_*_dispatch_*` directory and keep the complete PMC bundle beside
the ATT capture tree:

```bash
RUN_DIR="$(dirname "$TRACE_DIR")"
python3 dsl_docs/optimization/utilities/tools/wavescope/capture_wavescope_pmc.py \
  --output-dir "$RUN_DIR/pmc_bundle" --trace-dir "$TRACE_DIR" \
  --arch gfx950 --op gemm --shape '{"M":512,"N":512,"K":512}' \
  --kernel-name my_gemm --match-kernel my_gemm --repeats 3 --warmup 5 \
  --per-dispatch -- python3 /path/to/bench.py
```

Choose the actual operation (attention, convolution, GEMM, etc.), shape, target,
kernel and warmup count. The launcher runs in your current working directory.
The adjacent perf package is added to the child `PYTHONPATH`; existing entries
are preserved. Add the rocKE `library` path yourself if your launcher needs it.
Every bundle output directory must be new. `--trace-dir` must name an existing
WaveScope ATT dispatch folder containing `code.json`, `filenames.json` and
`occupancy.json`; existing counter sidecars are never overwritten.

**A successful PMC capture produces this CSV and JSON bundle:**

```text
gemm-pmc-before/
  manifest.json                 versioned entry point, file hashes and relative paths
  measurement.json              aggregate measurement/v1 (medians and spread)
  comparison.json               existing perf selfcheck result
  samples/0000/
    measurement.json            single-repeat measurement and capture metadata
    raw/pmc.txt                 requested hardware-counter recipe
    raw/prof/.../*counter_collection.csv
  samples/0001/...
```

With `--trace-dir`, the utility selects the first successful profiler repeat and
copies each replay-pass CSV beside `code.json` using distinct
`rocke_pmc_*_counter_collection.csv` names. WaveScope discovers and merges those
top-level sidecars automatically when it opens the trace folder. Keep the complete
bundle as a sibling of the ATT capture tree; nesting it below `TRACE_DIR` would let
the browser folder picker recursively load retained repeats in addition to the
published sidecars. The bundle preserves every repeat, JSON record and hash.

Without `--trace-dir`, open the ATT dispatch, select **Bottlenecks**, and manually
upload a recommended CSV printed by the utility. Each manual upload replaces the
previous one. With the full CDNA selection, `pmc_1` contains the LDS, L2, VALU and
MFMA rule inputs; `pmc_2` contains LDS instructions and wait cycles. Check
`profile_capture.counter_groups` for the actual selection. Keep repeats separate.
Raw CSVs retain warmup and other-kernel dispatches; JSON medians select the target
and exclude warmup. Ratios of raw sums and ratios of medians can differ.

To consume the **JSON contract**, retain the entire bundle and read
`manifest.json`. It uses `rocke.bench.artifacts/v1`; measurements use
`rocke.bench.measurement/v1`.
See the [artifact contract](../../../../../python/rocke/benchmark/perf/README.md#portable-artifacts-counter-csvs-and-measurement-json)
for selection, repeat identity, SHA-256 inventory and timing-source semantics.
The bundle layout contains original profiler artifacts and versioned measurements;
each consumer reads the corresponding entries in the manifest.
WaveScope's Bottlenecks visualizer derives PMC state from counter CSVs, not from
rocKE `measurement.json`. Keep JSON for agents and other schema-aware consumers;
publish the selected CSV sidecars for WaveScope.

The utility defaults to **export-only** (no history writes). Add `--store-history`
to compare later captures against stored baselines; `--cache`, `--threshold`,
`--noise-k` and `--json` pass through to the perf CLI. Progress and import paths
go to stderr so `--json` stdout stays machine-readable. Regressions retain exit 1.

The manifest distinguishes complete measurement export from profiler success.
A launcher emitting `PerfJSON:` can yield a complete wall-only export when the
profiler fails. Failed profiler files remain available for troubleshooting;
upload guidance selects CSVs only from successful profiler samples. An export
failure leaves a failed or incomplete manifest rather than a finalized baseline.

**Association is UNBOUND.** These counters were collected separately from ATT.
Check workload, shape, GPU and build before correlating them. File hashes prove
artifact integrity, not that two captures ran the same binary. The counter set
includes LDS-conflict and CDNA VALU/MFMA inputs; full roofline analysis requires
additional counters. Returned zero-valued counters do not establish hardware support.

## Open it

Command palette (`Ctrl+Shift+P`) → **WaveScope: Open Trace Folder…** → pick a
`ui_output_*_dispatch_*` directory. One dispatch per folder; they are
self-contained, so a folder copied off the machine still opens.

| Command | Does |
| --- | --- |
| `WaveScope: Open Trace Folder…` | pick a decoded dispatch directory |
| `WaveScope: Open Trace Viewer` | empty viewer; drop a folder onto it |
| `WaveScope: Reveal Viewer` | focus it again — `Ctrl+Alt+W` / `Cmd+Alt+W` |

Another extension can drive it without the palette, either through the exported
API or by running `wavescope.openTraceDir` with a path:

```js
const api = await vscode.extensions.getExtension("flydsl.wavescope").activate();
api.openTrace("/path/to/ui_output_..._dispatch_0");
```

## What the viewer reads

Useful when a folder looks wrong or is bigger than you expect:

| File | Role |
| --- | --- |
| `code.json` | the ISA, with per-instruction hits and stall totals |
| `filenames.json` | names the per-wave files to load |
| `se*_sm*_sl*_wv*.json` | one wave's timeline |
| `occupancy.json` | occupancy over time |
| `source_<n>_<name>` | source snapshots rocprofv3 copies in when the code object had DWARF |
| `inline_frames.json` | optional; the inlining call stack (see below) |
| `wstates*.json`, `realtime.json` | **never read** — often ~18% of the bytes |

Source text comes from those `source_*` snapshots, not from your working tree, so
the Source tab is empty when the kernel was built without `ROCKE_DEBUG_LOC=1` —
there was no DWARF for rocprofv3 to copy sources from.

## Source tab: `self` vs `+ inlined`

rocprofv3 keeps only the innermost DWARF frame per instruction. On a kernel
assembled out of helpers that is close to useless: most of the stall cycles land
on one line of some masking helper, which says nothing about which phase issued
the loads.

`inline_frames.json` restores the rest, and the tab then offers two attributions.
**self** charges each line for the instructions the compiler credits to it — the
view you get with no sidecar. **+ inlined** also charges each line with everything
inlined into it, so call sites light up, files containing nothing but calls appear
as tabs, and selecting an instruction shows the frames it came from, each
clickable. `capture_wavescope_trace.py` writes the sidecar for you;
`emit_inline_frames.py <capture-generation-dir>` regenerates it against a trace
you already have. For a legacy trace with no capture sentinel, review the trace
first and opt in explicitly with `--assume-complete`. The output root is not an
implicit alias for one of its `capture-*` children: choose one generation
explicitly so dispatches and code objects from separate captures are never mixed.

Re-running the sidecar producer over a completed generation is expected. Each
capture itself uses a fresh `capture-<trace-id>` directory, so it never mutates
or mistakes an older dispatch for current output. Sidecar regeneration removes
only sidecar-owned files before anything that can fail:

- `emit_inline_frames.py` drops them before it looks for a code object, so each
  dispatch ends with a sidecar from this run or none, never the previous one;
- `emit_inline_frames.py <dir> --invalidate-only` is that step on its own.

The capture sentinel is never removed or promoted by sidecar generation. A
running or truncated capture is refused, and a cleanup failure stops regeneration
without changing capture status.

## When it doesn't work

| Symptom | Cause |
| --- | --- |
| No **WaveScope** commands in the palette | not installed on this side of the SSH connection, or the window needs a reload |
| Viewer opens empty | folder is the `rocprofv3 -d` output root, not the `ui_output_*_dispatch_*` directory inside it |
| Source tab empty | kernel built without `ROCKE_DEBUG_LOC=1`, so no DWARF and no source snapshots |
| One helper line owns most of the stalls | no `inline_frames.json`; re-run `emit_inline_frames.py`, then use `+ inlined` |
| Console warns the sidecar matched few or no instructions | it was built from a different build of the kernel — re-run `emit_inline_frames.py` against *this* trace |
| No dispatch folder decoded at all | the kernel regex matched nothing, or the trace decoder is missing — the capture script says which |
| `emit_inline_frames.py` skipped a dispatch | it ran a code object none of the dumped DWARF belongs to; pass `--code-object` to name the right one. Skipping is deliberate — addresses repeat across objects, so a guess would attribute another kernel's source rather than fail. A skipped dispatch is left with no sidecar, including one from an earlier run, and the run still succeeds if any other dispatch resolved |
| Every dispatch skipped as "ran several dumped code objects" | several objects in one generation use the same decoder id; pass `--code-object` only when you can identify the correct object |
| Stall totals exceed wall-clock | `code.json` columns are totals over every execution; divide by `Hit`, don't multiply |

## Related

- `../stage2_capture/capture_att_trace.py` — the ATT capture this wraps, usable on
  its own for an ISA-level trace with no source correlation.
- [`../../skills/capture-kernel-trace-rocke.md`](../../skills/capture-kernel-trace-rocke.md)
  — the underlying rocprofv3 flags and the PMC fallback when the trace decoder is
  unavailable.
- [`../../../../architecture/wavescope_integration.md`](../../../../architecture/wavescope_integration.md)
  — how the pieces fit together, and how to drive the viewer during optimization.
