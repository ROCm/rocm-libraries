# WaveScope

**WaveScope** is the viewer that turns an ATT trace into a per-wave timeline, an
ISA listing, and a Source tab that maps instructions back to the Python that
authored them. The two scripts here produce what it reads.

Both are stdlib-only, import no rocke, and run from any directory against any
trace, so the commands below work as written from wherever you are once you point
at them. From `platform/`, that is
`dsl_docs/optimization/utilities/tools/wavescope/`. Each takes `--help`.

- `capture_wavescope_trace.py` — capture a source-correlated trace end to end.
- `emit_inline_frames.py` — write the inline-frames sidecar for a trace you
  already decoded.

## What WaveScope shows

`rocprofv3 --att` records what every wave executed, cycle by cycle, on one CU.
Decoded, that becomes a folder of JSON the viewer reads. Opened in the editor you
get the wave timeline (where the stalls are), the ISA with per-instruction hit and
stall totals, and — if the kernel was built with source locations — a Source tab
that highlights the Python lines those instructions came from, clickable in both
directions.

WaveScope lives in its own repo, <https://github.com/aghamari/WaveScope>. It is
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
first and opt in explicitly with `--assume-complete`.

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
