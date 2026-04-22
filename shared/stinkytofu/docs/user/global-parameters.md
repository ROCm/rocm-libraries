# StinkyTofu Configuration via GlobalParameters

Control StinkyTofu behavior through Tensile's `GlobalParameters` system -- either via `--global-parameters` on the CLI or in the YAML configuration file.

## Quick Reference

| Parameter | Values | Default | Purpose |
|---|---|---|---|
| `StinkyTofuOptLevel` | `0`, `1`, `2`, `3` | `0` | Pipeline optimization level |
| `StinkyTofuDebugLevel` | `0`, `1`, `2` | `0` | Diagnostic output verbosity |
| `StinkyTofuPrintBeforePass` | comma-separated pass names | `""` | Print IR before specific passes |
| `StinkyTofuPrintAfterPass` | comma-separated pass names | `""` | Print IR after specific passes |
| `StinkyTofuDebugPass` | comma-separated pass names | `""` | PASS_DEBUG logging + instruction-order snapshot allow-list |
| `StinkyTofuPassOrderSnapshotJson` | file path | `""` | Before/after instruction-order JSON for stinkytofu-analysis |

---

## `StinkyTofuOptLevel`

Maps directly to the pipeline's `OptLevel` enum (`O0`--`O3`). Set to `None` in YAML to disable the StinkyTofu feature entirely.

---

## `StinkyTofuDebugLevel`

Debug output is **per-PM** -- the outer PassManager and each ScopeAdaptor's inner PassManager are configured independently.

| Level | Output | Destination |
|---|---|---|
| `0` (default) | Silent | -- |
| `1` | Pass names in execution order | stdout |
| `2` | Initial IR before passes + IR after each pass | per-PM files (see below) |

### DebugLevel 2 file naming

Each PM writes to its own file, named by scope:

| Scope | File prefix |
|---|---|
| Outer PM | `kernel-OuterPM` |
| Single-region adapter | `<groupName>` (e.g. `loopWithPrefetch`) |
| Multi-region adapter | `<group1>+<group2>` (e.g. `loopWithPrefetch+noLoadLoopBody`) |
| Whole-kernel adapter | `wholeKernel` |

Example files for a gfx1250 kernel:

```
kernel-OuterPM-before_passes.txt
kernel-OuterPM-after_passes.txt
loopWithPrefetch-before_passes.txt
loopWithPrefetch-after_passes.txt
noLoadLoopBody-before_passes.txt
noLoadLoopBody-after_passes.txt
loopWithPrefetch+noLoadLoopBody-before_passes.txt
loopWithPrefetch+noLoadLoopBody-after_passes.txt
```

---

## `StinkyTofuPrintBeforePass` / `StinkyTofuPrintAfterPass`

Comma-separated list of pass names (case-sensitive) to selectively dump IR before or after. Output goes to per-PM files using the same naming convention as DebugLevel 2.

These work independently of `StinkyTofuDebugLevel`. When both are set, they compose without duplication. Unmatched pass names are silently ignored.

Pass names target the actual pass (e.g. `CFGBuilderPass`, `StinkyDAGSchedulerPass`), not the `ScopeAdaptor` wrapper. The pass runs inside an inner PM, so the IR dump shows the extracted region, not the full kernel.

---

## `StinkyTofuDebugPass`

Comma-separated list of pass names (case-sensitive). Serves two purposes:

1. **PASS_DEBUG logging** -- prints pass-internal debug information (e.g. DAG graphs, scheduling decisions) to stderr.
2. **Instruction-order snapshot allow-list** -- when `StinkyTofuPassOrderSnapshotJson` is also set, records before/after instruction order for the listed passes. If this parameter is empty and a snapshot path is set, only `StinkyDAGSchedulerPass` is recorded by default.

Unmatched pass names are silently ignored.

---

## `StinkyTofuPassOrderSnapshotJson`

File path for before/after instruction-order JSON consumed by `tools/stinkytofu-analysis`. When empty (default), no snapshot is written. The passes recorded are controlled by `StinkyTofuDebugPass`.

Note: multiple kernels may overwrite the same file unless you use a unique path per build.

**Global scope**: `DebugPass` is a global setting that applies to all PMs regardless of nesting. No per-PM configuration needed.

---

## Usage

### Via CLI

String values require both outer single quotes and inner double quotes (`'Key="value"'`) so that Python's `eval()` parses them correctly. This applies even for a single pass name.

```bash
Tensile.sh config.yaml output/ --global-parameters StinkyTofuOptLevel=3 StinkyTofuDebugLevel=2
Tensile.sh config.yaml output/ --global-parameters StinkyTofuOptLevel=3 'StinkyTofuPrintAfterPass="CFG Builder, StinkyDAGSchedulerPass"'
Tensile.sh config.yaml output/ --global-parameters StinkyTofuOptLevel=3 'StinkyTofuDebugPass="StinkyDAGSchedulerPass"' 'StinkyTofuPassOrderSnapshotJson="dag.json"'
```

### Via YAML

```yaml
GlobalParameters:
  StinkyTofuOptLevel: 3
  StinkyTofuDebugLevel: 2
  StinkyTofuPrintAfterPass: "CFG Builder, StinkyDAGSchedulerPass"
  StinkyTofuDebugPass: "StinkyDAGSchedulerPass"
  StinkyTofuPassOrderSnapshotJson: "dag.json"
```

---

## Scope

These parameters apply to the **Tensile/KernelWriter integration path only** -- they have no effect on `stinkytofu-opt` (see [stinkytofu-opt README](../../tools/stinkytofu-opt/README.md)).
