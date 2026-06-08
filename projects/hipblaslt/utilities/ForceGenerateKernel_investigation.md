# `ForceGenerateKernel` YAML Parameter Investigation

## Verdict

**(A) Live reader — the strict validation gate is exposing a routing bug, not dead config.**

`ForceGenerateKernel` was deliberately removed from `globalParameters` in commit `dc2c963c` (March 25, 2025, "Remove global variables") and simultaneously re-routed to a new `DebugConfig` struct populated by `makeDebugConfig()`. The YAML key is therefore **alive and functional** — it is read directly from `config["GlobalParameters"]` by `makeDebugConfig()` in `Tensile/Common/Types.py` (lines 96–97). The strict gate in `assignGlobalParameters` fires on it because `makeDebugConfig` runs _after_ `assignGlobalParameters` in `Tensile.py` (lines 654, 658), and the key was never added to the gate's `ignoreKeys` list.

This is structurally identical to the `PrintSolutionRejectionReason` case documented in `PrintSolutionRejectionReason_investigation.md`, and was predicted there at the end of the Recommendation section.

---

## Step 1 — Live-code search at HEAD

`git grep -n ForceGenerateKernel -- '*.py' '*.cpp' '*.cc' '*.h' '*.hpp' '*.cu' '*.cmake' '*.json' '*.toml' '*.md' '*.rst'` returned three hits in code files:

| File | Lines | Role |
|------|-------|------|
| `Tensile/Common/Types.py` | 96–97 | **Active YAML reader** — `makeDebugConfig()` reads `config["ForceGenerateKernel"]` directly from the raw `GlobalParameters` dict and assigns it to `DebugConfig.forceGenerateKernel` |
| `Tensile/KernelWriter.py` | 6649 | `self.states.preventVgprOverflowDuringNewTile = 0 and not self.debugConfig.forceGenerateKernel` — live conditional |
| `Tensile/KernelWriter.py` | 9632–9633 | `if self.debugConfig.forceGenerateKernel:` — suppresses `RuntimeError` on kernel-generation failure and prints a warning instead |

The flag controls two behaviors: (1) whether VGPR overflow detection is suppressed during new-tile address setup, and (2) whether a kernel-generation error is fatal. Both are live code paths that fire when the flag is set.

**Conclusion from Step 1: a live reader exists.** The YAML key `ForceGenerateKernel` is fully wired to `debugConfig.forceGenerateKernel`. The strict gate is incorrectly firing on a valid operational parameter.

---

## Step 2 — Git history

### Original introduction

**SHA:** `f8246afafa09`  
**Date:** 2022-11-11  
**Message:** "Add tensilelite"

`ForceGenerateKernel` was present from day one in `globalParameters`:

```python
# even if error occurs in kernel generation (ie due to resource overflow),
# generate the kernel source anyway.  Tensile will also attempt to run
# the kernel.  Useful to examine and debug overflow errors.
globalParameters["ForceGenerateKernel"] = 0
```

`KernelWriter.py` at that point read it as `globalParameters["ForceGenerateKernel"]` in two places (same logical sites as today).

### Removal from `globalParameters` / move to DebugConfig

**SHA:** `dc2c963c892457151df6f08a687bcb47912ed3f3`  
**Date:** 2025-03-25  
**Author:** David Dixon  
**Message:** "Remove global variables (#1677)"

The diff for `Tensile/Common/GlobalParameters.py` removed:

```python
-# even if error occurs in kernel generation (ie due to resource overflow),
-# generate the kernel source anyway.  Tensile will also attempt to run
-# the kernel.  Useful to examine and debug overflow errors.
-globalParameters["ForceGenerateKernel"] = 0
```

The same commit added `DebugConfig` (a `NamedTuple`) and `makeDebugConfig()` to `Tensile/Common/Types.py`. The new `makeDebugConfig()` reads `ForceGenerateKernel` directly from the raw config dict (the YAML `GlobalParameters:` block), bypassing `globalParameters` entirely. Both `KernelWriter.py` call sites were updated from `globalParameters["ForceGenerateKernel"]` to `self.debugConfig.forceGenerateKernel`.

The commit also updated `Tensile.py` to call `makeDebugConfig(config["GlobalParameters"])` immediately after `assignGlobalParameters(config.get("GlobalParameters", {}), isaInfoMap)`. **The YAML corpus was correct to keep the key; the routing simply was not reflected in `ignoreKeys`.**

### The new strict gate

The strict gate (added June 2026, "input-yaml validation — Step 5: assignGlobalParameters strict gate") promotes unknown keys to a `ConfigTypeError`. Because `ForceGenerateKernel` is not in `globalParameters` and is not in `ignoreKeys`, it triggers the error — even though `makeDebugConfig` downstream consumes it correctly.

---

## Active YAML files

9 YAML files use `ForceGenerateKernel` without being commented out:

| File | Value |
|------|-------|
| `Tests/common/gemm/gfx12/tdm_multicast_gfx1250.yaml` | `[True]` |
| `Tests/common/gemm/gfx950/general_wgm.yaml` | `1` |
| `Tests/common/streamk/gfx1250/sk_mxf4gemm_explicit.yaml` | `True` |
| `Tests/common/streamk/gfx1250/sk_mxf4gemm_quick.yaml` | `True` |
| `Tests/common/streamk/gfx1250/sk_mxf4gemm_tdm.yaml` | `True` |
| `Tests/common/streamk/gfx1250/sk_mxf8f4gemm_tdm.yaml` | `True` |
| `Tests/common/streamk/gfx1250/sk_mxf8gemm_explicit.yaml` | `True` |
| `Tests/common/streamk/gfx1250/sk_mxf8gemm_quick.yaml` | `True` |
| `Tests/common/streamk/gfx1250/sk_mxf8gemm_tdm.yaml` | `True` |

Note: `general_wgm.yaml` uses integer `1` and `tdm_multicast_gfx1250.yaml` uses list `[True]` instead of bare `True`/`False`. These are type mismatches relative to `DebugConfig.forceGenerateKernel: bool=False`; however, both are truthy and `makeDebugConfig` does not validate types itself, so they work in practice. The list form `[True]` is unusual and may not behave as expected (a non-empty list is truthy in Python, but the value stored would be a list, not a bool).

---

## Recommendation

**Do not delete the YAML entries.** The parameter is live. There are two parts to the fix:

### Part 1: Add `ForceGenerateKernel` to `ignoreKeys`

In `Tensile/Common/GlobalParameters.py`, `assignGlobalParameters()`, extend `ignoreKeys` so the strict gate skips it:

```python
ignoreKeys = [
    "Architecture",
    "PrintLevel",
    "Device",
    "UseCompression",
    "CxxCompiler",
    "CCompiler",
    "OffloadBundler",
    "Assembler",
    "LogicPath",
    "LogicFilter",
    "OutputPath",
    "Experimental",
    "GenSolTable",
    # Moved to DebugConfig (makeDebugConfig in Common/Types.py); not a globalParameter.
    "PrintSolutionRejectionReason",
    "ForceGenerateKernel",
]
```

All other `DebugConfig` members (`EnableAsserts`, `EnableDebugA`, `EnableDebugB`, `EnableDebugC`, `ExpectedValueC`, `ForceCExpectedValue`, `DebugKernel`, `SplitGSU`, `PrintIndexAssignmentInfo`) should be added in the same sweep — they have the same structural situation and will trigger the same strict-gate error if any YAML exercises them.

### Part 2: Fix YAML type mismatches

- `general_wgm.yaml` line 23: `ForceGenerateKernel: 1` → `ForceGenerateKernel: True`
- `tdm_multicast_gfx1250.yaml` line 30: `ForceGenerateKernel: [True]` → `ForceGenerateKernel: True`

These are the same class of type-mismatch errors the strict gate was introduced to catch. The integer and list forms work by accident (Python truthiness), but should be normalized to the canonical bool form.

---

## Worktree path

`/home/alvasile/rocm-libraries/.claude/worktrees/agent-a710c08a75a9a1f02`
