# `DeviceLDS` YAML Parameter Investigation

## Verdict

**(C) Once-live, now dead.** `DeviceLDS` was a genuine `GlobalParameters` key that was read from YAML by `makeDepthUConfig()` in `Tensile/Common/Types.py`. The reader — and the `DepthUConfig` named-tuple it populated — were explicitly removed in commit `7770c97e13a3` (May 8, 2025, "Fix device LDS size (#2035)"), which moved LDS-size knowledge from YAML-supplied global parameters to hardware-detected `archCaps`. The YAML corpus was not cleaned up at that time, and contributors continued adding `DeviceLDS:` to new test files after the removal. The parameter has been inert since May 8, 2025.

**Important disambiguation:** `archCaps["DeviceLDS"]` is a live, critical value used in five places in the Python/C++ source, but it is populated by `rocisa`'s hardware-detection code (`hardware_caps.hpp`), not by reading the YAML `GlobalParameters: DeviceLDS` field. The two uses of the string `"DeviceLDS"` are entirely separate data paths.

---

## Step 1 — Live-code search at HEAD

`git grep -n DeviceLDS -- '*.py' '*.cpp' '*.cc' '*.h' '*.hpp' '*.cu' '*.cmake' '*.json' '*.toml' '*.md' '*.rst'` returns nine matches:

| File | Line | What it is |
|------|------|-----------|
| `tensilelite/Tensile/KernelWriter.py` | 6200 | Reads `self.states.archCaps["DeviceLDS"]` to check LDS overflow |
| `tensilelite/Tensile/KernelWriter.py` | 7262 | Reads `self.states.archCaps["DeviceLDS"]` for max LDS constant offset check |
| `tensilelite/Tensile/KernelWriterAssembly.py` | 211 | Reads `self.states.archCaps["DeviceLDS"]` for occupancy calculation |
| `tensilelite/Tensile/SolutionStructs/Solution.py` | 1431 | Reads `isaInfoMap[isa].archCaps["DeviceLDS"]` to set `MaxLDS` |
| `tensilelite/Tensile/SolutionStructs/Solution.py` | 4538 | Reads `isaInfoMap[isa].archCaps["DeviceLDS"]` for LDS occupancy |
| `tensilelite/Tensile/SolutionStructs/Solution.py` | 4549 | Reads `isaInfoMap[isa].archCaps["DeviceLDS"]` for 1LDSBuffer auto-detect |
| `tensilelite/rocisa/rocisa/include/hardware_caps.hpp` | 531 | **Populates** `rv["DeviceLDS"] = deviceLDS` in C++ hardware-caps dictionary |
| `shared/stinkytofu/python_module/src/HardwareCaps.cpp` | 246 | **Populates** `rv["DeviceLDS"] = deviceLDS` in a parallel stinkytofu caps dict |
| `shared/stinkytofu/python_module/tests/test_comgr.py` | 230 | Asserts `arch["DeviceLDS"] == 327680` in a hardware-caps unit test |

None of these touch `globalParameters["DeviceLDS"]`. The `archCaps` dictionary is built by `rocisa.rocIsa.getInstance()` during capability detection (`Tensile/Common/Capabilities.py:46-53`), sourced from `hardware_caps.hpp` — it never reads from the YAML `GlobalParameters:` block.

No code at HEAD reads `globalParameters["DeviceLDS"]` or parses `DeviceLDS` from a YAML config.

Conclusion from Step 1: **no live YAML reader exists**.

---

## Step 2 — Git history archaeology

### Introduction of `DeviceLDS` in YAML

The earliest commits introducing `DeviceLDS: 163840` in test YAML files cluster around March 2025 when gfx950 support was first added:

| Sample SHA | Date | Message |
|-----------|------|---------|
| `7501df0a0b76` | 2025-03-09 | "Adding initial support for gfx950 (#1710)" |

At introduction time, `DeviceLDS` was a live `GlobalParameters` entry read by `makeDepthUConfig()` (see below). The value `163840` (160 KB) is the gfx950's expanded LDS size and `327680` (320 KB) is gfx1250's.

### Where `DeviceLDS` was read from YAML

Before removal, `Tensile/Common/Types.py` contained a `DepthUConfig` named tuple and a factory function:

```python
class DepthUConfig(NamedTuple):
    deviceLDS: int = 65536
    maxLDS: int = 65536

def makeDepthUConfig(config: dict) -> DepthUConfig:
    deviceLDS = maxLDS = 65536
    if "DeviceLDS" in config:
        deviceLDS = config["DeviceLDS"]
    if "MaxLDS" in config:
        maxLDS = config["MaxLDS"]
    return DepthUConfig(deviceLDS, maxLDS)
```

`makeDepthUConfig` received the parsed YAML `GlobalParameters` dict and extracted `DeviceLDS` from it, passing the result into the depth-U iteration logic (used in `BenchmarkProblems.py` and elsewhere to constrain kernel search). This was a genuine semantic read: the YAML value directly affected kernel generation.

### Removal commit

**SHA:** `7770c97e13a35e44372cc0f75f4d32c2b0831c76`
**Date:** 2025-05-08
**Author:** Alex Brown
**Message:** "Fix device LDS size (#2035)"
**Upstream hipBLASLt SHA:** `bcfd619a4c5acd8f8fcdf5fb8c0f51344cbb3469`

The commit message explicitly states: *"Move DeviceLDS to ArchCaps and MaxLDS to ForkParameters to fix kernel gen bug at build time."*

This commit simultaneously:
- Deleted the entire `DepthUConfig` named-tuple and `makeDepthUConfig` function from `Types.py`
- Removed all call sites of `makeDepthUConfig` across `BenchmarkProblems.py`, `Contractions.py`, `LibraryLogic.py`, `SolutionLibrary.py`, `Solution.py`, and `Tensile.py`
- Added `MaxLDS: [-1]` to `defaultBenchmarkCommonParameters` (promoting it to a solution fork parameter)
- Wired all LDS-size reads to `archCaps["DeviceLDS"]` (hardware-detected) instead of the YAML-supplied value
- Did **not** touch any YAML test files

After this commit, `DeviceLDS` in any `GlobalParameters:` YAML block was silently ignored (stored into `globalParameters` as an unknown key via the then-warning-only unknown-key path, but never consumed).

### Post-removal YAML additions

Contributors continued adding `DeviceLDS:` to new YAML test files after the removal, unaware the key had been retired:

| SHA | Date | Message |
|-----|------|---------|
| `6d4d1e6874c9` | 2025-06-04 | "MX: add test yaml" |
| `0884ba846ed3` | (post May 2025) | "Add custom main loop scheduling (#1502)" |
| `89b3fc4c2800` | 2026-05-02 | "[hipblaslt] Add support for gfx950 mxfp4 (#6499)" |
| `e53dc7fc7104` | 2026-05-25 | "[tensilelite][SPMM] Enable DirectToLds A/B/Metadata for SPMM (#7261)" |
| `42e06b9f3f26` | 2026-06-03 | "[SubtileImpl] Add support for MXFP8 MT128x128 with DU=256 (#7656)" |

This explains why 45 YAML files carry the key today despite it being dead — it was propagated by copy-paste from existing YAML templates.

### Why it went undetected for 13 months

At the time of removal, `assignGlobalParameters` handled unknown keys with `printWarning(...)` only — not an error. The unknown key was silently stored into `globalParameters` and never consumed. The strict gate (commit that introduced `ConfigTypeError` for unknown keys, June 2026) finally surfaced the stale key.

### Commits summary

| Role | SHA | Date | Message |
|------|-----|------|---------|
| Earliest YAML introduction | `7501df0a0b76` | 2025-03-09 | "Adding initial support for gfx950 (#1710)" |
| Reader removed | `7770c97e13a3` | 2025-05-08 | "Fix device LDS size (#2035)" |
| Latest post-removal YAML addition | `42e06b9f3f26` | 2026-06-03 | "[SubtileImpl] Add support for MXFP8 MT128x128 with DU=256 (#7656)" |
| Strict gate that exposed the stale key | (input_yaml branch) | 2026-06-05 | "input-yaml validation — Step 5: assignGlobalParameters strict gate" |

---

## Recommendation

Delete `DeviceLDS:` from every `GlobalParameters:` block in the YAML corpus. The field has had no effect since May 8, 2025. **Do not** add `DeviceLDS` to `globalParameters`, `ignoreKeys`, or `globalParameterTypeOverrides` — all three options would perpetuate dead config.

The functional LDS-size information now lives exclusively in `archCaps["DeviceLDS"]`, which is populated at runtime by hardware detection and requires no YAML override. If a test genuinely needs to constrain LDS, `MaxLDS` (now a fork parameter in `defaultBenchmarkCommonParameters`) is the correct knob.

A sweep to remove all occurrences:

```
git grep -rn 'DeviceLDS' -- '*.yaml' '*.yml'
```

All 45 affected files are under `projects/hipblaslt/tensilelite/Tensile/Tests/common/`.

---

## Worktree path

`/home/alvasile/rocm-libraries/.claude/worktrees/agent-a2d8e44e7b1e9f379`
