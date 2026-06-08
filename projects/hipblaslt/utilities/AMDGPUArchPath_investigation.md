# `AMDGPUArchPath` YAML Parameter Investigation

## Verdict

**(C) Once-live, now dead.** `AMDGPUArchPath` was a genuine global parameter, with a registry entry and an active reader that used it to invoke the `amdgpu-arch` tool for ISA detection. Both the registry entry and the reader were removed in commit `dc2c963c892` (March 25, 2025, "Remove global variables"). The sole YAML file that now carries this key — `streamk/gfx1250/sk_mxf8f4gemm_quick.yaml` — was not created until March 17, 2026, nearly a year after the parameter was dead. The key has been inert since the removal commit.

There is an additional red flag: the value in the YAML is `rocm_agent_enumerator` — the name of the *other* tool (`ROCmAgentEnumeratorPath`'s tool, not `AMDGPUArchPath`'s). At the time the parameter was live, the correct value would have been a path to `amdgpu-arch` (e.g., `/opt/rocm/llvm/bin/amdgpu-arch`). The YAML author appears to have confused the two companion parameters.

---

## Step 1 — Live-code search at HEAD

`git grep -n AMDGPUArchPath -- '*.py' '*.cpp' '*.cc' '*.h' '*.hpp' '*.cu' '*.cmake' '*.json' '*.toml' '*.md' '*.rst'` returned **zero results**.

The YAML corpus search returns exactly one hit:

| File | Line | Value |
|------|------|-------|
| `tensilelite/Tensile/Tests/common/streamk/gfx1250/sk_mxf8f4gemm_quick.yaml` | 32 | `AMDGPUArchPath: rocm_agent_enumerator` |

No Python or C++ code reads `globalParameters["AMDGPUArchPath"]` at HEAD. `AMDGPUArchPath` does not appear in the `globalParameters` registry in `Tensile/Common/GlobalParameters.py` (lines 43–366 as of HEAD), and it is not in the `ignoreKeys` list in `assignGlobalParameters`. Under the current strict gate (commit `0ce0829c`, June 5, 2026), any config file that names this key will raise `ConfigTypeError`.

Conclusion from Step 1: **no live reader** — proceeding to Step 2.

---

## Step 2 — Git history archaeology

### Introduction of `AMDGPUArchPath`

**SHA:** `2af824ee536e`
**Date:** 2024-10-24
**Author:** who who who (fsx950223)
**Message:** "make tensile detect local device correctly (#1252)"

This commit renamed the existing `ROCmAgentEnumeratorPath` parameter to `AMDGPUArchPath` in the old `Tensile/Common.py` and updated the reader to invoke `amdgpu-arch` instead of `rocm_agent_enumerator`. The rationale was that `amdgpu-arch` is the preferred tool for GPU detection (over the older `rocm_agent_enumerator`).

The parameter's meaning was a **filesystem path** to the `amdgpu-arch` tool binary, e.g., `/opt/rocm/llvm/bin/amdgpu-arch`.

### Reader location at introduction time

The reader lived in `detectGlobalCurrentISA()` inside `Tensile/Common.py`:

```python
if globalParameters["CurrentISA"] == (0,0,0) and globalParameters["AMDGPUArchPath"]:
    process = subprocess.run([globalParameters["AMDGPUArchPath"]], stdout=subprocess.PIPE)
```

It also appeared in `assignGlobalParameters`:

```python
if "AMDGPUArchPath" in config:
    globalParameters["AMDGPUArchPath"] = config["AMDGPUArchPath"]
```

And in `initGlobalParameters()` where the path was auto-discovered:

```python
globalParameters["AMDGPUArchPath"] = locateExe(globalParameters["ROCmPath"], "llvm/bin/amdgpu-arch")
```

### Intermediate refactoring — Static build (#1283)

**SHA:** `fac41feafa9c`
**Date:** 2024-11-25
**Message:** "Static build (#1283)"

This commit added `ROCmAgentEnumeratorPath` back as a **separate** companion parameter alongside `AMDGPUArchPath`. The `detectGlobalCurrentISA()` function was refactored to try `AMDGPUArchPath` (i.e., `amdgpu-arch`) first, then fall back to `ROCmAgentEnumeratorPath` (i.e., `rocm_agent_enumerator`) if detection failed. This two-tool fallback design is also what the `Tensile/Common/Architectures.py` comment still describes at HEAD.

### Module split — Move Common.py to module (#1607)

**SHA:** `d170037bd4fe`
**Date:** 2025-02-12
**Message:** "Move Common.py to module (#1607)"

This commit split the monolithic `Common.py` into a `Common/` package. Both `AMDGPUArchPath` and `ROCmAgentEnumeratorPath` were moved into `Tensile/Common/GlobalParameters.py`, with their registry entries and all readers intact.

### Removal commit

**SHA:** `dc2c963c892`
**Date:** 2025-03-25
**Author:** David Dixon
**Message:** "Remove global variables (#1677)"

This commit deleted all traces of `AMDGPUArchPath` from `GlobalParameters.py`:

- `globalParameters["AMDGPUArchPath"] = None` from the registry
- `detectGlobalCurrentISA_()` call using `globalParameters["AMDGPUArchPath"]`
- The `locateExe(...)` calls that set the path during startup
- The `if "AMDGPUArchPath" in config:` block in `assignGlobalParameters`

The same commit removed `ROCmAgentEnumeratorPath` and `CurrentISA` from the registry, and moved the architecture detection responsibility entirely to the `Tensile/Toolchain/` module (`Validators.py:ToolchainDefaults.DEVICE_ENUMERATOR`). No YAML files were touched in this commit.

### The YAML file — created after removal

**SHA:** `fea3f70e02cb`
**Date:** 2026-03-17
**Author:** Henderson, Nathan
**Message:** "Add mixed GEMM + StreamK tests with and without TDM"

The sole offending YAML (`streamk/gfx1250/sk_mxf8f4gemm_quick.yaml`) was created nearly a year after `AMDGPUArchPath` was removed. The PR that merged it into `develop` is `cfb02486ac04` (April 21, 2026, "Enable StreamK for gfx1250 in TensileLite (#6432)").

The value `rocm_agent_enumerator` in that YAML is also wrong for this parameter. `AMDGPUArchPath` expected a path to `amdgpu-arch`; `rocm_agent_enumerator` is the tool name belonging to the companion `ROCmAgentEnumeratorPath` parameter. The author appears to have conflated the two. (The `ROCmAgentEnumeratorPath` parameter suffered the same post-removal re-introduction problem; see `ROCmAgentEnumeratorPath_investigation.md` for details.)

### No YAML file ever correctly used `AMDGPUArchPath`

The git history of the parameter registry shows `AMDGPUArchPath` was removed in March 2025. The only YAML file ever to carry this key was created in March 2026. There are no YAML files in the historical corpus that set `AMDGPUArchPath` while the reader was alive.

---

## Commits summary

| Role | SHA | Date | Message |
|------|-----|------|---------|
| Parameter introduced (renamed from ROCmAgentEnumeratorPath) | `2af824ee536e` | 2024-10-24 | "make tensile detect local device correctly (#1252)" |
| Fallback two-tool design added | `fac41feafa9c` | 2024-11-25 | "Static build (#1283)" |
| Module split — moved to GlobalParameters.py | `d170037bd4fe` | 2025-02-12 | "Move Common.py to module (#1607)" |
| Parameter + reader removed | `dc2c963c892` | 2025-03-25 | "Remove global variables (#1677)" |
| YAML file created (parameter already dead) | `fea3f70e02cb` | 2026-03-17 | "Add mixed GEMM + StreamK tests with and without TDM" |
| YAML merged to develop | `cfb02486ac04` | 2026-04-21 | "Enable StreamK for gfx1250 in TensileLite (#6432)" |
| Strict gate that exposed the stale key | `0ce0829c` | 2026-06-05 | "input-yaml validation — Step 5: assignGlobalParameters strict gate" |

---

## Recommendation

Delete `AMDGPUArchPath: rocm_agent_enumerator` from line 32 of:

```
projects/hipblaslt/tensilelite/Tensile/Tests/common/streamk/gfx1250/sk_mxf8f4gemm_quick.yaml
```

That is the only file in the corpus that contains this key. There is no need to add `AMDGPUArchPath` back to `globalParameters` or to the `ignoreKeys` list. The architecture detection it once drove is now handled entirely by the `Toolchain` layer.

Do **not** substitute the intended behavior with `ROCmAgentEnumeratorPath: rocm_agent_enumerator` — that parameter was also removed and is also dead (see `ROCmAgentEnumeratorPath_investigation.md`).

---

## Worktree path

`/home/alvasile/rocm-libraries/.claude/worktrees/agent-a8cdf5d7e05a3e22d`
