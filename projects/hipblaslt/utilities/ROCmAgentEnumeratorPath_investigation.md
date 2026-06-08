# ROCmAgentEnumeratorPath Investigation

**Verdict: (C) Once-live, now dead — parameter was intentionally removed from `GlobalParameters.py` but the YAML test files were not cleaned up.**

---

## Step 1: Live-code search at HEAD

`git grep -n ROCmAgentEnumeratorPath` over all non-YAML source files (`.py`, `.cpp`, `.cc`, `.h`, `.hpp`, `.cu`, `.cmake`, `.json`, `.toml`, `.md`, `.rst`) returns **zero results**.

The underlying tool `rocm_agent_enumerator` is still referenced in multiple live files:

- `tensilelite/Tensile/Toolchain/Validators.py` — uses it as a fallback enumerator
- `tensilelite/Tensile/Common/Architectures.py` — falls back to it when `amdgpu-arch` fails
- `tensilelite/Tensile.py` — accepts `--rocm-agent-enumerator` CLI argument
- `clients/scripts/performance/generator.py` — hardcodes `/opt/rocm/bin/rocm_agent_enumerator`

But none of these read the YAML key `ROCmAgentEnumeratorPath` from `globalParameters`. The tool path is resolved via the `Toolchain` module and CLI flags, not via a GlobalParameters YAML key.

The `ignoreKeys` list in `assignGlobalParameters` (line 795) does **not** include `ROCmAgentEnumeratorPath`, so with `TENSILE_STRICT_TYPE_CHECK=strict` (the default), any config file containing this key will raise `ConfigTypeError`.

---

## Step 2: Git history

### Introduction

Commit `07c4c9b37cd1` ("Manually sync from gfx1250", Author: Serge Lu, Aug 26, 2025) introduced:

- `globalParameters["ROCmAgentEnumeratorPath"] = None` in `GlobalParameters.py`
- A reader in `assignGlobalParameters`:  
  ```python
  if "ROCmAgentEnumeratorPath" in config:
      globalParameters["ROCmAgentEnumeratorPath"] = config["ROCmAgentEnumeratorPath"]
  ```
- The comment: `# if ROCmAgentEnumeratorPath is "rocm_agent_enumerator", the arch path is /opt/rocm/bin/rocm_agent_enumerator; otherwise it is /opt/rocm/llvm/bin/amdgpu-arch`

This was part of a "gfx1250 emulator" feature. `ClientWriter.py` also emitted the parameter into the benchmark client config ini at the time.

### Removal

Commit `4a5aa3cb7fbe` ("Revert modifications for emulator (#1001)", Author: Cai Meng-Zhe (Ray), Mar 27, 2026) removed:

- `globalParameters["ROCmAgentEnumeratorPath"] = None` from `GlobalParameters.py`
- The `assignGlobalParameters` reader block
- The `ClientWriter.py` `param("ROCmAgentEnumeratorPath", ...)` call
- Emulator-related env vars in `tox.ini` and build workarounds in CMake

The commit message explicitly states: **"Remove emulator parameter: ROCmAgentEnumeratorPath."**

The removal commit (`4a5aa3cb7fbe`) did clean `ROCmAgentEnumeratorPath` out of one YAML file (`Tensile/Tests/common/gemm/gfx12/fp16_gfx1250.yaml` — confirmed in the diff). However, subsequent PRs re-introduced the key into gfx1250 test YAML files without noticing it had been removed from `GlobalParameters.py`:

- Commit `342a2dc54655` ("[hipblaslt][tensilelite] Enable StreamK for gfx1250 in TensileLite (#6432)", Apr 21, 2026) — added `ROCmAgentEnumeratorPath: rocm_agent_enumerator` back to `fp16_gfx1250.yaml` and to `sk_mxf6b8gemm_quick.yaml` / `sk_mxf8b6gemm_quick.yaml`.
- Commits `5064321f54d5`, `65db794f3fb8`, `8559733113a2`, `621e7bc8fb71` — added it to the remaining five YAML files (`nt_th_nv_gfx1250.yaml`, `stinky_sia4.yaml`, `tdm_gfx1250.yaml`, `spmm_tdm_all.yaml`, `spmm_tdm_f16_transposes.yaml`).

All eight current offenders arrived after the parameter was removed from `GlobalParameters.py`.

---

## Summary of affected YAML files

All under `projects/hipblaslt/tensilelite/Tensile/Tests/common/`:

```
gemm/gfx12/fp16_gfx1250.yaml
gemm/gfx12/nt_th_nv_gfx1250.yaml
gemm/gfx12/stinky_sia4.yaml
gemm/gfx12/tdm_gfx1250.yaml
sparse/gfx1250/spmm_tdm_all.yaml
sparse/gfx1250/spmm_tdm_f16_transposes.yaml
streamk/gfx1250/sk_mxf6b8gemm_quick.yaml
streamk/gfx1250/sk_mxf8b6gemm_quick.yaml
```

---

## Recommendation

**Remove `ROCmAgentEnumeratorPath` from all 8 YAML files.** The parameter has no live reader. The intent — selecting between `rocm_agent_enumerator` and `amdgpu-arch` — is now handled entirely by the `Toolchain` layer (`Validators.py:ToolchainDefaults.DEVICE_ENUMERATOR`) and the `--rocm-agent-enumerator` CLI flag; there is no need to replumb this through a GlobalParameters key.

Do **not** add it back to `GlobalParameters.py` or the `ignoreKeys` list; it was deliberately removed as part of the emulator revert and restoring it would re-open dead infrastructure.

---

## Worktree path

Investigation conducted in:
`/home/alvasile/rocm-libraries/.claude/worktrees/agent-a000660ff4752944f`
