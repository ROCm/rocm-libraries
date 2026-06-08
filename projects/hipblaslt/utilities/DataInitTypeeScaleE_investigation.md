# `DataInitTypeeScaleE` YAML Parameter Investigation

## Verdict

**(B) Dead-on-arrival — typo, never had a consumer.**

`DataInitTypeeScaleE` (double-e in `Typee`) is a misspelling of a parameter that was **never registered** in `globalParameters`. The correctly-spelled counterpart `DataInitTypeScaleE` also does not exist: there is no `DataInitTypeScaleE` entry in `GlobalParameters.py`, no reader in `ClientWriter.py`, and no reference anywhere in the Python or C++ codebase. The closest real parameters are `DataInitTypeScaleA/B/C/D` and `DataInitTypeAlphaVec` — a `ScaleE` variant was simply never implemented. The typo entered the corpus at YAML-authorship time and has been silently ignored (swallowed as an unrecognised key) ever since. The strict gate in `assignGlobalParameters` (commit `0ce0829c`, June 5 2026) is the first mechanism that has ever flagged it.

---

## Step 1 — Live-code search at HEAD

### Search for the misspelled key (`DataInitTypeeScaleE`)

`grep -r "DataInitTypeeScaleE"` across all `.py`, `.cpp`, `.cc`, `.h`, `.hpp`, `.cu`, `.cmake`, `.md`, `.rst`, `.yaml`, `.yml` (excluding `.tox/`, `.worktrees/`, `node_modules/`, `build/`, `__pycache__/`, `.git/`) returned matches only in:

| File | Nature |
|------|--------|
| `tensilelite/Tensile/Tests/common/gemm/gfx950/f8f16mix_f8s.yaml` | YAML `GlobalParameters:` block — data only |
| `tensilelite/Tensile/Tests/common/gemm/fp8nfp16mix_hfp8ns.yaml` | YAML `GlobalParameters:` block — data only |
| `utilities/input_yaml_validation_implementation_audit.md` | Audit doc — documentation only |

No Python module, C++ file, CMake file, or test code at HEAD reads or references `DataInitTypeeScaleE`.

### Search for the correctly-spelled variant (`DataInitTypeScaleE`)

`grep -r "DataInitTypeScaleE"` across the same file set returned **zero matches** — not in `GlobalParameters.py`, not in `ClientWriter.py`, not in any YAML or source file.

### Registry audit

`GlobalParameters.py` (lines 183–198) registers the following `DataInitType*` keys:

```
DataInitTypeAB, DataInitTypeA, DataInitTypeB, DataInitTypeC, DataInitTypeD,
DataInitTypeE, DataInitTypeAlpha, DataInitTypeBeta, DataInitTypeBias,
DataInitTypeScaleA, DataInitTypeScaleB, DataInitTypeScaleC, DataInitTypeScaleD,
DataInitTypeScaleAlphaVec, DataInitTypeMXSA, DataInitTypeMXSB
```

Neither `DataInitTypeScaleE` nor `DataInitTypeeScaleE` appears in this list. `DataInitTypeScaleE` is a missing entry — the scale-for-E initialisation type was simply never implemented as a global parameter.

### ClientWriter audit

`ClientWriter.py` reads `DataInitTypeScaleA/B/C/D` and `DataInitTypeScaleAlphaVec` explicitly (lines 515–519) and passes them as `init-scale-a`, `init-scale-b`, etc., to the benchmark client. There is no `init-scale-e` argument, no `initScaleE` variable, and no `DataInitTypeScaleE` read anywhere.

Conclusion from Step 1: **no live reader exists — for either spelling**.

---

## Step 2 — Git history

### Introduction of `DataInitTypeeScaleE` in the YAML files

**File 1:** `tensilelite/Tensile/Tests/common/gemm/fp8nfp16mix_hfp8ns.yaml`

| Role | SHA | Date | Message |
|------|-----|------|---------|
| Introduction commit | `da601974bcf4` | 2025-01-28 | "Incorporate new F8 design (#1577)" |

The initial patch added the file from scratch with `DataInitTypeeScaleE: 21` already present in the `GlobalParameters:` block. The same commit did **not** touch `GlobalParameters.py` or `ClientWriter.py` for any `ScaleE` registration — the parameter was typed into the YAML without a corresponding implementation being landed at the same time.

**File 2:** `tensilelite/Tensile/Tests/common/gemm/gfx950/f8f16mix_f8s.yaml`

| Role | SHA | Date | Message |
|------|-----|------|---------|
| Introduction commit | `7501df0a0b76` | 2025-03-09 | "Adding initial support for gfx950 (#1710)" |

Same pattern: the file was created with `DataInitTypeeScaleE: 21` in its `GlobalParameters:` block. The commit did touch `GlobalParameters.py` (adding `PrintTensorScaleAlphaVec`) and `ClientWriter.py` (adding `print-tensor-scale-alpha-vec`) — but nothing related to a `ScaleE` init type was added.

### Was a `DataInitTypeScaleE` reader ever present?

Searches (`git log -S 'DataInitTypeScaleE' -- '*.py'` and `git log -S 'DataInitTypeeScaleE' -- '*.py'`) over the full history returned **no results** — neither spelling ever appeared in Python source files. There was never a `globalParameters["DataInitTypeScaleE"]` assignment, no ClientWriter read, and no CLI `--init-scale-e` argument.

### Why it went undetected

At the time of introduction, `assignGlobalParameters` handled unknown keys with `printWarning(...)` only — not an error. Every YAML run emitted a warning for `DataInitTypeeScaleE`, but tests passed and CI did not enforce clean output. The strict gate (commit `0ce0829c`, June 5 2026) upgraded this from a warning to a `ConfigTypeError`, finally surfacing the stale key.

### Commits summary

| Role | SHA | Date | Message |
|------|-----|------|---------|
| First YAML introduction | `da601974bcf4` | 2025-01-28 | "Incorporate new F8 design (#1577)" |
| Second YAML introduction | `7501df0a0b76` | 2025-03-09 | "Adding initial support for gfx950 (#1710)" |
| Strict gate that exposed it | `0ce0829c` | 2026-06-05 | "input-yaml validation — Step 5: assignGlobalParameters strict gate" |

---

## Recommendation

**Delete `DataInitTypeeScaleE: 21` from both YAML files** — it is a typo of a parameter that was never implemented. The fix is removal, not rename.

The typo hypothesis (double-e) is confirmed: the parameter was almost certainly intended as `DataInitTypeScaleE`, modelled on `DataInitTypeScaleA/B/C/D`. However, that correctly-spelled form was **also never registered or consumed** — so renaming to `DataInitTypeScaleE` would still produce a `ConfigTypeError`. Unless there is a plan to implement `DataInitTypeScaleE` as a new global parameter with a corresponding `ClientWriter` reader and benchmark-client `--init-scale-e` flag, the correct fix is simple deletion.

The two affected files are:

- `projects/hipblaslt/tensilelite/Tensile/Tests/common/gemm/fp8nfp16mix_hfp8ns.yaml` (line 18)
- `projects/hipblaslt/tensilelite/Tensile/Tests/common/gemm/gfx950/f8f16mix_f8s.yaml` (line 18)

---

## Worktree path

`/home/alvasile/rocm-libraries/.claude/worktrees/agent-aeaa83deef6b2fc0b`
