# `MaxFileName` YAML Parameter Investigation

## Verdict

**(C) Once-live, now dead.** `MaxFileName` was a real global parameter read by `KernelWriter.py` to cap generated file name lengths. It was deliberately removed from both `globalParameters` and the reader call site in commit `d170037bd4fe` ("Move Common.py to module", Feb 12 2025), which replaced it with a hard-coded constant `MAX_FILENAME_LENGTH = 64` in `Tensile/Common/Constants.py`. A backward-compatibility deprecation warning was added to `Tensile/Tensile.py` at that same commit but was itself removed in a later commit (`843fe258090c`) on a branch not yet merged to `develop`. The YAML test corpus was never cleaned up.

---

## Step 1 — Live-code search at HEAD

`git grep -n MaxFileName -- '*.py' '*.cpp' '*.cc' '*.h' '*.hpp' '*.cu' '*.cmake' '*.json' '*.toml' '*.md' '*.rst'` returned **two matches**, both in `Tensile/Tensile.py`:

| File | Lines | What it is |
|------|-------|-----------|
| `tensilelite/Tensile/Tensile.py` | 664–665 | Deprecation-warning guard: `if "MaxFileName" in globalParameters or "MaxFileName" in config: printWarning(...)` |

No code reads `globalParameters["MaxFileName"]` as a governing value. No C++ or CMake file references it. The only use is the obsolete deprecation warning.

**The constant that replaced `MaxFileName` is:**

| File | What it is |
|------|-----------|
| `tensilelite/Tensile/Common/Constants.py:19` | `MAX_FILENAME_LENGTH: int = 64` (hard-coded replacement) |
| `tensilelite/Tensile/SolutionStructs/Naming.py:210,214` | Reader of `MAX_FILENAME_LENGTH` — the live filename-shortening logic |

The parameter has no live reader. Proceeding to Step 2.

---

## Step 2 — Git history archaeology

### Removal commit

`git log --all --oneline -S 'MaxFileName' -- '*.py'` returned three relevant commits:

| SHA | Date | Message |
|-----|------|---------|
| `d170037bd4fe` | 2025-02-12 | "Move Common.py to module (#1607)" |
| `c47b1d2a13a6` | (repo reorg) | "Reorganize project folders (#6)" |
| `843fe258090c` | 2025-09-18 | "SQLite vulnerability fix" (large batch commit) |

### What happened in `d170037bd4fe`

This is the authoritative removal commit. Its diff shows:

1. `globalParameters["MaxFileName"] = 64` was deleted from `Tensile/Common/GlobalParameters.py`.
2. `KernelWriter.py`'s `_shortenFileBase()` was updated from `globalParameters["MaxFileName"]` to `MAX_FILENAME_LENGTH`, a new constant in `Tensile/Common/Constants.py`.
3. A deprecation warning block was added to `Tensile/Tensile.py` (the check currently at lines 664–665):

```python
if "MaxFileName" in globalParameters or "MaxFileName" in config:
    printWarning("MaxFileName is no longer configurable, it will be automatically set to 64")
```

The commit message squash log explicitly includes: `feat: remove 'MaxFileName' from global params`.

### Why the deprecation warning is now inert — and misleading

The deprecation warning check runs at line 664, *after* `assignGlobalParameters` is called at line 654. Once the strict gate in `assignGlobalParameters` is active (Step 5 of the input-yaml validation series), a YAML file carrying `MaxFileName: 256` under `GlobalParameters:` will cause `assignGlobalParameters` to raise `ConfigTypeError` before execution ever reaches line 664. The warning is dead code under strict mode.

Additionally, `"MaxFileName" in config` checks the top-level YAML dict — not `config["GlobalParameters"]` — so it would never match (the value lives under `GlobalParameters:`). Only the `"MaxFileName" in globalParameters` branch could ever fire, and only if `assignGlobalParameters` did not raise.

### The `843fe258090c` commit

This large batch commit (on branch `origin/users/alaayala/fix_bdba_sqlite`, not yet merged to `develop`) removes the deprecation warning entirely from `Tensile/Tensile.py`. That branch has not reached `develop`, so the current HEAD on `users/alvasile/input_yaml` still carries the warning.

### Commits summary

| Role | SHA | Date | Message |
|------|-----|------|---------|
| `MaxFileName` added to globalParameters (oldest visible) | `c47b1d2a13a6` | (repo init) | Reorganize project folders |
| Reader + registry entry removed; constant + deprecation warning added | `d170037bd4fe` | 2025-02-12 | Move Common.py to module (#1607) |
| Deprecation warning removed (not yet on develop) | `843fe258090c` | 2025-09-18 | SQLite vulnerability fix |
| Strict gate that exposed stale key | `0ce0829c1642` | 2026-06-05 | input-yaml validation — Step 5 |

---

## Recommendation

Remove `MaxFileName: <N>` from all YAML files in the test corpus. The parameter has had no effect since Feb 12, 2025. The value is hard-coded at 64 in `Tensile/Common/Constants.py`; YAML overrides are silently ignored (or now raise `ConfigTypeError`). Do not add `MaxFileName` back to `globalParameters` or to the `ignoreKeys` list.

The affected YAML files at HEAD are (active, not commented-out):

- `Tensile/Tests/common/client/rotate_mode0.yaml`
- `Tensile/Tests/common/client/rotate_mode0_gfx12.yaml`
- `Tensile/Tests/common/client/rotate_mode1.yaml`
- `Tensile/Tests/common/client/rotate_mode1_gfx12.yaml`
- `Tensile/Tests/common/gemm/gfx950/lds160K.yaml`
- `Tensile/Tests/common/gemm/gfx950/mx32f4_tn.yaml`
- `Tensile/Tests/common/gemm/gfx950/mx32f8_tn.yaml`
- `Tensile/Tests/common/gemm/gfx950/plr_zero.yaml`
- `Tensile/Tests/common/gemm/wgm.yaml`
- `Tensile/Tests/common/sparse/gfx94x/spmm_vw_lg_one.yaml`
- `Tensile/Tests/common/sparse/gfx94x/spmm_vw_lg_one_sb.yaml`
- `Tensile/Tests/common/sparse/gfx94x/use_sgpr_for_gro.yaml`
- `Tensile/Tests/common/sparse/gfx94x/use_sgpr_for_gro_sb.yaml`
- `Tensile/Tests/common/sparse/gfx950/spmm_vw_lg_one.yaml`
- `Tensile/Tests/common/sparse/gfx950/spmm_vw_lg_one_sb.yaml`
- `Tensile/Tests/common/streamk/sk_hgemm_race.yaml`
- `Tensile/Tests/common/streamk/sk_mx32f4_quick.yaml`
- `Tensile/Tests/common/streamk/sk_mx32f8_quick.yaml`

Three additional files have `#MaxFileName: 256` (commented out): `dtl.yaml`, `dtv.yaml`, `dtv_gfx90a.yaml`, `swizzleA.yaml`, `swizzleB.yaml`. Those comments are harmless but can also be removed for cleanliness.

Additionally, the deprecation warning in `Tensile/Tensile.py` lines 664–665 is dead code under strict mode and should be deleted.

---

## Worktree path

`/home/alvasile/rocm-libraries/.claude/worktrees/agent-a699e9b60acce3919`
