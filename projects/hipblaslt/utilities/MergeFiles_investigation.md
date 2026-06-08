# `MergeFiles` YAML Parameter Investigation

## Verdict

**(C) Once-live, now dead.** `MergeFiles` was a real global parameter with a real reader that controlled whether generated kernel source files were merged into a single file or kept separate. Both the registry entry and all readers were removed in commit `2d2e1496a9d7` (January 2, 2025, "Remove merge-files option (#1407)"). That commit cleaned up ~45 YAML files but missed several, including `largeMT.yaml`. Subsequent new test YAML files added between April–June 2026 also inadvertently included the stale key. The parameter has been inert since the removal — assignments in YAML were silently swallowed as unknown-key warnings until the strict gate landed in June 2026.

---

## Step 1 — Live-code search at HEAD

`git grep -n MergeFiles -- '*.py' '*.cpp' '*.cc' '*.h' '*.hpp' '*.cu' '*.cmake' '*.json' '*.toml' '*.md' '*.rst'` returned **no matches**.

No Python, C++, CMake, or other source file at HEAD references `MergeFiles` in any way. The string does not appear in `Tensile/Common/GlobalParameters.py`, in any Python module under `tensilelite/`, or in any C++ file.

Conclusion from Step 1: **no live reader exists** — proceeding to Step 2.

---

## Step 2 — Git history archaeology

### Where `MergeFiles` was defined and read

At `dbc5e593321c` (December 23, 2024 — the oldest accessible commit where `largeMT.yaml` exists), `MergeFiles` was a live global parameter defined in `Tensile/Common.py`:

```
projects/hipblaslt/tensilelite/Tensile/Common.py:221:
    globalParameters["MergeFiles"] = True  # F=store every solution and kernel in separate file; T=store all solutions in single file
```

The reader logic in the same file enforced consistency between `MergeFiles` and `NumMergedFiles`:

```
projects/hipblaslt/tensilelite/Tensile/Common.py:1730:
    if "MergeFiles" in config and "NumMergedFiles" in config:
        if not config["MergeFiles"] and config["NumMergedFiles"] > 1:
            # ... emit warning/error about conflicting flags
```

The parameter also flowed through `KernelWriter.py`, `ClientWriter.py`, `BenchmarkProblems.py`, and `AssemblyCommands.py` — all readers that controlled the actual file layout of generated kernel code.

`install.sh` exposed it as a CLI flag pair `--merge-files` / `--no-merge-files`.

### Removal commit

**SHA:** `2d2e1496a9d7a152da829861bd6ed5216c7d2235`
**Date:** 2025-01-02
**Author:** David Dixon
**Message:** "Remove merge-files option (#1407)"

This commit simultaneously:
- Deleted `globalParameters["MergeFiles"] = True` from `Tensile/Common.py`
- Removed the `MergeFiles` / `NumMergedFiles` consistency-check block from `Common.py`
- Removed all `MergeFiles`-conditional branches from `KernelWriter.py`, `ClientWriter.py`, `BenchmarkProblems.py`, and `AssemblyCommands.py`
- Removed `--merge-files` / `--no-merge-files` CLI flags from `install.sh`
- Cleaned `MergeFiles:` from ~45 YAML test files in the commit message line "Update yaml files in tests"

### What was missed

The cleanup in `2d2e1496a9d7` was incomplete in two ways:

1. **`largeMT.yaml` was not cleaned up.** The file existed at the time of the removal commit (confirmed by `git show 2d2e1496a9d7:projects/hipblaslt/tensilelite/Tensile/Tests/common/gemm/largeMT.yaml | grep MergeFiles` returning `MergeFiles: False`) but was not listed in the commit's changed-file set. It has carried the stale key ever since.

2. **New test YAMLs added after the removal re-introduced the key.** Four commits added new YAMLs containing `MergeFiles: False` long after it was removed from code:

| Commit | Date | Author | File(s) |
|--------|------|--------|---------|
| `7925a41496fb` | 2026-04-08 | Alex Vasile | `Tests/mxfp4/slow.yaml` |
| `3492f3599c3b` | 2026-06-02 | Brad Nemanich | `Tests/common/gemm/gfx950/subtile_mxfp8.yaml` |
| `d67460539a0f` | 2026-06-02 | Brad Nemanich | `Tests/common/gemm/gfx950/subtile_mxfp8.yaml` |
| `b76503f63b42` | 2026-06-03 | Prabhjot Sandhu | `Tests/common/gemm/gfx950/subtile_fp8.yaml` |

These were written by authors who likely copied from existing YAMLs that still contained the stale key, or used it from memory without realizing it had been removed 16 months earlier.

### Why it went undetected for 16 months

After `2d2e1496a9d7`, `assignGlobalParameters` in `GlobalParameters.py` handled unknown keys with `printWarning(...)` only — not an error. Every YAML run would have printed a warning line for `MergeFiles`, but tests passed and CI did not enforce clean output. The strict gate (commit `0ce0829c1642`, June 5 2026, "input-yaml validation — Step 5: assignGlobalParameters strict gate") upgraded this from a warning to a `ConfigTypeError`, finally surfacing the stale key.

### Commits summary

| Role | SHA | Date | Message |
|------|-----|------|---------|
| Last commit with reader intact | `dbc5e593321c` | 2024-12-23 | "Use archVGPR when accVGPR is not enough. (#1460)" |
| Reader + registry entry removed; partial YAML sweep | `2d2e1496a9d7` | 2025-01-02 | "Remove merge-files option (#1407)" |
| New YAML re-introducing stale key (oldest) | `7925a41496fb` | 2026-04-08 | "Add test yaml" |
| New YAMLs re-introducing stale key (most recent) | `b76503f63b42` | 2026-06-03 | "[SubtileImpl] Add subtile_fp8_256x288_vgpr_overflow test YAML" |
| Strict gate that exposed the stale key | `0ce0829c1642` | 2026-06-05 | "input-yaml validation — Step 5" |

---

## Recommendation

Delete `MergeFiles:` (and any commented-out `# MergeFiles:`) from every YAML in the corpus. The line has had no effect since January 2, 2025. There is no need to add `MergeFiles` back to `globalParameters` or to any `ignoreKeys` list — those options would perpetuate dead config rather than clean it up.

The affected files (16 active occurrences plus 4 commented-out) are all under `projects/hipblaslt/tensilelite/Tensile/Tests/`. A targeted sweep:

```
git grep -n 'MergeFiles' -- '*.yaml' '*.yml'
```

will locate every occurrence (active and commented-out) for the cleanup.

---

## Worktree path

`/home/alvasile/rocm-libraries/.claude/worktrees/agent-af22449af7b297094`
