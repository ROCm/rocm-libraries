# `NewClient` YAML Parameter Investigation

## Verdict

**(C) Once-live, now dead.** `NewClient` was a real global parameter with a real reader, both of which were removed in commit `dc2c963c` (March 25, 2025, "Remove global variables"). The YAML corpus was not cleaned up at that time. The parameter has been inert ever since — assignments in YAML were silently swallowed as unknown-key warnings until the strict gate landed in June 2026.

---

## Step 1 — Live-code search at HEAD

`git grep -n NewClient -- '*.py' '*.cpp' '*.cc' '*.h' '*.hpp' '*.cu' '*.cmake' '*.json' '*.toml' '*.md' '*.rst'` returned **four matches**, none of which is a parameter reader:

| File | Line | What it is |
|------|------|-----------|
| `tensilelite/Tensile/ClientWriter.py` | 211 | Function **named** `runNewClient` — unrelated to the `NewClient` YAML key |
| `tensilelite/Tensile/GenerateSummations.py` | 134 | Call to `ClientWriter.runNewClient(...)` — same function, not the key |
| `tensilelite/Tensile/Tests/common/flags/gfx942/test_build_only.py` | 43 | Inline YAML string `NewClient: 2` embedded in a Python test constant `_CONFIG` — data, not code |
| `tensilelite/Tensile/Tests/common/flags/gfx942/test_use_cache.py` | 43 | Same pattern |

No Python or C++ code at HEAD reads `globalParameters["NewClient"]` or any equivalent. No C++ files, CMake files, or config files reference `NewClient` at all.

Conclusion from Step 1: **no live reader exists** — proceeding to Step 2.

---

## Step 2 — Git history archaeology

### Introduction of `NewClient` in code

The oldest accessible commit with `GlobalParameters.py` is `d170037b` ("Move Common.py to module", Feb 12 2025). At that commit, the parameter was already present:

```
projects/hipblaslt/tensilelite/Tensile/Common/GlobalParameters.py:281:
    globalParameters["NewClient"] = 2  # Old client deprecated: NewClient must be set to 2.
```

The comment itself says the **old client** was deprecated and `NewClient` was the **replacement** — meaning `NewClient` was introduced upstream in the original Tensile project when a new benchmark client was written and the parameter was used as an assertion that callers had updated their configs. By the time TensileLite was imported into this repo (the earliest YAML commit is `c9a10ea5`, Feb 21 2023), `NewClient` was already in the registry and the YAMLs already carried `NewClient: 2`.

### Where `NewClient` was read

At commit `faf16a8c` ("dot2 fp16 mac kernel for gfx942", March 14 2025 — the last commit before the removal), the reader lived in `SolutionStructs.py`:

```
projects/hipblaslt/tensilelite/Tensile/SolutionStructs.py:1304:
    if globalParameters["NewClient"] != 2:
        print("WARNING: Old client deprecated, NewClient parameter being set to 2.")
        globalParameters["NewClient"] = 2
```

This was a one-way enforcement guard: if someone accidentally passed `NewClient: 1` (the old client), it would print a warning and force the value back to 2. With the old client fully gone, the guard was meaningless.

### Removal commit

**SHA:** `dc2c963c892457151df6f08a687bcb47912ed3f3`  
**Date:** 2025-03-25  
**Author:** David Dixon  
**Message:** "Remove global variables (#1677)"

This commit simultaneously deleted:
- `globalParameters["NewClient"] = 2` from `Tensile/Common/GlobalParameters.py`
- The three-line guard block from `Tensile/SolutionStructs.py`

The commit touched a large number of files (moving away from module-level global variable reads) and did **not** touch any YAML test files. The `NewClient: 2` stanza in ~183 YAML files was left as orphaned config.

### Why it went undetected for 14 months

At the time of removal, `assignGlobalParameters` in `GlobalParameters.py` handled unknown keys with `printWarning(...)` only — not an error. The relevant code at commit `faf16a8c` (line 1783–1784):

```python
if key not in globalParameters:
    printWarning("Global parameter %s = %s unrecognised." % (key, value))
globalParameters[key] = value  # silently stored anyway
```

After `dc2c963c`, every YAML run would have printed a warning line for `NewClient`, but tests passed and CI did not enforce clean output. The strict gate (commit `0ce0829c`, June 5 2026, "input-yaml validation — Step 5: assignGlobalParameters strict gate") upgraded this from a warning to a `ConfigTypeError`, finally surfacing the stale key.

### Commits summary

| Role | SHA | Date | Message |
|------|-----|------|---------|
| First YAML introduction (sample) | `c9a10ea5` | 2023-02-21 | "Add Tests for tensilelite" |
| Last commit with reader intact | `faf16a8c` | 2025-03-14 | "dot2 fp16 mac kernel for gfx942" |
| Reader + registry entry removed | `dc2c963c` | 2025-03-25 | "Remove global variables (#1677)" |
| Strict gate that exposed the stale key | `0ce0829c` | 2026-06-05 | "input-yaml validation — Step 5" |

---

## Recommendation

Delete `NewClient: 2` from every YAML (and the two embedded Python test configs) across the corpus. The line has had no effect since March 25, 2025. There is no need to add `NewClient` back to `globalParameters` or to the `ignoreKeys` list — those options would perpetuate dead config rather than clean it up.

The two Python test files (`test_build_only.py` and `test_use_cache.py`) embed the YAML as a string constant and must be edited directly (not by a YAML-only sweep tool).

A `grep -rn 'NewClient' --include='*.yaml' --include='*.py'` confined to the test trees will locate all occurrences.

---

## Worktree path

`/home/alvasile/rocm-libraries/.claude/worktrees/agent-a3d8c2a90667c7024`
