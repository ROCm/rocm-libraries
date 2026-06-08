# `MinKForGSU` YAML Parameter Investigation

## Verdict

**(C) Once-live, now dead.** `MinKForGSU` was a real global parameter with a real reader in `Tensile/Contractions.py`. Both were removed in commit `dc2c963c` (March 25, 2025, "Remove global variables (#1677)") — the reader was replaced with a module-level constant `MIN_K_FOR_GSU = 32` and the registry entry was commented out in `GlobalParameters.py`. The YAML corpus was not cleaned up at that time. The parameter has been inert ever since — assignments in YAML were silently swallowed as unknown-key warnings until the strict gate landed in June 2026.

---

## Step 1 — Live-code search at HEAD

`git grep -n MinKForGSU -- '*.py' '*.cpp' '*.cc' '*.h' '*.hpp' '*.cu' '*.cmake' '*.json' '*.toml' '*.md' '*.rst'` returned **one match**:

| File | Line | What it is |
|------|------|-----------|
| `tensilelite/Tensile/Common/GlobalParameters.py` | 280 | **Commented-out** entry: `#globalParameters["MinKForGSU"] = 32  # min K size to use GlobalSplitU algorithm (only for HPA now)` |

No Python or C++ code at HEAD reads `globalParameters["MinKForGSU"]` or any equivalent. No C++ files, CMake files, or config files reference `MinKForGSU` as a live symbol.

The actual business logic that was controlled by this parameter is now hardcoded as a module constant in `Contractions.py`:

```python
# tensilelite/Tensile/Contractions.py:39
MIN_K_FOR_GSU = 32
```

This constant is used at line 577 in the same file to generate a `GlobalSplitUCheckMinK` predicate. The YAML key can no longer influence this value.

Conclusion from Step 1: **no live reader exists** — proceeding to Step 2.

---

## Step 2 — Git history archaeology

### Introduction of `MinKForGSU` in code

The initial "Add tensilelite" commit (`f8246afa`, Nov 11 2022) introduced `MinKForGSU` as an **active** registry entry with value 256, and a reader in `Contractions.py`:

```python
# GlobalParameters.py (at f8246afa)
globalParameters["MinKForGSU"] = 256  # min K size to use GlobalSplitU algorithm (only for HPA now)

# Contractions.py (at f8246afa)
value = globalParameters['MinKForGSU'] * state['GlobalSplitU']
rv += [cls('GlobalSplitUCheckMinK', value=[value, state["GlobalSplitU"]])]
```

The parameter allowed callers to tune the minimum K dimension threshold required before GlobalSplitU would be attempted. The default changed from 256 to 32 at some point between the initial commit and the reader removal.

### YAML test files

The first YAML file to set `MinKForGSU` in the test corpus was introduced in commit `46b746db` ("update tensilelite ci yaml for new parameter", May 16 2023), setting `MinKForGSU: 1` to force GSU eligibility for test problems with small K. This was a legitimate use of the parameter at the time — the YAML value controlled the threshold at runtime.

Additional YAML files carrying `MinKForGSU` were added over subsequent months (e.g., `92d79dff`, Jul 28 2023, "add bias stride test"; `78993f18`, "[Tensilelite] Enable Sparse I8/H/S").

### Where `MinKForGSU` was read

At the time of removal, the sole reader was in `Tensile/Contractions.py`:

```python
if '_GlobalAccumulation' in state and state['_GlobalAccumulation'] != None and not state["StreamK"]:
    value = globalParameters['MinKForGSU']
    rv += [cls('GlobalSplitUCheckMinK', value=[value, state["GlobalSplitU"]])]
```

This generated a kernel predicate checking whether the problem's K dimension met the threshold. The YAML-provided `MinKForGSU: 1` in test files was lowering the threshold to ensure GSU kernels were considered even for small-K test shapes.

### Removal commit

**SHA:** `dc2c963c892457151df6f08a687bcb47912ed3f3`
**Date:** 2025-03-25
**Author:** David Dixon
**Message:** "Remove global variables (#1677)"

This commit simultaneously:
- Changed the reader in `Tensile/Contractions.py` from `globalParameters['MinKForGSU']` to the module constant `MIN_K_FOR_GSU`
- Commented out `globalParameters["MinKForGSU"] = 32` in `Tensile/Common/GlobalParameters.py`

The commit touched a large number of files as part of a broader effort to eliminate runtime reads of module-level global state, and did **not** touch any YAML test files. The `MinKForGSU` stanza in ~102 YAML files was left as orphaned config.

Note: because the replacement `MIN_K_FOR_GSU = 32` is hardcoded, the YAML values (almost all `MinKForGSU: 1`) no longer have any effect on kernel selection. Tests that depend on GSU kernels being considered for small-K shapes now rely on `MIN_K_FOR_GSU = 32` being a low enough threshold, not on the YAML override.

### Why it went undetected for 14 months

At the time of removal, `assignGlobalParameters` handled unknown keys with `printWarning(...)` only, silently storing the unknown key into `globalParameters` anyway:

```python
if key not in globalParameters:
    printWarning("Global parameter %s = %s unrecognised." % (key, value))
globalParameters[key] = value  # stored anyway, never read again
```

After `dc2c963c`, every YAML run would have printed a warning for `MinKForGSU`, but tests passed and CI did not enforce clean output. The strict gate (Step 5 of the input-yaml validation work, June 2026) upgraded this from a warning to a `ConfigTypeError`, finally surfacing the stale key.

### Commits summary

| Role | SHA | Date | Message |
|------|-----|------|---------|
| First introduction as active parameter | `f8246afa` | 2022-11-11 | "Add tensilelite" |
| First YAML use (sample) | `46b746db` | 2023-05-16 | "update tensilelite ci yaml for new parameter" |
| Reader replaced with constant; registry entry commented out | `dc2c963c` | 2025-03-25 | "Remove global variables (#1677)" |
| Strict gate that exposed the stale key | (Step 5 of input-yaml validation) | 2026-06 | — |

---

## Recommendation

Delete `MinKForGSU: <value>` from every YAML in the test corpus. The line has had no effect since March 25, 2025. There is no need to add `MinKForGSU` back to `globalParameters` or to the `ignoreKeys` list — those options would perpetuate dead config rather than clean it up.

The hardcoded replacement `MIN_K_FOR_GSU = 32` in `Contractions.py` already covers the original intent. Test files that previously set `MinKForGSU: 1` to lower the GSU eligibility threshold no longer benefit from this — the hardcoded threshold of 32 should be sufficient for any reasonable test K size. If a test actually needs a threshold below 32, that would need a code change to `MIN_K_FOR_GSU` or a rework of the predicate, but this is not expected.

A `grep -rn 'MinKForGSU' --include='*.yaml'` confined to the test trees will locate all ~102 occurrences.

---

## Worktree path

`/home/alvasile/rocm-libraries/.claude/worktrees/agent-a7accad731801910d`
