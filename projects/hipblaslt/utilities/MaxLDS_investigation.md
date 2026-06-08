# `MaxLDS` YAML Parameter Investigation

## Verdict

**(A) Live reader exists.** `MaxLDS` is a valid *solution/kernel* parameter that is actively read and acted upon by kernel generation code. It is correctly registered in `defaultBenchmarkCommonParameters` and `validParameters`, and is actively consumed by `SolutionStructs/Solution.py`. The problem is that it has been misplaced in the `GlobalParameters:` YAML block (where it has no effect and now triggers `ConfigTypeError`) rather than in `BenchmarkCommonParameters:` or `ForkParameters:` within `BenchmarkProblems:`.

---

## Step 1 — Live-code search at HEAD

`git grep -n MaxLDS -- '*.py' '*.cpp' '*.cc' '*.h' '*.hpp' '*.cu' '*.cmake' '*.json' '*.toml' '*.md' '*.rst'`

Returned multiple live reader matches:

| File | Location | What it does |
|------|----------|-------------|
| `Tensile/Common/GlobalParameters.py:412` | `defaultBenchmarkCommonParameters` list | Registers `MaxLDS: [-1]` as default solution parameter (NOT in `globalParameters` dict) |
| `Tensile/Common/ValidParameters.py:655` | `validParameters["MaxLDS"]` | Declares the valid value set `[-1, 65536, 163840, 327680]` |
| `Tensile/Common/ValidParameters.py:302,305` | comments for `1LDSBuffer` | References `MaxLDS` semantics in docs |
| `Tensile/SolutionStructs/Solution.py:1430–1431` | `if state["MaxLDS"] == -1:` | Auto-resolves -1 to `isaInfoMap[isa].archCaps["DeviceLDS"]` |
| `Tensile/SolutionStructs/Solution.py:3073,3166` | LDS feasibility checks | Rejects solutions where `ldsNumBytesAlignedA + ldsNumBytesAlignedB > state["MaxLDS"]` |
| `Tensile/SolutionStructs/Solution.py:4499,4502,4512,4516` | `DtlPlusLdsBuf` logic | Decrements `numLdsBlk` when LDS exceeds `MaxLDS` |
| `Tensile/SolutionStructs/Solution.py:4532–4535` | `LocalSplitUReuseLDS` | Computes LDS reuse count from `MaxLDS` |
| `Tensile/SolutionStructs/Solution.py:4613,4876–4877` | Final LDS rejection | Rejects kernel if `ldsSize > state["MaxLDS"]` |
| `Tensile/Tests/unit/test_TensileLibLogicToYaml.py:237,420` | Unit test fixture | Uses `MaxLDS: 163840` as a solution parameter in test data |

Conclusion from Step 1: **live readers exist** — `MaxLDS` is a real, actively used solution parameter. Verdict is (A). Step 2 (history archaeology for dead parameter) is not applicable.

---

## The actual problem: wrong YAML block

`MaxLDS` is a **solution parameter**, not a global parameter. The parameter registries confirm this:

- `defaultBenchmarkCommonParameters` (line 412 of `GlobalParameters.py`): a list of per-solution defaults used by `BenchmarkStructs.py` to build `defaultSolution`. This is NOT the `globalParameters` `OrderedDict` that `assignGlobalParameters` validates.
- `validParameters["MaxLDS"]` in `ValidParameters.py:655`: solution-level validation.
- `globalParameters` (the dict starting at line 43): does NOT contain `MaxLDS` at any point.

The ~44 YAML files under `Tensile/Tests/common/` (plus files under `streamk/` and `sparse/`) set `MaxLDS` inside their `GlobalParameters:` top-level block. For example, in `gfx950/lds160K.yaml`:

```yaml
GlobalParameters:
  ...
  MaxLDS: 163840   # ← wrong section; not a global parameter
```

When `assignGlobalParameters` processes this, it finds `MaxLDS` not in `globalParameters` and now raises `ConfigTypeError` (the strict gate that landed in commit `0ce0829c`).

The correct placement is inside `BenchmarkProblems:` as a `BenchmarkCommonParameters` or `ForkParameters` entry — as is already done correctly in `gfx12/1024_vgpr_gfx1250.yaml`:

```yaml
BenchmarkProblems:
  - ...
    BenchmarkCommonParameters:
      ...
    ForkParameters:
      - MaxLDS: [327680]   # ← correct: solution parameter
```

---

## What `MaxLDS` does

When set to 163840 (160 KiB) or 327680 (320 KiB) in a solution, it overrides the default device LDS cap (`-1` auto-resolves to `isaInfoMap[isa].archCaps["DeviceLDS"]`). This is used to test kernels that require the expanded LDS available on gfx950 and gfx1250, which have 160 KiB or 320 KiB LDS respectively rather than the baseline 64 KiB. Without setting `MaxLDS`, the kernel generator would reject solutions that use more than 65536 bytes of LDS.

In the `GlobalParameters:` block the value was **silently ignored** — it was stored into `globalParameters` dict as an unknown key (the old warning-only path did `globalParameters[key] = value` after printing a warning), but `Solution.py` reads from `state["MaxLDS"]` (the solution state dict, not `globalParameters`), so no effective constraint was applied. The tests likely passed for a different reason: the `BenchmarkCommonParameters` default for `MaxLDS` is -1, which auto-resolves to the device's actual LDS cap via `isaInfoMap`, which on gfx950/gfx1250 is already 163840/327680. So the YAML setting in `GlobalParameters:` was redundant and unread, while the solution's auto-detection produced the correct answer anyway.

---

## Recommendation

**Remove `MaxLDS` from all `GlobalParameters:` sections** in the test YAML corpus. Do not add `MaxLDS` to the `globalParameters` dict or to the `ignoreKeys` list — the parameter is legitimate but it belongs in `BenchmarkCommonParameters:` / `ForkParameters:`, not `GlobalParameters:`.

For files where the only purpose of `MaxLDS: 163840` in `GlobalParameters:` was to test 160 KiB LDS kernels, verify that the corresponding `BenchmarkProblems:` section already has `MaxLDS` in `BenchmarkCommonParameters` or `ForkParameters` (or relies on auto-detection, which already works correctly for gfx950/gfx1250). If tests pass without the `GlobalParameters:` entry — and they should, because auto-detection resolves to the correct device LDS cap — simply delete the misplaced line.

The `git grep` targeting the wrong-block occurrences:

```bash
git grep -n 'MaxLDS' -- 'projects/hipblaslt/tensilelite/Tensile/Tests/common/**/*.yaml'
```

All matches at indentation level matching `GlobalParameters:` child keys (two-space or four-space indent, unindented relative to their parent `GlobalParameters:` block) are the ones to remove.

---

## Worktree path

`/home/alvasile/rocm-libraries/.claude/worktrees/agent-a86455e99fab251b9`
