# `PrintIndexAssignments` YAML Parameter Investigation

## Verdict

**(C) Once-live, now dead — rename artifact, not a typo.**

`PrintIndexAssignments` was a real `globalParameters` entry with real readers from its introduction (commit `bffa850d`, June 2023) through the "Remove global variables" refactor (commit `dc2c963c`, March 25, 2025). In that refactor the global-variable approach was replaced with a typed `DebugConfig` named-tuple plumbed explicitly through function signatures. The new YAML key for the same concept became `PrintIndexAssignmentInfo` (note the added `Info` suffix), read in `Types.py:makeDebugConfig()`. The old YAML key `PrintIndexAssignments` was **not cleaned up** from the one YAML file that carried it (`Tensile/Tests/common/sgemm_xf32_asm.yaml`). Since then it has been inert — the strict gate introduced in June 2026 now rejects it.

The key is **not** a near-typo of `printIndexAssignmentInfo`; it is the direct predecessor. The correct resolution is to rename it to `PrintIndexAssignmentInfo` in the YAML file, not to delete it.

---

## Step 1 — Live-code search at HEAD

### Search for `PrintIndexAssignments` (PascalCase, no `Info` suffix)

```
git grep -n PrintIndexAssignments -- '*.py' '*.cpp' '*.cc' '*.h' '*.hpp' '*.cu' '*.cmake' '*.json' '*.toml' '*.md' '*.rst'
```

**Zero matches.** No Python or C++ code at HEAD reads `PrintIndexAssignments` from global parameters or anywhere else.

YAML search:

```
git grep -rn PrintIndexAssignments -- '*.yaml' '*.yml'
```

Single match:

```
projects/hipblaslt/tensilelite/Tensile/Tests/common/sgemm_xf32_asm.yaml:19:  PrintIndexAssignments: 1
```

Conclusion: the YAML key has **no live reader**. Proceeding to Step 2.

### Search for `printIndexAssignmentInfo` (camelCase with `Info` suffix)

`git grep -n printIndexAssignmentInfo -- '*.py' ...` returned **46 matches** spanning:
- `Tensile/Common/Types.py` — `DebugConfig` named-tuple definition and `makeDebugConfig()` reader (reads YAML key `PrintIndexAssignmentInfo`).
- `Tensile/SolutionStructs/Problem.py` — `ProblemType.__init__`, `assignDerivedParameters`, conditional print logic.
- `Tensile/BenchmarkProblems.py`, `BenchmarkStructs.py`, `Contractions.py`, `LibraryIO.py`, `LibraryLogic.py`, `SolutionLibrary.py`, `SolutionStructs/Solution.py`, `TensileCreateLibrary/Run.py`, `ClientWriter.py`, `Tensile.py`.

`PrintIndexAssignmentInfo` (PascalCase, the YAML key for the live flag) is read at:

```
projects/hipblaslt/tensilelite/Tensile/Common/Types.py:102:
    if "PrintIndexAssignmentInfo" in config:
        printIndexAssignmentInfo = config["PrintIndexAssignmentInfo"]
```

This is inside `makeDebugConfig(config)` which reads from the `GlobalParameters:` YAML block. `PrintIndexAssignmentInfo` **is** the current live key; it is not in `globalParameters` (the registry) because the `DebugConfig` subsystem explicitly sidesteps the registry — see `ignoreKeys` discussion below.

---

## Step 2 — Git history archaeology

### Introduction of `PrintIndexAssignments` in code

Commit `bffa850dfe3e` (June 5, 2023, "[Tensilelite] Support XFloat32 data type and instruction") added both:

1. `globalParameters["PrintIndexAssignments"] = 0  # Print the tensor index assignment info` in `Common.py`.
2. Three reader sites in `SolutionStructs.py`:
   ```python
   if globalParameters["PrintIndexAssignments"]:
   ```
3. The YAML key `PrintIndexAssignments: 1` in `Tensile/Tests/common/sgemm_xf32_asm.yaml`.

The key was therefore **live from day one** in that YAML file.

### Module refactor (Feb 12, 2025)

Commit `d170037bd4fe` ("Move Common.py to module") moved `Common.py` into the `Common/` package. `PrintIndexAssignments` was faithfully copied to `Common/GlobalParameters.py`:

```python
globalParameters["PrintIndexAssignments"] = 0  # Print the tensor index assignment info
```

### Removal and rename (Mar 25, 2025)

Commit `dc2c963c892457151df6f08a687bcb47912ed3f3` ("Remove global variables", by David Dixon) simultaneously:

- **Removed** `globalParameters["PrintIndexAssignments"]` from `Common/GlobalParameters.py`.
- **Added** `DebugConfig` named-tuple in `Common/Types.py` with field `printIndexAssignmentInfo: bool = False`.
- **Added** `makeDebugConfig()` in `Types.py` which reads the new YAML key `PrintIndexAssignmentInfo` (with `Info` suffix).
- **Refactored** all call sites in `BenchmarkProblems.py`, `Problem.py`, `Solution.py`, etc. to pass `printIndexAssignmentInfo` explicitly rather than reading from global state.

The YAML file `sgemm_xf32_asm.yaml` was **not updated** in this commit. The old key `PrintIndexAssignments: 1` was left behind while the reader now expects `PrintIndexAssignmentInfo`.

### No subsequent reader

No commit after `dc2c963c` reintroduces any reader for the old name. The key has been dead — and silently tolerated by the permissive unknown-key path — since March 2025.

---

## Key distinction: `PrintIndexAssignmentInfo` vs. the `globalParameters` registry

`PrintIndexAssignmentInfo` is read from the YAML `GlobalParameters:` block by `makeDebugConfig()` in `Types.py`, **not** via the `globalParameters` dict in `GlobalParameters.py`. The strict gate in `assignGlobalParameters` only validates keys against the `globalParameters` registry. `makeDebugConfig()` is called separately (in `Tensile.py` and `BenchmarkProblems.py`) and consumes the block before (or independently of) the registry check. Neither `PrintIndexAssignments` nor `PrintIndexAssignmentInfo` is in `globalParameters`; they live in the `DebugConfig` subsystem.

This means:
- `PrintIndexAssignmentInfo: true` in a YAML file will be read correctly by `makeDebugConfig()` and will also trigger the "unknown key" error in the strict gate — it is in the same position as `PrintIndexAssignments`.
- The YAML file under test uses the **old name** (`PrintIndexAssignments`) which no code reads at all; it should be renamed to `PrintIndexAssignmentInfo`.

---

## Recommendation

Rename the key in `Tensile/Tests/common/sgemm_xf32_asm.yaml` from:

```yaml
PrintIndexAssignments: 1
```

to:

```yaml
PrintIndexAssignmentInfo: true
```

Notes:
- The value `1` should become `true` (a bool) to match the `bool` type expected by `makeDebugConfig()` and to satisfy the strict type-gate if `PrintIndexAssignmentInfo` is ever added to `globalParameters` or the `DebugConfig` gate has type checking added.
- If `PrintIndexAssignmentInfo` is intended to be validated by the strict gate, it should be added to the `ignoreKeys` list in `assignGlobalParameters` (like `PrintLevel` and `Device`) or moved into the `globalParameters` registry. Currently both old and new keys fall through to "unknown key" because the `DebugConfig` subsystem reads independently from the registry. This is a pre-existing structural issue not introduced by this YAML file.
- Do **not** delete the line — the test was intentionally exercising this debug flag.

---

## Worktree path

`/home/alvasile/rocm-libraries/.claude/worktrees/agent-a84f15f1275088fa7`
