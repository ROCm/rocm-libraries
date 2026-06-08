# `PrintSolutionRejectionReason` YAML Parameter Investigation

## Verdict

**(A) Live reader — and the strict validation gate is exposing a routing bug, not dead config.**

`PrintSolutionRejectionReason` was deliberately removed from `globalParameters` in commit `dc2c963c` (March 25, 2025, "Remove global variables") and simultaneously re-routed to a new `DebugConfig` struct populated by `makeDebugConfig()`. The YAML key is therefore **alive and functional** — it is read directly from `config["GlobalParameters"]` by `makeDebugConfig()` in `Tensile/Common/Types.py` (line 98–99). The strict gate in `assignGlobalParameters` fires on it because `makeDebugConfig` runs _after_ `assignGlobalParameters` in `Tensile.py` (lines 654, 658), and the key was never added to the gate's `ignoreKeys` list.

---

## Step 1 — Live-code search at HEAD

`git grep -n PrintSolutionRejectionReason -- '*.py' '*.cpp' ... '*.rst'` returned six hits in code files (not counting YAML tests):

| File | Lines | Role |
|------|-------|------|
| `Tensile/Common/Types.py` | 98–99 | **Active YAML reader** — `makeDebugConfig()` reads `config["PrintSolutionRejectionReason"]` directly from the raw `GlobalParameters` dict |
| `Tensile/TensileLogic/Run.py` | 169–170 | Injects `{"PrintSolutionRejectionReason": True}` into `gp_config` when verbose; explicitly comments "Only set ... when verbose to avoid 'unrecognised' warning in quiet mode" — aware of the gap |
| `Tensile/BenchmarkProblems.py` | 588 | User-facing help string: "You should re-run with `PrintSolutionRejectionReason: True`" |

`git grep -n printSolutionRejectionReason -- '*.py'` returned 40+ hits across `BenchmarkProblems.py`, `ClientWriter.py`, `Contractions.py`, `KernelWriterAssembly.py`, `LibraryIO.py`, `LibraryLogic.py`, `SolutionLibrary.py`, `SolutionStructs/Solution.py`, `SolutionStructs/Utilities.py`, `SolutionStructs/Validators/MatrixInstruction.py`, `Tensile.py`, `TensileCreateLibrary/Run.py`, `Tests/unit/test_LibraryLogic_types.py`, and `Tests/unit/test_validateParameterTypes.py`.

The flag is threaded through every major code path that selects and rejects kernels. It is a first-class feature.

**Conclusion from Step 1: a live reader exists.** The YAML key `PrintSolutionRejectionReason` is fully wired to the camelCase flag `printSolutionRejectionReason`. The strict gate is incorrectly firing on a valid operational parameter.

---

## Step 2 — Git history

### Removal from `globalParameters`

**SHA:** `dc2c963c892457151df6f08a687bcb47912ed3f3`  
**Date:** 2025-03-25  
**Author:** David Dixon  
**Message:** "Remove global variables (#1677)"

The diff for `Tensile/Common/GlobalParameters.py` in this commit removed:

```python
-globalParameters["PrintSolutionRejectionReason"] = (
-    False  # when a solution is marked as invalid, print why
-)
```

The same commit added `DebugConfig` (a `NamedTuple`) and `makeDebugConfig()` to `Tensile/Common/Types.py`. The new `makeDebugConfig()` reads `PrintSolutionRejectionReason` directly from the raw config dict (the YAML `GlobalParameters:` block), bypassing `globalParameters` entirely. The commit also updated `Tensile.py` to call `makeDebugConfig(config["GlobalParameters"])` immediately after `assignGlobalParameters(config.get("GlobalParameters", {}), isaInfoMap)`.

This was an intentional architectural move: the parameter was migrated from the global-variable system to a typed `DebugConfig` struct. The YAML key spelling was preserved (PascalCase) and the route was updated. **The YAML corpus was correct to keep the key; the routing simply was not reflected in `ignoreKeys`.**

### The new strict gate

The strict gate (added June 2026, "input-yaml validation — Step 5: assignGlobalParameters strict gate") promotes unknown keys to a `ConfigTypeError`. Because `PrintSolutionRejectionReason` is not in `globalParameters` and is not in `ignoreKeys`, it triggers the error — even though `makeDebugConfig` downstream consumes it correctly.

`TensileLogic/Run.py` line 169 already acknowledged this tension: it avoids passing the key to `assignGlobalParameters` in quiet mode specifically "to avoid 'unrecognised' warning in quiet mode". That comment predates the strict gate and confirms the architectural gap was known but not fully closed.

---

## Recommendation

**Do not delete the YAML entries.** The parameter is live. The fix is to add `"PrintSolutionRejectionReason"` to `ignoreKeys` in `assignGlobalParameters` (in `Tensile/Common/GlobalParameters.py`, around line 795), following the same pattern as other retired-from-registry-but-still-valid keys (e.g. `"Architecture"`, `"PrintLevel"`).

The rationale: `makeDebugConfig` is the correct consumer of this key; `assignGlobalParameters` should skip it rather than error on it. The key belongs to `DebugConfig`, not `globalParameters`.

### Recommended fix

In `Tensile/Common/GlobalParameters.py`, `assignGlobalParameters()`, extend `ignoreKeys`:

```python
ignoreKeys = [
    "Architecture",
    "PrintLevel",
    "Device",
    "UseCompression",
    "CxxCompiler",
    "CCompiler",
    "OffloadBundler",
    "Assembler",
    "LogicPath",
    "LogicFilter",
    "OutputPath",
    "Experimental",
    "GenSolTable",
    # Moved to DebugConfig (makeDebugConfig in Common/Types.py); not a globalParameter.
    "PrintSolutionRejectionReason",
]
```

Other keys from `DebugConfig` (`EnableAsserts`, `EnableDebugA`, `EnableDebugB`, `EnableDebugC`, `ExpectedValueC`, `ForceCExpectedValue`, `DebugKernel`, `ForceGenerateKernel`, `SplitGSU`, `PrintIndexAssignmentInfo`) should be audited in the same sweep — they have the same structural situation and will trigger the same strict-gate error if any YAML exercises them.

---

## Worktree path

`/home/alvasile/rocm-libraries/.claude/worktrees/agent-a5b65d5e44f3ce8b9`
