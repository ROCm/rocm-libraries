# Item 6: Naming.getKernelNameMin splitGSU TypeError

## Verdict

Open.

## Current Source References

- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/SolutionStructs/Naming.py:230-231`: `getKernelNameMin(kernel, splitGSU)` calls `_getName(kernel, getRequiredParametersMin(), splitGSU, True)`, so this path always enters `_getName` with `ignoreInternalArgs=True`.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/SolutionStructs/Naming.py:149-155`: `_getName` backs up `GlobalSplitU`, then with `ignoreInternalArgs=True` and `splitGSU=True` rewrites `state["GlobalSplitU"]` to the string `"M"` when the original value is `> 1` or `== -1`.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/SolutionStructs/Naming.py:157-161`: immediately after the rewrite, `_getName` evaluates `state["GlobalSplitU"] > 0 or state["GlobalSplitU"] == -1`; for split-GSU values rewritten to `"M"`, the first comparison is `"M" > 0`, which raises `TypeError` on Python 3.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/SolutionStructs/Naming.py:202-203`: the restore of `GlobalSplitU` and `ProblemType["GroupedGemm"]` happens after the failing comparison, so the exception path can leave the caller's state mutated.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/SolutionStructs/Naming.py:208-226`: `getKernelFileBase()` routes through `shortenFileBase()` and `getKernelNameMin()`, so filename generation can hit the same crash.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/TensileCreateLibrary/Run.py:223`: `processKernelSource()` calls `getKernelFileBase(splitGSU, kernel)` before source generation.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/TensileCreateLibrary/Run.py:294-303`: `passPostKernelInfoToSolution()` calls `getKernelNameMin(..., splitGSU)` for generated kernels and solution kernels.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/LibraryLogic.py:127-132`: `analyzeProblemType()` assigns `KernelNameMin` using `getKernelNameMin(s, splitGSU)`.

## Characterization

- The referenced characterization file is available at `/home/alvasile/repos/rocm-libraries-investigation/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/Naming/test_naming_char.py:191-198`.
- The same characterization also exists in the current target source at `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/Naming/test_naming_char.py:191-198`.
- It pins `GlobalSplitU` values `4` and `-1` with `splitGSU=True` and expects `N.getKernelNameMin(...)` to raise `TypeError`, matching the current source.

## Evidence

Static trace for `getKernelNameMin(state, splitGSU=True)` with `state["GlobalSplitU"] == 4`:

```text
getKernelNameMin -> _getName(..., ignoreInternalArgs=True)
line 155: state["GlobalSplitU"] = "M"
line 160: state["GlobalSplitU"] > 0
         -> "M" > 0
         -> TypeError: '>' not supported between instances of 'str' and 'int'
```

The same trace applies to original `GlobalSplitU == -1`, because line 155 also rewrites it to `"M"` before line 160 compares it with `0`.

I did not run pytest. A direct no-bytecode Python reproduction attempt was blocked during import by the local environment before reaching `Naming.py` execution:

```text
ImportError: cannot import name 'rocIsa' from 'rocisa' (unknown location)
```

## Impact

This can crash any path that asks for a minimal kernel name with `splitGSU=True` for split-GSU kernels where `GlobalSplitU > 1` or `GlobalSplitU == -1`. That includes library-logic naming, kernel filename generation, and post-codegen kernel-result mapping. Because `_getName` mutates the input state before the failing comparison and restores it only at the end, the crash path can also leave `state["GlobalSplitU"] == "M"` and force `ProblemType["GroupedGemm"]` to `False` in the caller's object.

## Recommended Fix

Avoid comparing the masked string value as a number. The least invasive fix is to make the discard decision from the backed-up numeric value before or independent of the `"M"` rewrite, for example:

```python
gsuBackup = state["GlobalSplitU"]
...
if ignoreInternalArgs:
  if gsuBackup > 0 or gsuBackup == -1:
    requiredParametersTemp.discard("GlobalSplitU")
```

Also wrap the temporary state mutations in `try/finally` so `GlobalSplitU` and `ProblemType["GroupedGemm"]` are restored even if naming raises for some other reason.

## Recommended Test

Replace the current pinned `pytest.raises(TypeError)` characterization with assertions that `getKernelNameMin(make_state(GlobalSplitU=4), splitGSU=True)` and `getKernelNameMin(make_state(GlobalSplitU=-1), splitGSU=True)` return stable names without raising. Add an explicit restoration assertion for the exception-safe path, or extend `test_name_does_not_mutate_state` to cover `getKernelNameMin(..., splitGSU=True)` with `GlobalSplitU` values `4` and `-1`.
