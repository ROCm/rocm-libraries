# Investigation: TensileLibLogicToYaml.formGroups("None") on skip-MI path

## Verdict

Open.

The current target source still assigns the string `"None"` when MI handling is skipped or disabled, then unconditionally passes that string into `formGroups()`, whose implementation calls `.items()`.

## Current Source References

- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/TensileLibLogicToYaml.py:138` defines `formGroups(MIInstruction9Bits: dict)`.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/TensileLibLogicToYaml.py:142` iterates `MIInstruction9Bits.items()`, so a string input raises `AttributeError`.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/TensileLibLogicToYaml.py:197` starts the skip-MI / MI-enabled decision block.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/TensileLibLogicToYaml.py:214` only builds MI groups when `skipMI != True and isMatrixInsEnabled`.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/TensileLibLogicToYaml.py:217` sets `temp = "None"` otherwise.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/TensileLibLogicToYaml.py:219` unconditionally calls `forkData.append(formGroups(temp))`.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/TensileLibLogicToYaml.py:378` routes `TensileLibLogicToYaml(..., skipMI)` through `formForkParams(currentIndexSolution, skipMI)`.

## Characterization Comparison

The referenced investigation files are present and match the report:

- `/home/alvasile/repos/rocm-libraries-investigation/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/DECISIONS.md:239` records D14 as the `formGroups("None")` crash.
- `/home/alvasile/repos/rocm-libraries-investigation/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/DECISIONS.md:241` to `:245` describes `formForkParams(sol, skipMI=True)` setting `temp = "None"` and crashing at `.items()`.
- `/home/alvasile/repos/rocm-libraries-investigation/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/TensileLibLogicToYaml/test_tensile_lib_logic_to_yaml_char.py:109` to `:115` pins `formForkParams({"EnableMatrixInstruction": False}, skipMI=True)` as raising `AttributeError`.

## Evidence

Static evidence is sufficient: the target source still has the exact crash path described above.

A direct normal import of `Tensile.TensileLibLogicToYaml` in this checkout currently fails earlier because the local Python environment cannot import `rocisa.rocIsa`. To exercise only the target functions, I stubbed the unrelated heavy imports and called the current module functions. The result was:

```text
formGroups: AttributeError: 'str' object has no attribute 'items'
formForkParams: AttributeError: 'str' object has no attribute 'items'
```

This confirms the target function behavior once import-only dependencies are bypassed.

## Impact

The `--skipMI` path is still broken: `skipMI=True` makes `formForkParams()` take the `temp = "None"` branch and crash before YAML generation can complete. The same failure also affects MI-disabled solutions, including `EnableMatrixInstruction` missing, false, or paired with a falsey `MatrixInstruction`, because those states leave `isMatrixInsEnabled` false and hit the same unconditional `formGroups(temp)` call.

## Recommended Fix

Do not call `formGroups()` when MI parameters are intentionally skipped or disabled. The likely fix is to append the MI `Groups` entry only inside the enabled path:

```python
if not skipMI and isMatrixInsEnabled:
    forkData.append(formGroups(form9BitMIInst(currentIndexSolution)))
```

If the generated YAML schema requires an explicit empty group, use a dictionary sentinel such as `{}` instead of the string `"None"` and document the intended output shape. Given the CLI help says `--skipMI` skips the MatrixInstruction field, omitting the MI group is the cleaner behavior.

## Recommended Test

Add unit coverage that asserts `formForkParams({"EnableMatrixInstruction": False}, skipMI=True)` does not raise and does not emit a `MatrixInstruction` group. Also cover `formForkParams({"EnableMatrixInstruction": False}, skipMI=False)` and a top-level `TensileLibLogicToYaml(..., skipMI=True)` fixture so the CLI/orchestrator path cannot regress.
