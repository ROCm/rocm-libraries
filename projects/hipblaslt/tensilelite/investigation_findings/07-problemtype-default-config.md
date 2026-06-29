# ProblemType.FromDefaultConfig latent signature quirk

## Verdict

Open.

## Current source references

- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/SolutionStructs/Problem.py:821-823`: `FromDefaultConfig` is decorated with `@classmethod`, but declares only `printIndexAssignmentInfo`.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/SolutionStructs/Solution.py:500-508`: `Solution.__init__` calls `ProblemType.FromDefaultConfig(printIndexAssignmentInfo)` when the solution config omits `ProblemType`.
- `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/ProblemType/test_problemtype_char.py:185-190`: the current target tree has the same characterization note and calls `ProblemType.FromDefaultConfig()` with no explicit args.
- `/home/alvasile/repos/rocm-libraries-investigation/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/ProblemType/test_problemtype_char.py:185-190`: the referenced external characterization is available and matches the same no-argument characterization.

## Static evidence

Current implementation:

```python
@classmethod
def FromDefaultConfig(printIndexAssignmentInfo: bool):
  return ProblemType(_defaultProblemType, printIndexAssignmentInfo)
```

Because this is a `classmethod`, Python binds the owning class as the first positional argument. That means `ProblemType.FromDefaultConfig()` succeeds only because `printIndexAssignmentInfo` receives the `ProblemType` class object, not a boolean. The constructor then receives that class object as the print flag, which is truthy.

The actual target call site in `Solution.__init__` passes an explicit boolean:

```python
self["ProblemType"] = ProblemType.FromDefaultConfig(printIndexAssignmentInfo)
```

With the current one-parameter classmethod, that call would bind two positional arguments (`ProblemType` and the explicit boolean) to a function that accepts one, producing a `TypeError` before the default problem type can be constructed.

## Impact

The no-argument characterization documents the quirk rather than the intended API. It can mask the broken explicit-argument path and also changes behavior by passing a truthy class object into `ProblemType.__init__` as `printIndexAssignmentInfo`.

The higher-risk path is solution construction without an explicit `ProblemType`: `Solution.__init__` currently attempts the explicit-argument call and should fail with `TypeError`. If this fallback is reachable from YAML or generated solution configs, default problem type construction is broken.

## Recommended fix and test

Fix the method signature to accept the implicit class argument and give the print flag its intended default:

```python
@classmethod
def FromDefaultConfig(cls, printIndexAssignmentInfo: bool = False):
  return cls(_defaultProblemType, printIndexAssignmentInfo)
```

Add or update tests so both supported call forms are covered:

- `ProblemType.FromDefaultConfig()` returns a `ProblemType` and does not enable index-assignment printing.
- `ProblemType.FromDefaultConfig(False)` returns a `ProblemType` without raising.
- A minimal `Solution` construction path with no `ProblemType` in the config reaches the fallback without raising, if such a minimal fixture is practical in the existing unit test setup.
