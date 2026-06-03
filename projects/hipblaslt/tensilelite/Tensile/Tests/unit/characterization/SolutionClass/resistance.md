# Resisting items — `Solution` class slice 2

The targeted slice-2 **surface** symbols (construction, Mapping interface,
identity/hash/equality, the simple statics, `getKernels`) are fully covered.
This file records the one dead line, the two pipeline-dependent accessors, and
the scoping boundary to slice 3. New file in the per-target dir per the add-only
rule.

## Dead LINE (1) — counted as Miss, unreachable

| Line | Code | Why unreachable |
|---|---|---|
| 5229 | `__ne__`: `if result is NotImplemented: return result` | `__eq__` here always returns a concrete `bool` (its `isinstance` guard returns `False`, never `NotImplemented`), so `result is NotImplemented` is never true. Dead defensive arm (same shape as `ProblemType.__ne__`). |

## Pipeline-dependent accessors (characterized as AttributeError)

| Symbol | Behaviour | Test |
|---|---|---|
| `getKernelBetaOlnyObjects` (L544-545) | returns `self.betaOnlyKernelObjects` | `test_kernel_betaonly_conversion_accessors_need_pipeline_state` asserts `AttributeError` |
| `getKernelConversionObjects` (L550-553) | returns `self.conversionKernelObjects` | same |

These attributes are populated by a **later pipeline stage** (kernel
beta-only / conversion generation), not by parse/construction, so on a freshly
built `Solution` they raise `AttributeError`. Pinned as current behaviour; the
populated paths belong to a higher-level (TensileCreateLibrary) flow outside the
`Solution` unit slice. `getKernels` (L534-543), by contrast, returns `[self]`
and is fully exercised.

## Out of slice — deferred to slice 3 (not resistance)

`Solution.__init__` is exercised on the **happy path** (the gfx942 HSS_BH
fixture config), which incidentally covers a swath of the derivation. But
comprehensive coverage of the construction/derivation surface —
`assignProblemIndependentDerivedParameters`, `assignDerivedParameters`,
`setGlobalReadVectorWidth`, `setGlobalLoadTileDimClassic`,
`checkAndAssignWaveSeparateGlobalRead`, `isDirectToVgpr/LdsDoable`,
`depthUIteration`, and `_deriveAndValidateMXScaleLayoutAndTransport` — needs a
matrix of valid+invalid configs across ISAs and is the **slice 3** target. The
bulk of the whole-file Missing lines are those reject-heavy, cap-gated paths.

## Determinism technique (not a gap)

- A real `Solution` is parsed from the committed vendored logic fixture via the
  reused cap fixtures; the construction snapshot pins the **schema** (sorted key
  set) + curated **stable** fields, not the toolchain-derived values, so it is
  reproducible in the dev container (same rationale as the LibraryIO suite).
- The different-name `__eq__` branch (L5219-5220) is reached with a shallow copy
  of the solution whose cached `_name` is overridden (the fixture yields a
  single solution); `a == a` covers the DeviceNames branch (L5222-5224).
- `__setitem__` mutates the session-shared solution; the test saves/restores.
