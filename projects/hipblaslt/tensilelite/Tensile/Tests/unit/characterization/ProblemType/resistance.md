# Resisting lines — `Tensile/SolutionStructs/Problem.py` (ProblemType slice)

The suite reaches **~97% line** standalone (601 stmts, ~18 miss) and the slice
clears the ≥95% line bar. Unlike the prior pure modules, this slice has a
genuine residue: a set of lines that are **dead or unreachable** for any
GEMM `ProblemType` constructed through the public path. They are catalogued
below so the residual is auditable. New file in the per-target dir per the
add-only rule.

## Unreachable / dead LINES (counted as Miss, cannot be hit here)

| Line(s) | Code | Why unreachable |
|---|---|---|
| 859 | `raise Exception("NO dest data type or data type specified")` | In the `DestDataType` resolution `else`. Reached only if neither `DestDataType` **nor** `DataType` is in `config` — but a missing `DataType` already raised at L826. Dead. |
| 870, 875 | `self["DataType"] = DataType(0)` after `raise` in `ComputeDataType` resolution | Same shape: the lines sit **after** an unconditional `raise` (L869/L874) and on a `DataType`-absent path that L826 already rejects. Dead. |
| 257 | `raise RuntimeError("...does not have enough indices...")` (GEMM arm of `ExactDict`) | For a GEMM problemType, `convertLeadingDims` always returns exactly `NumIndicesC+1+4` = `TotalIndices+NumIndicesLD` entries, so the length check never fails. (The **non-GEMM** sibling raise at L259-260 IS covered, via a synthetic non-GEMM problemType dict.) |
| 997-998 | gradient `elif ActivationType=='none' and UseE==True: UseE=False` | Dead by ordering: when activation is `none`, the earlier `UseE` block (L980-982) has already forced `UseE=False`, so this `elif` can never see `UseE==True`. |
| 1081 | `def isGEMM(self): return self.operationType == 0` | References `self.operationType`, which is never set (the value lives in `self["OperationType"]`). Calling it would `AttributeError`; nothing in the slice calls it. Dead/buggy. |
| 1109 | `raise Exception("invalid index ... (inC but not (inA or inB))")` | `initGEMM` always assigns every C index to A and/or B, so no C index is orphaned. Unreachable for GEMM. |
| 1118 | `raise Exception("... expected summation but not (inA and inB)")` | `initGEMM` puts the summation index in both A and B. Unreachable for GEMM. |
| 1130 | `raise Exception("duplicate index in ...")` | `initGEMM` never produces duplicate indices in A/B. Unreachable for GEMM. |
| 1136 | `raise Exception("Tensile requires >= 2 free indices ...")` | A 2D GEMM always yields exactly 2 free indices (A owns one, B the other), and `AllowNoFreeDims` only relaxes this further. Unreachable for GEMM. |
| 1188-1193 | the `Index0/Index1/Tensor*/Tile*` **else** branch (`Index01A >= Index01B`) | For GEMM, A always owns free index 0 and B owns free index 1 (`initGEMM`), so `Index01A (0) < Index01B (1)` always — the `else` is unreachable. |
| 1359 | `__ne__`: `if result is NotImplemented: return result` | `__eq__` here always returns a concrete `bool` (its `isinstance` guard returns `False`, never `NotImplemented`), so this arm is dead. |

That is the entire residue: every other statement in the slice is covered.

## Reachable branches that DID resist a naive first attempt (now covered)

| Item | How it was finally reached |
|---|---|
| dtype changes (HHS/double/complex/F8/...) | The config builder had to mirror **real YAML**: a *minimal* dict (only the keys set), not a copy of `_defaultProblemType`. Starting from the full default pinned `MacDataTypeA`/`DataTypeA/B` to `0`, silently overriding any `DataType` change and tripping the GEMM-type check. |
| activation reg-size guard (L972-976) | Needs `ActivationComputeDataType == DestDataType` with `Dest.numRegisters != Compute.numRegisters` **and** `DataType.numRegisters < Dest.numRegisters`: config `F8/H/S` with `ActivationComputeDataType=h`. |
| `AllowNoFreeDims` dimList (L1167) | A dedicated `AllowNoFreeDims=True` config. |
| `_populateLookupTable`-style guard raises | The reachable ones (`ExactList` length / `-1`, `ProblemSizeRange` >4-descriptor, `ExactDict` bad-field & non-GEMM count, all the `ProblemSizes` `printExit` paths) are driven with crafted inputs; `printExit` is `sys.exit(-1)` so they are pinned via `pytest.raises(SystemExit)`. |

## Determinism technique (not a gap)

- `ProblemType` holds live `DataType` / `ActivationType` objects in `state`; the
  `conftest.norm` helper renders these to `"<DataType Float>"` / `"<ActivationType
  None>"` strings and sorts keys, so snapshots are object-free and stable.
- `validateProblemTypeParameterTypes` mutates a **module-global collector in
  `Solution`**; the test clears it, runs, captures the delta, and restores it in
  a `finally`, so the snapshot is just the delta and the shared session is
  unaffected (verified: full `-m unit` stays green at 1670 passed).
- `ProblemType` configs are **minimal** (mirroring YAML), so dtype-derivation
  guards behave as in production rather than being short-circuited by defaults.
