# hipBLASLt Integration Status — PartialRMS Fused GEMM

Audit of what is complete and what remains before the PartialRMS fused GEMM
epilogue variants can be consumed through the hipBLASLt public API.

## TensileLite side — complete

| Component | File | Detail |
|---|---|---|
| Kernel variants | `epilogues/yaml/tune_gemm_variants.yaml` | Three variants tuned for gfx950: plain GEMM, PartialRMS, PartialRMS+Residual |
| Problem setters / getters | `tensilelite/include/Tensile/ContractionProblem.hpp:765` | `setUsePartialRMS`, `setPartialRMSResidualAdd` and matching getters |
| Predicates | `tensilelite/include/Tensile/ContractionProblemPredicates.hpp:2089` | `UsePartialRMSEqual`, `UsePartialRMSResidualAddEqual` |
| Predicate serialisation | `tensilelite/include/Tensile/Serialization/ContractionPredicates.hpp:112` | Both predicates registered in the subclass map and MappingTraits |
| Predicate emission | `Tensile/Contractions.py:424` | Python emits `UsePartialRMS` and `UsePartialRMSResidualAdd` predicates into library logic YAML |
| Kernel argument slots | `tensilelite/src/ContractionSolution.cpp:1115` | `RMSNormGamma`, `PartialBuf`, `ResidualBuf` (conditional on `partialRMSResidualAdd`) appended when `sizeMapping.partialRMS` |
| Input struct fields | `tensilelite/include/Tensile/ContractionProblem.hpp:1624` | `void* partialBuf`, `void const* rmsGamma`, `void const* residual` in `ContractionInputs` |
| Client config gen | `Tensile/ClientWriter.py:625` | `partial-rms-mt0/mt1` emitted from `partialRMSMT0/1` args; tile sizes derived from `originalSolution["MacroTile0/1"]` at line 810 |

## hipBLASLt host side — missing

### 1. Public API epilogue enum — `library/include/hipblaslt/hipblaslt.h:85`

`hipblasLtEpilogue_t` has no PartialRMS values. Required additions (names
indicative — final names are a design decision):

```c
HIPBLASLT_EPILOGUE_PARTIAL_RMS          = ...,
HIPBLASLT_EPILOGUE_PARTIAL_RMS_RESIDUAL = ...,
```

### 2. C++ ext API — `library/include/hipblaslt/hipblaslt-ext.hpp`

`GemmEpilogue` and `GemmInputs` classes expose no PartialRMS settings or
auxiliary buffer pointers. Required additions:

- `GemmEpilogue`: flag or mode field selecting the PartialRMS variant.
- `GemmInputs`: pointers `void* partialBuf`, `const void* rmsGamma`, `const void* residual`.

### 3. rocblaslt matmul descriptor — `library/src/amd_detail/rocblaslt/`

Two structs need new fields:

| Struct | File | Missing |
|---|---|---|
| `_rocblaslt_matmul_desc` | `src/include/handle.h:166` | PartialRMS mode flag; `partialBuf`, `rmsGamma`, `residual` pointers |
| `RocblasltContractionProblem` | `include/rocblaslt-types.h:485` | Same fields; used to carry inputs into `tensile_host.cpp` |

### 4. Problem construction — `library/src/amd_detail/rocblaslt/src/tensile_host.cpp:1951`

`ConstructTensileProblem` reads the epilogue descriptor and calls the
corresponding `tensileProblem.set*` methods. It currently has no handling
for PartialRMS. Required additions:

```cpp
// analogous to is_bias_enabled(), is_act_enabled(), etc.
if (is_partial_rms_enabled(prob)) {
    tensileProblem.setUsePartialRMS(true);
    tensileProblem.setPartialRMSResidualAdd(prob.partialRMSResidualAdd);
}
```

### 5. Input pointer wiring — `tensile_host.cpp:2408`

`GetTensileInputs` fills the `TensileInputs` struct from the descriptor.
`inputs.partialBuf`, `inputs.rmsGamma`, and `inputs.residual` are never set;
they must be wired from the descriptor fields added in item 3.

## Summary

```
User call
  └─ hipblasLtMatmul (epilogue enum) ──── MISSING: no PartialRMS enum value
       └─ _rocblaslt_matmul_desc      ──── MISSING: no PartialRMS fields
            └─ ConstructTensileProblem ─── MISSING: setUsePartialRMS not called
            └─ GetTensileInputs        ─── MISSING: partialBuf/rmsGamma/residual not set
                 └─ ContractionSolution.cpp:1115  ← READY: packs args if inputs are set
                 └─ ContractionProblemPredicates  ← READY: predicates select correct kernel
```

Everything below `GetTensileInputs` is in place. The work remaining is entirely
in the hipBLASLt public/host layer: exposing the API surface, plumbing the
descriptor fields, and wiring the five call sites listed above.
