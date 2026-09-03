# rocKE MMA operand metadata cleanup

## Scope

- Keep kernel and operator terminology unchanged. GEMM/CK names such as `C`,
  `D`, `c_frag`, and bias-loop variables remain local to those abstractions.
- Model instruction metadata in machine order: three matrix data sources and
  one destination.
- Give every matrix source an optional associated scale operand. Ordinary
  MFMA/WMMA sources are unscaled; scaled MX may attach scales to source 0 and
  source 1 without pretending the scale values are additional homogeneous
  matrix fragments.
- Preserve the legacy three-dtype catalog query as a compatibility shorthand
  where destination dtype defaults to source 2 dtype. Add a separately named
  indexed query for callers that distinguish source 2 from destination. Do not
  make source 2 and destination equality a catalog invariant.
- Do not extend unpack signatures and do not rename kernel-level variables.

## Implementation order

1. Change the Python and C platform metadata structures and architecture SSOT.
2. Keep compatibility accessors for existing A/B/C-oriented consumers.
3. Update only platform consumers proven by compilation to require direct
   source/destination access.
4. Add narrow Python and C tests for indexed catalog round trips,
   source/destination distinction, legacy query behavior, and optional source
   scale parsing. Do not add broad recurrence guards over the real catalogs.
5. Confirm no `library/` or Python instance changes remain.

## Validation

- Load `ubuntu-24` and `rocm/10`, recording the resolved ROCm module.
- Configure/build the platform and run CTest in a fresh `mktemp -d` directory.
- Run focused and full platform Python core tests.
- Exercise CPU-only catalog construction, query, and lowering for both gfx950
  and gfx1250 so MFMA/MX and WMMA metadata paths are covered without a GPU.
- Check Python/C emitted-byte identity for LLVM 20, 22, and 23.
- Run repository pre-commit hooks and `git diff --check`.
- Review the final changed-file inventory before committing.
