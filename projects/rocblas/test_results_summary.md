# rocBLAS Quick Test Partial Results Summary for MI430

## Environment

- **GPU**: AMD mi430 gfx1251
- **rocBLAS**: 5.3.0, built **without Tensile** and **without hipBLASLt**

---

## Fully Passed

All BLAS-1 scalar/vector ops passed cleanly.

| Test | Tests Passed |
|------|-------------|
| asum / _batched / _strided_batched | 212 / 276 / 324 |
| axpy / _batched / _strided_batched | 171 / 441 / 549 |
| copy / _batched / _strided_batched | 156 / 108 / 108 |
| iamax / _batched / _strided_batched | 212 / 276 / 324 |
| iamin / _batched / _strided_batched | 212 / 276 / 324 |
| nrm2 / _batched / _strided_batched | 212 / 276 / 324 |

Auxiliary tests also passed: `half_operators` (2), `helper_utilities` (7), `complex_operators` (2), `set_get_vector*` (159), `set_get_matrix*` (81), `set_get_pointer_mode` (1), `set_get_atomics_mode` (1), `check_numerics_vector` (21).

---

## Failures (numerical correctness issues)

| Test | Failed / Total | Root cause |
|------|---------------|------------|
| `dot` / `_batched` / `_strided_batched` | 113 / 415, 37 / 699, 41 / 759 | Imaginary part mismatch (`std::imag`) |
| `dotc` / `_batched` / `_strided_batched` | 57 / 126, 37 / 282, 41 / 306 | Same imaginary mismatch |
| `axpy_ex` / `_batched_ex` / `_strided_batched_ex` | 57 / 502, 49 / 1311, 28 / 1635 | Scalar rounding (e.g. 10.5 vs 9) |
| `rot` | 16 / 582 | Imaginary part mismatch |
| `trsv` / `_batched` | 48 / 1474, 36 / 4354 | Forward error exceeds tolerance (complex f64, UCU triangular solve) |

The dot/dotc/axpy_ex failures appear to be complex arithmetic precision issues on gfx1251. The trsv failures show numerics well outside tolerance (~1.0 vs ~3.5e-14) for complex double with upper triangular/unit diagonal — likely a missing or incorrect kernel for this architecture.

---

## Still Running / Incomplete / Not Run

These files had no final GTest summary line at the time of capture:

- **In progress**: `rot_batched`, `rot_strided_batched`, `rotg`, `rotg_batched`, `rotg_strided_batched`
- **Incomplete**: `logging`, `check_numerics_matrix`, `check_numerics_matrix_batched`
- **Skipped/Not run**: BLAS-2 matrix-vector operations:`gemv`, `gbmv`, `ger` / `gerc` / `geru`, `hemv`, `her` / `her2`, `hbmv`, `hpmv`, `hpr` / `hpr2`, `sbmv`, `spmv`, `spr` / `spr2`, `symv`, `syr` / `syr2`, `tbmv`, `tbsv`, `tpmv`, `trmv`, `trsv_strided_batched`
