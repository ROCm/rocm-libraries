# Changelog for the hipBLAS Fortran binding

## Unreleased

### Added

* Generated Fortran bindings, as a single `hipblas` module: `use hipblas`, link
  `roc::hipblas_fortran`. Controlled by `BUILD_FORTRAN_BINDINGS` (ON on Linux, OFF on Windows),
  `BUILD_FORTRAN_CLIENTS` and `FORTRAN_ARRAY_INTERFACES`
  (`assumed-shape` by default; `none` or `assumed-rank`). Found with
  `find_package(hipblas-fortran)`; the archive and `.mod` files install per compiler under
  `<libdir>/fortran/<compiler>` and `<includedir>/fortran/<compiler>`.
  The archive has one section per procedure: link with `-Wl,--gc-sections` to keep only the wrappers
  you call.
