# Changelog for the hipSPARSE Fortran binding

## Unreleased

### Added

* Generated Fortran bindings, as a single `hipsparse` module: `use hipsparse`, link
  `roc::hipsparse_fortran`. Controlled by `BUILD_FORTRAN_BINDINGS` (ON on Linux, OFF on Windows),
  `BUILD_FORTRAN_CLIENTS` (the binding tests in `test/`) and `FORTRAN_ARRAY_INTERFACES`
  (`assumed-shape` by default; `none` or `assumed-rank`). Found with `find_package(hipsparse-fortran)`; the archive and `.mod` files
  install per compiler under `<libdir>/fortran/<compiler>` and `<includedir>/fortran/<compiler>`.
  Pointer-mode-dependent scalars (`alpha`, `beta`, `nnzTotalDevHostPtr`, dot results) are
  `type(c_ptr), value`, so pass `c_loc(x)`; one interface covers both pointer modes.
  The archive has one section per procedure: link with `-Wl,--gc-sections` to keep only the wrappers
  you call.
