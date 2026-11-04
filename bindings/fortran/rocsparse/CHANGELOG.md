# Changelog for the rocSPARSE Fortran binding

## Unreleased

### Added

* Generated Fortran bindings, as a single `rocsparse` module: `use rocsparse`, link
  `roc::rocsparse_fortran`. Controlled by `BUILD_FORTRAN_BINDINGS` (ON on Linux, OFF on Windows),
  `BUILD_FORTRAN_CLIENTS` and `FORTRAN_ARRAY_INTERFACES` (`assumed-shape` by default; `none` or
  `assumed-rank`). Found with `find_package(rocsparse-fortran)`; the archive and `.mod` files
  install per compiler under `<libdir>/fortran/<compiler>` and `<includedir>/fortran/<compiler>`.
  The module binds `rocsparse_int` as `integer(c_int)`, so it matches a rocSPARSE built without
  `BUILD_ROCSPARSE_ILP64`. An installed rocSPARSE does not record that option: configure the
  bindings with `-DROCSPARSE_ILP64=ON` against an ILP64 build, and nothing Fortran is built.
  The archive has one section per procedure: link with `-Wl,--gc-sections` to keep only the wrappers
  you call.
