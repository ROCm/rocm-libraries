# Changelog for the rocRAND Fortran binding

## Unreleased

### Added

* Generated Fortran bindings, as a single `rocrand` module: `use rocrand`, link
  `roc::rocrand_fortran`. Controlled by `BUILD_FORTRAN_BINDINGS` (`ON` on Linux, `OFF` on Windows,
  where ROCm ships no Fortran compiler), `BUILD_FORTRAN_CLIENTS` and `FORTRAN_ARRAY_INTERFACES`
  (`assumed-shape` by default; `none` or `assumed-rank`). Found with
  `find_package(rocrand-fortran)`; the archive and the `.mod` files install per compiler, under
  `<libdir>/fortran/<compiler>` and `<includedir>/fortran/<compiler>`.
  The archive has one section per procedure: link with `-Wl,--gc-sections` to keep only the wrappers
  you call.
