# Changelog for the rocSOLVER Fortran binding

## Unreleased

### Added

* Generated Fortran bindings, as a single `rocsolver` module: `use rocsolver`, link `roc::rocsolver_fortran`.
  Controlled by `BUILD_FORTRAN_BINDINGS` (defaults to `${UNIX}`: ON on Linux, OFF on Windows) and
  `BUILD_FORTRAN_CLIENTS` and `FORTRAN_ARRAY_INTERFACES` (`assumed-shape` by default; `none` or
  `assumed-rank`). Found with
  `find_package(rocsolver-fortran)`; the archive and the `.mod` files install per compiler, under
  `<libdir>/fortran/<compiler>` and `<includedir>/fortran/<compiler>`.
  The archive has one section per procedure: link with `-Wl,--gc-sections` to keep only the wrappers
  you call.
