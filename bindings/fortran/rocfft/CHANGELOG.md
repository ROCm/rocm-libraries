# Changelog for the rocFFT Fortran binding

## Unreleased

### Added

* Generated Fortran bindings, as a single `rocfft` module: `use rocfft`, link `roc::rocfft_fortran`.
  Controlled by `BUILD_FORTRAN_BINDINGS` (on by default on Linux, off on Windows),
  `BUILD_FORTRAN_CLIENTS` and `FORTRAN_ARRAY_INTERFACES` (`assumed-shape` by default; `none` or `assumed-rank`).
  Found with `find_package(rocfft-fortran)`; the archive and the `.mod` files install per compiler,
  under `<libdir>/fortran/<compiler>` and `<includedir>/fortran/<compiler>`.
  The archive has one section per procedure: link with `-Wl,--gc-sections` to keep only the wrappers
  you call.
