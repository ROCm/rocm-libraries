# Changelog for the hipFFT Fortran binding

## Unreleased

### Added

* Generated Fortran bindings, as two independent modules: `use hipfft` and link
  `hip::hipfft_fortran` for the hipFFT API, or `use hipfftw` and link `hip::hipfftw_fortran` for the
  FFTW compatibility API.
  Controlled by `BUILD_FORTRAN_BINDINGS` (on by default on Linux, off on Windows),
  `BUILD_FORTRAN_CLIENTS` and `FORTRAN_ARRAY_INTERFACES` (`assumed-shape` by default; `none` or
  `assumed-rank`). Found with `find_package(hipfft-fortran)` and
  `find_package(hipfftw-fortran)`; the archives and the `.mod` files install per compiler,
  under `<libdir>/fortran/<compiler>` and `<includedir>/fortran/<compiler>`.
  The archive has one section per procedure: link with `-Wl,--gc-sections` to keep only the wrappers
  you call.
