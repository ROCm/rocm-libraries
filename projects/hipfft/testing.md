# hipFFT CPU reference backends

The client tests select their CPU numerical reference with
`HIPFFT_REFERENCE_BACKEND=FFTW|FFTZ`. The default remains `FFTW`.
`FFTZ` requires the native `aocl-fftz` CMake package. This selection applies
to the reference computation, including hipFFTW functional accuracy tests;
the GPU libraries under test are unchanged.

The reference interface lives in `shared/reference_fft.h`. Its adapters own
plans, translate dimensions and batch strides, and support execution on new
buffers. Both providers use single precision for half reference calculations.
The existing result cache, host callbacks, and numerical tolerances remain in
the test framework. `reference_backend.*` compares small transforms against
an independent direct DFT and checks real/complex transforms, dimensions,
strides, batches, half conversion, and repeated execution.

The historical GoogleTest names containing `vs_fftw` and the `--fftw_compare`
option remain valid. `--reference_compare` is an alias for that option.
The executable prints its selected backend at startup. FFTZ has no FFTW wisdom
support; `--wise` is rejected with that backend.

hipFFTW API validation remains enabled. Its helper dynamically loads the GPU
implementation. With FFTZ selected, declarations come from AMD's
`hipfft/hipfftw.h` and no linked FFTW function addresses are required.
The projects currently keep separate copies of their common test helpers,
including the reference adapters, to preserve standalone builds.

For a TheRock build, set `THEROCK_ENABLE_FFTZ=ON` and
`THEROCK_HIPFFT_REFERENCE_BACKEND=FFTZ`, then build the top-level `hipFFT+dist`
target. The native FFTZ runtime is installed beside the tests. TheRock's FFT
artifact topology still enables FFTW independently of this link selection;
the experimental option alone does not remove FFTW from the full build.

Run `hipfft-test --gtest_filter=reference_backend.* --test_prob=0 --unittest_prob=0` for adapter checks. Existing `manual/accuracy_test.vs_fftw/*` cases and
`hipfftw_functional_validation_*` tests exercise the chosen provider against
GPU execution. Use the project's `test_categories.yaml` for broader CI suites.

The reference memory estimate retains the previous conservative heuristic;
it does not measure FFTZ's peak allocation. Linux, threaded FFTZ, MPI, and
large-transform coverage require separate validation.

Generated with Codex.
