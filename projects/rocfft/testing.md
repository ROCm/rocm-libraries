# rocFFT testing strategy

This document describes how you test rocFFT, including unit tests, integration tests, accuracy
tests, and performance tests.

## Component overview

rocFFT is a FFT GPU library.

The HIP kernels are produced via a code generator embedded in the library, which are compiled using
hipRTC.  Transforms are available for single-process single-device, single-process multi-device, and
multi-process single/multi-device.

rocFFT tests use googletest and some python scripts included in the rocFFT git repository.

## Development workflow

Build tests with the CMake options `BUILD_CLIENTS_TESTS` and `BUILD_CLIENTS_BENCH`, or with
`BUILD_CLIENTS` to enable both.

# Testing strategy and layers

## Test naming

rocFFT transforms can take a variety of parameters, such as length, precision, various data layout
options, etc.  These are encoded into a human-readable string referred to as a FFT token.  For
example,

```
complex_forward_len_8_single_ip_batch_1_istride_1_CI_ostride_1_CI_idist_8_odist_8_ioffset_0_0_ooffset_0_0
```

describes a complex forward FFT of length 8, in-place, using single-precision, and input and output
strides of 1 (with other information about batch distance and offset included as well).  This maps
1:1 with information provided to the library via the public API.  Multi-device transforms are also
covered by this tokenization format (though the format becomes increasingly verbose as more
information is added).  The token format is not part of the public API of rocFFT, but can be useful
for users when providing bug reports or otherwise inspecting logs.  The tokens are used for both
accuracy and performance tests.

## Resources and runtime robustness

rocFFT tests try to use all GPU memory and all host memory so they can cover as many transform sizes
as possible. `rocfft-test` queries the device and the host to see whether a problem size fits. It
either does not generate the test if the memory footprint is too large, or it skips the test if it
detects at runtime that not enough memory is available. Available-memory queries and the actual
allocations happen at different times, and there is no mutex on host or device allocation, so memory
that a query reports as free might be gone at allocation time. If a host or device allocator fails,
`rocfft-test` skips that test. The host reference FFT library also allocates memory
internally. `rocfft-test` cannot recover if that library hits an out-of-memory (OOM) error. The
tests keep a safety margin on allocated memory to improve robustness.

The number of failed allocations is tracked and reported at the end of `rocfft-test`'s execution.
There is a command-line option to report hip runtime errors as failures instead of just skipped
tests.

On an accelerated processing unit (APU), host memory and device memory share one pool, which
complicates accounting. Host and device allocations go through structs that track that shared
pool. Extra care is required because the HIP runtime can be optimistic and might not track host
allocations.


## Testing strategy

### Unit test strategy

#### Environment support

ocFFT supports a variety of OSes, as defined in the [ROCm compatibility
documentation](https://rocm.docs.amd.com/en/latest/compatibility/compatibility.html). Continuous
integration (CI) in TheRock currently covers four device architectures and two operating
systems. Support for later versions of C++ is often unavailable in older Linux OSes; not testing
compilation and smoke tests in these environments may break compilation or execution in these cases.

#### Unit tests

it tests are correctness tests that verify library infrastructure. For example, they cover API
behavior for invalid parameters and whether internal infrastructure behaves as expected.

API correctness tests live in `rocfft-test`. GoogleTest names use the `rocfft_UnitTest.*` filter.

Internal library correctness tests use an internal CTest framework. Enable it with the CMake option
`ROCFFT_BUILD_INTERNAL_TESTS`. The executable is `library/src/tests/rocfft-internal-test`. More
expensive array format validation tests run in `rocfft-test` under the GoogleTest filter
`reference_test/valid_length_stride.*`.

### Integration test strategy

#### Bitwise reproducibility tests

rocFFT tests bitwise reproducibility by hashing each output and re-running the suite to confirm that
the hashes match. The hash is SHA-256. The implementation is in the rocFFT repository; tests do not
use `std::hash`, which is not stable across runs.  Bitwise reproducibility requires the same rocFFT
version, the same ROCm compiler, runtime, and driver, and the same GPU model.


#### Memory canary tests

rocFFT transforms do not modify data which is not part of the transform - this includes memory which
is skipped over when the length+batch/stride+dist combination implies that the data layout is
discontiguous.  We have a number of ad-hoc lengths which verify this behaviour in the checkstride
accuracy test suite, and manually-specified test cases can verify this with the --checkstride
command-line option.

In addition, out-of-place complex-to-complex transforms preserve the input data.  There is a testing
gap for this in rocFFT.

#### Samples

Samples are currently located in a separate repository. The purpose of the samples repository is
pedagogical; as such, it should work with the latest release of rocFFT. Latest API features and
behaviour in the rocFFT library will therefore reside in the rocFFT repository, and will be moved to
the samples repository after the library changes are made available in a public release. In order to
ensure that these samples behave correctly, these should be tested as part of the pre-commit and
nightly testing.

Combinations of flags are not currently run in TheRock CI, which is a testing gap.

#### Feature flags

Features under development, or for specific use cases may be controlled by feature flags at either
the cmake or execution stage. These options need to be tested, and, in some cases, combinations of
these flags need to be tested.

There is a gap in that combinations of flags are not currently run in the TheRock's CI.

#### Accuracy tests

Accuracy tests use the `accuracy_test.vs_fftw` filter and Fastest Fourier Transform in the West
(FFTW) for reference. L2 and L-infinity error-bound scaling for FFT is known analytically and sets
the allowable numerical error. In `rocfft-test` you can set the precision-based constant multiplier
(machine epsilon) for half, single, and double precision. If you set that value, numerical error is
less than or equal to it. The precision-based epsilon values reported at the end of a run can bound
a later run.

`rocfft-test` uses randomization. You can pass a static random seed on the command line, but do not
use a fixed seed in general testing. The seed is for reproducing failures. Randomized test selection
is stable when tests are added or removed, but not when test names change. GoogleTest uses the test
name and the random seed to decide whether each test runs.

Tests are selected at random from a predefined parameter set. `rocfft-test` options control that
selection. The suite also generates random tests. A command-line option controls how many. Because
you can fully specify data decomposition (lengths and strides) in rocFFT, the suite must reject
invalid configurations, mainly self-aliasing arrays. The array format validator in rocFFT guarantees
that those configurations are not used.

Multi-process tests use a different execution path than single-process tests. Accuracy tests are
handled by `rocfft-mpi-worker` with the `--accuracy` option. Multi-process tests cover a large range
of hardware configurations, and it is wasteful to reserve on the order of 1000 GPUs for a two-GPU
test. The script `scripts/rocfftslurmtest.py` splits accuracy tests by hardware configuration and
submits them to Slurm. The script `scripts/rocfft_mpi_test.py` then launches
`rocfft-mpi-worker`. The Python wrapper can recover from hangs and crashes, which are common in
distributed computing. Multi-process accuracy testing still uses reference computation on a single
host node, which limits problem size.

TheRock CI does not cover multi-process tests because the infrastructure is not available.

Unless a developer specifies otherwise, run accuracy tests on every architecture that rocFFT
supports. TheRock currently tests only a subset of those architectures.

Do not re-run accuracy tests until you get the result you want. Treat an error as an error so
that intermittent failures still show up in CI.


#### Downstream dependency tests

rocFFT is used by several downstream projects, and you need to verify that infrastructure changes do
not break them. Skip these tests for most pre-submit checks. Run them in a targeted way on a slower
cadence, for example monthly.

### ASAN / TSAN / Sanitizer Coverage

Address sanitizer (ASAN) coverage is enabled in rocFFT with `BUILD_ADDRESS_SANITIZER`.

Thread sanitizer (TSAN) and other sanitizers are not enabled in rocFFT.

clang-format checks code format.

cppcheck runs static analysis.


### Code coverage

Since rocFFT uses a code generator and hipRTC to produce device kernels, device-side code coverage
is nonsensical.

Host-side code coverage is useful in showing testing gaps, but the effectiveness of a
percent-coverage target is controversial, particularly in the context of Goodhart's law.  For
example, we also need to be able to handle failures in the hip runtime.  If `hipMalloc` fails, then
this needs to be handled, but that code path is never tested; in order to satisfy code coverage
targets, developers are incentivized to reduce or eliminate code that handles `hipMalloc` errors.
This issue can be alleviated by mocking the hip runtime so that we can inject errors, but such
infrastructure does not exist, which constitutes a gap in testing.  Similar mocking is also
unavailable for the compiler and driver.

rocFFT tests use randomization by design, and the library covers single-GPU, single-process
multi-GPU, and multi-process multi-GPU transforms. Code coverage must combine results from those
cases. TheRock CI does not do that today because of infrastructure issues.

### Benchmarking and performance validation

The goal of pre-commit testing is to determine how a commit will impact performance in the target
branch after the commit is merged.  Therefore, pre-commit performance testing must test the
difference in performance between the target branch with and without the commit applied.

Performance tests focus on a different parameter space than the accuracy tests.  For example, while
it's important that the length-1 transforms perform correctly (accuracy), the transform is actually
the identity operation, and the highest-performance option for software that needs to perform a
length-1 transform is to not use rocFFT, but to perform no operation at all.

Performance tests must deal with jitter.  Jitter is noise in execution time, and the GPU is not
immune to this behaviour.  The results of naive performance regression tests often consist almost
entirely of false-positive results if jitter is not accounted for in the experimental design and
analysis of results.  rocFFT manages jitter via experimental design and a multi-hypothesis
statistical testing framework.

In order to eliminate the correlation between jitter and testing case, the performance-testing
experimental design is to load both the control and test versions of the rocFFT library into the
same executable and randomize the execution order for the two cases.  This is handled by
`dyna-rocfft-bench` (and `dyna_rocfft_mpi_worker` for the multi-process case) using `dlopen` for
Linux and `LoadLibraryA` for Windows.  For cases where this framework does not apply (eg comparing
performance between two different devices) `rocfft-bench` is usable as a single-library client,
though the risk of false-positives is naturally higher.  On the other hand, one is unlikely to be
testing the performance impact of a software change between two different devices, so the importance
of false-positives is fairly small in this case.

Post-processing of the data uses statistical tests, and we have implemented the T-test, Mood's
median test, and the Mann-Whitney U test (also known as the Wilcoxon rank-sum test).  While the data
distribution of execution times tends to not follow a normal distribution, the T-test only requires
that the sampling distribution of the difference in means be approximately normal, which the central
limit theorem justifies for sufficiently large sample sizes (which we take to be at least 20).  The
three tests answer subtly different questions, ie the differences of the mean, the median, or the
rank, though, for realistic data, these tend to all agree.  Since we also test multiple points in
parameter space together, it's also important to use a multi-hypothesis testing framework in order
to avoid p-hacking oneself.  rocFFT implements the Bonferroni correction and the Benjamini–Hochberg
procedure in order to reduce the false-positive rate.

In addition to statistical testing, post-processing provides a measure of central tendency and
confidence intervals for transform execution time.  Since the data tends to not be normally
distributed and have long tails, the central tendency of the execution time is better represented by
the median than by the mean, which has the added benefit of being invariant under monotonic
transformation (eg from time to gflops or bandwidth).  We also do not use the standard deviation to
express confidence intervals; not only is the data not normal, but the mean minus the standard
deviation is often negative, which is not only unphysical, but quite difficult to plot when using a
logarithmic scale.  Instead, we use bootstrap resampling to compute the confidence intervals on the
median execution time.

These features are implemented in the python script `scripts/perf/rocfft-perf`, which also includes
`scripts/perf/suites.py` to define the rocFFT performance suites.  Multi-process performance testing
is still in development.

Unless a developer specifies otherwise, run performance tests on every architecture that rocFFT
supports.

Since we perform null-hypothesis testing, we do not use a percentage cutoff.

Do not re-run performance tests until you get the result you want.

While every effort has been made to reduce false positives, these will still inevitably occur; the
performance tests therefore cannot be gating.  We will trust developers to use their judgement to
identify false positives.

Performance testing is not implemented in TheRock because of infrastructure issues.


## Other testing infrastructure

### Kernel test harness

rocFFT can generate a stand-alone kernel harness for debugging. Enable it by setting the environment
variable `ROCFFT_DEBUG_GENERATE_KERNEL_HARNESS=1`, where `1` turns generation on. You still need to
set kernel launch parameters. Those parameters appear in plan debugging when `ROCFFT_LAYER=8`, where
`8` selects the plan-debug layer. This is mainly a developer tool for isolating which layer of the
stack is involved in a bug.

## Pre-submit and CI Gates

Pre-submit tests currently cover unit tests and accuracy tests for:

- gfx94X, gfx950, gfx125X on Linux in a docker image
- gfx1151 on Windows

That is a small subset of the architectures rocFFT supports, which is a testing gap caused by
TheRock CI infrastructure. Run at least smoke tests on all architectures, and run performance tests
when changes might affect performance.

TheRock CI reduces test probability to a few percent because of performance issues in the CI
infrastructure, so too few tests run. The probability used to be 100%. Increase it to at least 50%
by raising test timeouts and using faster host hardware.

Static analysis (formatting) is gating for all pull requests.

Multi-GPU tests are run when a pull request is manually tagged with `ci:multi-gpu`; at least a
smoketest-level sample should be run on any relevant pull request, automatically.  This is a gap in
TheRock CI infrastructure.

In general, performance tests and multi-process tests are not run. Those gaps are due to
infrastructure availability.

### Desired testing standard 

The goal is targeted static analysis, unit tests, integration tests, and performance tests on every
architecture combination that rocFFT supports. Run a targeted set of tests before submit (for
example, documentation-only changes do not need performance tests) and confirm with a weekly build.
