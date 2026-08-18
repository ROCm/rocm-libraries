# rocFFT Testing Strategy

This document describes the testing strategy for rocFFT.

## Component Overview

rocFFT is a FFT GPU library.

The HIP kernels are produced via a code generator embedded in the library, which are compiled using
hipRTC.  Transforms are available for single-process single-device, single-process multi-device, and
multi-process single/multi-device.

rocFFT tests rely on googletest and some python scripts included in the rocFFT git repository.

## Develoment Workflow

Tests are compiled with `BUILD_CLIENTS_TESTS` and `BUILD_CLIENTS_BENCH`, or just `BUILD_CLIENTS` to
cover both.

# Testing Strategy and Layers

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
information is added).  The token format is not part of the public API or rocFFT, but can be useful
for users when providing bug reports or otherwise inspecting logs.  The tokens are used for both
accuracy and performance tests.

## Resources and runtime robustness

rocFFT tests will attempt to use as all of GPU's memory, and all of the host's memory in order to
test as many transform sizes as possible. rocfft-test will query the device and the host in order to
determine if a given problem size will fit into the available hardware resources, and will either
not generate the test (if the memory footprint exceeds the hardware's availability) or skips the
tests (if rocfft-test detects at runtime that not enough memory is available).  Since the runtime
code to detect available memory and the actual memory allocation occur at different times, and we
don't have a mutex on allocating memory on the host or the device, it may be that the available
memory, as reported by runtime queries, is not actually available at allocation time.  For both host
and device allocations, if the allocator returns failure, then that test is skipped.  However, the
host reference FFT library also allocates memory internally, which, in the case of a OOM error, is
not something from which rocfft-test can recover.  (Host-side OOM errors are generally bad news.)
We also maintain a safety margin on how much memory we allocate in order to improve test robustness.

The number of failed allocations is tracked and reported at the end of `rocfft-test`'s execution.
There is a command-line option to report allocation errors as failures instead of just skipped
tests.

In and API setting, the accounting of host and device memory is complicated by the fact that this
is, in fact, a shared memory pool.  To deal with this, host memory and device memory are allocated
via structs that track this accounting, with extra care given to the somewhat overly optimistic hip
runtime, which may not track host memory allocations.


## Testing strategy

### Unit test strategy

#### Environment support

rocFFT supports a variety of OSes, as defined in the ROCm documentation.  The current CI in TheRock
only covers 4 device architectures, and 2 OSes.  Support for later versions of C++ are often
unavailable in older Linux OSes; not testing compilation+smoke-test in these environments may break
compilation or execution in these cases.

#### Unit tests

Unit tests are correctness tests verify the behaviour of the library infrastructure.  For example,
correctness tests cover API behaviour for cases where a user man provide invalid parameters, or
whether internal library infrastructure behaves as expected.

The API correctness is handled by rocfft-test, where gtest names are `rocfft_UnitTest.*`

Testing for internal library correctness is handled by an internal ctest framework, which is
controlled by the cmake option `ROCFFT_BUILD_INTERNAL_TESTS`, and the test executable is
`library/src/tests/rocfft-internal-test`.  Some more computationally expensive tests (the array
format validation tests) are provided by rocfft-test under the gtest filter
`reference_test/valid_length_stride.*`.

### Integration test strategy

#### Bit-wise reproducibility tests

rocFFT offers bit-wise reproducibility!  We test this by hashing the output using, and re-running
the test suite to verify bit-wise reproducibility.  Bit-wise reproducibility requires that one be
running the same version of rocFFT, identical ROCm stacks (compiler/runtime/driver), and the same
GPU model.

#### Accuracy tests

The accuracy tests are all given as accuracy_test.vs_fftw, using FFTW for reference computation.  L2
and L-infinity error bound scaling for FFT is known analytically, and we use this fact for
determining allowable numerical error for a given FFT.  In particular, one can specify the
precision-based constant multiplier ("machine-epsilon") in rocfft-test for half, single, and double
precision.  These bounds are designed so that if one sets the value, then numerical error will be <=
that value. Thus, the reported precision-based epsilon values reported at the end of the test run
can be used as a bound for running the tests again.

rocfft-test makes use of randomization.  A random static random seed can be provided via
command-line options, but should not be used in a general testing environment; the purpose here is
to reproduce tests in the case of failures.  Randomized test selection is stable under test addition
or removal, but not under test name changes (the test name, via gtest, is used in conjunction with
the random seed to determine the random number generator which determines whether an individual test
is run).

We will randomly select tests to run from a predefined set of parameters.  These tests can be
controlled by options in rocfft-test.  In addition to a random selection of tests, we also randomly
generate tests, and the number of these randomly-generated tests can also be controlled via
command-line.  Since one can specify the data decomposition (ie lengths and strides) fully in
rocFFT, care must be taken to avoid invalid configurations, which is basically when arrays
self-alias.  This is handled by the array format validator algorithm in rocFFT, which guarantees
that this does not occur.

Multi-process tests require a different execution path than single-process tests.  Accuracy tests
are handled by rocfft-mpi-worker using the --accuracy option.  Multi-process tests target a large
range of hardware configurations, and it is wasteful to reserve O(10^3) GPUs when running a two-GPU
test.  The script `scripts/rocfftslurmtest.py` divides the accuracy tests by hardware
configurations, and submits this to slurm, where the script `scripts/rocfft_mpi_test.py` launches
`rocfft-mpi-worker` to test the transforms.  The use of a python script also allows recovery from
hangs and crashes, which are unfortunately common in distributed computing software development.
The multi-process accuracy testing framework currently relies on reference computation on a single
host node, which restricts the size of problems which can be tested.

Multi-process tests are currently not covered by CI in TheRock due to a lack of infrastructure.

Accuracy tests should be run, unless otherwise specified by the developer, on all architectures that
are supported by the rocFFT library.  Currently, TheRock only tests on a subset of the supported
architectures.

One should not re-run accuracy tests until the desired result is achieved.

### ASAN / TSAN / Sanitizer Coverage

ASAN (address sanitizer) coverage is enabled in rocFFT via `BUILD_ADDRESS_SANITIZER`.

TSAN (thread sanitizer) and other sanitizer enabled in rocFFT.

clange-format is run to check for code format issues.

cppcheck is run for static analysis.


### Benchmarking and Performance Validation

Performance tests focus on a different parameter space than the accuracy tests.  For example, while
it's important that the length-1 transforms perform correctly (accuracy), the transform is actually
the identity operation, and the highest-performance option for software that needs to perform a
length-1 transform is to not use rocFFT, but to perform no operation at all.

Performance tests must deal with jitter.  Jitter is the tendency for execution time to include some
level of noise, and the GPU is not immune to this behaviour.  The results of naive performance
regression tests will consist almost entirely of false-positive results if jitter is not accounted
for in the experimental design and analysis of results.  rocFFT manages jitter via experimental
design and a multi-hypothesis statistical testing framework.

In order to eliminate the correlation between jitter and testing case, the performance-testing
experimental design is to load both the control and test versions of the rocFFT library into the
same executable and randomize the execution order for the two cases.  This is handled by
dyna-rocfft-bench (and dyna_rocfft_mpi_worker for the multi-process case) using dlopen for Linux and
/LoadLibraryA for Windows.  For cases where this framework does not apply (eg comparing performance
between two different devices) rocfft-bench is usable as a single-library client, though the risk of
false-positives is naturally higher.  On the other hand, one is unlikely to be testing the
performance impact of a software change between two different devices, so the importance of
false-positives is fairly small in this case.

Post-processing of the data from the test is handled by statistical tests, and we have implemented
the T-test, Mood's median test, and the Mann-Whitney U test (also known as the Wilcoxon rank-sum
test).  While the data distribution of execution times does not follow a normal distribution, the
T-test only requires that the difference between the distributions follows is normally distributed,
which is generally accepted to be true when thee sample size is at least 20.  The three tests answer
subtly different questions, ie the differences of the mean, the median, or the rank, though, for
realistic data, these tend to be all agree.  Since we also test multiple points in parameter space
together, it's also important to use a multi-hypothesis testing framework in order to avoid
p-hacking oneself.  rocFFT implements the Bonferroni correction and the Benjamini–Hochberg procedure
in order to reduce the false-positive rate.

In addition to statistical testing, rocFFT provides confidence intervals for transform execution
time.  Since the data tends to not be normally distributed and have long tails, the central tendency
of the execution time is better represented by the median than by the mean, which has the added
benefit of being invariant under monotonic transformation (eg time -> gflops or bandwidth).  We also
do not use the standard deviation to express confidence intervals; not only is the data not normal,
but the mean minus the standard deviation is often negative, which is not only unphysical, but quite
difficult to plot when using a logarithmic scale.  Instead, we use bootstrap resampling to compute
the confidence intervals on the median execution time.

The above features are implemented in the python script `scripts/perf/rocfft-perf`, which also
includes `scripts/perf/suites.py`, which defines the performance testing suites for rocFFT.
Multi-process performance testing is still in development.

Performance tests should be run, unless otherwise specified by the developer, on all
architectures that are supported by the rocFFT library.

Since we perform null-hypothesis testing, we do not use a percentage cutoff.

One should not re-run performance tests until the desired result is achieved.

While every effort has been made to reduce false positives, these will still inevitably occur; the
performance tests therefore cannot be gating.  We will trust developers to use their judgement to
deal with these cases.

Performance testing is currently not implemented in TheRock due to infrastructure issues.


## Pre-submit / CI Gates

Pre-submit tests currently cover unit tests and accuracy tests for:

** gfx94X, gfx950, gfx125X on Linux in a docker image
** gfx1151 on Windows

The CI tests in TheRock reduce the test probability to 1% due to performance issues in the CI
infrastucture, which results in an uncomfortably low number of tests being run.  Previously the test
probability was at 100%.

Static analysis (formatting and cppcheck) is gating for PRs.

There are no multi-gpu tests run, performance tests isn't run, and multi-process tests are not run.
These gaps are due to infractucture availability issues.

### Desired testing standard 

Our objective is to have targetted static analysis, unit test, integration tests, and performance
tests on all architectures combinations that rocFFT supports.  Tests should be performed pre-submit
using a targetted testing strategy (eg documentation builds don't need performance testing), with a
weekly build to confirm.
