# rocFFT Testing Strategy

This document describes the testing strategy for rocFFT.

## Component Overview

rocFFT is a FFT gpu library.

The HIP kernels are produced via a code generator embedded in the library, which are compilbed using
hipRTC.  Transforms are available for single-process single-device, single-process multi-device, and
multi-process single/multi-device.

rocFFT tests rely on googletest and some python scripts included in the rocFFT git repository.

# Testing Strategy and Layers

## Test naming

TODO: talk about test name tokenization

## Resources and runtime robustness

TODO: talk about how memory footprint restrictions, and hipskip.

## Testing objectives

### Correctness tests

These tests verify the API behaviour, as well as some internal structure.  Some of this is coverd by
rocfft-test (gtest names are roccfft_UnitTest.*, but also reference_test/valid_length_stride.*)

TODO: internal ctest stuff.

### Bitwise reproducibility tests

rocFFT offers bitwise reproducibility!  We test this by hashing the output using, and re-running the
test suite to verify bitwise reproducbility.

### Accuracy tests

The accuracy tests are all given as accuracy_test.vs_fftw, using FFTW for reference computation.  L2
and L-infinity error bound scaling for FFT is known analytically, and we use this fact for
determining allowable numerical error for a given FFT.  In particular, one can specify the
precision-based constant multiplier ("machine-epsilon") in rocfft-test for half, single, and double
precision.  These bounds are designed so that if one sets the value, then numerical error will be <=
that value. Thus, the reported precision-based epsilon values reported at the end of the test run
can be used as a bound for running the tests again.

rocfft-test makes use of randomization.  A random static random seed can be provided via
command-line options, but should not be used in a general testing environment; the purpose
here is to reproduce tests in the case of failures.  Randomized test selection is stable
under test addition or removal, but not under test name changes (the test name, via gtest, is
used in conjunction with the random seed to determine the random number generator which determines
whether the individual test is run). 

We will randomly select tests to run from a predefined set of parameters.  These tests can be
controlled by options in rocfft-test.  In addition to a random selection of tests, we also randomly
generate tests, and the number of these randomly-generated tests can also be controlled via
command-line.  Since one can specify the data decomposition (ie lengths and strides) fully in
rocFFT, care must be taken to avoid invalid configurations, which is basically when arrays
self-alias.  This is handled by the array validator algorithm in rocFFT, which guarantees that
this does not occur.

TODO: multi-proc accuracy tests.

### Performance tests

TODO: 
