# Contributing

## Philosophy

AMD welcomes contributions from the community. Whether those contributions are bug reports,
bug fixes, documentation additions, performance notes, or other improvements, we value
collaboration with our users. We can build better solutions together.

# Submitting a Pull Request

To contribute changes to rocSOLVER, open a pull request targeting the `develop` branch. Pull
requests will be tested and reviewed by the AMD development team. AMD may request changes or
modify the submission before acceptance.

## Interface requirements

The public interface must be:

- C99 compatible
- Source and binary compatible with previous releases
- Fully documented with Doxygen and Sphinx

All identifiers in the public headers must be prefixed with `rocblas`, `ROCBLAS`, `rocsolver`,
or `ROCSOLVER`.

All user-visible symbols must be prefixed with `rocblas` or `rocsolver`.

## Style guide

In general, follow the style of the surrounding code. All code is auto-formatted using clang-format.
To apply the rocsolver formatting, run `clang-format -i -style=file <files>` on any files you've
changed. You can install git hooks to do this automatically upon commit by running
`scripts/install-hooks --get-clang-format`. If you find you'd rather not use the hooks, they can
be removed using `scripts/uninstall-hooks`.

## Tests

To run the rocSOLVER test suite, first build the rocSOLVER test client following the instructions in
[Building and Installation][1]. Then, run the `rocsolver-test` binary. For a typical build, the test
binary will be found at `./build/release/clients/staging/rocsolver-test`.

The full test suite is quite large and may take a long time to complete, so passing the
[`--gtest_filter=<pattern>`][2] option to rocsolver-test may be useful during development. A fast
subset of tests can be run with `--gtest_filter='checkin*'`, while the extended tests can be run
with `--gtest_filter='daily*'`.

## Rejected contributions

Unfortunately, sometimes a contribution cannot be accepted. The rationale for a decision may or may
not be disclosed.

[1]: https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/userguide/install.html
[2]: https://github.com/google/googletest/blob/release-1.10.0/googletest/docs/advanced.md#running-a-subset-of-the-tests
