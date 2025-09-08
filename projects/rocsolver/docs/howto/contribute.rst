.. meta::
  :description: rocSOLVER contribution guide
  :keywords: rocSOLVER, ROCm, documentation, contributing

.. _contribute:

*******************************
Contribute to rocSOLVER
*******************************

AMD welcomes community contributions, including bug reports,
bug fixes, documentation additions, performance notes, and other improvements.

Submitting a pull request
---------------------------

To contribute changes to rocSOLVER, open a pull request targeting the ``develop`` branch. Pull
requests will be tested and reviewed by the AMD development team. AMD might request changes or
modify the submission before acceptance.

Interface requirements
---------------------------

The public interface must be:

- C99 compatible
- Source and binary compatible with previous releases
- Fully documented with Doxygen and Sphinx

All identifiers in the public headers must be prefixed with ``rocblas``, ``ROCBLAS``, ``rocsolver``,
or ``ROCSOLVER``. All user-visible symbols must be prefixed with ``rocblas`` or ``rocsolver``.

Style guide
---------------------------

Follow the style of the surrounding code. All code is auto-formatted using clang-format.
To apply the rocSOLVER formatting, run ``clang-format -i -style=file <files>`` on any files you've
changed. You can install Git hooks to do this automatically upon commit by running
``scripts/install-hooks --get-clang-format``. If you don't want to use the hooks, they can
be removed using ``scripts/uninstall-hooks``.

Tests
---------------------------

To run the rocSOLVER test suite, first build the rocSOLVER test client following the instructions in
the :doc:`Installation guide <../installation/installlinux>`. Then run the ``rocsolver-test`` binary. For a typical build, the test
binary can be found at ``./build/release/clients/staging/rocsolver-test``.

The full test suite is quite large and can take a long time to complete. During development, it might be useful to 
`run a subset of the tests <https://github.com/google/googletest/blob/release-1.10.0/googletest/docs/advanced.md#running-a-subset-of-the-tests>`_ 
by passing the ``--gtest_filter=<pattern>`` option to ``rocsolver-test``. A quick
subset of tests can be run with ``--gtest_filter='checkin*'``, while the extended tests can be run
using ``--gtest_filter='daily*'``.

Rejected contributions
---------------------------

Unfortunately, sometimes a contribution cannot be accepted. The rationale for this decision is not always disclosed.



