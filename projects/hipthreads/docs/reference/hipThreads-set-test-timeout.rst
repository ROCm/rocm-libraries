.. meta::
  :description: Reference for the hipThreads test suite, its environment variables, and timeout behavior
  :keywords: hipThreads, ROCm, testing, lit, timeout, HIPTHREADS_TEST_TIMEOUT, CI

.. _testing:

**********************************************
hipThreads testing timeout
**********************************************

The hipThreads test suite is adapted from the libc++ thread tests.  Each test compiles to its own executable and runs the primitives inside a GPU kernel before checking its results.

The hipThreads test suite runs under `lit, the LLVM Integrated Tester <https://llvm.org/docs/CommandGuide/lit.html>`_. lit compiles each test with ``hipcc`` and runs the executables using the ``run.py`` wrapper.

Because the tests in the suite run concurrency primitives on the device, a test can hang instead of finishing. For example, a thread that waits on a mutex that's never released will stay blocked.

To prevent hanging, tests will be terminated after 30s and marked as having failed.

The timeout value is configurable through the ``HIPTHREADS_TEST_TIMEOUT`` :doc:`environment variable <environment-variables>`.

For this variable to take effect, it must be called before lit:

.. code-block:: bash

  export HIPTHREADS_TEST_TIMEOUT=120
  lit -a -v -j 1 test/

Setting ``HIPTHREADS_TEST_TIMEOUT`` to ``0`` disables the timeout.
