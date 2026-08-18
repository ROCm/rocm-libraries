.. meta::
   :description: hipThreads environment variables
   :keywords: hipThreads, environment variables, ROCm, AMD, vcores, concurrency

.. _environment-variables:

******************************************
hipThreads environment variables
******************************************

The following environment variables affect the hipThreads runtime.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Environment variable
     - Value
   * - ``HIPTHREADS_VCORES_PER_WGP``
     - | Sets the number of scheduler virtual cores (vcores) launched per workgroup processor (WGP).
       | Must be a positive integer greater than zero. Non-numeric, zero, or empty values are treated as invalid and ignored.
       | Default: ``16``, or the value of ``-DHIPTHREADS_DEFAULT_VCORES_PER_WGP`` when hipThreads is built from source.

The test suite reads a separate set of environment variables when it runs under `lit, the LLVM Integrated Tester <https://llvm.org/docs/CommandGuide/lit.html>`_.

On Windows, running the tests requires ``ROCM_PATH`` or ``HIP_PATH`` to point to the ROCm or HIP install root. Configuration fails when neither is set.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Environment variable
     - Value
   * - ``HIPTHREADS_TEST_TIMEOUT``
     - | Sets the number of seconds a single test can run before the runner terminates it.
       | Fractional seconds are allowed.
       | Setting to ``0`` disables the timeout.
       | Default: ``30``.
   * - ``ROCM_PATH``
     - | Path to the ROCm install root.
       | Default: ``/opt/rocm`` on Linux. Falls back to ``HIP_PATH`` on Windows.
   * - ``HIP_PATH``
     - | Windows path to the HIP install root, used when ``ROCM_PATH`` is unset.
       | Configuration fails when neither ``ROCM_PATH`` nor ``HIP_PATH`` is set on Windows.
   * - ``HIPTHREADS_SOURCE_DIR``
     - | Path to the hipThreads source root.
       | Default: auto-detected from the location of ``test/lit.cfg``.
   * - ``HIPTHREADS_BUILD_DIR``
     - | Path to the build directory that holds the built tests and the ``hipthreads`` library.
       | Default: ``HIPTHREADS_SOURCE_DIR/build``
   * - ``HIP_ARCHITECTURES``
     - | Sets the GPU target.
       | If this variable is left unset on Windows, it defaults to ``gfx906``.
       | The architecture is autodetected on Linux.
