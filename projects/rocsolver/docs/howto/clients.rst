.. meta::
  :description: rocSOLVER documentation and API reference library
  :keywords: rocSOLVER, ROCm, API, documentation

.. _clients:

******************************
rocSOLVER Client Applications
******************************

rocSOLVER has an infrastructure for testing and benchmarking similar to that of
`rocBLAS's testing and benchmarking <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/Programmers_Guide.html#rocblas-benchmarking-and-testing>`_,
as well as sample code illustrating basic use of the library.

Client binaries are not built by default; they require specific flags to be passed to the install script
or CMake system. If the ``-c`` flag is passed to ``install.sh``, the client binaries will be located in the
directory ``<rocsolverDIR>/build/release/clients/staging``. If both the ``-c`` and ``-g`` flags are passed to
``install.sh``, the client binaries will be located in ``<rocsolverDIR>/build/debug/clients/staging``.
If the ``-DBUILD_CLIENTS_TESTS=ON`` flag, the ``-DBUILD_CLIENTS_BENCHMARKS=ON`` flag, and/or the
``-DBUILD_CLIENTS_SAMPLES=ON`` flag are passed to the CMake system, the relevant client binaries will normally
be located in the directory ``<rocsolverDIR>/build/clients/staging``. See the :ref:`Installation
section <installation>` for more information on building the library and its clients.

Testing rocSOLVER
==========================

The ``rocsolver-test`` client executes a suite of `Google tests <https://github.com/google/googletest>`_ (*gtest*) that
verifies the correct functioning of the library. The results computed by rocSOLVER, given random input data,
are normally compared with the results computed by `NETLib LAPACK <https://www.netlib.org/lapack/>`_ on the CPU, or tested implicitly
in the context of the solved problem. It will be built if the ``-c`` flag is passed to ``install.sh`` or if the ``-DBUILD_CLIENTS_TESTS=ON`` flag is
passed to the CMake system.

Calling the rocSOLVER gtest client with the ``--help`` flag

.. code-block:: bash

    ./rocsolver-test --help

returns information on different flags that control the behavior of the gtests.

One of the most useful flags is the ``--gtest_filter`` flag, which allows the user to choose which tests to run
from the suite. For example, the following command will run the tests for only geqrf:

.. code-block:: bash

    ./rocsolver-test --gtest_filter=*GEQRF*

Note that rocSOLVER's tests are divided into two separate groupings: ``checkin_lapack`` and ``daily_lapack``.
Tests in the ``checkin_lapack`` group are small and quick to execute, and verify basic correctness and error
handling. Tests in the ``daily_lapack`` group are large and slower to execute, and verify correctness of
large problem sizes. Users may run one test group or the other using ``--gtest_filter``, e.g.

.. code-block:: bash

    ./rocsolver-test --gtest_filter=*checkin_lapack*
    ./rocsolver-test --gtest_filter=*daily_lapack*


Benchmarking rocSOLVER
==================================

The ``rocsolver-bench`` client runs any rocSOLVER function with random data of the specified dimensions. It compares basic
performance information (i.e. execution times) between `NETLib LAPACK <https://www.netlib.org/lapack/>`_ on the
CPU and rocSOLVER on the GPU. It will be built if the ``-c`` flag is passed to ``install.sh`` or if the
``-DBUILD_CLIENTS_BENCHMARKS=ON`` flag is passed to the CMake system.

Calling the rocSOLVER bench client with the ``--help`` flag

.. code-block:: bash

    ./rocsolver-bench --help

returns information on the different parameters and flags that control the behavior of the benchmark client.

Two of the most important flags for ``rocsolver-bench`` are the ``-f`` and ``-r`` flags. The ``-f`` (or
``--function``) flag allows the user to select which function to benchmark. The ``-r`` (or ``--precision``)
flag allows the user to select the data precision for the function, and can be one of s (single precision),
d (double precision), c (single precision complex), or z (double precision complex).

The non-pointer arguments for a function can be passed to ``rocsolver-bench`` by using the argument name as
a flag (see the :ref:`Reference <reference>` sections for more information on the function arguments and
their names). For example, the function ``rocsolver_dgeqrf_strided_batched`` has the following method signature:

.. code-block:: cpp

    rocblas_status
    rocsolver_dgeqrf_strided_batched(rocblas_handle handle,
                                     const rocblas_int m,
                                     const rocblas_int n,
                                     double* A,
                                     const rocblas_int lda,
                                     const rocblas_stride strideA,
                                     double* ipiv,
                                     const rocblas_stride strideP,
                                     const rocblas_int batch_count);

A call to ``rocsolver-bench`` that runs this function on a batch of one hundred 30x30 matrices could look like this:

.. code-block:: bash

    ./rocsolver-bench -f geqrf_strided_batched -r d -m 30 -n 30 --lda 30 --strideA 900 --strideP 30 --batch_count 100

Generally, ``rocsolver-bench`` will attempt to provide or calculate a suitable default value for these arguments,
though at least one size argument must always be specified by the user. Functions that take m and n as arguments
typically require m to be provided, and a square matrix will be assumed. For example, the previous command is
equivalent to:

.. code-block:: bash

    ./rocsolver-bench -f geqrf_strided_batched -r d -m 30 --batch_count 100

Other useful benchmarking options include the ``--perf`` flag, which will disable the LAPACK computation and only time \
and print the rocSOLVER performance result; the ``-i`` (or ``--iters``) flag, which indicates the number of times to run the
GPU timing loop (the performance result would be the average of all the runs); and the ``--profile``
flag, which enables :ref:`profile logging <log_profile>` indicating the maximum depth of the nested output.

.. code-block:: bash

    ./rocsolver-bench -f geqrf_strided_batched -r d -m 30 --batch_count 100 --perf 1
    ./rocsolver-bench -f geqrf_strided_batched -r d -m 30 --batch_count 100 --iters 20
    ./rocsolver-bench -f geqrf_strided_batched -r d -m 30 --batch_count 100 --profile 5

In addition to the benchmarking functionality, the rocSOLVER bench client can also provide the norm of the error in the
computations when the ``-v`` (or ``--verify``) flag is used; and return the amount of device memory required as workspace for the given function, if the
``--mem_query`` flag is passed.

.. code-block:: bash

    ./rocsolver-bench -f geqrf_strided_batched -r d -m 30 --batch_count 100 --verify 1
    ./rocsolver-bench -f geqrf_strided_batched -r d -m 30 --batch_count 100 --mem_query 1



rocSOLVER sample code
==================================

rocSOLVER's sample programs provide illustrative examples of how to work with the rocSOLVER library. They will be
built if the ``-c`` flag is passed to ``install.sh`` or if the ``-DBUILD_CLIENTS_SAMPLES=ON`` flag is passed to the
CMake system.

Currently, sample code exists to demonstrate the following:

* Basic use of rocSOLVER in C, C++ using the example of :ref:`rocsolver_geqrf <geqrf>`;
* Use of batched and strided_batched functions, using :ref:`rocsolver_geqrf_batched <geqrf_batched>` and :ref:`rocsolver_geqrf_strided_batched <geqrf_strided_batched>` as examples;
* Use of rocSOLVER with the Heterogeneous Memory Management (HMM) model; and
* Use of rocSOLVER's :ref:`multi-level logging <logging-label>` functionality.
