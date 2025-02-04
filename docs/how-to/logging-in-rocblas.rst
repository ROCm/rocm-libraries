.. meta::
  :description: how to use logging in rocBLAS
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation, logging, reference

.. _logging:

********************************************************************
Logging in rocBLAS
********************************************************************

**Note that performance will degrade when logging is enabled.**

You can set four environment variables to control logging:

* ``ROCBLAS_LAYER``

* ``ROCBLAS_LOG_TRACE_PATH``

* ``ROCBLAS_LOG_BENCH_PATH``

* ``ROCBLAS_LOG_PROFILE_PATH``

``ROCBLAS_LAYER`` is a bitwise OR of zero or more bit masks as follows:

*  If ``ROCBLAS_LAYER`` is not set, then there is no logging.

*  ``(ROCBLAS_LAYER & 1) != 0`` enables trace logging.

*  ``(ROCBLAS_LAYER & 2) != 0`` enables bench logging.

*  ``(ROCBLAS_LAYER & 4) != 0`` enables profile logging.

*  ``(ROCBLAS_LAYER & 8) != 0`` enables internal API logging, for example, the GEMM backend.

See the ``rocblas_layer_mode`` enumeration for these values as constants.

Trace logging outputs a line each time a rocBLAS function is called. The
line contains the function name and the values of arguments.

Bench logging outputs a line each time a rocBLAS function is called. The
line can be used with the executable ``rocblas-bench`` to call the
function with the same arguments.

Profile logging, at the end of program execution, outputs a YAML
description of each rocBLAS function called, the values of its
performance-critical arguments, and the number of times it was called
with those arguments (the ``call_count``). Some arguments, such as
``alpha`` and ``beta`` in GEMM, are recorded with a value representing
the category that the argument falls into, such as ``-1``, ``0``, ``1``,
or ``2``. The number of categories, and the values representing them,
might change over time, depending on how many categories are needed to
adequately represent all the values that can affect the performance
of the function.

Internal API logging outputs information like the GEMM backend used for a particular GEMM call.
Not all internal APIs are logged. The log output goes to the same stream as trace logging, but the
output can be controlled.

The default stream for logging output is standard error. Three
environment variables can set the full path name for a log file:

*  ``ROCBLAS_LOG_TRACE_PATH`` sets the full path name for trace logging.
*  ``ROCBLAS_LOG_BENCH_PATH`` sets the full path name for bench logging.
*  ``ROCBLAS_LOG_PROFILE_PATH`` sets the full path name for profile logging.

For example, in a Bash shell, use the following to output bench logging to the file
``bench_logging.txt`` in your present working directory:

.. code-block:: shell

   export ROCBLAS_LOG_BENCH_PATH=$PWD/bench_logging.txt

A full path is required, not a relative path. In the command above,
``$PWD`` expands to the full path of your present working directory.
If the paths are not set, then the logging output is streamed to standard error.

When profile logging is enabled, memory usage increases. If the
program exits abnormally, it is possible that profile logging will
not be outputted before the program exits.


rocTX support in rocBLAS
========================

The `rocTX <https://rocm.docs.amd.com/projects/roctracer/en/latest/reference/roctx-spec.html>`_ library contains application code
instrumentation APIs to support high-level correlation of runtime API or activity events.
When integrated with rocBLAS, rocTX enables users to capture detailed logs, like ``ROCBLAS_TRACE`` or ``ROCBLAS_BENCH``, and view
them in profiling tools such as rocProf,
offering better insights into runtime behavior and performance bottlenecks.

The following steps describe how to enable logging:

.. code-block:: shell

   # To view trace logging

   export ROCBLAS_LAYER=1
   rocprof --hip-trace --roctx-trace ./rocblas-bench -f geam

   # To view bench logging

   export ROCBLAS_LAYER=2
   rocprof --hip-trace --roctx-trace ./rocblas-bench -f geam

These settings activate the corresponding logging layers in rocBLAS, allowing users to capture either trace-level
information (for function calls) or bench-level information (for benchmarking purposes) during profiling.

.. note::

   rocTX support in rocBLAS is unavailable on Windows and is not supported in the static library version on Linux.
