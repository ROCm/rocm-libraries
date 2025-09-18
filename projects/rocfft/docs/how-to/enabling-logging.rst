.. meta::
  :description: Enabling logging in rocFFT
  :keywords: rocFFT, ROCm, API, documentation, logging


.. _enabling-logging:

********************************************************************
Enabling logging in rocFFT
********************************************************************

rocFFT can write a variety of log messages to aid troubleshooting. Here are the different
logs that rocFFT supports.

*  **Trace logging**: Logs the library entry points (for example, ``rocfft_plan_create`` or ``rocfft_execute``) and their parameter
   values when they are called. Error messages during plan creation and execution are also logged here.
*  **Benchmark logging**: Logs the ``rocfft-bench`` command line when a plan is created.
   You can use this command to rerun the same transform later.
*  **Profile logging**: Logs a message for each kernel launched during plan execution.
   This message contains the following elements:

   *  Kernel duration
   *  The size of the user data buffers seen by the kernel
   *  Estimates for the observed memory bandwidth and bandwidth efficiency

   .. note::
   
      To provide the kernel duration, rocFFT must use ``hipEvents`` and wait for each kernel to complete.
      This might interfere with time measurement at higher levels, for example, for ``rocfft-bench``.

*  **Plan logging**: Logs the plan details when a transform is executed, including the following:

   *  Each TreeNode in the plan
   *  The work buffer size required by the plan
   *  The kernel grid and block dimensions
   *  The kernel maximum occupancy (estimated by HIP)

*  **Kernel I/O logging**: Logs the kernel details during plan execution, including the input to each
   kernel (the data provided by the user) and the final output of the transform.

   .. note::

      The amount of data logged can become very large, particularly for 2D and 3D transforms, so
      logging it to a file instead of stderr is usually a good idea. See the next section for more details.
      
      Writing the data involves extra ``hipMemcpy`` operations and serializing the
      data to the log can also take a significant amount of time. Both of these factors affect performance.

*  **Runtime compilation logging**: Logs details about runtime compilation during plan creation,
   including the following:

   *  The source code
   *  Messages indicating a kernel was found in a cache, and did not need to be compiled at runtime
   *  Compilation errors (if any)
   *  Duration measurements indicating the time it took to generate source code for the kernel and compile the kernel

   The source code for the kernels is delimited by lines containing the strings ``ROCFFT_RTC_BEGIN``
   and ``ROCFFT_RTC_END``. This lets you isolate the source code for each kernel if a
   single log contains code for multiple kernels.

   .. note::

      All non-code messages (except for compile errors) are written as C++ comments, so 
      you can pass the whole file to clang-format to inspect the source code.

      The source code details for the runtime compilation can be very large, so consider writing
      this log to a file instead of stderr.

*  **Tuning logging**: Logs details about any kernels that are tried and rejected while tuning is running.
   It also logs messages when tuned solutions are used during plan building.
*  **Graph logging**: Logs the graph of subplans during multi-GPU or multi-process plan execution.
   Subplans include FFT plans, transpose plans (to reshape data for communication), and communication steps.
   This is written as Graphviz data. The view of the global graph might be slightly different from
   different nodes. This is because the current node has more visibility into subplans that run locally
   than those that run on other nodes.

Configuring the logging output
==============================

The logging output can be controlled using the ``ROCFFT_LAYER`` environment variable.
``ROCFFT_LAYER`` is a numerical bitmask, where zero or more bits can be set to enable one or more logging layers.
The log output is written to stderr by default.

The following table maps the different logging layers to a ``ROCFFT_LAYER`` bit field value.
To determine what value to set for ``ROCFFT_LAYER``, add up the values of all the layers you want to see.
For example, to see the output for trace, profile, and plan logging, set ``ROCFFT_LAYER`` to ``13``
(``1`` + ``4`` + ``8``).

.. csv-table::
   :header: "Log type","ROCFFT_LAYER bit field value"
   :widths: 20, 20

   "Trace logging","1"
   "Benchmark logging","2"
   "Profile logging","4"
   "Plan logging","8"
   "Kernel I/O logging","16"
   "Runtime compilation logging","32"
   "Tuning logging","64"
   "Graph logging","128"

Logging to a file
=============================

By default, messages are written to stderr, but they can be redirected to
output files using the environment variables described in this section.
Each type of log can be redirected separately using a unique environment variable.
The corresponding log must be enabled using the ``ROCFFT_LAYER`` variable
before any details can be logged to the file.
For example, to redirect the trace log to a file, trace logging must
also be enabled in the  ``ROCFFT_LAYER`` bit field.

.. note::

   Some log types, such as kernel I/O logging and runtime compilation logging, can generate
   a large number of log entries, so redirecting their output to a file is recommended.

The following table lists the environment variable to redirect logging for each
log type. Set this variable to a valid file path to redirect the output of the corresponding log type.
For example, to send the trace logging output to a file, enable the trace log, then set the
``ROCFFT_LOG_TRACE_PATH`` variable to the name of the destination file.

.. csv-table::
   :header: "Log type","File redirection variable"
   :widths: 20, 30

   "Trace logging","``ROCFFT_LOG_TRACE_PATH``"
   "Benchmark logging","``ROCFFT_LOG_BENCH_PATH``"
   "Profile logging","``ROCFFT_LOG_PROFILE_PATH``"
   "Plan logging","``ROCFFT_LOG_PLAN_PATH``"
   "Kernel I/O logging","``ROCFFT_LOG_KERNELIO_PATH``"
   "Runtime compilation logging","``ROCFFT_LOG_RTC_PATH``"
   "Tuning logging","``ROCFFT_LOG_TUNING_PATH``"
   "Graph logging","``ROCFFT_LOG_GRAPH_PATH``"
