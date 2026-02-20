.. meta::
  :description: rocSOLVER logging documentation
  :keywords: rocSOLVER, ROCm, API, documentation, logging, tracing

.. _logging-label:

******************************
rocSOLVER multi-level logging
******************************

rocSOLVER provides logging facilities that can be used to output information on rocSOLVER function calls. The infrastructure is 
similar to `rocBLAS logging <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/logging.html>`_. 
Three logging modes are supported: trace logging, bench logging, and profile logging.

.. note::

   Performance will degrade when logging is enabled.

Logging modes
================================================

This section discusses the three logging modes.

Trace logging
--------------

Trace logging outputs a line each time an internal rocSOLVER or rocBLAS routine is called.
The output includes the function name and the values of the arguments (excluding stride arguments). You
can specify the maximum depth of nested function calls that can appear in the log.

Bench logging
----------------

Bench logging outputs a line each time a public rocSOLVER routine is called (excluding
auxiliary library functions). It outputs a line that can be used with the
``rocsolver-bench`` executable to call the function with the same size arguments.

.. _log_profile:

Profile logging
-------------------

Profile logging is triggered whenever you call ``rocsolver_log_write_profile`` or ``rocsolver_log_flush_profile``,
or terminate the logging session using ``rocsolver_log_end``. It outputs statistics on each
internal rocSOLVER and rocBLAS routine that was called. These metrics include the number of times each function
was called, the total program runtime occupied by the function, and the total program runtime
occupied by its nested function calls. Similar to trace logging, you can specify the maximum depth of the nested output.

.. note::

   When profile logging is enabled, the stream is synchronized after every internal function call.

Initialization and setup
================================================

To use the rocSOLVER logging facilities, first call ``rocsolver_log_begin``
to allocate the internal data structures and begin the logging session.
You can then specify a layer mode and max level depth, either programmatically using
``rocsolver_log_set_layer_mode`` and ``rocsolver_log_set_max_levels`` or by setting the corresponding
environment variables.

The layer mode specifies which logging types are activated. It can be set to ``rocblas_layer_mode_none``,
``rocblas_layer_mode_log_trace``, ``rocblas_layer_mode_log_bench``, ``rocblas_layer_mode_log_profile``,
or a bitwise combination of these values. The max level depth specifies the default maximum depth for any nested
function calls that might appear in the trace and profile logging.

The default layer mode and max level depth can be specified using the following environment variables:

*  ``ROCSOLVER_LAYER``
*  ``ROCSOLVER_LEVELS``

If these variables are not set, the layer mode defaults to ``rocblas_layer_mode_none`` and the
maximum level depth defaults to ``1``. These defaults can be restored by calling the function
``rocsolver_log_restore_defaults``.

``ROCSOLVER_LAYER`` is a bitwise OR of zero or more bit masks as follows:

*  If ``ROCSOLVER_LAYER`` is not set, then there is no logging.
*  If ``(ROCSOLVER_LAYER & 1) != 0``, then there is trace logging.
*  If ``(ROCSOLVER_LAYER & 2) != 0``, then there is bench logging.
*  If ``(ROCSOLVER_LAYER & 4) != 0``, then there is profile logging.

Three environment variables are available to set the full path name for a log file:

*  ``ROCSOLVER_LOG_TRACE_PATH`` sets the full path name for trace logging.
*  ``ROCSOLVER_LOG_BENCH_PATH`` sets the full path name for bench logging.
*  ``ROCSOLVER_LOG_PROFILE_PATH`` sets the full path name for profile logging.

If any of these three environment variables have not been set, then the full path
for the corresponding log file is set to ``ROCSOLVER_LOG_PATH``. If ``ROCSOLVER_LOG_PATH`` also isn't
set, the corresponding logging output is streamed to standard error.

The results of profile logging, if enabled, can be printed using ``rocsolver_log_write_profile``
or ``rocsolver_log_flush_profile``. When logging facilities are no longer required, for example, at
program termination, the user must call ``rocsolver_log_end`` to free the data structures used
for logging. If the profile log has not been flushed beforehand, then ``rocsolver_log_end``
also outputs the profile logging results.

For more details on these logging functions, see the :ref:`rocSOLVER Logging functions <api_logging>`
reference section.


Example code
================================================

Code examples that illustrate the use of the rocSOLVER multi-level logging facilities can be found
in this section or in the ``example_logging.cpp`` file in the ``clients/samples`` directory.

The following example shows a basic use case. It enables trace and profile logging and sets the
maximum depth for the logging output.

.. code-block:: cpp

   // initialization
   rocblas_handle handle;
   rocblas_create_handle(&handle);
   rocsolver_log_begin();

   // begin trace logging and profile logging (max depth = 5)
   rocsolver_log_set_layer_mode(rocblas_layer_mode_log_trace | rocblas_layer_mode_log_profile);
   rocsolver_log_set_max_levels(5);

   // call rocSOLVER functions...

   // terminate logging and print profile results
   rocsolver_log_flush_profile();
   rocsolver_log_end();
   rocblas_destroy_handle(handle);

Alternatively, you can control which logging modes are enabled by using environment variables.
The benefit of this approach is that you don't need to recompile the program to use a different
logging environment. With this approach, however, you cannot call ``rocsolver_log_set_layer_mode`` and
``rocsolver_log_set_max_levels`` in the code. Here's an example:

.. code-block:: cpp

   // initialization
   rocblas_handle handle;
   rocblas_create_handle(&handle);
   rocsolver_log_begin();

   // call rocSOLVER functions...

   // termination
   rocsolver_log_end();
   rocblas_destroy_handle(handle);

You can then set the desired logging modes and maximum depth on the command line as follows:

.. code-block:: bash

   export ROCSOLVER_LAYER=5
   export ROCSOLVER_LEVELS=5


Kernel logging
================================================

To add kernel launches from within rocSOLVER to the trace and profile logs, use an
additional layer mode flag. The flag ``rocblas_layer_mode_ex_log_kernel`` can be combined with the
``rocblas_layer_mode`` flags and passed to ``rocsolver_log_set_layer_mode`` to enable
kernel logging. Alternatively, the environment variable ``ROCSOLVER_LAYER`` can be set so that
``(ROCSOLVER_LAYER & 16) != 0``, as follows:

*  If ``(ROCSOLVER_LAYER & 17) != 0``, then kernel calls are added to the trace log
*  If ``(ROCSOLVER_LAYER & 20) != 0``, then kernel calls are added to the profile log

Multiple host threads
================================================

The logging facilities for rocSOLVER assume that each ``rocblas_handle`` is associated with at
most one host thread. When using the rocSOLVER multi-level logging setup, it's recommended to
create a separate ``rocblas_handle`` for each host thread.

The ``rocsolver_log_*`` functions are not thread safe. Calling a log function while any rocSOLVER
routine is executing on another host thread results in undefined behavior. After it's enabled,
logging data collection is thread safe. However, if rocSOLVER routines are called from multiple host threads,
trace logging is likely to result in garbled trace trees.
