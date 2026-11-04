.. meta::
   :description: API reference for Primbench benchmark types, executor, state, JSON builder, macros, and utility functions.
   :keywords: Primbench, benchmark_interface, executor, settings, state, json, macros, API, ROCm, HIP, GPU benchmarking

***********************
Primbench API
***********************

The core types for defining and running GPU benchmarks with Primbench include the ``state`` object passed into ``run()``, the JSON builder, and the user-facing macros and utility functions. For command-line option details, see :doc:`Command-line options </reference/cli-options>`. For correctness validation workflows, see :doc:`Validate benchmark output </how-to/validate-output>`.

Command-line values override any programmatic ``settings`` values passed to the ``executor`` constructor.

Flags
=====

Enumeration and wrapper for combining benchmark flags.

.. doxygenenum:: primbench::detail::flags::Flags
.. doxygenstruct:: primbench::detail::flags::FlagTag
   :members:

Settings
========

All tunable parameters for benchmark execution. Each field has a default that can be overridden programmatically or from the command line.

.. doxygenstruct:: primbench::settings
   :members:

Benchmark interface
===================

Abstract base class that users subclass to define a benchmark specialization. Implement ``meta()`` to describe the specialization and ``run()`` to execute it.

.. doxygenstruct:: primbench::benchmark_interface
   :members:

Executor
========

Manages the benchmark lifecycle: parses command-line arguments, queues benchmark specializations, and runs them.

.. doxygenclass:: primbench::executor
   :members:

Benchmark state
===============

The ``state`` object serves as the primary interface for declaring throughput metrics, registering the kernel lambda, setting up per-iteration callbacks, and running correctness tests. It is passed to ``benchmark_interface::run()``, and exposes the GPU stream and the current input size.

Public fields
-------------

.. doxygenclass:: primbench::detail::state
   :members:

Throughput declarations
-----------------------

These methods declare how many logical items, read bytes, and written bytes each kernel invocation processes. The executor uses these values to compute throughput metrics in the output.

.. doxygenfunction:: primbench::detail::state::set_items
.. doxygenfunction:: primbench::detail::state::add_reads
.. doxygenfunction:: primbench::detail::state::add_writes

Kernel registration
-------------------

Register the kernel lambda that the executor times, and an optional callback that runs before every iteration, for example to reset output buffers.

.. doxygenfunction:: primbench::detail::state::run
.. doxygenfunction:: primbench::detail::state::run_before_every_iteration

Correctness testing
-------------------

Register a callable that validates kernel output. The callable runs once after the warmup batch, before timed iterations begin. Use ``PRIMBENCH_ASSERT`` inside the test callable to check results.

.. doxygenfunction:: primbench::detail::state::test

JSON builder
============

The ``json`` struct is a lightweight builder used inside ``benchmark_interface::meta()`` to attach algorithm names, type names, and custom fields to a benchmark specialization. Calls to ``add()`` can be chained, and nested ``json`` objects are supported.

.. doxygenstruct:: primbench::detail::json
   :members:

Size constants
==============

.. doxygenvariable:: primbench::KiB

.. doxygenvariable:: primbench::MiB

.. doxygenvariable:: primbench::GiB

Macros
======

The following macros configure type names, error checking, correctness assertions, and compile-time behavior.

Type registration
-----------------

.. doxygendefine:: PRIMBENCH_REGISTER_TYPE

Registers a human-readable display name for a C++ type so that ``primbench::name<T>()`` returns it. The macro must be invoked at namespace scope.

Error checking
--------------

.. doxygendefine:: PRIMBENCH_CHECK

Wraps a HIP API call. If the call returns a failure status, the macro prints the file, line, and error string to ``stderr`` and exits the program.

Correctness assertions
----------------------

.. doxygendefine:: PRIMBENCH_ASSERT

Asserts equality between an input value and an expected value. Overloads handle scalar arithmetic types, iterable containers, and brace-enclosed initializer lists. An optional tolerance parameter defaults to ``0.0`` and controls the maximum allowed difference for floating-point comparisons. On mismatch the macro prints file, line, and a diagnostic message to ``stderr`` and exits.

Compile-time configuration
--------------------------

.. doxygendefine:: PRIMBENCH_GPU_CACHE_SIZE

Sets the size of the buffer used to evict GPU caches before kernel launches. Override by defining the macro before including the header or passing it as a compiler flag.

``PRIMBENCH_NO_MONITORING``
   When defined, disables GPU temperature monitoring. The library compiles without a monitoring dependency.

``PRIMBENCH_NO_TEST``
   When defined, disables correctness-test execution.

Version-control metadata
~~~~~~~~~~~~~~~~~~~~~~~~

``BRANCH_NAME`` and ``COMMIT_HASH`` are optional compile-time macros. Pass them with ``-DBRANCH_NAME=...`` and ``-DCOMMIT_HASH=...``. Their values are embedded in the ``context.general`` section of the JSON output so that benchmark results can be traced back to a specific source revision.

Free functions
==============

.. doxygenfunction:: primbench::log

.. doxygenfunction:: primbench::name
