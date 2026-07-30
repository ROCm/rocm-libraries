.. meta::
   :description: API reference for Primbench benchmark types, executor, state, macros, and utility functions.
   :keywords: Primbench, benchmark_interface, executor, settings, state, macros, API, ROCm, HIP, GPU benchmarking

====================
Primbench API
====================

This page documents the core types for defining and running GPU benchmarks with Primbench, the ``state`` object passed into ``run()``, and the user-facing macros and utility functions. For command-line option details, see :doc:`Command-line options </reference/cli-options>`. For correctness validation workflows, see :doc:`Validate benchmark output </how-to/validate-output>`.

Command-line values override any programmatic ``settings`` values passed to the ``executor`` constructor.

Flags
*****

Enumeration and wrapper for combining benchmark flags.

.. doxygenenum:: primbench::flags::Flags
.. doxygenstruct:: primbench::flags::FlagTag
   :members:

Settings
********

All tunable parameters for benchmark execution. Each field has a default that can be overridden programmatically or from the command line.

.. doxygenstruct:: primbench::settings
   :members:

Benchmark interface
*******************

Abstract base class that users subclass to define a benchmark specialization. Implement ``meta()`` to describe the specialization and ``run()`` to execute it.

.. doxygenclass:: primbench::benchmark_interface
   :members:

Executor
********

Manages the benchmark lifecycle: parses command-line arguments, queues benchmark specializations, and runs them.

.. doxygenclass:: primbench::executor
   :members:

Benchmark state
***************

The ``state`` object is passed to ``benchmark_interface::run()`` and serves as the primary interface for declaring throughput metrics, registering the kernel lambda, setting up per-iteration callbacks, and running correctness tests. It also exposes the GPU stream and the current input size.

Public fields
-------------

.. doxygenstruct:: primbench::state
   :members:

Stream type
-----------

.. doxygentypedef:: primbench::stream_t

Throughput declarations
-----------------------

These methods declare how many logical items, read bytes, and written bytes each kernel invocation processes. The executor uses these values to compute throughput metrics in the output.

.. doxygenfunction:: primbench::state::set_items
.. doxygenfunction:: primbench::state::add_reads
.. doxygenfunction:: primbench::state::add_writes

Kernel registration
-------------------

Register the kernel lambda that the executor times, and an optional callback that runs before every iteration, for example to reset output buffers.

.. doxygenfunction:: primbench::state::run
.. doxygenfunction:: primbench::state::run_before_every_iteration

Correctness testing
-------------------

Register a callable that validates kernel output. The callable runs once after timing completes. Use ``PRIMBENCH_ASSERT`` inside the test callable to check results.

.. doxygenfunction:: primbench::state::test

Macros
******

.. doxygendefine:: PRIMBENCH_GPU_CACHE_SIZE

.. doxygendefine:: PRIMBENCH_REGISTER_TYPE

.. doxygendefine:: PRIMBENCH_ASSERT

.. doxygendefine:: PRIMBENCH_CHECK

Free functions
**************

.. doxygenfunction:: primbench::log

.. doxygenfunction:: primbench::name
