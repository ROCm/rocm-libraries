.. meta::
   :description: Configure Primbench benchmark settings including programmatic defaults, custom CLI arguments, specialization filtering, output control, dry runs, and build-time metadata.
   :keywords: Primbench, benchmark, settings, CLI, filter, JSON, CSV, dry run, ROCm, HIP, configuration

******************************
Configure benchmark settings
******************************

Primbench benchmarks accept configuration through the ``primbench::settings`` struct, command-line arguments, and compile-time macros. For the full list of command-line options, see :doc:`Command-line options </reference/cli-options>`. For details on JSON and CSV output structure, see :doc:`JSON output format </reference/primbench-json-output>`.

Pass settings programmatically
================================

Set default values for any benchmark option by constructing a ``primbench::settings`` struct and passing it to the ``primbench::executor`` constructor. Command-line arguments supplied at runtime override programmatic defaults.

C++17 style
-----------

Create a ``primbench::settings`` object, assign the fields you want to change, and pass it as the third argument:

.. code:: cpp

   primbench::settings settings;
   settings.min_gpu_ms_per_batch    = 100;
   settings.noise_tolerance_percent = 2;
   primbench::executor executor(argc, argv, settings);

C++20 designated initializers
------------------------------

With C++20 or later, use designated initializers for a more compact form:

.. code:: cpp

   primbench::executor executor(argc,
                                argv,
                                {
                                    .min_gpu_ms_per_batch = 100,
                                    .noise_tolerance_percent = 2,
                                });

All settings are written verbatim to ``context.settings`` in the JSON output. If an option is provided both programmatically and on the command line, the command-line value takes precedence.

Add custom CLI arguments
=========================

Benchmarks can register additional options that appear in ``--help`` output. Call ``executor.get<T>()`` with the option name, a default value, and a description string:

.. code:: cpp

   size_t dimensions = executor.get<size_t>("dimensions", 3, "The number of dimensions");

When you pass ``--dimensions 5`` on the command line, ``dimensions`` receives the value ``5``. If there are any custom options, they are written to ``context.custom_settings`` in the JSON output.

Filter specializations with ``--filter``
==========================================

Use the ``--filter`` option with a regex pattern to run only a subset of queued specializations. The pattern matches against each specialization's name, the string built from its ``meta()`` return value.

.. code:: shell

   ./copy_benchmark --filter 'type: long long'

   ./copy_benchmark --filter long

   ./copy_benchmark --filter 'l.*g'

   ./copy_benchmark --filter '^type: long long$'

Specializations that don't match the regex are skipped entirely.

Control output files and formatting
=====================================

Primbench writes results to JSON by default and optionally to CSV.

JSON output
-----------

By default, Primbench writes results to ``results.json``. Change the path with ``--json-out``:

.. code:: shell

   ./copy_benchmark --json-out my_results.json

To suppress JSON output entirely, redirect it:

.. code:: shell

   ./copy_benchmark --json-out /dev/null

CSV output
----------

Produce a CSV file alongside JSON by passing ``--csv-out``:

.. code:: shell

   ./copy_benchmark --csv-out results.csv

The CSV contains a condensed view: ``index``, ``name``, ``bytes_per_second``, ``gib_per_second``, ``items_per_second``, ``noise_timeout``, and ``noise_percent``.

Per-batch details
-----------------

Add ``--output-batches`` to include a ``batches`` array for each specialization in the JSON output, containing per-batch timing details:

.. code:: shell

   ./copy_benchmark --output-batches

JSON indentation
----------------

Adjust indentation with ``--spaces-per-indent``. The default is ``4``. Set to ``0`` for compact, unindented JSON:

.. code:: shell

   ./copy_benchmark --spaces-per-indent 0

Perform a dry run
==================

Use ``--dry`` to skip benchmark setup and execution while still producing JSON and CSV output files. Use a dry run to confirm that your specializations are queued correctly and that output paths are writable.

.. code:: shell

   ./copy_benchmark --dry

Embed branch name and commit hash
===================================

Define the ``BRANCH_NAME`` and ``COMMIT_HASH`` macros at compile time to record version-control metadata in the JSON output under ``context.general.branch_name`` and ``context.general.commit_hash``.

The following command embeds the current Git branch and short commit hash. In a detached HEAD state, such as during CI, ``BRANCH_NAME`` is set to ``DETACHED``:

.. code:: shell

   hipcc -o copy_benchmark examples/hip/copy_benchmark.cpp \
     -I. \
     -lamd_smi \
     -DBRANCH_NAME=\"$(git symbolic-ref -q --short HEAD || echo DETACHED)\" \
     -DCOMMIT_HASH=\"$(git rev-parse --short HEAD)\" \
     && ./copy_benchmark
