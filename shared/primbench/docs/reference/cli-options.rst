.. meta::
   :description: Complete reference of all command-line options accepted by Primbench benchmarks, including size suffixes, noise-reduction tuning, temperature control, and output formatting.
   :keywords: Primbench, CLI, command-line, options, ROCm, HIP, GPU, benchmark, settings

**********************
Command-line options
**********************

Primbench benchmarks accept command-line options that override the corresponding fields in the ``settings`` struct. When a value is set both programmatically through ``settings`` and on the command line, the command-line value takes precedence.

For details on setting values programmatically before parsing the command line, see :doc:`Configure benchmark settings </how-to/configure-settings>`. For the ``settings`` struct and the rest of the benchmark API, see :doc:`Primbench API </reference/primbench-api>`.

Options reference
=================

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Option
     - Type / Default
     - Description

   * - ``--help``
     - Flag
     - Prints usage information and exits.

   * - ``--size``
     - ``size_t`` / ``128 * MiB``
     - Input array size in bytes. Must be greater than ``0``. Accepts ``KiB``, ``MiB``, or ``GiB`` suffixes, for example ``--size 256MiB``.

   * - ``--hot``
     - ``bool`` / ``false``
     - When set, cache clearing before kernel launches is skipped. Boolean options take no value.

   * - ``--seed``
     - ``uint32_t`` / ``42``
     - Seed used for input array generation.

   * - ``--json-out``
     - ``string`` / ``"results.json"``
     - Output JSON file path.

   * - ``--csv-out``
     - ``string`` / ``""``
     - Output CSV file path. No CSV is written when empty.

   * - ``--filter``
     - ``string`` / ``""``
     - Regex filter applied to specialization names. Only matching specializations are benchmarked.

   * - ``--dry``
     - ``bool`` / ``false``
     - Perform a dry run without executing kernels. Boolean options take no value.

   * - ``--min-gpu-ms-per-batch``
     - ``double`` / ``10.0``
     - Minimum GPU time in milliseconds for each batch. Must be greater than ``0``.

   * - ``--min-secs``
     - ``double`` / ``1.0``
     - Minimum total benchmark duration in seconds. Must be greater than ``0``. Must be less than or equal to ``--noise-timeout-secs``.

   * - ``--noise-timeout-secs``
     - ``double`` / ``10.0``
     - Maximum duration in seconds before a noisy benchmark times out. Must be greater than ``0``.

   * - ``--batch-window-size``
     - ``size_t`` / ``10``
     - Number of recent batches used in the noise-reduction sliding window. Must be greater than ``0``.

   * - ``--noise-tolerance-percent``
     - ``double`` / ``1.0``
     - Noise tolerance percentage for early stopping. Must be greater than ``0``. Batching continues until the measurement noise falls below this threshold or the noise timeout is reached.

   * - ``--min-gpu-temp``
     - ``uint16_t`` / ``50``
     - Minimum GPU temperature in degrees Celsius. Must be less than or equal to ``--max-gpu-temp``. If the GPU is below this temperature, Primbench runs warmup workloads until it reaches this threshold before starting a specialization.

   * - ``--max-gpu-temp``
     - ``uint16_t`` / ``60``
     - Maximum GPU temperature in degrees Celsius. If the GPU exceeds this temperature, Primbench waits for it to cool below this threshold before starting a specialization.

   * - ``--max-warming-secs``
     - ``double`` / ``60.0``
     - Maximum time in seconds to spend warming the GPU. Must be greater than ``0``.

   * - ``--max-cooling-secs``
     - ``double`` / ``60.0``
     - Maximum time in seconds to spend cooling the GPU. Must be greater than ``0``.

   * - ``--output-batches``
     - ``bool`` / ``false``
     - When set, per-batch details are included in the output. Boolean options take no value.

   * - ``--spaces-per-indent``
     - ``uint32_t`` / ``4``
     - Number of spaces per indentation level in JSON output. Must be less than or equal to ``8``. Set to ``0`` for compact, unformatted JSON.

   * - ``--stream-blocking-timeout-secs``
     - ``double`` / ``10.0``
     - Maximum duration in seconds before stream blocking times out. Must be greater than ``0``.

Size suffixes
=============

The ``--size`` option accepts an optional suffix to specify the unit:

- ``KiB``, kibibytes, 1024 bytes
- ``MiB``, mebibytes, 1024 × 1024 bytes
- ``GiB``, gibibytes, 1024 × 1024 × 1024 bytes

When no suffix is provided, the value is interpreted as raw bytes.

Programmatic defaults and CLI precedence
========================================

Each option corresponds to a field in the ``settings`` struct. You can set default values programmatically before passing ``settings`` to the executor. Any value supplied on the command line overrides the programmatic default. See :doc:`Configure benchmark settings </how-to/configure-settings>` for examples of combining programmatic and command line configuration.
