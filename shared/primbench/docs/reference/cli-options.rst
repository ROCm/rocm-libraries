.. meta::
   :description: Complete reference of all command-line options accepted by Primbench benchmarks, including size suffixes, noise-reduction tuning, temperature control, and output formatting.
   :keywords: Primbench, CLI, command-line, options, ROCm, HIP, GPU, benchmark, settings

**********************
Command-line options
**********************

Primbench benchmarks accept command-line options that override the corresponding fields in the ``settings`` struct. When a value is set both programmatically (through ``settings``) and on the command line, the command-line value takes precedence.

For details on setting values programmatically before parsing the command line, see :doc:`Configure benchmark settings </how-to/configure-settings>`. For the ``settings`` struct and the rest of the benchmark API, see :doc:`Primbench API </reference/primbench-api>`.

Options reference
*****************

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
     - Input array size in bytes. Accepts ``KiB``, ``MiB``, or ``GiB`` suffixes, for example ``--size 256MiB``.

   * - ``--hot``
     - ``bool`` / ``false``
     - When set, the GPU cache is not cleared between batches.

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
     - Perform a dry run without executing kernels.

   * - ``--min-gpu-ms-per-batch``
     - ``double`` / ``10.0``
     - Minimum GPU time in milliseconds for each batch.

   * - ``--min-secs``
     - ``double`` / ``1.0``
     - Minimum total benchmark duration in seconds.

   * - ``--noise-timeout-secs``
     - ``double`` / ``10.0``
     - Maximum duration in seconds before a noisy benchmark times out.

   * - ``--batch-window-size``
     - ``size_t`` / ``10``
     - Number of recent batches used in the noise-reduction sliding window.

   * - ``--noise-tolerance-percent``
     - ``double`` / ``1.0``
     - Noise tolerance percentage for early stopping. Batching continues until the measurement noise falls below this threshold or the noise timeout is reached.

   * - ``--min-gpu-temp``
     - ``uint16_t`` / ``50``
     - Minimum GPU temperature (°C). The benchmark waits for the GPU to cool to or below this temperature before starting a specialization.

   * - ``--max-gpu-temp``
     - ``uint16_t`` / ``60``
     - Maximum GPU temperature (°C). The benchmark warms the GPU to at least this temperature before starting a specialization.

   * - ``--max-warming-secs``
     - ``double`` / ``60.0``
     - Maximum time in seconds to spend warming the GPU.

   * - ``--max-cooling-secs``
     - ``double`` / ``60.0``
     - Maximum time in seconds to spend cooling the GPU.

   * - ``--output-batches``
     - ``bool`` / ``false``
     - When set, per-batch details are included in the output.

   * - ``--spaces-per-indent``
     - ``uint32_t`` / ``4``
     - Number of spaces per indentation level in JSON output. Set to ``0`` for compact, unformatted JSON.

   * - ``--stream-blocking-timeout-secs``
     - ``double`` / ``10.0``
     - Maximum duration in seconds before stream blocking times out.

Size suffixes
*************

The ``--size`` option accepts an optional suffix to specify the unit:

- ``KiB`` — kibibytes (1024 bytes)
- ``MiB`` — mebibytes (1024 × 1024 bytes)
- ``GiB`` — gibibytes (1024 × 1024 × 1024 bytes)

When no suffix is provided, the value is interpreted as raw bytes.

Programmatic defaults and CLI precedence
****************************************

Each option corresponds to a field in the ``settings`` struct. You can set default values programmatically before passing ``settings`` to the executor; any value supplied on the command line overrides the programmatic default. See :doc:`Configure benchmark settings </how-to/configure-settings>` for examples of combining programmatic and command-line configuration.

For background on how the noise-reduction options (``--noise-timeout-secs``, ``--batch-window-size``, ``--noise-tolerance-percent``) interact, see :doc:`Noise reduction in primbench </conceptual/noise-reduction>`. For details on the JSON and CSV output controlled by ``--json-out``, ``--csv-out``, ``--output-batches``, and ``--spaces-per-indent``, see :doc:`JSON output format </reference/json-output-format>`.
