.. meta::
   :description: Describes the JSON and CSV output formats produced by Primbench, including the context object, specializations array, and compile-time branch and commit embedding.
   :keywords: Primbench, JSON, CSV, output format, results, ROCm, benchmark, specializations, context

*****************
Primbench output
*****************

Primbench writes benchmark results to a JSON file and, optionally, a CSV file. The JSON file records environment, configuration, and per-specialization measurements. CSV output provides a condensed tabular view.

JSON output
===========

By default, Primbench writes results to ``results.json``. The output path is controlled by ``--json-out`` and the ``settings.json_out`` field. The JSON file contains three top-level keys: ``context``, ``specializations``, and ``summary``.

The ``context`` object
-----------------------

The ``context`` object captures every detail of the environment and configuration used for the benchmark run. It contains four sub-objects: ``results_version``,  ``general``, ``settings``, and ``custom_settings``.

``results_version`` is the results schema version. This field appears at the top level of ``context``.

The ``general`` sub-object records information about the GPU, backend, monitoring library, and host:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Description
   * - ``algorithm``
     - Algorithm name shared by all queued specializations. Matches the ``algo`` key returned by ``meta()``.
   * - ``specialization_count``
     - Number of specializations queued on the executor.
   * - ``library_build_type``
     - | ``"release"`` when compiled with ``NDEBUG``
       | ``"debug"`` otherwise
   * - ``gpu``
     - GPU information. Object with ``name``, ``arch``, and ``pci_bus_id`` fields describing the active GPU.
   * - ``backend``
     - Backend used. Object with ``name``, version strings for the runtime and driver, a nested ``compiler`` object with ``name`` and ``version``, and a ``hip_version`` field.
   * - ``monitoring``
     - GPU monitoring. Object with ``name``and ``version``. Omitted when monitoring is disabled through ``-DPRIMBENCH_NO_MONITORING``.
   * - ``temperature_type``
     - GPU temperature sensor used. Omitted when monitoring is disabled.
   * - ``host_name``
     - Hostname of the machine.
   * - ``date``
     - Local timestamp in RFC 3339 format, ``yyyy-mm-ddTHH:MM:SS±HH:MM``.
   * - ``branch_name``
     - Present only when the ``BRANCH_NAME`` macro is defined at compile time.
   * - ``commit_hash``
     - Present only when the ``COMMIT_HASH`` macro is defined at compile time.

The ``settings`` sub-object is a verbatim serialization of the ``primbench::settings`` struct. It includes every configurable field in ``primbench::settings``, including ``size``, ``hot``, ``seed``, ``json_out``, ``csv_out``, ``filter``, ``dry``, ``min_gpu_ms_per_batch``, ``min_secs``, ``noise_timeout_secs``, ``batch_window_size``, ``noise_tolerance_percent``, ``min_gpu_temp``, ``max_gpu_temp``, ``max_warming_secs``, ``max_cooling_secs``, ``output_batches``, ``spaces_per_indent``, and ``stream_blocking_timeout_secs``. When a setting is provided both programmatically and on the command line, the command-line value takes precedence.


When the benchmark registers additional command-line arguments through ``executor.get<T>()``, a ``custom_settings`` object appears in ``context``. Each key is the argument name, and the value is the argument's parsed value. This object is omitted when no custom arguments are registered.

The ``specializations`` array
------------------------------

Each element in the ``specializations`` array corresponds to one queued specialization and contains the following fields:

.. list-table::
   :header-rows: 1
   :widths: 25 55

   * - Field
     - Description
   * - ``index``
     - Zero-based position of the specialization in the queue.
   * - ``name``
     - Display name derived from the ``meta()`` return value, excluding the ``algo`` key.
   * - ``bytes_per_second``
     - Measured byte throughput.
   * - ``items_per_second``
     - Measured item throughput.
   * - ``bytes_per_item``
     - Number of bytes transferred per item, reads plus writes.
   * - ``items``
     - Total number of items processed per kernel call.
   * - ``noise_timeout``
     - ``true`` if the run timed out before noise fell below the tolerance.
   * - ``noise_percent``
     - Final coefficient of variation across the last batch window, expressed as a percentage.
   * - ``meta``
     - The JSON object returned by the specialization's ``meta()`` method.
   * - ``elapsed_secs``
     - An object with ``host`` and ``gpu`` durations in seconds. ``host`` is wall time and ``gpu`` is device time.
   * - ``gpu_temp_celsius``
     - An object with ``start`` and ``end`` GPU temperatures in degrees Celsius. Present only when monitoring is available.
   * - ``calls``
     - An object with ``kernel_calls_per_batch``, ``ms_per_batch``, ``batches``, and ``kernel_calls``.
   * - ``batches``
     - An optional array of per-batch details. Present only when ``output_batches`` is true.

The ``summary`` object
-----------------------

The ``summary`` object provides aggregate statistics. The ``noise_timeouts`` field counts specializations that timed out due to noise. The ``elapsed_secs`` field is an object with ``host`` and ``gpu`` totals across all specializations.

CSV output
==========

CSV output is enabled when ``--csv-out`` is set to a file path. The CSV is a condensed view with one row per specialization. Columns are ``index``, ``name``, ``bytes_per_second``, ``gib_per_second``, ``items_per_second``, ``noise_timeout``, and ``noise_percent``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Column
     - Description
   * - ``index``
     - Zero-based specialization index.
   * - ``name``
     - Specialization display name, quoted in the file.
   * - ``bytes_per_second``
     - Measured byte throughput.
   * - ``gib_per_second``
     - Throughput converted to GiB per second.
   * - ``items_per_second``
     - Measured item throughput.
   * - ``noise_timeout``
     - ``0`` or ``1``.
   * - ``noise_percent``
     - Final noise percentage.

JSON output can be suppressed by setting ``--json-out`` to ``/dev/null`` while CSV output remains enabled.

Embedding branch name and commit hash
======================================

Defining the ``BRANCH_NAME`` and ``COMMIT_HASH`` macros at compile time will include them in ``context.general.branch_name`` and ``context.general.commit_hash``. These fields tie benchmark results to a specific source revision. In a detached HEAD state, such as a CI pipeline, ``BRANCH_NAME`` is typically set to ``DETACHED``. When the macros aren't defined, the corresponding fields are omitted from the JSON output.
