.. meta::
   :description: JSON and CSV output formats produced by Primbench, including the json builder, context object, specializations array, and compile-time branch and commit embedding.
   :keywords: Primbench, JSON, CSV, output format, results, ROCm, benchmark, specializations, context, meta

******************************
Primbench JSON output format
******************************

Primbench writes benchmark results to a JSON file and, optionally, a CSV file. The JSON file records environment, configuration, and per-specialization measurements. CSV output provides a condensed tabular view.

JSON builder
============

The ``json`` struct is a lightweight JSON builder passed to ``benchmark_interface::meta()`` so that each benchmark specialization can describe itself with algorithm names, type names, and custom fields. These key-value pairs appear in the ``meta`` field of each specialization in the output file. Calls to ``add()`` can be chained, and nested ``json`` objects are supported.

.. doxygenstruct:: primbench::detail::json
   :members:

Results file structure
======================

By default, Primbench writes results to ``results.json``. The output path is controlled by ``--json-out`` and the ``settings.json_out`` field. The JSON file contains three top-level keys: ``context``, ``specializations``, and ``summary``.

The ``context`` object
----------------------

The ``context`` object captures every detail of the environment and configuration used for the benchmark run. It includes ``results_version``, ``general``, ``settings``, ``custom_settings`` when custom arguments are registered, and ``flags``.

``results_version``
~~~~~~~~~~~~~~~~~~~

A string indicating the results schema version, for example ``"4.0.0"``. This field appears at the top level of ``context``, not inside a sub-object.

``general``
~~~~~~~~~~~

The ``general`` sub-object records information about the GPU, backend, monitoring library, and host:

- The ``algorithm`` field holds the algorithm name shared by all queued specializations. It matches the ``algo`` key returned by ``meta()``.
- The ``specialization_count`` field records the number of specializations queued on the executor.
- The ``library_build_type`` field is ``"release"`` when compiled with ``NDEBUG`` defined and ``"debug"`` otherwise.
- The ``gpu`` field is an object with ``name``, ``arch``, and ``pci_bus_id`` fields describing the active GPU.
- The ``backend`` field is an object with a ``name`` of ``"hip"`` or ``"cuda"``, version strings for the runtime and driver, and a nested ``compiler`` object with ``name`` and ``version``. For HIP backends, a ``hip_version`` field is also present.
- The ``monitoring`` field is an object with ``name`` and ``version``. On HIP, ``name`` is ``"amdsmi"``. On CUDA, ``name`` is ``"nvml"``. This field is omitted when monitoring is disabled through ``-DPRIMBENCH_NO_MONITORING``.
- The ``temperature_type`` field names the GPU temperature sensor in use, for example ``"edge"`` or ``"hotspot"``. Omitted when monitoring is disabled.
- The ``host_name`` field records the hostname of the machine.
- The ``date`` field is a local timestamp in RFC 3339 format, ``yyyy-mm-ddTHH:MM:SS±HH:MM``.
- The ``branch_name`` field is present only when the ``BRANCH_NAME`` macro is defined at compile time.
- The ``commit_hash`` field is present only when the ``COMMIT_HASH`` macro is defined at compile time.

``settings``
~~~~~~~~~~~~

The ``settings`` sub-object is a verbatim serialization of the ``primbench::settings`` struct. It includes every configurable field in ``primbench::settings``, including ``size``, ``hot``, ``seed``, ``json_out``, ``csv_out``, ``filter``, ``dry``, ``min_gpu_ms_per_batch``, ``min_secs``, ``noise_timeout_secs``, ``batch_window_size``, ``noise_tolerance_percent``, ``min_gpu_temp``, ``max_gpu_temp``, ``max_warming_secs``, ``max_cooling_secs``, ``output_batches``, ``spaces_per_indent``, and ``stream_blocking_timeout_secs``. When a setting is provided both programmatically and on the command line, the command-line value takes precedence.

Field descriptions for each setting appear in :doc:`Configure benchmark settings </how-to/configure-settings>` and :doc:`Command-line options </reference/cli-options>`.

``custom_settings``
~~~~~~~~~~~~~~~~~~~

When the benchmark registers additional command-line arguments through ``executor.get<T>()``, a ``custom_settings`` object appears in ``context``. Each key is the argument name, and the value is the argument's parsed value. This object is omitted when no custom arguments are registered.

``flags``
~~~~~~~~~

The ``flags`` sub-object records which runtime flags were active. Currently the only flag is ``sync``, a boolean.

The ``specializations`` array
-----------------------------

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
----------------------

After the ``specializations`` array, a ``summary`` object provides aggregate statistics:

- The ``noise_timeouts`` field counts specializations that timed out due to noise.
- The ``elapsed_secs`` field is an object with ``host`` and ``gpu`` totals across all specializations.

CSV output
==========

CSV output is enabled when ``--csv-out`` is set to a file path. The CSV is a condensed view with one row per specialization. Columns are ``index``, ``name``, ``bytes_per_second``, ``gib_per_second``, ``items_per_second``, ``noise_timeout``, and ``noise_percent``.

- The ``index`` column holds the zero-based specialization index.
- The ``name`` column holds the specialization display name, quoted in the file.
- The ``bytes_per_second`` column holds measured byte throughput.
- The ``gib_per_second`` column holds throughput converted to GiB per second.
- The ``items_per_second`` column holds measured item throughput.
- The ``noise_timeout`` column is ``0`` or ``1``.
- The ``noise_percent`` column holds the final noise percentage.

JSON output can be suppressed by setting ``--json-out`` to ``/dev/null`` while CSV output remains enabled.

Embedding branch name and commit hash
=====================================

Defining the ``BRANCH_NAME`` and ``COMMIT_HASH`` macros at compile time causes Primbench to include them in ``context.general.branch_name`` and ``context.general.commit_hash``. These fields tie benchmark results to a specific source revision. In a detached HEAD state, such as a CI pipeline, ``BRANCH_NAME`` is typically set to ``DETACHED``. When the macros aren't defined, the corresponding fields are omitted from the JSON output. Compile-time embedding of branch and commit metadata is documented in :doc:`Configure benchmark settings </how-to/configure-settings>`.
