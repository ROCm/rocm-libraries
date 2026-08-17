.. meta::
   :description: CSV output format produced by Primbench, including column descriptions.
   :keywords: Primbench, CSV, output format, results, ROCm, benchmark

****************************
Primbench CSV output format
****************************

Primbench can write a CSV file alongside JSON results. CSV output is enabled when ``--csv-out`` is set to a file path. The CSV is a condensed view with one row per specialization.

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
