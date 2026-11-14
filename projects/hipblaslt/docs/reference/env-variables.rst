.. meta::
  :description: hipBLASLt environment variables reference
  :keywords: hipBLASLt, ROCm, API, environment variables, environment, reference

.. _environment-variables:

********************************************************************
hipBLASLt environment variables
********************************************************************

This section describes the important hipBLASLt environment variables,
which are grouped by functionality.

Logging and debugging
=====================

The logging and debugging environment variables for hipBLASLt are collected in the following table.
For more information, see :doc:`Use logging and heuristics <../how-to/use-logging-heuristics>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``HIPBLASLT_LOG_LEVEL``
        | Controls the verbosity level of hipBLASLt logging output.
        | Levels are cumulative: each one also enables the levels below it.
      - | 0: Off (logging disabled, default)
        | 1: Error (only errors are logged)
        | 2: Trace (API calls with kernel launches log parameters)
        | 3: Hints (performance improvement suggestions)
        | 4: Info (general library execution information)
        | 5: API trace (detailed API call parameters)

    * - | ``HIPBLASLT_LOG_MASK``
        | Controls logging output using bit mask flags (can be combined).
        | Consulted only when ``HIPBLASLT_LOG_LEVEL`` is unset.
      - | 0: Off
        | 1: Error
        | 2: Trace
        | 4: Hints
        | 8: Info
        | 16: API trace
        | 32: Bench
        | 64: Profile
        | 128: Extended profile

    * - | ``HIPBLASLT_LOG_FILE``
        | Specifies path to logging file. Can contain ``%i`` for process ID replacement.
        | Has no effect unless a level or mask has enabled logging.
      - | Path to log file (for example, ``logfile_%i.log``)
        | If not defined: log messages printed to stderr

    * - | ``HIPBLASLT_ENABLE_MARKER``
        | Enables marker trace for ROCProfiler profiling.
      - | 0 or unset: Disable marker trace
        | 1: Enable marker trace


Offline tuning
===============

The offline tuning environment variables for hipBLASLt are collected in the following table.
For more information, see :doc:`Use hipBLASLt offline tuning <../how-to/how-to-use-hipblaslt-offline-tuning>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``HIPBLASLT_TUNING_FILE``
        | Specifies file to store tuning results with best solution indices for GEMM problems.
      - | Path to tuning file (for example, ``tuning.txt``)
        | File stores optimal kernel indices for reuse

    * - | ``HIPBLASLT_TUNING_OVERRIDE_FILE``
        | Specifies file to load tuning results and override default kernel selection.
      - | Path to tuning file (for example, ``tuning.txt``)
        | Loads previously saved optimal kernel choices

    * - | ``HIPBLASLT_TUNING_USER_MAX_WORKSPACE``
        | Sets maximum workspace size constraint during tuning stage.
      - | Integer value in bytes (default: 128 * 1024 * 1024)
        | Limits workspace size for solution selection

Runtime tuning
==============

Runtime tuning is opt-in and off by default. ``HIPBLASLT_TUNING_MODE`` and
``HIPBLASLT_TUNING_CACHE_PATH`` are read the first time they are needed in a process, so set them
before the first hipBLASLt call. The scratch cap is read on the first scratch allocation. A process
running in a secure execution context (set-user-ID, set-group-ID, or another credential-changing exec
such as file capabilities) ignores every variable in this section, so tuning stays off. For more
information, see :doc:`Use hipBLASLt offline tuning <../how-to/how-to-use-hipblaslt-offline-tuning>`.

``online`` mode never waits on the GPU. It dispatches each of the top
``HIPBLASLT_TUNING_MAX_CANDIDATES`` in turn ``HIPBLASLT_TUNING_HOT_ITERS`` times on the caller's own
calls, reads each timing back on a later call under the same key, and records the fastest. Like
``tune``, it tunes each cache key once and writes one row for it. Calls that share a key but differ
in what it leaves out, such as beta, run default selection while the first of them is explored, and
replay its row afterwards. A shape the cache file already serves is replayed, not explored. Exploring
calls share a process-wide lock.

Each key spends ``HIPBLASLT_TUNING_COLD_ITERS`` untimed calls, then ``HIPBLASLT_TUNING_MAX_CANDIDATES``
times ``HIPBLASLT_TUNING_HOT_ITERS`` timed calls exploring, which may run slower candidates, and holds
memory proportional to the timed calls until it is done; values set for ``tune`` apply to ``online``
too. The startup notice shows the values ``online`` uses as ``K``, ``R`` and ``cold``.

While a key is explored, which takes about ``HIPBLASLT_TUNING_MAX_CANDIDATES`` times
``HIPBLASLT_TUNING_HOT_ITERS`` calls, consecutive calls of a problem run different kernels, so their
outputs can differ bitwise. Those calls are also slower: they run lower-ranked candidates, which can
be much slower than the kernel default selection picks. The call that records a key's winner appends
its row to the file synchronously, as ``tune`` does. For reproducible results or steady latency, run
with a cache file that already serves the shapes.

A call is not timed while its thread holds a HIP error it has not read, or when it launches on a
stream of a device other than the thread's current device. It still runs, but a key explored only by
such calls can end with no row.

``online`` mode measures real launches. Settings that serialize or skip launches, such as
``AMD_SERIALIZE_KERNEL``, ``HIP_LAUNCH_BLOCKING``, a profiler, or a ``TENSILE_DB`` setting that skips
launches, distort its timings, so do not use ``online`` mode under them.

Each process explores and records its own keys, so give each rank or process its own cache file, or
share a file that already serves the shapes. Processes that share a path each explore and append
rows of their own.

``online`` mode tunes only ``hipblasLtMatmul`` calls that pass no algorithm (``algo`` is
``nullptr``). Nothing else is tuned online:

* ``hipblasLtMatmul`` with an algorithm the caller supplies launches it as given, untimed.
* ``hipblasLtMatmulAlgoGetHeuristic`` is never handed a candidate to try. It is answered from the
  cache file, as in ``cache`` mode, and otherwise by its usual ranking.
* The ``hipblaslt_ext`` GEMM and grouped GEMM interfaces, and pointer-array batched problems.
* Calls with a ``HIPBLASLT_ADAPTIVE_SM_COUNT`` hint, which run as with ``online`` off.
* Fused GEMM and all-to-all calls, which write to their peer ranks, as in ``tune`` mode.
* Calls dispatched to rocRoller: block-scaled (MX) problems by default, or every problem with
  ``HIPBLASLT_USE_ROCROLLER=1``. They use rocRoller's own selection, as in ``tune`` mode.

A call captured into a HIP graph is never explored or timed, and gets the measured winner if there
is one. Exploring calls leave a capture in progress valid in every capture mode, whether it is on
their own thread or another.

The winner is appended to ``HIPBLASLT_TUNING_CACHE_PATH`` in the row format ``cache`` replays,
marked incomplete if a candidate could not be timed. The row records the mode that measured it, and
``tune`` never treats an ``online`` row as covering its own search. When a file holds several rows for
a key, replay prefers a complete row to an incomplete one, and the newest within each, so a complete
``online`` row is replayed ahead of a newer incomplete ``tune`` row. ``online`` mode does not read
``HIPBLASLT_TUNING_ALL_KERNELS``, ``HIPBLASLT_TUNING_ROTATING_MB``, ``HIPBLASLT_TUNING_FLUSH_ICACHE``,
``HIPBLASLT_TUNING_BUDGET_MS_PER_SHAPE`` or ``HIPBLASLT_TUNING_SCRATCH_MAX_BYTES``.

``cache``, ``tune`` and ``online`` mode write a few notices without any logging variable. For
``tune``, whose search can block the first call on a new shape for minutes, they keep a silent pause
from looking like a hang. The notices are bounded: one line naming the mode and what loaded, one start
and one result per shape that ``tune`` actually tunes, one result per key that ``online`` tunes (it
prints no start line), and one closing summary. Replaying a cache adds no output per call. Where the
notices go depends on logging:

* No level or mask: stderr. ``HIPBLASLT_LOG_FILE`` alone does not open a log file.
* ``HIPBLASLT_LOG_LEVEL`` 1 to 3, or a mask without the info bit: the stream logging already opened,
  including ``HIPBLASLT_LOG_FILE``.
* ``HIPBLASLT_LOG_LEVEL=4`` or higher, or ``HIPBLASLT_LOG_MASK`` including ``8``: the same stream,
  formatted like every other log line, plus a cache hit, miss or invalidation line once per problem,
  the scratch and candidate setup, a progress heartbeat, and measurement results.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``HIPBLASLT_TUNING_MODE``
        | Selects runtime tuning behavior.
      - | ``off``: Disable runtime tuning (default)
        | ``cache``: Replay valid entries from the cache file
        | ``tune``: Benchmark uncached supported problems and append winners
        | ``online``: Time top ranked candidates on live ``hipblasLtMatmul`` calls that pass no
          algorithm, and record the winner

    * - | ``HIPBLASLT_TUNING_CACHE_PATH``
        | Specifies the runtime cache file.
      - | Path to a tuning file
        | Required for ``cache``, ``tune``, and ``online`` modes

    * - | ``HIPBLASLT_TUNING_ALL_KERNELS``
        | Selects exhaustive or ranked-prefix candidate enumeration.
      - | 1: Enumerate every candidate (default)
        | 0: Use a ranked prefix

    * - | ``HIPBLASLT_TUNING_MAX_CANDIDATES``
        | Limits the ranked prefix when exhaustive enumeration is disabled.
      - | Positive integer (``tune`` default: 128, ``online`` default: 5)
        | Below 2 switches ``online`` mode off

    * - | ``HIPBLASLT_TUNING_COLD_ITERS``
        | Sets untimed warm-up launches per candidate (``online``: untimed calls under a key before
          exploration starts).
      - | Non-negative integer (``tune`` default: 1000, ``online`` default: 3)
        | 0 disables warm-up
        | ``online`` runs default selection's pick on these calls, so clocks settle at no extra cost

    * - | ``HIPBLASLT_TUNING_HOT_ITERS``
        | Sets timed launches per candidate.
      - | Positive integer (``tune`` default: 1000, ``online`` default: 3)

    * - | ``HIPBLASLT_TUNING_MAX_EVENT_PAIRS``
        | Caps the HIP event pairs ``online`` mode keeps per device; a call that finds every pair
          in flight runs untimed.
      - | Positive integer (default: 4096)

    * - | ``HIPBLASLT_TUNING_MIN_IMPROVEMENT_PCT``
        | ``online`` only: how much faster than default selection's pick, in percent, another
          candidate must measure to be recorded instead of it.
      - | Integer from 0 to 100 (default: 0, the fastest candidate wins)

    * - | ``HIPBLASLT_TUNING_ROTATING_MB``
        | Sets the target rotating-buffer footprint.
      - | MiB (default: 512)
        | 0 disables rotation

    * - | ``HIPBLASLT_TUNING_FLUSH_ICACHE``
        | Invalidates the instruction cache between timed launches, matching the bench client.
      - | 1: Flush (default); costs about 5% of tuning time
        | 0: Do not flush; winners become less reproducible

    * - | ``HIPBLASLT_TUNING_BUDGET_MS_PER_SHAPE``
        | Sets a soft wall-clock limit on one shape's search, checked between candidates.
      - | Milliseconds (default: 300000, five minutes; 0 is unlimited)
        | A truncated search records its best candidate as incomplete
        | A single candidate can overrun the limit

    * - | ``HIPBLASLT_TUNING_SCRATCH_MAX_BYTES``
        | Caps library-owned tuning scratch per device.
      - | Bytes (default: 1 GiB)

Origami with Stream-K configuration
===================================

The Origami with Stream-K configuration environment variables for hipBLASLt are collected in the following table.
These variables apply to all GEMMs in an application.
For more information, see :doc:`Use Stream-K with hipBLASLt <../how-to/how-to-use-streamk>`.

The ``TENSILE_PERSISTENT_*`` launch controls apply to persistent kernels, including Stream-K and persistent DataParallel.
The legacy aliases listed below remain supported. When both names are set, the preferred name takes precedence,
even if its value is invalid; it does not fall back to the legacy value.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``TENSILE_SOLUTION_SELECTION_METHOD``
        | Controls hipBLASLt kernel selection strategy for GEMM operations.
      - | 0: Default (standard tuned libraries, no Stream-K)
        | 2: Origami with Stream-K (enables Origami solution selection for consistent performance)
        | This variable has no effect on the AMD Instinct™ MI350 series. Stream-K is always used.

    * - | ``TENSILE_PERSISTENT_DYNAMIC_GRID``
        | Legacy alias: ``TENSILE_STREAMK_DYNAMIC_GRID``.
        | Controls persistent grid size selection behavior.
      - | 0: Disable dynamic grid (use all available compute units)
        | 1: Only reduce compute units for small problems to the number of output tiles when ``num_tiles < CU count``
        | 2: Also reduce compute units for large sizes to improve the data-parallel portion and reduce power
        | 3: Analytically predict the best grid size by weighing the cost of the fix-up step and the cost of processing MACs 
        | 4: The Stream-K algorithm behaves like data parallel (Launch WGs =  number of CUs)
        | 5: The Stream-K algorithm uses the Origami ``select_best_grid_size`` function
        | 6: Default (automatically pick the optimal workgroup count)

    * - | ``TENSILE_GRIDBASED_KDTREE``
        | Force-enables the k-d tree index on grid-based solution-selection tables, which are otherwise scanned linearly.
        | Enable-only: it can switch the index on for tables that do not ask for it, but it can never switch it off. A table that declares ``UseKdTree: true`` in its library-logic file already uses the index with this variable unset, and no value of this variable changes that.
        | Affects which kernel is selected, not the result it computes.
      - | Unset or 0: Each grid-based table uses the ``UseKdTree`` value declared in its library-logic file. Tables that do not declare it are scanned linearly.
        | Non-zero: Enable the index for every grid-based table, in addition to those already declaring ``UseKdTree: true``.
        | To disable the index for a table that declares ``UseKdTree: true``, remove the key (or set it to ``false``) in that table's library-logic file and rebuild the device library. This cannot be done at runtime.

    * - | ``TENSILE_PERSISTENT_FIXED_GRID``
        | Legacy alias: ``TENSILE_STREAMK_FIXED_GRID``.
        | Overrides the default persistent grid size, subject to the kernel's launch limits.
      - | Positive integer specifying the number of workgroups
        | Default: 0 (no override)
        | Example: 64 (limits GEMM kernels to 64 workgroups)

    * - | ``TENSILE_PERSISTENT_MAX_CUS``
        | Legacy alias: ``TENSILE_STREAMK_MAX_CUS``.
        | Sets the compute-unit budget used for persistent grid sizing.
      - | Positive integer specifying the compute-unit budget
        | Example: 32
        | Default: 0 (all available compute units)

    * - | ``TENSILE_PERSISTENT_GRID_MULTIPLIER``
        | Legacy alias: ``TENSILE_STREAMK_GRID_MULTIPLIER``.
        | Multiplies the compute-unit count to size the persistent grid when dynamic grid selection and the CU cap are disabled and no fixed grid is set.
        | The resulting grid remains subject to the kernel's launch limits.
      - | Integer greater than 1 to enable multiplication
        | Default: 1

    * - | ``TENSILE_PERSISTENT_DYNAMIC_WGM``
        | Legacy alias: ``TENSILE_STREAMK_DYNAMIC_WGM``.
        | Retained for compatibility; currently has no effect on workgroup mapping.
      - | Default: 0

    * - | ``TENSILE_PERSISTENT_HYBRID_FORCE_MODE``
        | Legacy alias: ``TENSILE_STREAMK5_FORCE_MODE``.
        | Debug override for the mode of an already selected Hybrid kernel; does not select the kernel family.
      - | -1: Respect the normal assignment policy (default)
        | 0: Force static assignment
        | 1: Force dynamic work-queue assignment
        | Malformed or out-of-range values are ignored.

Adaptive compute-unit target
============================

The adaptive ``sm_count_target`` environment variables for hipBLASLt are collected in the following table.
They are read once per process. Boolean values accept ``1``, ``on``, ``true``, ``0``, ``off``, and ``false``
(case-insensitive). Invalid or out-of-range values keep the default and log a warning.
For requirements and behavior, see :ref:`Adapting to the available compute units <adaptive-sm-count>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``HIPBLASLT_ADAPTIVE_SM_COUNT``
        | Measures the compute units available to GEMMs and uses the count as an implicit ``sm_count_target``.
        | Applies to C API calls with Stream-K tile scheduling ``AUTO`` or ``ON`` and no explicit ``sm_count_target``.
        | The hint only moves when the launched kernels carry the occupancy probe (a probe-built device library and gfx950 Stream-K Hybrid kernels); otherwise it stays 0.
      - | 0: Off (default)
        | 1: On

    * - | ``HIPBLASLT_ADAPTIVE_SM_COUNT_PROBE_EVERY``
        | Launches per stream between probes.
      - | Integer, 1 or greater
        | Default: 8

    * - | ``HIPBLASLT_ADAPTIVE_SM_COUNT_TINY_MFLOPS``
        | GEMMs below this many MFLOPs (``2*M*N*K*batch / 10^6``) never probe.
      - | Non-negative integer
        | Default: 256

    * - | ``HIPBLASLT_ADAPTIVE_SM_COUNT_PROBE_WINDOW``
        | Number of latest probe readings that decide the hint. The hint changes only when all of them are below it or all are above it, and then takes the smallest reading. No hint counts as the device's CU count. A single reading of the device's CU count clears the hint at once, and a new hint then needs a full set of readings below that count.
      - | Integer from 1 to 16
        | Default: 3

    * - | ``HIPBLASLT_ADAPTIVE_SM_COUNT_LOG``
        | Logs adaptive mode activity to ``HIPBLASLT_LOG_FILE``, or stderr if unset.
      - | 0: Off (default)
        | 1: Configuration, hint changes, re-selections, and failed probe events
        | 2: Also every probe reading

    * - | ``HIPBLASLT_ADAPTIVE_SM_COUNT_FORCE``
        | Uses a fixed CU count as the hint instead of measuring it. A positive value enables adaptive mode on its own.
      - | Integer from 0 to 65535
        | Default: 0 (measure)

    * - | ``HIPBLASLT_ADAPTIVE_SM_COUNT_FORCE_PROBE``
        | With ``HIPBLASLT_ADAPTIVE_SM_COUNT_FORCE``, still runs the probes and discards their readings, so timing includes the probe cost.
        | Ignored with a warning without ``HIPBLASLT_ADAPTIVE_SM_COUNT_FORCE``.
      - | 0: Off (default)
        | 1: On

.. _env-type_overrides:

Type overrides
======================

Overrides for specific types.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32``
        | Overrides the compute type used for GEMMs which specify a compute type of ``XF32``.
      - | -1: Off (Default)
        | 0: F32
        | 1: XF32(eg TF32)
        | 2: F32_BF16
