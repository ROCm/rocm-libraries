.. meta::
   :description: Sources of GPU benchmark noise and how Primbench addresses each through batch-based timing, stream blocking, thermal management, and cache clearing.
   :keywords: Primbench, GPU benchmarking, noise reduction, ROCm, HIP, batch timing, coefficient of variation, GPU warming, GPU cooling, cache clearing

********************************
Noise reduction in Primbench
********************************

GPU microbenchmarks are susceptible to several categories of measurement noise. Uncontrolled noise can mask real performance differences. For example, a benchmark with 10% noise cannot distinguish a 4% performance improvement from random variation. Primbench applies a layered set of techniques to reduce each noise source.

Batch-based timing
==================

Individual GPU kernel calls often complete in microseconds, and the time recorded for a single call varies substantially from one invocation to the next. Measuring noise across individual kernel calls amplifies variance rather than averaging it out.

Primbench groups kernel calls into batches and measures the total GPU time of each batch. Noise is computed across batches, not individual calls.

Dynamic batch sizing
--------------------

The number of kernel calls per batch is determined dynamically. Starting from a small count, Primbench doubles the number of kernel calls until the batch takes at least ``min_gpu_ms_per_batch`` milliseconds of GPU time. The threshold is configurable through ``--min-gpu-ms-per-batch``. This keeps each batch large enough to produce a stable timing signal regardless of kernel duration.

Batch window and stopping criterion
------------------------------------

Noise is measured as the coefficient of variation of GPU times across the most recent batches. The coefficient of variation is standard deviation divided by the mean. The number of batches in this sliding window is controlled by ``--batch-window-size``.

When the coefficient of variation drops below ``--noise-tolerance-percent`` and the benchmark has run for at least ``--min-secs`` seconds of wall time, Primbench stops the benchmark early. This avoids running longer than necessary once stable results have been reached.

Before timing the first batch, Primbench issues a single unmeasured kernel launch as a warmup so that the GPU caches the kernel instructions before real measurements begin.

Stream blocking for atomic event recording
==========================================

GPU event recording is asynchronous. The recorded start time might not correspond to the moment the kernel begins executing if other work is still being processed in the stream. This introduces timing noise.

Primbench addresses this by enqueuing the start event, the kernel, and the stop event as one atomic sequence. It briefly blocks GPU execution with a lightweight spinlock kernel, enqueues all three operations, and then releases the block:

.. code:: cpp

   block(stream);
   hipEventRecord(start, stream);
   kernel(stream);
   hipEventRecord(stop, stream);
   unblock(stream);
   elapsed = stop - start;

Because the spinlock holds the stream, no other work can interleave between the start event and the kernel launch.

Stream blocking requires the benchmarked kernel to be asynchronous. The kernel must return control to the host before finishing on the GPU. A synchronous kernel, such as one that calls device synchronization internally, causes the spinlock to deadlock. Constructing the executor with ``primbench::flags::sync`` disables stream blocking for synchronous kernels. The stream-blocking spinlock times out after ``--stream-blocking-timeout-secs`` seconds.

GPU warming and cooling
=======================

GPU clock frequencies shift with temperature. A cold GPU boosts above its sustained frequency, inflating early results. A hot GPU throttles, adding variance across batches. Both effects introduce noise.

Primbench keeps the GPU within a configurable temperature range before each batch:

- If the GPU temperature is below ``--min-gpu-temp``, short warmup workloads run until the temperature rises into range.
- If the GPU temperature exceeds ``--max-gpu-temp``, Primbench waits for the GPU to cool naturally.

If warming takes longer than ``--max-warming-secs`` or cooling takes longer than ``--max-cooling-secs``, Primbench aborts with an error.

Temperature sensor selection
-----------------------------

On AMD GPUs, temperatures are read through AMD SMI. Primbench probes available sensor types, ``edge`` and ``hotspot``, in priority order at startup and caches the first one that returns a reading. If no supported sensor can be read, Primbench terminates with an error. The selected sensor type is recorded as ``context.general.temperature_type`` in the JSON output.

.. note::

   GPU monitoring requires AMD SMI on HIP. When monitoring is unavailable, such as on Windows, temperature-based warming and cooling must be disabled by compiling with ``-DPRIMBENCH_NO_MONITORING``.

GPU cache clearing
==================

Residual data in GPU caches from a previous batch can artificially speed up the next batch, adding variance between the first batch and later ones.

To simulate a cold-cache scenario, Primbench writes to a large GPU buffer before each kernel launch, evicting cached data. The buffer size is controlled by the ``PRIMBENCH_GPU_CACHE_SIZE`` compile-time macro.

When ``settings.hot`` is true, cache clearing before kernel launches is skipped. This mode measures cache-warm performance. The ``hot`` setting is recorded in ``context.settings.hot`` in the JSON output.

Noise timeout
=============

Some benchmarks remain noisy even when applying noise reduction strategies. Primbench times out any benchmark that stays above the noise tolerance for longer than ``--noise-timeout-secs``. The timed-out specialization is flagged with ``noise_timeout: true`` in the JSON output, and the summary reports the total number of noise timeouts.
