.. meta::
  :description: Using the MIOpen performance database
  :keywords: MIOpen, ROCm, API, documentation, performance database

************************************************************************************************
Using the performance database
************************************************************************************************

Many MIOpen kernels have parameters that affect their performance. Setting these parameters to
optimal values allows for the best possible throughput. The optimal values depend on many factors,
including the network configuration, GPU type, clock frequencies, and ROCm version.

Due to the large number of possible configurations and settings, MIOpen provides a set of pre-tuned
values for the "most applicable" network configurations and a method for expanding the set of
optimized values. MIOpen's performance database (PerfDb) contains these pre-tuned parameter values
in addition to any user-optimized parameters.

The PerfDb consists of two parts:

* **System PerfDb**: A system-wide storage that holds pre-run values for the most applicable
  configurations.
* **User PerfDb**: A per-user storage that holds optimized values for arbitrary configurations.

The User PerfDb *always takes precedence* over System PerfDb.

MIOpen also has auto-tuning functionality, which is able to find optimized kernel parameter values for
a specific configuration. The auto-tune process might take a long time, but after the optimized values are
found, they're stored in the User PerfDb. MIOpen then automatically reads and uses these parameter
values.

By default, System PerfDb resides within the MIOpen install location, while User PerfDb resides in your
home directory. For more information, see :ref:`setting up locations <setting-up-locations>`.

The System PerfDb is not modified during the MIOpen installation.

Auto-tuning kernels
==========================================================

MIOpen performs auto-tuning during the these API calls:

* ``miopenFindConvolutionForwardAlgorithm()``
* ``miopenFindConvolutionBackwardDataAlgorithm()``
* ``miopenFindConvolutionBackwardWeightsAlgorithm()``

Auto-tuning is performed for only one "problem configuration", which is implicitly defined by the
tensor descriptors that are passed to the API function.

In order for auto-tuning to begin, the following conditions must be met:

* The applicable kernels must have tuning parameters
* The value of the ``exhaustiveSearch`` parameter is set to ``true``
* Neither the System nor User PerfDb can contain values for the relevant "problem configuration".

You can override the latter two conditions and force the search using either the API call
``miopenSetTuningPolicy()`` or the ``-MIOPEN_FIND_ENFORCE`` environment variable. In addition to
controlling the auto-tuning behaviour of convolutions, both ``miopenSetTuningPolicy()`` and
``-MIOPEN_FIND_ENFORCE`` can be used to control the tuning for batch normalization.
See the following section for more details.

To optimize performance, MIOpen provides several find modes to accelerate find API calls.
These modes include:

*  normal find
*  fast find
*  hybrid find
*  dynamic hybrid find

For more information about the MIOpen find modes, see :ref:`Find modes <find_modes>`.

Using MIOPEN_FIND_ENFORCE or miopenSetTuningPolicy() to control auto-tuning
----------------------------------------------------------------------------------------------------------

``MIOPEN_FIND_ENFORCE`` supports case-insensitive symbolic and numeric values. The possible values
are:

* ``NONE``/``(1)``: No change in the default behavior.
* ``DB_UPDATE``/``(2)``: Do not skip auto-tune (even if PerfDb already contains optimized values). If you
  request auto-tune via API, MIOpen performs it and updates PerfDb. You can use this mode for
  fine-tuning the MIOpen installation on your system. However, this mode slows down the processes.
* ``SEARCH``/``(3)``: Perform auto-tune even if not requested via API. In this case, the library behaves as
  if the ``exhaustiveSearch`` parameter is set to ``true``. If PerfDb already contains optimized values,
  auto-tune is not performed. You can use this mode to tune applications that don't anticipate any means
  of getting the best performance from MIOpen. When in this mode, your application's first run might
  take substantially longer than expected.
* ``SEARCH_DB_UPDATE``/``(4)``: A combination of ``DB_UPDATE`` and ``SEARCH``. MIOpen performs
  auto-tune and updates User PerfDb on each ``miopenFindConvolution*()`` call. This mode is
  only recommended for debugging purposes.
* ``DB_CLEAN``/``(5)``: Removes optimized values related to the "problem configuration" from User
  PerfDb. Auto-tune is blocked, even if explicitly requested. System PerfDb is left intact.

  .. caution::

      Use the ``DB_CLEAN`` option with care.

Note that the API call miopenSetTuningPolicy() can be used to set the same modes as
``MIOPEN_FIND_ENFORCE``.  For example, to set the ``SEARCH`` mode, code like the following could be used:
.. code-block:: c

    miopenSetTuningPolicy(handle, miopenTuningPolicySearch);
    miopenBatchNorm*()
    miopenSetTuningPolicy(handle, miopenTuningPolicyNone);

Note that this API method is supported for both convolutions and batchnorms, although batchnorm does
not support a policy of ``DB_UPDATE`` (this will be a no-op and the user should specify ``SEARCH_DB_UPDATE``
instead if they want ``DB_UPDATE`` behavior).

If both the API method and environment variable are used, then the API method takes precedence.

Spatial batch normalization tuning
-----------------------------------

Packed NCHW BF16 forward training with FP32 parameters and batches of 8–64
includes split-batch spatial candidates. These shorten each thread's batch loop;
the existing untuned default is unchanged. Retune each shape and statistics mode
to compare the additional candidates against existing configurations.

On gfx1250, the single-kernel spatial Variant1 path omits its optional
full-chunk normalization barrier. Other architectures and the tail path retain
their synchronization. Retuning may still prefer the split-kernel Variant2 path.

gfx1250 wave32 also supports BF16 spatial vector8 candidates. These use 16-byte
input/output vectors while retaining FP32 normalization arithmetic and the
existing BF16 conversion behavior. Other devices retain their existing vector
choices. Spatial statistics for scalar-channel gfx1250 workgroups use compact
wave/block reductions instead of a full-workgroup LDS tree.

All batchnorm kernel builds propagate ``MIOPEN_USE_RNE_BFLOAT16`` from the library
configuration, including explicit zero for truncation. The mode is part of the
kernel compile options and cache key, so device output conversion agrees with
host verification.

For bounded gfx1250 FP16/BF16 packed-NCHW problems, spatial Variant4 retains input in
packed registers and computes centered local variance before merging partitions
with Welford's algorithm. It performs one global input read and one output write
per element, supports exact in-place execution, and retains the existing activation
and running/saved statistics behavior. Partial input/output overlap is unsupported.

The buffered path supports vector4/vector8 and 256/512/1024-thread groups with at
most 256 retained input bytes per thread, batches up to 64, and spatial size up
to 4800. The untuned vector8/512-thread default applies when NHW is at most 65536,
HW is at least 512, and there are at least 128 channels. Other shapes retain the
existing default; tuning can compare eligible buffered candidates.

gfx1250 wave32 backward tuning includes exact-divisor batch splits for packed
NCHW FP32 and mixed FP16/BF16 with batches of 8–64. Padded batch lanes contribute
zero to reductions and do not write input gradients. Saved-statistics problems
with batches of 16–64, at most 64 channels, and spatial size 4096–8192 use a
256-by-2 workgroup with a short batch loop by default. Smaller spatial problems
and other devices retain their existing defaults. Retune each saved/recomputed
statistics mode separately; batch splitting is not uniformly faster at higher
channel counts.

Saved-statistics backward Variant5 retains packed BF16 input and output gradients
before reducing parameter gradients and writing input gradients. It supports exact
DX=X or DX=DY aliasing, but not partial overlap, and uses at most 128 total retained
BF16 values per thread. Eligible problems with at least 128 channels and HW at least
512 use vector8/1024 threads by default.

Forward Variant6 streams three vectorized passes rather than retaining a large
input array: relative mean, centered local variance, then normalization. Its
Welford merge preserves stable variance for high-offset inputs. The measured
vector8/1024-thread default applies to C=128–256, HW=4096–8192, and
NHW=65537–262144. Other shapes retain the buffered or split-kernel defaults.
Backward vector8 split-kernel candidates remain tuning-only.

Saved-statistics backward Variant8 streams parameter-gradient accumulation and
input-gradient normalization in two passes without retained-input arrays.
It preserves FP32 activation/normalization and BF16 conversion, supports exact
DX=X or DX=DY aliasing, and uses the vector8/1024-thread default in the same
medium-spatial range as forward Variant6. Low-channel forward problems with
batches of 16–64 and HW=4096–8192 use finer 64-by-2 spatial/batch workgroups.

On gfx1250 saved-statistics backward paths, immutable statistics and parameters
are loaded directly rather than broadcast through shared memory. Recomputed
statistics retain stash-read synchronization and their existing behavior.

The wave32 scalar-channel reduction is implemented once in the shared reduction
helper and reused by spatial forward/backward statistics and parameter-gradient
kernels. Channel-vector reductions retain their layout-specific LDS path.
Saved-parameter direct loads support FP32 and mixed FP16/BF16 in both NCHW and NHWC.

Per-activation forward training on gfx1250 retains packed FP16/BF16 input for
batches up to 64, preserving centered variance, batch-order accumulation and
output conversion while avoiding repeated input reads. Larger batches and FP32
retain their existing path. Fused inference in both normalization modes splits
batch work across the z grid using the same occupancy heuristic as standalone
spatial inference; the final partial batch is covered by the grid-stride loop.

Search ranks a filtered mean of short candidate measurements, whereas the driver
also reports minimum steady-state latency. Its warm-up timing is included in the
score, and forward batchnorm search omits running-statistics updates. Consequently,
a fresh search is not guaranteed to improve an existing database entry in a later
benchmark. Compare repeated runs with the same statistics mode before replacing
validated records. For closer candidate sampling, ``MIOPEN_TUNING_ITERATIONS``
increases the sample count and ``MIOPEN_TUNING_FOLLOWUP_TOLERANCE_PCT`` widens the
gate for additional sampling; both increase tuning work.

For example, use an isolated user database and force a fresh search:

.. code-block:: bash

    MIOPEN_USER_DB_PATH=/tmp/miopen-bn-tuning \
    MIOpenDriver bnormbfp16 -n 42 -c 128 -H 30 -W 40 -m 1 \
        --forw 1 -b 0 -r 1 -s 1 --layout NCHW -t 1 -i 100 -V 1 \
        --tuning_policy 4

Reuse that database without ``--tuning_policy 4`` for subsequent measurements.
Running and saved statistics are part of the tuning key. ``--forw 2`` without
``-r 1`` recalculates statistics through the training kernel; it does not measure
inference with precomputed statistics.

Updating MIOpen and User PerfDb
==========================================================

If you install a new version of MIOpen, it is recommended that you move or delete your old User
PerfDb file. This prevents older database entries from affecting configurations within the newer system
database. The User PerfDb is named ``miopen.udb`` and can be found at the User PerfDb path location.
