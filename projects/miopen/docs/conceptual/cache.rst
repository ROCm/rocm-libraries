.. meta::
  :description: Using the MIOpen kernel cache
  :keywords: MIOpen, ROCm, API, documentation, kernel cache

********************************************************************
Kernel cache
********************************************************************

MIOpen caches binary kernels to disk so they don't need to be compiled the next time you run the
application. This cache is stored in ``$HOME/.cache/miopen`` by default, but you can change this at
build time by setting the ``MIOPEN_CACHE_DIR`` CMake variable.

Clear the cache
====================================================

You can clear the cache by deleting the cache directory (for example, ``$HOME/.cache/miopen``). However,
you should only do this for development purposes or to free disk space. You don't need to
clear the cache when upgrading MIOpen.

Disabling the cache
====================================================

Disabling the cache is generally useful for development purposes. You can disable the cache
in this following situations:

*  During the build, either set ``MIOPEN_CACHE_DIR`` to an empty string or set
   ``BUILD_DEV=ON`` when configuring CMake.
*  At runtime, set the ``MIOPEN_DISABLE_CACHE`` environment variable to ``true``.

Kernel cache writes during tuning
====================================================

Outside of tuning, MIOpen commits each newly compiled kernel to the user kernel cache in its own
SQLite transaction. During tuning (``GenericSearch``) and during the parallel precompilation step
of Find, MIOpen compiles many kernels in a row, so it groups their cache writes into bounded
transactions instead:

* Each compiled kernel is compressed and hashed as usual, then held in memory until its group is
  committed. The kernel is available to the same process immediately, without recompilation.
* A group is committed in one transaction as soon as it holds 64 kernels or 32 MiB of compressed
  code, or its oldest kernel has waited for two seconds, and when the tuning or precompilation step
  ends. The transaction contains only the database writes, never kernel compilation.
* Other processes see the kernels of a group once it's committed.
* If a commit fails, the whole group is rolled back and discarded, and the error is reported.
  Kernels committed earlier stay in the cache, and discarded kernels are compiled again the next
  time they're needed, as with any cache miss.

Batching reduces the number of commits and the file synchronization calls that come with them,
which matters most when the cache is on a network file system. It doesn't change the SQLite
journal or synchronization settings, so each commit is as durable as before.

You can change the limits with the ``MIOPEN_DEBUG_KERNEL_CACHE_BATCH_MAX_RECORDS``,
``MIOPEN_DEBUG_KERNEL_CACHE_BATCH_MAX_BYTES``, and ``MIOPEN_DEBUG_KERNEL_CACHE_BATCH_MAX_AGE_MS``
environment variables. Setting ``MIOPEN_DEBUG_KERNEL_CACHE_BATCH_MAX_RECORDS=1`` commits every
kernel on its own.

Updating MIOpen and removing the cache
===============================================================

For MIOpen version 2.4 and later, MIOpen's kernel cache directory is versioned, so any existing cached kernels
won't collide when upgrading.

.. note::

   For MIOpen version 2.3 and earlier, if the compiler changes or you modify the kernels, then you must
   delete the cache for the existing MIOpen version using the command
   ``rm -rf $HOME/.cache/miopen/<miopen-version-number>``.

Installing precompiled kernels
====================================================

GPU architecture-specific, precompiled kernel packages are available in the ROCm package
repositories. These packages reduce the startup latency of MIOpen kernels. They contain a kernel cache file,
which they install in the ROCm installation directory along with other MIOpen artifacts. When MIOpen launches a
kernel, it first checks for a kernel in the kernel cache within the MIOpen installation directory. If
the file doesn't exist, or the required kernel isn't found, it compiles the kernel and places it in the
kernel cache.

These packages are optional and must be separately installed from MIOpen. To conserve
disk space, you can choose not to install these packages (which would result in higher
startup latency). You also have the option to only install kernel packages for your device architecture,
which helps save disk space.

If the MIOpen kernels package is not installed, or if the kernel doesn't match the GPU, you'll get a
warning message similar to:

.. code:: bash

   > MIOpen: Warning [SQLiteBase] Missing system database file:gfx906_60.kdb Performance may degrade

The performance degradation mentioned in the warning only affects the network start-up time (the
"initial iteration time") and can be safely ignored.

Refer to the :doc:`installation instructions <../install/install>` for guidance on installing the MIOpen
kernels package.
