.. meta::
  :description: Using the MIOpen kernel cache
  :keywords: MIOpen, ROCm, API, documentation, kernel cache

********************************************************************
Kernel cache
********************************************************************

MIOpen caches binary kernels to disk so they don't need to be compiled the next time you run the
application. This cache is stored in ``$HOME/.cache/miopen`` by default, but you can change this at
build time by setting the ``MIOPEN_CACHE_DIR`` CMake variable.
At runtime, ``MIOPEN_CUSTOM_CACHE_DIR`` overrides this location. MIOpen appends a version-specific
subdirectory to the selected kernel cache directory.

.. _cache-storage:

Choosing cache and database storage
====================================================

Use low-latency, node-local storage for the writable kernel cache and user find/performance
databases. Network or parallel filesystems can make algorithm finds, compilation, and tuning
extremely slow, cause timeouts, and expose databases to corruption from unreliable locking.
The SQLite kernel cache uses write-ahead logging (WAL), which requires same-host shared-memory
coordination and is `not supported on network filesystems <https://www.sqlite.org/wal.html>`_.
Moving only the find/performance databases does not address a network-backed kernel cache.

Set both locations before starting the application, using paths on verified node-local storage:

.. code:: bash

   export MIOPEN_CUSTOM_CACHE_DIR="/local/scratch/$USER/miopen/cache"
   export MIOPEN_USER_DB_PATH="/local/scratch/$USER/miopen/db"

Replace ``/local/scratch`` with the appropriate local filesystem on your machine. Persistent local
storage preserves compiled kernels and tuning results between runs. Temporary storage such as
tmpfs or ``/tmp`` may be cleared, requiring subsequent runs to compile and tune again. ``/tmp`` and
``$TMPDIR`` are not guaranteed to be node-local; check their filesystem before using them.

On Linux, MIOpen warns when a selected user cache/database path is on a recognized network
filesystem, including explicitly configured paths. The warning identifies the path and the
environment variable used to relocate it. Explicit overrides remain unchanged.

In non-development builds, a recognized network-backed *default* kernel cache is redirected to
``temp_directory_path()/.cache/miopen/<version>``. The default user find/performance database path
is similarly redirected to ``temp_directory_path()/.config/miopen``. These redirects emit warnings
because the temporary caches and tuning results may not persist. Development builds warn without
redirecting; their kernel cache is disabled by default unless enabled at configuration time.
Warnings follow the normal ``MIOPEN_LOG_LEVEL`` setting and are emitted when the cached paths are
initialized, not on every database access.

Detection recognizes NFS, SMB/SMB2/CIFS, Coda, OCFS2, AFS, Lustre, Ceph, GPFS, BeeGFS, PanFS,
and 9p. Other filesystems, including network-backed FUSE mounts, might not be recognized.
Absence of a warning does not guarantee adequate storage performance or safe network sharing.
The warnings do not change SQLite's network-filesystem limitations.


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
