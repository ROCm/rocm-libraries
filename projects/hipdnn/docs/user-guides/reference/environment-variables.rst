.. meta::
  :description: Learn about the environment variables used by hipDNN for logging, plugins, tests, and more.
  :keywords: hipDNN, ROCm, environment, API, variables

.. _variables:

****************************
hipDNN environment variables
****************************

Learn about the environment variables used by hipDNN for logging, plugins, tests, and more.

Logging configuration
=====================

hipDNN provides two environment variables to control logging behavior:

- ``HIPDNN_LOG_LEVEL``
- ``HIPDNN_LOG_FILE``

``HIPDNN_LOG_LEVEL``
--------------------

Sets the minimum severity for which logs will be emitted. Levels are inclusive: choosing a level enables messages at that level and all higher severities.

- ``off``: Disables all logging (default)
- ``info``: General informational messages
- ``warn``: Potential issues that do not interrupt execution
- ``error``: Recoverable errors that may affect results or performance
- ``fatal``: Unrecoverable errors; the operation will not continue

Here's an example:

.. code:: bash

  export HIPDNN_LOG_LEVEL=info

``HIPDNN_LOG_FILE``
-------------------

Specifies the file path where logs will be appended. If this variable isn't set, logs are written to ``stderr``.

Here's an example:

.. code:: bash

  export HIPDNN_LOG_FILE=/path/to/hipdnn.log

.. tip::

  When using the MIOpen legacy plugin, you can use MIOpen-specific environment variables to control the underlying library's logging behavior.


.. _plugin-loading-variables:

Plugin loading
==============

The following environment variable can be used to control which folders hipDNN will scan for plugins to load.

``HIPDNN_PLUGIN_DIR``
---------------------

By default, hipDNN loads plugins from ``./hipdnn_plugins/engines/``.
This path is relative to the hipDNN backend shared library location in the ROCm install folder, typically ``/opt/rocm/lib/`` on Linux.

Default structure example (Linux):

.. code::

  /opt/rocm/lib/
  └── hipdnn_plugins/
      └── engines/
          ├── miopen_plugin.so
          └── other_plugin.so

When ``HIPDNN_PLUGIN_DIR`` is set, hipDNN *only* loads plugins from the specified directory and supplementary custom paths, ignoring the default location.
This allows complete control over which plugins are loaded.

.. code:: bash

  # Load plugins from a custom directory
  export HIPDNN_PLUGIN_DIR=/path/to/test/plugins

Path resolution
~~~~~~~~~~~~~~~

The ``HIPDNN_PLUGIN_DIR`` paths can be:

- **Relative paths**: Resolved from the backend shared library location (typically ``/opt/rocm/lib`` on Linux, or ``C:\TheRock\bin`` on Windows if ROCm is installed to ``C:\TheROck``).

  - For example, if ``HIPDNN_PLUGIN_DIR`` is set to ``./test_plugins``, then hipDNN tries to load all plugins from ``/opt/rocm/lib/./test_plugins``.

- **Absolute paths**: Used as specified.

For both relative paths and absolute paths:

- If the path specifies a folder, hipDNN tries to load all ``.so`` files (Linux) or ``.DLL`` files (Windows) from that folder as plugins.
- If the path specifies a filename ending in ``.so`` (Linux) or ``.DLL`` (Windows), then only that plugin will be loaded.
- If the path specifies a filename without an extension, hipDNN prefixes the filename with ``lib`` and adds the ``.so`` suffix (Linux), or adds the ``.DLL`` suffix (Windows) and only loads that file.

See :ref:`plugin-loading` for API functions that provide additional control over which folders plugins are loaded from.

.. _backend-library-variables:

Backend library discovery
=========================

A consumer that links ``hipdnn_frontend_dynamic`` resolves the hipDNN backend shared library at first use rather than through a link-time dependency.
hipDNN computes the path itself, in this order:

#. ``hipdnn_frontend::setBackendLibraryPath_ext()`` if the calling module has called it, otherwise ``HIPDNN_BACKEND_LIBRARY_PATH``.
#. The directory of the calling module (executable or shared library).
#. That directory's sibling ``../lib`` and ``../lib64``.
#. The directory the HIP runtime was loaded from.
#. The bare library name, left to the system loader.

Locations 2 and 3 require the calling module's own directory.
When the loader reports no origin for it, or one that cannot be trusted -- a relative name, which would be interpreted against whatever working directory the process happens to have -- both are skipped and resolution continues with the remaining locations.

.. warning::

  On Windows, whichever location supplied the directory the backend is loaded from -- the setter, ``HIPDNN_BACKEND_LIBRARY_PATH``, the calling module's directory, its sibling ``../lib`` or ``../lib64``, or the HIP runtime's directory -- is searched for the backend's own first-level dependents ahead of ``System32``, so it must not be writable by lower-privileged principals.
  The altered order does not apply transitively to those dependents' own dependencies, and KnownDLLs still resolve from the system directory.

``HIPDNN_BACKEND_LIBRARY_PATH``
-------------------------------

The directory holding the backend shared library. The filename is always hipDNN's own -- ``libhipdnn_backend.so`` on Linux, ``hipdnn_backend.dll`` on Windows -- so this variable selects a location, never a particular file.

.. code:: bash

  export HIPDNN_BACKEND_LIBRARY_PATH=/opt/rocm/lib

The value must be a non-empty absolute directory; any other value is reported on ``stderr`` and ignored.
If the backend is absent or fails to load, resolution continues to the next location.

A consumer that links ``hipdnn_frontend_dynamic`` can call ``hipdnn_frontend::setBackendLibraryPath_ext()``, declared in ``<hipdnn_frontend/BackendLibraryPath.hpp>`` and reachable through ``<hipdnn_frontend.hpp>`` in a runtime-load build.
It applies to one calling module (executable or shared library) rather than the whole process, and takes precedence over this variable for that module.
The setter's directory is validated by the same rule -- non-empty and absolute, anything else reported on ``stderr`` and ignored -- and a stored value suppresses this variable, which is read only when the calling module stored none.
A module that stored an empty or relative directory therefore gets neither its own override nor the variable's.

Both are read once, at the calling module's first backend call.

.. note::

  That first call caches the outcome, failure included, and closes the setter before it starts searching.
  After a failed load, ``setBackendLibraryPath_ext()`` returns ``false`` and stores nothing, so trying a different directory requires a new process.

Secure execution
================

On Linux, in a secure execution environment -- a set-user-ID or set-group-ID process, or one that gained capabilities across ``execve`` -- hipDNN ignores every environment variable that steers what code it loads: ``HIPDNN_BACKEND_LIBRARY_PATH``, ``HIPDNN_PLUGIN_DIR``, and ``HIPDNN_HEURISTIC_PLUGIN_DIR``.
Backend resolution skips module-relative and HIP-runtime locations, but still honors an explicit ``setBackendLibraryPath_ext()`` override before the system loader's hardened search.
Variables that do not select code, such as the logging variables above, are unaffected.
Windows has no equivalent execution mode, so these three variables are always honored there.
