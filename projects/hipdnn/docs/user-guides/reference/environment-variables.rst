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
