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

Sets the minimum severity that will be emitted. Levels are inclusive: choosing a level enables messages at that level and all higher severities.

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

Specifies the file path where logs will be appended. If not set, logs are written to ``stderr``.

Here's an example:

.. code:: bash

  export HIPDNN_LOG_FILE=/path/to/hipdnn.log

.. tip::

  When using the MIOpen legacy plugin, you can use MIOpen-specific environment variables to control the underlying library's logging behavior.

Plugin loading
==============

The following environment variable can be used to control which folders hipDNN looks in to load plugins.

``HIPDNN_PLUGIN_DIR``
---------------------

Sets the folder where the hipDNN plugins are located. When set, hipDNN only loads plugins from this folder, ignoring any plugins that are installed with the ROCm distribution.

Here's an example:

.. code:: bash

  set HIPDNN_PLUGIN_DIR=/dev/custom_plugins
