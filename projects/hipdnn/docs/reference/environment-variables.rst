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

Frontend and plugin logging
===========================

The frontend and plugins can be configured to use the same logging destination as the backend, which is lazy-initialized automatically:

1. Initialize logging using the ``initializeCallbackLogging`` function.
2. Pass ``hipdnnLoggingCallback_ext`` as the callback function (accessible via plugin API or backend header).

This ensures all components log to the same destination.

MIOpen plugin logging
=====================

.. tip::

  When using the MIOpen legacy plugin, you can use MIOpen-specific environment variables to control the underlying library's logging behavior.

For more details about MIOpen logging, see the latest `MIOpen Debug and Logging documentation <https://rocm.docs.amd.com/projects/MIOpen/en/develop/how-to/debug-log.html>`_. 
All MIOpen environment variables remain compatible with hipDNN's MIOpen legacy plugin.

Test configuration
==================

``HIPDNN_GLOBAL_TEST_SEED``
---------------------------

Controls the random number generator seed used across hipDNN tests. This allows for reproducible test runs or full randomization when needed.

- No set value: Uses default seed value of ``1`` (default behavior)
- ``<number>``: Uses the specified numeric seed (for example, ``42``, ``12345``)
- ``RANDOM``: Generates a random seed using ``std::random_device``

.. note::

  The ``RANDOM`` value is case-insensitive (``random``, ``Random``, ``RANDOM`` all work).

Here are some examples:

.. code:: bash

  # Use a specific seed for consistent results
  export HIPDNN_GLOBAL_TEST_SEED=42

  # Use default seed (1) for reproducible tests
  unset HIPDNN_GLOBAL_TEST_SEED

  # Use random seed for each test run
  export HIPDNN_GLOBAL_TEST_SEED=RANDOM


Best practices
--------------

- Use the default seed (1) for CI/CD pipelines to ensure consistent test results.
- Use a specific numeric seed when debugging to reproduce exact test conditions.
- Use ``RANDOM`` during development to catch edge cases with different data patterns.

Error handling
==============

hipDNN provides functions for retrieving error information.

Getting error strings
---------------------

.. code:: c

  // Convert status code to string
  const char* error_str = hipdnnGetErrorString(status);

  // Get detailed error message for the current thread
  char message[HIPDNN_ERROR_STRING_MAX_LENGTH];
  hipdnnGetLastErrorString(message, sizeof(message));


Best practices
--------------

- Check return status codes from all hipDNN API calls.
- Use ``hipdnnGetLastErrorString`` for detailed error context.
- Enable appropriate logging levels during development and debugging.
- Configure logging to files for production deployments.