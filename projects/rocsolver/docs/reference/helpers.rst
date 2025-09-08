.. meta::
  :description: rocSOLVER library and logging functions API documentation
  :keywords: rocSOLVER, ROCm, API, documentation, library functions, logging

.. _helpers:

*****************************************************
rocSOLVER library and logging functions
*****************************************************

These are helper functions that retrieve information and control some functions of the library.
The helper functions are divided into the following categories:

* :ref:`lib_info`: Return information about the library version.
* :ref:`algo_select`: Select different algorithm modes of certain APIs.
* :ref:`api_logging`: Control the :ref:`logging-label` capabilities.



.. _lib_info:

Library information
===============================

.. contents:: List of library information functions
   :local:
   :backlinks: top

rocsolver_get_version_string()
------------------------------------
.. doxygenfunction:: rocsolver_get_version_string

rocsolver_get_version_string_size()
------------------------------------
.. doxygenfunction:: rocsolver_get_version_string_size



.. _algo_select:

Algorithm selection
===============================

.. contents:: List of algorithm selection functions
   :local:
   :backlinks: top

rocsolver_set_alg_mode()
------------------------------------
.. doxygenfunction:: rocsolver_set_alg_mode

rocsolver_get_alg_mode()
------------------------------------
.. doxygenfunction:: rocsolver_get_alg_mode



.. _api_logging:

Logging functions
===============================

.. contents:: List of logging functions
   :local:
   :backlinks: top

rocsolver_log_begin()
---------------------------------
.. doxygenfunction:: rocsolver_log_begin

rocsolver_log_end()
---------------------------------
.. doxygenfunction:: rocsolver_log_end

rocsolver_log_set_layer_mode()
---------------------------------
.. doxygenfunction:: rocsolver_log_set_layer_mode

rocsolver_log_set_max_levels()
---------------------------------
.. doxygenfunction:: rocsolver_log_set_max_levels

rocsolver_log_restore_defaults()
---------------------------------
.. doxygenfunction:: rocsolver_log_restore_defaults

rocsolver_log_write_profile()
---------------------------------
.. doxygenfunction:: rocsolver_log_write_profile

rocsolver_log_flush_profile()
---------------------------------
.. doxygenfunction:: rocsolver_log_flush_profile

