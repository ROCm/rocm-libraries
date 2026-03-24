.. meta::
  :description: hipDNN engine configuration knobs provide a flexible mechanism for controlling the runtime behavior of hipDNN engines.
  :keywords: hipDNN, ROCm, API, knobs

.. _knobs:

*********************************
hipDNN engine configuration knobs
*********************************

Engine configuration knobs provide a flexible mechanism for controlling the runtime behavior of hipDNN engines. They let you tune performance, configure algorithmic choices, and adjust memory usage without recompiling code.

There's distinction between ``Knob`` and ``KnobSetting``.
A ``Knob`` is a description of a specific configuration setting, whereas a ``KnobSetting`` is used to specify the value to set for a specific ``Knob`` when the plan is executed by the specific engine.

This topic is about the engine ``Knob``. See :ref:`set-engine-knobs` for examples using both ``Knob`` and ``KnobSetting``.

This system is designed to be:

- **Optional**: Plugins can opt-in to exposing knobs.
- **Flexible**: Support for multiple data types beyond ``int64_t``.
- **Namespace-safe**: Plugin-specific human-readable knob identifiers.
- **Extensible**: New knob types can be added without breaking existing code.

Knobs are runtime-configurable parameters that affect engine behavior. Each knob has a:

- **Unique identifier**: A string-based ID (for example, ``"global.benchmarking"``).
- **Type**: Integer (int64), Float (double), or String.
- **Default value**: The value used when not explicitly set.
- **Constraints**: Valid ranges or allowed values.
- **Description**: Human-readable explanation of purpose.

Knobs enable you to:

- Enable or disable features (for example, benchmarking mode).
- Tune performance parameters (for example, tile sizes and workspace limits).
- Select algorithmic variants (for example, solver selection).
- Control memory usage (for example, workspace size limits).

Knob categories
===============

Different engines expose different knobs. However, hipDNN supports two specific categories of knobs: global and custom knobs.


Global knobs
------------

Global knobs are standard knobs to provide common functionality across all or most engines.
These knobs use the namespace prefix ``global.*``.
Plugins can include these global knob definitions when the common functionality defined by the knob is supported by the plugin.

**Examples**:

- ``global.benchmarking``: Enable/disable solver benchmarking.
- ``global.workspace_size_limit``: Set the maximum workspace memory.

Custom knobs
------------

Custom knobs are engine-specific or plugin-specific parameters. Plugin developers can register custom knobs to expose their own tuning parameters.
These knobs use the namespace prefix ``<plugin>.*`` or ``<plugin>.<operation>.*``. They configure engine-specific behavior, are defined by plugin developers, and they extend hipDNN functionality for specific use cases.

**Examples**:

- ``miopen.conv.tile_size``: Set the convolution tile size for MIOpen.
- ``rocblas.gemm.algo``: Select the GEMM algorithm for ROCm BLAS operations.

Naming conventions
==================

Knobs follow a hierarchical naming scheme to avoid conflicts and improve organization: ``<namespace>.<category>.<knob_name>``.

Global namespace (reserved):

.. code::

  global.benchmarking
  global.workspace_size_limit
  global.deterministic

Plugin-specific namespace:

.. code::

  miopen.conv.tile_size
  rocblas.gemm.transpose_algorithm
  custom_plugin.matmul.block_size

.. important::

  The ``global.*`` namespace is reserved for standard knobs.

Standard global knobs
=====================

These are the global knobs available in hipDNN:

.. list-table::
   :widths: 3 3 3 5
   :header-rows: 1

   * - Knob
     - Type
     - Default
     - Description
   * - ``global.benchmarking``
     - Integer (int64)
     - 0 (disabled)
     - Enable benchmarking mode for kernel selection. When enabled, engines can run multiple kernel variants and select the fastest. The first run might be slower due to benchmarking overhead.
   * - ``global.workspace_size_limit``
     - Integer (int64)
     - The maximum size needed for optimal performance.
     - Limits the maximum workspace memory that solvers can use for convolution operations (Forward, Backward Data, Backward Weights). Refer to :ref:`plugin-support` for specific details.

.. note::

  Additional global knobs may be available depending on the engine. Use ``get_knobs_for_engine()`` to discover all available knobs for a specific engine.

Provider-specific knobs
=======================

Different engine providers may expose their own custom knobs. See :ref:`plugin-support` for specific details.

.. tip::

  When developing with multiple providers, use ``get_knobs_for_engine()`` to programmatically discover the available knobs rather than hard-coding knob names.

Validation
==========

The backend:

- Deserializes the knob settings from the ``EngineConfig``.
- Queries the engine for its supported knobs.
- Validates each setting against the knob's constraints.
- Uses the default values for any unspecified knob settings.

If the validation fails, hipDNN returns ``HIPDNN_STATUS_BAD_PARAM``.

Deprecation
===========

The ``Knob::isDeprecated()`` method returns ``true`` if the knob shouldn't be used (hipDNN logs a warning if you add a ``KnobSetting`` for a deprecated ``Knob``).
A deprecated knob indicates the knob will be removed in a future hipDNN or plugin release. The knob will otherwise continue to operate while deprecated.

Code that uses the deprecated knob should be updated at earliest opportunity to ensure compatibility with future hipDNN and plugin releases.

API reference
=============

See :ref:`knob-api` for the API reference information.

Best practices
==============

For users
---------

- **Query before setting**: Always call ``get_knobs_for_engine()`` to understand the available knobs and their constraints before setting values.
- **Validate constraints**: Check the constraint object to ensure your values are valid before creating execution plans.
- **Use default values when possible**: Only customize knobs when you have a specific performance or behavior requirement.
- **Handle errors gracefully**: Always check the error return value when setting knobs or creating execution plans.
- **Be aware of deprecated knobs**: Watch for deprecation warnings and update your code to use recommended alternatives.
- **Profile before tuning**: Measure performance impact when changing knob values to ensure improvements.

For plugin developers
---------------------

Plugin developers can expose custom knobs using the Plugin SDK utilities:

- `KnobFactory <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/KnobFactory.hpp>`_: Helper class to create knob definitions.
- `IPlanBuilder::getCustomKnobs() <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>`_: Interface method for exposing knobs.
- `GlobalKnobDefines <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/GlobalKnobDefines.hpp>`_: Constants for standard global knob names.

For comprehensive guidance on exposing knobs in your plugin, see `Providing Knobs <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/PluginDevelopment.md#providing-knobs>`_.

Examples
========

For complete working examples, see:

- `Knobs Usage Sample <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/samples/knobs>`_: A comprehensive example demonstrating knob discovery and configuration.
- `Frontend Tests <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/frontend/tests/TestKnob.cpp>`_: Unit tests showing knob API usage.
