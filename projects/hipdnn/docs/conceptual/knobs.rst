.. meta::
  :description: hipDNN engine configuration knobs provide a flexible mechanism for controlling runtime behavior of hipDNN engines.
  :keywords: hipDNN, ROCm, API,

.. _knobs:

*********************************
hipDNN engine configuration knobs
*********************************

Engine configuration knobs provide a flexible mechanism for controlling runtime behavior of hipDNN engines. They allow you to tune performance, configure algorithmic choices, and adjust memory usage without recompiling code.

The knobs system is designed to be:

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
- Tune performance parameters (for example, tile sizes, workspace limits).
- Select algorithmic variants (for example, solver selection).
- Control memory usage (for example, workspace size limits).

Knob types
==========

hipDNN supports two categories of knobs: Global and Custom knobs.

Global knobs
------------

Global knobs are standard knobs available across all or most engines. They're defined in the ``global.*`` namespace and provide common functionality.
These knobs use the namespace prefix: ``global.*``. They have consistent behavior across engines, they're defined by the hipDNN specification, and can't be registered by custom plugins.

**Examples**:

- ``global.benchmarking``: Enable/disable solver benchmarking.
- ``global.workspace_size_limit``: Set the maximum workspace memory.

Custom knobs
------------

Custom knobs are engine-specific or plugin-specific parameters. Plugin developers can register custom knobs to expose their own tuning parameters.
These knobs use the namespace prefix: ``<plugin>.*`` or ``<plugin>.<operation>.*``. They configure engine-specific behavior, they're defined by plugin developers, and they extend hipDNN functionality for specific use cases.

**Examples**:

- ``miopen.conv.tile_size``: Set the convolution tile size for MIOpen.
- ``rocblas.gemm.algo``: Select the GEMM algorithm for ROCm BLAS.

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

  The ``global.*`` namespace is reserved for standard knobs. Custom plugins can't register knobs in this namespace.

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
     - Enable benchmarking mode for kernel selection. When enabled, engines may run multiple kernel variants and select the fastest. First run may be slower due to benchmarking overhead.
   * - ``global.workspace_size_limit``
     - Integer (int64)
     - The maximum size needed for optimal performance.
     - Limits the maximum workspace memory that solvers can use for convolution operations (Forward, Backward Data, Backward Weights). Refer to :ref: `plugin-support` for specific details.

.. note::

  Additional global knobs may be available depending on the engine. Use ``get_knobs_for_engine()`` to discover all available knobs for a specific engine.

Provider-specific knobs
=======================

Different engine providers may expose their own custom knobs. Refer to the o :ref: `plugin-support` for specific details.

.. tip::

  When developing with multiple providers, use ``get_knobs_for_engine()`` to programmatically discover available knobs rather than hard-coding knob names.

Validation
==========

The backend:

- Deserializes the knob settings from the ``EngineConfig``.
- Queries the engine for its supported knobs.
- Validates each setting against the knob's constraints.
- Use the default values for any unspecified knob settings.

If the validation fails, hipDNN returns ``HIPDNN_STATUS_BAD_PARAM``.

API reference
=============

See :ref:`knob-api` for the API reference information.

Best practices
==============

For users
---------

- **Query before setting**: Always call ``get_knobs_for_engine()`` to understand available knobs and their constraints before setting values.
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

For comprehensive guidance on exposing knobs in your plugin, see the `Providing Knobs <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/PluginDevelopment.md#providing-knobs>`_.

Examples
========

For complete working examples, see:

- `Knobs Usage Sample <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/samples/knobs>`_: Comprehensive example demonstrating knob discovery and configuration.
- `Frontend Tests <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/frontend/tests/TestKnob.cpp>`_: Unit tests showing knob API usage.
