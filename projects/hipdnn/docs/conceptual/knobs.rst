.. meta::
  :description: hipDNN engine configuration knobs provide a flexible mechanism for controlling runtime behavior of hipDNN engines.
  :keywords: hipDNN, ROCm, API, 

.. _knobs:

*********************************
hipDNN engine configuration knobs
*********************************

Engine configuration knobs provide a flexible mechanism for controlling runtime behavior of hipDNN engines. They allow you to tune performance, configure algorithmic choices, and adjust memory usage without recompiling code.

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

Query available knobs
=====================

You can discover what knobs an engine supports and their constraints using the hipDNN frontend:

.. code:: cpp

  #include <hipdnn_frontend.hpp>

  using namespace hipdnn_frontend;

  // After building the graph
  Graph graph;
  // ... setup and build graph ...

  // Get available knobs for an engine
  std::vector<Knob> knobs;
  auto error = graph.get_knobs_for_engine(engineId, knobs);

  if (error.is_good()) {
      for (const auto& knob : knobs) {
          std::cout << "Knob ID: " << knob.knobId() << "\n";
          std::cout << "Description: " << knob.description() << "\n";
          std::cout << "Type: " << static_cast<int>(knob.valueType()) << "\n";

          // Access default value (it's a variant)
          const auto& defaultVal = knob.defaultValue();
          if (std::holds_alternative<int64_t>(defaultVal)) {
              std::cout << "Default: " << std::get<int64_t>(defaultVal) << "\n";
          } else if (std::holds_alternative<double>(defaultVal)) {
              std::cout << "Default: " << std::get<double>(defaultVal) << "\n";
          } else if (std::holds_alternative<std::string>(defaultVal)) {
              std::cout << "Default: " << std::get<std::string>(defaultVal) << "\n";
          }

          // Check constraints
          const IConstraint* constraint = knob.constraint();
          if (constraint) {
              std::cout << "Constraint: " << constraint->toString() << "\n";
          }

          std::cout << "---\n";
      }
  } else {
      std::cerr << "Error getting knobs: " << error.get_message() << "\n";
  }

Use the knob lookup map
=======================

Use the lookup method to access the know ID:

.. code:: cpp

  std::unordered_map<std::string, Knob> knobMap;
  auto error = graph.get_knob_lookup_for_engine(engineId, knobMap);

  if (error.is_good()) {
      auto it = knobMap.find("global.benchmarking");
      if (it != knobMap.end()) {
          const Knob& benchmarkingKnob = it->second;
          // Use the knob...
      }
  }

Set knob values
===============

You can set knob values when creating an execution plan. Here's an example:

.. code:: cpp

  #include <hipdnn_frontend.hpp>

  using namespace hipdnn_frontend;

  Graph graph;
  // ... setup and build graph ...

  // Create knob settings
  std::vector<KnobSetting> settings;

  // Set integer knob
  settings.emplace_back("global.benchmarking", 1);

  // Set int64 knob
  settings.emplace_back("global.workspace_size_limit", 1024000LL);

  // Set float knob
  settings.emplace_back("some.float_knob", 0.5);

  // Set string knob
  settings.emplace_back("some.string_knob", std::string("value"));

  // Create execution plan with these settings
  auto error = graph.create_execution_plan_ext(engineId, settings);

  if (error.is_good()) {
      std::cout << "Execution plan created successfully with custom knob settings\n";
  } else {
      std::cerr << "Error: " << error.get_message() << "\n";
  }

Use type-safe knob setting
--------------------------

Refer to this sample code for type-safe knob setting:

.. code:: cpp

  // KnobSetting constructor is type-safe
  KnobSetting intSetting("test.knob", 42);                    // int64_t
  KnobSetting floatSetting("test.knob", 3.14);                // double
  KnobSetting stringSetting("test.knob", std::string("val")); // string

  // You can also update values later
  intSetting.setValue(100);

Use the default knob values
===========================

If you don't specify a knob setting, the engine will use the default value defined by the knob. To explicitly use defaults for all knobs:

.. code:: cpp

  // Option 1: Don't specify any settings (simplest)
  auto error = graph.create_execution_plan_ext(engineId, {});

  // Option 2: Specify only the knobs you want to customize
  std::vector<KnobSetting> settings;
  settings.emplace_back("global.benchmarking", 1);  // Only customize this one
  auto error = graph.create_execution_plan_ext(engineId, settings);

.. note::

  - All knob settings are validated against their constraints when creating an execution plan. Invalid values will result in an error with a descriptive message.
  - If you specify a knob that doesn't exist for the engine, hipDNN will log a warning but continue. This allows forward compatibility when new knobs are added.
  - If a knob is marked as deprecated, hipDNN will log a warning when you use it, but the knob will still function normally.

Standard global knobs
=====================

These are the global knobs are available in hipDNN:

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

.. note::

  Additional global knobs may be available depending on the engine. Use ``get_knobs_for_engine()`` to discover all available knobs for a specific engine.

Provider-specific knobs
=======================

Different engine providers may expose their own custom knobs. Refer to the provider-specific documentation for details:

- :ref:`miopen`
- :ref:`hipblaslt`

.. tip::

  When developing with multiple providers, use ``get_knobs_for_engine()`` to programmatically discover available knobs rather than hard-coding knob names.

API reference
=============

``Knob`` class
--------------

Describes metadata for an available knob.

**Key methods**:

- ``const std::string& knobId()``: Get the knob identifier.
- ``const std::string& description()``: Get a human-readable description.
- ``bool isDeprecated()``: Check if a knob is deprecated.
- ``KnobValueType valueType()``: Get value type (``INT64``, ``FLOAT64``, or ``STRING``).
- ``const KnobValueVariant& defaultValue()``: Get default value as a variant.
- ``const IConstraint* constraint()``: Get the constraint validator.
- ``Error validate(const KnobSetting& setting)``: Validate a setting.

``KnobSetting`` class
---------------------

Represents a knob value setting to apply.

**Constructors**:

.. code:: cpp

  KnobSetting(std::string knobId, KnobValueVariant value);
  template <typename T> KnobSetting(std::string knobId, const T& value);

**Key methods**:

- ``const std::string& knobId()``: Get the knob identifier.
- ``const KnobValueVariant& value()``: Get the knob value.
- ``template <typename T> void setValue(const T& value)``: Update the knob value.

Graph methods
-------------

**Querying knobs**:

.. code:: cpp

  Error get_knobs_for_engine(int64_t engineId, std::vector<Knob>& knobs) const;
  Error get_knob_lookup_for_engine(int64_t engineId, std::unordered_map<std::string, Knob>& knobs) const;

**Setting knobs**:

.. code:: cpp

  Error create_execution_plan_ext(int64_t engineId, const std::vector<KnobSetting>& settings);

Constraint classes
------------------

**Base interface**:

.. code:: cpp

  class IConstraint {
      virtual Error validateKnobSetting(const KnobSetting& setting) const = 0;
      virtual std::string toString() const = 0;
  };

**Implementations**:

- ``IntConstraint(int64_t minValue, int64_t maxValue, int64_t step, std::unordered_set<int64_t> validValues)``
- ``FloatConstraint(double minValue, double maxValue)``
- ``StringConstraint(int32_t maxLength, std::unordered_set<std::string> validValues)``
- ``EmptyConstraint()``: No constraints

Type definitions
----------------

.. code:: cpp

  using KnobValueVariant = std::variant<int64_t, double, std::string>;
  typedef std::string KnobType_t;

  enum class KnobValueType {
      NOT_SET = 0,
      INT64 = 1,
      FLOAT64 = 2,
      STRING = 3,
  };

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