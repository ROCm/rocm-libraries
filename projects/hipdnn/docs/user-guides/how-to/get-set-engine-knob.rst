.. meta::
  :description: Learn how to get and set engine knob configurations in hipDNN.
  :keywords: hipDNN, ROCm, knobs

.. _set-engine-knobs:

************************************************
Get and set engine knob configurations in hipDNN
************************************************

hipDNN has a flexible system of engine configuration knobs that lets hipDNN plugin developers expose custom runtime settings for end users can to adjust.

This topic demonstrates how to query ``Knob`` descriptions and how to use ``KnobSetting`` to set their values.

.. note::

    For a conceptual overview of configuring engine settings using hipDNN knobs, see :ref:`knobs`.

Query available knobs
=====================

The supported knobs for a specific engine can be retrieved as a list or a lookup table.

Retrieve knobs as a list
------------------------

Use the hipDNN frontend method ``get_knobs_for_engine()`` to retrieve a list of knobs an engine supports and their constraints.

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

Retrieve knobs as a lookup table
--------------------------------

Use the hipDNN frontend method ``get_knob_lookup_for_engine()`` to retrieve a set of knobs an engine supports and their constraints. This is useful for checking if a specific knob exists without iterating through a list.

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

You can set the knob values using ``KnobSetting`` when creating an execution plan.

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

  // KnobSetting constructor is type-safe
  KnobSetting intSetting("test.knob", 42);                    // int64_t
  KnobSetting floatSetting("test.knob", 3.14);                // double
  KnobSetting stringSetting("test.knob", std::string("val")); // string

  // You can also update values later
  intSetting.setValue(100);

Use the default knob values
===========================

If you don't specify a knob setting, the engine will use the default value defined by the knob.

.. code:: cpp

  // Option 1: Don't specify any settings (all knobs use default values)
  auto error = graph.create_execution_plan_ext(engineId, {});

  // Option 2: Specify only the knobs you want to customize
  std::vector<KnobSetting> settings;
  settings.emplace_back("global.benchmarking", 1);  // Only customize this one
  auto error = graph.create_execution_plan_ext(engineId, settings);

.. note::

  - All knob settings are validated against their constraints when creating an execution plan. Invalid values result in an error with a descriptive message.
  - If you specify a knob that doesn't exist for the engine, hipDNN logs a warning and continues. This allows forward compatibility when new knobs are added.
  - If a knob is marked as deprecated, hipDNN logs a warning when you use it. The knob functions normally.

Best practices
==============

For development
---------------

- **Start with defaults**: Use default knob values during initial development.
- **Profile first**: Measure baseline performance before tuning knobs.
- **Query knobs**: Always check the available knobs and their constraints using ``get_knobs_for_engine()``.
- **Test incremental changes**: Modify one knob at a time to understand the impact.

For production
--------------

- **Enable benchmarking during warm-up**:

  .. code:: cpp

    // Warm-up phase
    std::vector<KnobSetting> warmupSettings;
    warmupSettings.emplace_back("global.benchmarking", 1);
    graph.create_execution_plan_ext(engineId, warmupSettings);

    // Execute a few times to populate cache
    for (int i = 0; i < 5; i++) {
        graph.execute(handle, variantPack, workspace);
    }

- **Use cached results in production**: After warm-up, benchmarking can be disabled because the results are cached.
- **Document knob settings**: Keep a record of knob configurations used in production for reproducibility.

For memory-constrained environments
-----------------------------------

- **Query workspace ranges**:

  .. code:: cpp

    // Find minimum and maximum workspace for the operation
    std::vector<Knob> knobs;
    graph.get_knobs_for_engine(engineId, knobs);

    for (const auto& knob : knobs) {
        if (knob.knobId() == "global.workspace_size_limit") {
            // Log constraint to understand valid range
        }
    }

- **Set conservative limits**: Start with a lower workspace limit and increase if performance is insufficient.
- **Balance batch size and workspace**: Reducing workspace allows larger batch sizes, which may offset performance loss.

Error handling
--------------

.. code:: cpp

  auto error = graph.create_execution_plan_ext(engineId, settings);
  if (!error.is_good()) {
      std::cerr << "Failed to create execution plan: " << error.get_message() << "\n";

      // Common errors:
      // - Workspace limit below minimum
      // - Invalid knob ID
      // - Value outside valid range
  }
