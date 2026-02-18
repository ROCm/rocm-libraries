.. meta::
  :description: Learn how to get/set engine knob configurations in hipDNN
  :keywords: hipDNN, ROCm, API, how-to 

.. _set-engine-knobs:

********************************************
Get/set engine knob configurations in hipDNN
********************************************

hipDNN has a flexible engine configuration knobs system that allows plugin developers to expose custom runtime settings and enables end-users to adjust these settings. 
Unlike a more limited ``int64_t``-based knob system with min/max/stride constraints, this design leverages Flatbuffers' union types to support multiple value types (integers, floats, strings) while maintaining type safety.

The knobs system is designed to be:

- **Optional**: Plugins can opt-in to exposing knobs.
- **Flexible**: Support for multiple data types beyond ``int64_t``.
- **Namespace-safe**: Plugin-specific human-readable knob identifiers.
- **Extensible**: New knob types can be added without breaking existing code.

Use the frontend
================

You can get/set engine knobs with the hipDNN frontend:

.. code:: cpp

    std::vector<Knob> knobs;
    graph.get_knobs_for_engine(MIOPEN_ENGINE, knobs);

You can configure the knobs to use their default values, or customize specific knob settings:

.. code:: cpp 

    // Option 1: Use default knob values
    auto defaultKnobSettings = Graph::knobToDefaultKnobSettings(knobs);
    graph.create_execution_plan_ext(MIOPEN_ENGINE, defaultKnobSettings);

.. code:: cpp 
  
    // Option 2: Customize specific knobs
    std::unordered_map<int64_t, KnobSetting> customKnobSettings;
    customKnobSettings.insert(Knob::make_knob_id("miopen.conv.tile_size"), 32);
    graph.create_execution_plan_ext(MIOPEN_ENGINE, customKnobSettings);

- ``Knob`` includes the knob's display names, descriptions, constraints, and its default values.
- The ``KnobSetting`` class enforces type-safe value setting.

Here's a complete working example you can use for reference:

.. code:: cpp

    using namespace hipdnn::frontend;

    // Create and build graph
    Graph graph;

    // ... setup graph ...

    std::vector<Knob> knobs;
    graph.get_knobs_for_engine(MIOPEN_ENGINE, knobs);

    // Option 1: Use default knob values
    auto defaultKnobSettings = Graph::knobToDefaultKnobSettings(knobs);
    graph.create_execution_plan_ext(MIOPEN_ENGINE, defaultKnobSettings);

    // Option 2: Customize specific knobs
    std::unordered_map<int64_t, KnobSetting> customKnobSettings;
    customKnobSettings.insert(Knob::make_knob_id("miopen.conv.tile_size"), 32);
    graph.create_execution_plan_ext(MIOPEN_ENGINE, customKnobSettings);

Validation
----------

The backend:

- Deserializes the knob settings from the ``EngineConfig``.
- Queries the engine for its supported knobs.
- Validates each setting against the knob's constraints.
- Use the default values for any unspecified knob settings.

If the validation fails, hipDNN returns ``HIPDNN_STATUS_BAD_PARAM``.

Knob naming convention guide
============================

You should use a hierarchical naming scheme when naming your knobs: ``<plugin_name>.<engine_name>.<knob_name>``.

For example:

- ``miopen.conv.tile_size``
- ``miopen.conv.algorithm``
- ``rocblas.gemm.transpose_algorithm``
- ``custom_plugin.matmul.block_size``

The plugin name prefix (for example, ``miopen.``) ensures that different plugins can have semantically similar knobs.

There's a ``global.`` namespace that contains commonly used knobs. 
Here are some examples:

- ``global.deterministic``
- ``global.workspace``
- ``global.benchmarking``

.. warning::

  There's no support for custom knobs in the ``global.`` namespace. Custom knobs registered in the ``global.`` namespace will be rejected.



