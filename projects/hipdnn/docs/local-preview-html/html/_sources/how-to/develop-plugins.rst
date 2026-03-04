.. meta::
  :description: Learn how to develop plugins for hipDNN.
  :keywords: hipDNN, ROCm, API, how-to 

.. _develop-plugins:

**************************
Develop plugins for hipDNN
**************************

hipDNN supports a plugin architecture that allows for modular extensions to the framework. Plugins are designed to be separate projects that extend hipDNN's capabilities without being part of the core repository. 
The backend discovers and manages these plugins, leveraging them for different aspects of deep learning routines. This architecture provides flexibility in implementation choices and enables optimizations for specific hardware or use cases.

.. important::

  This page is for advanced users such as senior developers, engineers, and system administrators who are looking to extend hipDNN with customized plugins. Most users should use the default plugins described in :ref:`build-execute`.

Plugin Types
============

hipDNN will support three types of plugins, each serving a specific purpose:

- Engine heuristic and selection plugins (``hipdnn_plugins/heuristics/``): These plugins help determine the best execution strategy for a given operation or graph. They analyze the computation requirements and available resources to select optimal implementations.
- Benchmarking and tuning plugins (``hipdnn_plugins/benchmarking/``): These plugins focus on performance optimization by benchmarking different implementations and tuning parameters for specific hardware configurations.
- Kernel engine plugins (``hipdnn_plugins/engines/``): These plugins provide the actual kernel implementations for operations. They contain the compute kernels that execute on the target hardware (GPUs, accelerators, etc.).

.. note::

  Only kernel engine plugins are supported in hipDNN. 

SDK libraries
=============

hipDNN provides several C++ SDK libraries for plugin development.

Data SDK (``data_sdk``)
-----------------------

The Data SDK contains the FlatBuffers schemas and data structures for graph representation. It includes:

- FlatBuffers schema definitions for graphs, nodes, and attributes.
- Data structures for deserializing serialized graphs.
- Utilities for working with graph data.

See :ref:`add-operation` for more information on adding new operations to the Data SDK (schemas, nodes, attributes).

Plugin SDK (``plugin_sdk``)
---------------------------

The Plugin SDK contains the plugin API and utilities needed to create a plugin that hipDNN can consume. It includes:

- Plugin interface definitions.
- Base classes for engine implementation.
- Utilities for plugin development.

Test SDK (``test_sdk``)
-----------------------

The Test SDK provides utilities for testing plugins. It includes:

- CPU reference implementations for validation (convolution, batchnorm, etc.).
- Test utilities (tolerances, seeds, logging).
- Mock objects for unit testing.
- FlatBuffer test utilities.

Plugin API
==========

The plugin API defines how kernel engine plugins interact with hipDNN:

- **Graph Processing**: Topologically sorted graphs are passed in a serialized format to plugins using FlatBuffers.
- **Data SDK Objects**: Plugins use Data SDK objects to deserialize and process graphs.
- **Capability Reporting**: Plugins analyze graphs and report whether they can execute them.
- **Execution Interface**: Plugins provide execution methods for supported operations.

Engine IDs
==========

hipDNN uses a deterministic hash-based system for managing engine IDs. This system converts human-readable engine names to unique ``int64_t`` identifiers.

Here's the workflow:

1. **Engine Names**: Define human-readable string names for your engines (for example, ``MIOPEN_PLUGIN``, ``MY_CUSTOM_ENGINE``).
2. **Hash Function**: The ``hipdnn_plugin_sdk::engine_names::engineNameToId()`` function converts names to IDs using a FNV-1a hash algorithm.
3. **Registration**: Engine names are registered in the Plugin SDK header for discoverability.

Use Engine IDs
--------------

.. code:: cpp

  #include <hipdnn_plugin_sdk/EngineNames.hpp>

  // Option 1: Use a registered engine ID
  const int64_t engineId = hipdnn_plugin_sdk::engine_names::MIOPEN_PLUGIN_ID;

  // Option 2: Generate ID from custom name
  const int64_t customEngineId = hipdnn_data_sdk::engineNameToId("MY_CUSTOM_ENGINE");

  // In your engine implementation
  class MyEngine {
      int64_t _id;
  public:
      MyEngine(const char* engineName)
          : _id(hipdnn_data_sdk::engineNameToId(engineName)) {
          // Engine is now initialized with a unique ID
      }
  };


Register new engine names
----------------------------

To add your engine name to the official registry:

1. Choose a unique name.
   
   - Use ``UPPER_CASE`` with underscores.
   - Make the name match the value.

2. Add it to the registry. Submit a PR to add your engine name to `plugin_sdk/include/hipdnn_plugin_sdk/EngineNames.hpp <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/data_sdk/include/hipdnn_data_sdk/utilities/EngineNames.hpp>`_:
   
   .. code:: cpp

     HIPDNN_REGISTER_ENGINE(MY_NEW_ENGINE, "MY_NEW_ENGINE")

3. Test it locally. You can use unregistered names during development; they'll generate a warning, but they'll work correctly.

Benefits
--------

- **Deterministic**: Same name always produces same ID.
- **No Collisions**: Hash algorithm minimizes collision risk.
- **Human-Readable**: Debug logs can show meaningful engine names.
- **Forward Compatible**: New engines can be used without registry updates.

.. tip::

  The engine ID system ensures globally unique identifiers across all plugins. You can query registered engines using ``hipdnn_plugin_sdk::engine_names::getAllEngineNames()`` and check for name collisions using the provided test utilities.

Create a kernel engine plugin
=============================

This section focuses on developing kernel engine plugins.

Prerequisites
-------------

Before creating a plugin, ensure you've built and installed hipDNN. Plugins depend on the hipDNN Data SDK and Plugin SDK headers. 

Steps
-----

1. Create plugin structure.
   
   1. Create a new project/repository for your plugin.
   2. Implement the plugin interface defined in `plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginApi.h <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginApi.h>`_. See :ref:`miopen` for an implementation reference.

2. Implement the plugin API functions.

   The underlying implementation below the plugin API level is entirely at the developer's discretion. While the following architectural components are recommended for code organization and maintainability, the only true requirement is to implement the exported API functions defined in ``engine_plugin_api.h``. However, the common architectural pattern consists of:
   
   - **Engine manager**: Manages available engines and their capabilities.
   - **Engine**: Implements graph execution for specific operations (each engine must have a globally unique ``int64_t`` ID).
   - **Execution plans**: Define how operations are executed.
   - **Engine name and ID**: Name your engine and place it in the EngineNames registry

3. Build and deploy the plugin.
  
   - Configure CMake to build the plugin as a shared library.
   - Install to the appropriate plugin directory where hipDNN can discover it at runtime.

Implementation details
----------------------

The **Engine Manager** is responsible for:

- Creating and managing engine instances.
- Reporting supported operations.
- Handling resource allocation.
- Managing device-specific contexts.

For **Engine Implementations**:

- Each engine must have a unique inter-plugin ``int64_t`` identifier.
- Implement the ``execute()`` method for graph execution.
- Provide ``get_supported_operations()`` to report capabilities.
- Handle operation-specific kernel launches.
- Manage memory transfers and synchronization.

**Execution plans** for kernel engines:

- Map hipDNN operations to backend-specific kernel implementations.
- Define memory layouts and data transformations.
- Specify kernel launch configurations.
- Handle device-specific optimizations.

In general, the best practices consist of:

- Organizing kernels by operation type.
- Efficiently managing device memory allocations and transfers.
- Validating inputs and provide meaningful error messages and logs via the sdk.
- Properly managing compute streams for asynchronous execution.
- Profiling kernels and optimize for target hardware.
- Validating and documenting supported operations, hardware requirements, and limitations.
- Including unit tests and integration tests.

Key files reference
~~~~~~~~~~~~~~~~~~~

- `Plugin API Interface <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginApi.h>`_
- `Example Plugin Implementation <https://github.com/ROCm/rocm-libraries/blob/develop/dnn-providers/miopen-provider/MiopenLegacyPlugin.cpp>`_
- `Example Engine Manager <https://github.com/ROCm/rocm-libraries/blob/develop/dnn-providers/miopen-provider/EngineManager.cpp>`_
- `Example Engine Implementation <https://github.com/ROCm/rocm-libraries/blob/develop/dnn-providers/miopen-provider/engines/MiopenEngine.cpp>`_

Plugin architecture
-------------------

Directory structure for kernel engine plugins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your plugin should be structured as an independent project:

.. code::

  your_kernel_plugin_project/
  ├── CMakeLists.txt
  ├── YourPlugin.cpp           # Main plugin entry point
  ├── EngineManager.cpp        # Engine management
  ├── engines/
  │   ├── EngineInterface.hpp  # Engine interface
  │   ├── YourEngine.cpp       # Engine implementation
  │   └── plans/                # Internal plans
  ├── tests/                    # Plugin-specific tests
  └── integration_tests/        # End-to-end integration tests


Build configuration
~~~~~~~~~~~~~~~~~~~

Your plugin's ``CMakeLists.txt`` should:

- Build as a shared library.
- Link against hipDNN Data SDK and Plugin SDK.
- Set appropriate install paths.
- Link to required compute libraries (that is, HIP).

Use hipDNN SDKs in external plugins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When building an external plugin, the hipDNN Data SDK provides CMake variables to help you install your plugin in the correct location:

- Absolute path: (``HIPDNN_FULL_INSTALL_PLUGIN_ENGINE_DIR``):
  
  - Hardcoded at CMake configure time.
  - This is intended for developer use *only*.

- Relative path (``HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR``):
  
  - This is recommended for installations.
  - Automatically prepends the ``CMAKE_INSTALL_PREFIX`` of the consumer.
  - Remains correct when setting the prefix during the CMake install command.

.. code:: cmake

  find_package(hipdnn_data_sdk CONFIG REQUIRED) # or hipdnn_frontend which includes hipdnn_data_sdk

  # Example: Configure your plugin to install to the correct location
  install(
      TARGETS your_plugin_name
      LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
  )

This ensures your plugin will be installed to the same directory structure that hipDNN expects for plugin discovery.

Build and install directory structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The hipDNN build system maintains consistent directory structures for plugins:

Build directory structure:

.. code::

  build/lib/
  └── hipdnn_plugins/
      └── engines/
          └── your_plugin.so


Install directory structure:

.. code::

  /opt/rocm/lib/
  └── hipdnn_plugins/
      └── engines/
          └── your_plugin.so


External plugins should follow this same structure to ensure compatibility.

Plugin loading
==============

hipDNN supports dynamic plugin loading with configurable search paths.

Default plugin loading
----------------------

By default, hipDNN loads plugins from ``./hipdnn_plugins/engines/``.
This path is relative to the backend shared library location, typically: ``/opt/rocm/lib/``.

Default structure example:

.. code::

  /opt/rocm/lib/
  └── hipdnn_plugins/
      └── engines/
          ├── miopen_legacy_plugin.so
          └── other_plugin.so

Environment variable override
------------------------------

You can override the default plugin directory using the ``HIPDNN_PLUGIN_DIR`` environment variable. This is particularly useful for testing and development:

.. code:: bash

  # Load plugins from a custom directory
  export HIPDNN_PLUGIN_DIR=/path/to/test/plugins

  # Example: Load test plugins during testing
  export HIPDNN_PLUGIN_DIR=/home/user/hipDNN/build/lib/test_plugins

When ``HIPDNN_PLUGIN_DIR`` is set, hipDNN will *only* load plugins from the specified directory and supplementary custom paths, ignoring the default location. This allows complete control over which plugins are loaded, which is essential for:

- Running tests with test-specific plugins.
- Development and debugging of new plugins.
- Isolating production plugins from test plugins.

Custom plugin paths
-------------------

Prior to creating a hipDNN handle, you can specify custom plugin paths using the ``hipdnnSetEnginePluginPaths_ext`` function:

.. code:: c

  hipdnnStatus_t hipdnnSetEnginePluginPaths_ext(
      size_t num_paths,
      const char* const* plugin_paths,
      hipdnnPluginLoadingMode_ext_t loading_mode
  );

Plugin symbol resolution
------------------------

All plugins are loaded with ``RTLD_NOW`` to ensure that all symbols are resolved at load time. 
This means that all dependencies must be satisfied when the plugin is loaded. To avoid symbol conflicts, all plugins must be built with with ``-fvisibility=hidden`` to limit symbol exposure.

Path resolution
~~~~~~~~~~~~~~~

Custom paths can be:

- **Relative paths**: Resolved from the backend shared library location.
- **Absolute paths**: Used as specified.

Loading modes
~~~~~~~~~~~~~~

``HIPDNN_PLUGIN_LOADING_ADDITIVE``: Adds new paths to the existing plugin search paths. 
``HIPDNN_PLUGIN_LOADING_ABSOLUTE``: Only loads from the specified paths. 

Example usage
~~~~~~~~~~~~~

.. code:: c

  // Add custom plugin directories
  const char* custom_paths[] = {
      "/home/user/my_plugins",        // Absolute path
      "./local_plugins",              // Relative to backend shared library
      "/opt/custom/hipdnn/plugins"
  };

  hipdnnSetEnginePluginPaths_ext(
      3,                              // Number of paths
      custom_paths,                   // Array of path strings
      HIPDNN_PLUGIN_LOADING_ADDITIVE  // Add to existing paths
  );

Plugins are loaded according to the selected path schema during hipDNN handle creation. Changing paths after handle creation has no effect until another handle is created.

Query loaded plugins
--------------------

After creating a hipDNN handle, you can query which engine plugins were successfully loaded using the ``hipdnnGetLoadedEnginePluginPaths_ext`` function:

.. code:: c

  hipdnnStatus_t hipdnnGetLoadedEnginePluginPaths_ext(
      hipdnnHandle_t handle,
      size_t* num_plugin_paths,
      char** plugin_paths,
      size_t* max_string_len
  );

This function uses a two-call pattern:

- **First call**: Query the number of plugins and required buffer size:
  
  .. code:: cpp

    size_t num_plugins = 0;
    size_t max_len = 0;

    hipdnnGetLoadedEnginePluginPaths_ext(handle, &num_plugins, nullptr, &max_len);

- **Second call**: Retrieve the actual plugin paths:
  
  .. code:: cpp

    hipdnnGetLoadedEnginePluginPaths_ext(handle, &num_plugins, nullptr, &max_len);

    std::vector<std::vector<char>> buffers(num_plugins, std::vector<char>(max_len));
    std::vector<char*> ptrs;
    ptrs.reserve(num_plugins);
    for(size_t i = 0; i < num_plugins; ++i) ptrs.push_back(buffers[i].data());

    hipdnnGetLoadedEnginePluginPaths_ext(handle, &num_plugins, ptrs.data(), &max_len);

    for(size_t i = 0; i < num_plugins; ++i)
    {
        std::cout << "Loaded plugin: " << buffers[i].data() << '\n';
    }

Test plugins
============

Testing is crucial for ensuring plugin reliability and correctness. Plugins should include both unit tests and integration tests to validate their functionality.

Test structure
--------------

Following the `Testing Strategy <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/testing/TestingStrategy.md>`_, plugins should organize tests as follows:

.. code::

  your_kernel_plugin_project/
  ├── tests/                    # Unit tests
  │   ├── TestEngine.cpp
  │   ├── TestKernels.cpp
  │   └── TestUtilties.cpp
  └── integration_tests/        # End-to-end integration tests
      ├── Operation1Test.cpp
      └── Operation2Test.cpp

Unit tests
----------

Unit tests focus on the internal implementation of your plugin components:

- **Location**: ``plugins/<plugin_name>/tests/``
- **Purpose**: Test individual components in isolation (engines, utilities, kernel logic).
- **Requirements**:

  - Must be fast-running.
  - GPU operations must be marked with the ``SKIP_IF_NO_DEVICE()`` macro.
  - Use mocking/stubbing for dependencies where appropriate.
  - Should work on both Windows and Linux.

Integration tests
-----------------

Integration tests validate end-to-end functionality of your plugin:

- **Location**: ``plugins/<plugin_name>/integration_tests/``
- **Purpose**: Validate correctness of graph execution and accuracy of results.
- **Requirements**:

  - Test complete operation graphs.
  - Validate against reference implementations.
  - Test different data types, layouts, dimensions, and edge-cases for each.
  - Enable tests for all supported ASICs.
  - GPU typically required for meaningful validation.
  - Tests are divided into two categories described by the prefix argument passed to ``INSTANTIATE_TEST_SUITE_P``.

    - **Smoke** - These tests are designed to test features using the smallest possible shape and run quickly (combined smoke test run time must be under 5 mins).
    - **Full** - These tests can contain regression shapes, large shapes, or slow shapes.

For a comprehensive example of an integration test, see `IntegrationGpuBatchnormForwardInference.cpp <https://github.com/ROCm/rocm-libraries/blob/develop/dnn-providers/miopen-provider/integration_tests/IntegrationGpuBatchnormForwardInference.cpp>`_.

.. important::

  See our `general testing requirements <https://github.com/ROCm/rocm-libraries/blob/develop/dnn-providers/miopen-provider/integration_tests/IntegrationGpuBatchnormForwardInference.cpp>`_.

.. _example-miopen:

Example: MIOpen Provider Plugin
================================

`MIOpen Provider Plugin <https://github.com/ROCm/rocm-libraries/tree/develop/dnn-providers/miopen-provider>`_

The MIOpen Provider Plugin is a complete example of a kernel engine plugin. It demonstrates how a plugin integrates with hipDNN and delegates execution to a backend. Furthermore, it incorporates the recommended structure and best practices for kernel engine plugins.

At a high level, it:

- Initializes and manages the GPU context using MIOpen handles.
- Translates hipDNN tensors into MIOpen tensor descriptors.
- Dispatches MIOpen kernels to execute operations.
- Coordinates streams and handles synchronization.

Structure
---------

- `Main Plugin <https://github.com/ROCm/rocm-libraries/blob/develop/dnn-providers/miopen-provider/MiopenLegacyPlugin.cpp>`_: Entry point and plugin registration.
- `Engine Manager <https://github.com/ROCm/rocm-libraries/blob/develop/dnn-providers/miopen-provider/EngineManager.hpp>`_: Manages MIOpen engines.
- `MIOpen Engine <https://github.com/ROCm/rocm-libraries/blob/develop/dnn-providers/miopen-provider/engines/MiopenEngine.cpp>`_: Implements graph execution using MIOpen kernels.

Troubleshooting
===============

Plugin loading failures
-----------------------

When a plugin fails to load or initialize, hipDNN logs an error and continues loading other plugins. Common issues include the following.

Plugin handle creation fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you see errors like ``"Failed to create handle for plugin 'PluginName'"``, this typically indicates:

- Missing dependencies that the plugin requires at runtime.
- GPU initialization failures (e.g., no compatible device found).
- Plugin internal initialization errors.

**Solution**: Check that all plugin dependencies are satisfied and a compatible GPU device is available.

Null handle returned
~~~~~~~~~~~~~~~~~~~~~

If you see ``"Plugin 'PluginName' returned null handle"``, the plugin's ``hipdnnEnginePluginCreate`` function returned a null pointer without throwing an exception.

**Solution**: Review the plugin's handle creation logic to ensure it either returns a valid handle or throws an exception with a meaningful error message.

Symbol collisions between plugins
---------------------------------

When multiple plugins are loaded and one or more plugins don't properly hide their symbols, you may encounter:

- Handle collision errors: ``"Plugin 'PluginName' returned a handle that collides with another plugin"``
- Unexpected behavior where one plugin's functions are called instead of another plugin's functions.
- Crashes or undefined behavior during plugin operations.

This occurs because dynamically loaded shared libraries can inadvertently share symbols, causing one plugin's function to override another plugin's function.

Example error log
~~~~~~~~~~~~~~~~~

.. code::

  [ERROR] Plugin 'my_plugin' returned a handle that collides with another plugin.
          This may indicate a symbol collision between plugins.
          Ensure all plugins are built with -fvisibility=hidden.


Solution
~~~~~~~~

All plugins must be built with symbol visibility hidden to prevent symbol collisions:

1. Add this code to your plugin's ``CMakeLists.txt``:

   .. code:: cmake

     set(CMAKE_CXX_VISIBILITY_PRESET hidden)
     set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

   Alternatively, add ``-fvisibility=hidden`` to your compiler flags:

   .. code:: cmake

     target_compile_options(your_plugin PRIVATE -fvisibility=hidden)

2. Only export the required plugin API symbols. The plugin SDK macros handle this automatically when visibility is hidden by default.

Verification
~~~~~~~~~~~~

To verify your plugin has proper symbol visibility:

.. code:: bash

  # List exported symbols (should only show plugin API functions)
  nm -gD your_plugin.so | grep " T "

  # Expected output should only contain:
  # hipdnnEnginePluginCreate
  # hipdnnEnginePluginDestroy
  # hipdnnEnginePluginGetAllEngineIds
  # ... (other plugin API functions)

If you see many internal symbols exported, your visibility settings are incorrect.