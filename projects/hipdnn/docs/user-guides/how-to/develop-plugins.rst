.. meta::
  :description: Learn how to develop plugins for hipDNN.
  :keywords: hipDNN, ROCm, plugins

.. _develop-plugins:

**************************
Develop plugins for hipDNN
**************************

hipDNN supports a plugin architecture that allows for modular extensions to the framework. Plugins are designed to be separate projects that extend the capabilities of hipDNN without being part of the core repository.
The backend discovers and manages these plugins, leveraging them across different aspects of deep learning routines. This architecture provides flexibility in implementation choices and enables optimizations for specific hardware or use cases.

.. important::

  This topic is for advanced users such as senior developers, engineers, and system administrators who are looking to extend hipDNN with custom plugins. Most users should use the default plugins described in :ref:`build-execute`.

Review the :ref:`architecture` and :ref:`backend-architecture` topics for context before beginning plugin development.

.. important::

  Custom plugins installed in the ROCm distribution folder will be included by default when hipDNN graphs are built. If supported graphs fail to build or execute after installing your custom plugin, remove the custom plugin(s) from the ROCm distribution folder. If this resolves the problem, then the custom plugin implementation will need to be updated. Logging can be enabled using the ``HIPDNN_LOG_LEVEL`` environment variable to help in diagnosing any issues. See :ref:`variables` for more info.

Plugin types
============

Kernel engine plugins provide the actual kernel implementations for operations. They contain the compute kernels that execute on the target hardware (GPUs).

SDK libraries
=============

hipDNN provides several C++ SDK libraries for plugin development.

Data SDK (``data_sdk``)
-----------------------

The Data SDK contains the FlatBuffers schemas and data structures for graph representation. It includes:

- FlatBuffers schema definitions for graphs, nodes, and attributes.
- Data structures for deserializing serialized graphs.
- Utilities for working with graph data.

Plugin SDK (``plugin_sdk``)
---------------------------

The Plugin SDK contains the plugin API and utilities needed to create a plugin that hipDNN can consume. It includes:

- Plugin interface definitions.
- Base classes for engine implementation.
- Utilities for plugin development.

Test SDK (``test_sdk``)
-----------------------

The Test SDK provides utilities for testing plugins. It includes:

- `CPU reference implementation <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/OperationSupport-ReferenceImpl.md>`_ for validation (convolution, batchnorm, etc.). This implementation:

  - Provides ground-truth results for validating GPU implementations.
  - Supports core operations (convolution, batchnorm, pointwise).
  - Isn't intended for performance or production use.

- Test utilities (tolerances, seeds, logging).
- Mock objects for unit testing.
- FlatBuffer test utilities.

Plugin API
==========

The plugin API defines how kernel engine plugins interact with hipDNN:

- **Graph processing**: Topologically sorted graphs are passed in a serialized format to plugins using FlatBuffers.
- **Data SDK objects**: Plugins use Data SDK objects to deserialize and process graphs.
- **Capability reporting**: Plugins analyze graphs and report whether they can execute them.
- **Execution interface**: Plugins provide execution methods for supported operations.

Engine IDs
==========

Every engine used by hipDNN requires a unique engine ID. Plugins that provide more than one engine must have a unique ID for each engine provided by the plugin.

hipDNN uses a deterministic hash-based system for managing engine IDs. This system converts human-readable engine names to unique ``int64_t`` identifiers.
The engine ID system ensures globally unique identifiers across all plugins.

When creating a new engine, select a unique descriptive name.
During development, add the ``HIPDNN_REGISTER_ENGINE(MY_CUSTOM_ENGINE)`` macro to a source file in your project.
This verifies that the new plugin name doesn't conflict with plugin names from the official distribution and creates variables that can be used to retrieve the unique ID for this engine.

Benefits
--------

- **Deterministic**: The same name always produces the same ID.
- **No collisions**: Hash algorithm minimizes collision risk.
- **Human-readable**: Debug logs can show meaningful engine names.
- **Forward compatible**: New engines can be used without registry updates.

Use engine IDs
--------------

.. code:: cpp

  #include <hipdnn_data_sdk/utilities/EngineNames.hpp>

  // This macro registers the engine name and creates helper variables
  // such as MY_CUSTOM_ENGINE_ID for this engine.
  HIPDNN_REGISTER_ENGINE(MY_CUSTOM_ENGINE)

  class MyCustomEngine : public hipdnn_plugin_sdk::IEngine< ... >
  {
  public:
      explicit MyCustomEngine(int64_t id);

  private:
      int64_t _id;
  };
  ...
  auto engine = std::make_unique<MyCustomEngine>(MY_CUSTOM_ENGINE_ID);

Register new engine names
----------------------------

To add your engine name to the official registry, submit a GitHub pull request to add your engine name to `plugin_sdk/include/hipdnn_plugin_sdk/EngineNames.hpp <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/data_sdk/include/hipdnn_data_sdk/utilities/EngineNames.hpp>`_ in this format:

.. code:: cpp

  HIPDNN_REGISTER_ENGINE(MY_CUSTOM_ENGINE)

Test it locally. You can use unregistered names during development, but you'll need to remove the ``HIPDNN_REGISTER_ENGINE()`` macro from your plugin before it's added to the official registry.

Create a kernel engine plugin
=============================

This section focuses on developing kernel engine plugins.

Prerequisites
-------------

Before creating a plugin, ensure you've installed hipDNN. Plugins depend on the hipDNN Data SDK and Plugin SDK headers.

Steps
-----

1. Create the plugin structure.

   1. Create a new project or repository for your plugin.
   2. Add definitions for the plugin interface defined in `plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginApi.h <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginApi.h>`_. See :ref:`miopen-provider` for an implementation reference.

2. Implement the plugin API functions.

   The underlying implementation below the plugin API level is entirely at the developer's discretion. While the following architectural components are recommended for code organization and maintainability, the only true requirement is to implement the exported API functions defined in ``engine_plugin_api.h``. However, the common architectural pattern consists of:

   - **Engine manager**: Manages available engines and their capabilities.
   - **Engine**: Implements graph execution for specific operations (each engine must have a globally unique ``int64_t`` ID).
   - **Execution plans**: Define how operations are executed.
   - **Engine name and ID**: Name your engine and place it in the EngineNames registry

3. Build and deploy the plugin.

   - Configure CMake to build the plugin as a shared library.
   - Install it in the ROCm hipDNN plugin directory where hipDNN can discover it at runtime or use the ``HIPDNN_PLUGIN_DIR`` environment variable to force hipDNN to only load plugins from the folder specified in the environment variable.

Typical implementation details
------------------------------

The **Engine manager** is responsible for:

- Creating and managing engine instances.
- Reporting supported operations.
- Handling resource allocation.
- Managing device-specific contexts.

For **Engine implementations**:

- Each engine must have a globally unique ``int64_t`` identifier.
- Implement ``isApplicable()`` to check if the engine solves the given graph.
- Create execution contexts for executing plans.

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
- Validating inputs and providing meaningful error messages and logs via the SDK.
- Properly managing compute streams for asynchronous execution.
- Profiling kernels and optimizing for target hardware.
- Validating and documenting supported operations, hardware requirements, and limitations.
- Including unit tests and integration tests.

Key files reference
~~~~~~~~~~~~~~~~~~~

- `Plugin API interface <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginApi.h>`_
- `Example plugin implementation <https://github.com/ROCm/rocm-libraries/blob/develop/dnn-providers/hip-kernel-provider/src/HipKernelContainer.cpp>`_
- `Example engine manager <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/EngineManager.hpp>`_
- `Example engine implementation <https://github.com/ROCm/rocm-libraries/blob/develop/dnn-providers/miopen-provider/engines/MiopenEngine.cpp>`_


Build configuration
~~~~~~~~~~~~~~~~~~~

Your plugin's ``CMakeLists.txt`` must:

- Build as a shared library.
- Enable Position Independent Code (PIC) compilation for the library.
- Link against the hipDNN Data SDK and Plugin SDK.
- Set appropriate install paths.
- Link to the required compute libraries (that is, HIP).

Use hipDNN SDKs in external plugins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When building an external plugin, the hipDNN Data SDK provides CMake variables to help you install your plugin in the correct location:

- Absolute path: (``HIPDNN_FULL_INSTALL_PLUGIN_ENGINE_DIR``):

  - Computed at ``find_package()`` time relative to the installed hipDNN location.
  - This is intended for *developer use only*.

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

The hipDNN plugins are installed in the ROCm install folder:

.. code::

  /opt/rocm/lib/
  └── hipdnn_plugins/
      └── engines/
          └── your_plugin.so


Install your plugin to this folder to have it included automatically by hipDNN. Note that if the ``HIPDNN_PLUGIN_DIR`` environment variable is set, the plugins will only be loaded from that folder and not the ROCm folder.

.. _plugin-loading:

Plugin loading
==============

hipDNN supports dynamic plugin loading with configurable search paths.

Default plugin loading
----------------------

By default, hipDNN loads plugins from ``./hipdnn_plugins/engines/``.
This path is relative to the hipDNN backend shared library location in the ROCm install folder, typically ``/opt/rocm/lib/`` on Linux.

Default structure example:

.. code::

  /opt/rocm/lib/
  └── hipdnn_plugins/
      └── engines/
          ├── miopen_plugin.so
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

See :ref:`plugin-loading-variables` for details on using the ``HIPDNN_PLUGIN_DIR`` to control plugin loading.

Custom plugin paths
-------------------

Prior to creating a hipDNN handle, you can specify custom plugin paths using the ``hipdnnSetEnginePluginPaths_ext`` backend API function before the hipDNN handle is created:

.. code:: c

  hipdnnStatus_t hipdnnSetEnginePluginPaths_ext(
      size_t num_paths,
      const char* const* plugin_paths,
      hipdnnPluginLoadingMode_ext_t loading_mode
  );

Plugin symbol resolution
------------------------

On Linux, all plugins are loaded with ``RTLD_NOW | RTLD_LOCAL`` to ensure that all symbols are resolved at load time.
This means that all dependencies must be satisfied when the plugin is loaded. To avoid symbol conflicts, all plugins must be built with ``-fvisibility=hidden`` to limit symbol exposure.

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

After creating a hipDNN handle, you can query which engine plugins were successfully loaded using the ``hipdnnGetLoadedEnginePluginPaths_ext`` backend API function:

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

- **Location**: ``<plugin_name>/src/tests/``
- **Purpose**: Test individual components in isolation (engines, utilities, kernel logic).
- **Requirements**:

  - Must be fast-running.
  - Typically, unit tests should never access GPU hardware. If unit tests need to access the GPU hardware, use the ``SKIP_IF_NO_DEVICE()`` macro to automatically skip the test if no HIP devices are found.
  - Use mocking/stubbing for dependencies where appropriate.
  - Should work on both Windows and Linux.

Integration tests
-----------------

Integration tests validate end-to-end functionality of your plugin:

- **Location**: ``<plugin_name>/src/integration_tests/``
- **Purpose**: Validate correctness of graph execution and accuracy of results.
- **Requirements**:

  - Test complete operation graphs.
  - Validate against reference implementations.
  - Test different data types, layouts, dimensions, and edge-cases for each.
  - Enable tests for all supported ASICs.
  - A GPU is typically required for meaningful validation. Use the ``SKIP_IF_NO_DEVICE()`` macro to automatically skip the test if no HIP devices are found.
  - Tests are divided into two categories designated by the prefix argument passed to ``INSTANTIATE_TEST_SUITE_P``.

    - **Smoke**: These tests are designed to test features using the smallest possible shape and run quickly (the combined smoke test run time must be under 5 mins).
    - **Full**: These tests can contain regression shapes, large shapes, or slow shapes.

For a comprehensive example of an integration test, see `IntegrationGpuBatchnormForwardInference.cpp <https://github.com/ROCm/rocm-libraries/blob/develop/dnn-providers/miopen-provider/integration_tests/IntegrationGpuBatchnormForwardInference.cpp>`_.

.. note::

  See `general testing requirements <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/Testing.md#testing-requirements>`_.

Example: MIOpen provider plugin
================================

See :ref:`miopen-provider` for more information.

Troubleshooting
===============

Plugin loading failures
-----------------------

When a plugin fails to load or initialize, hipDNN logs an error and continues loading other plugins. Common issues include the following.

Plugin handle creation fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you see errors like ``"Failed to create handle for plugin 'PluginName'"``, this typically indicates:

- Missing dependencies that the plugin requires at runtime.
- GPU initialization failures (for example, no compatible device found).
- Plugin internal initialization errors.

**Solution**: Check that all plugin dependencies are satisfied, library load paths are set correctly, and a compatible GPU device is available.

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
If the plugin loads successfully in isolation, then this could be the issue.

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
