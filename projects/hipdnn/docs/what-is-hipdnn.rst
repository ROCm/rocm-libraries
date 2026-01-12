.. meta::
   :description: hipDNN is a plugin-based deep learning library that provides graph-based operation support through various backend plugins 
   :keywords: hipDNN, ROCm, library, API

.. _what-is:

******************
What is hipDNN?
******************

hipDNN is a graph-based deep learning library that enables multi-operation fusion for improved performance on AMD GPUs. 
It uses operation graphs as an intermediate representation to describe computations, allowing different backend engines to optimize and execute these graphs efficiently.

hipDNN has a plugin-based architecture which allows you to extend hipDNN without modifying the core library.

hipDNN allows developers to run deep learning workloads on the frontend on ROCm and AMD GPUs while maintaining compatibility with NVIDIA's cuDNN API.

Features
========

- **Graph-based API**: Operations are expressed as computational graphs rather than individual function calls, enabling optimization opportunities.
- **Plugin architecture**: Backend engines, heuristics, and benchmarking are implemented as plugins, allowing extensibility without modifying the core library.
- **Performance through Fusion**: Multiple operations can be fused into single kernels for better performance.
- **Engine selection**: Heuristics and benchmarking will be implemented as plugins, allowing extensibility without modifying the core library.
- **Industry standard API**: Provides a familiar interface that matches established deep learning library conventions.

MIOpen plugin architecture
==========================

The MIOpen legacy plugin serves as the kernel provider. It employs a modular C++ architecture, largely decoupled from the API layer.

- Dependency injection container (``MiopenContainer``): This is the root object that manages the lifecycle and dependencies of all other components. It initializes the ``EngineManager`` and ensures that all necessary services are correctly injected.
- Engine manager (``EngineManager``): The central registry for execution engines. It orchestrates the selection of the appropriate engine for a given operation graph by querying its registered engines.
- Plan builders (``IPlanBuilder``): Each engine is associated with a set of plan builders. These components are responsible for:

  - Applicability: Inspecting an operation graph to determine if the engine can execute it.
  - Resource estimation: Calculating the required workspace size.
  - Plan construction: Creating an executable IPlan object if the graph is supported.

- Plans (IPlan): An IPlan represents a strategy for executing a specific operation graph. It encapsulates all the necessary logic and state to run the routine, abstracting the details from the higher-level engine management.
- C-API Interface: A thin translation layer that exposes these internal C++ components to the backend via the required engine plugin C-API.

Execution flow
==============

When the backend requests a graph execution, the flow within the plugin is as follows:

- **Ingestion**: The C-API bridge receives the raw graph handle and forwards it to the MiopenContainer.
- **Selection**: The EngineManager iterates through registered engines to find a candidate.
- **Compilation**: The selected Engine's PlanBuilder validates the graph and constructs an IPlan.
- **Execution**: The IPlan executes the operation, marshaling pointers from the backend's VariantPack to the underlying device kernels.

This architecture effectively separates the plugin interface from the engine implementation details. However, currently, this infrastructure is largely internal to the MIOpen plugin. The goal of the Plugin SDK is to standardize and provide these as reusable components for plugin development, so developers can focus on the implementations of the underlying kernels and libraries.