.. meta::
  :description: Learn about common terms used in hipDNN.
  :keywords: hipDNN, ROCm, terms, definitions

.. _glossary:

***************
hipDNN glossary
***************

- **DAG (Directed Acyclic Graph)**: A graph structure representing tensor operations where edges indicate data flow and no cycles exist.
- **Engine**: A component capable of executing one or more types of operation graphs.
- **Engine Config**: Configuration parameters that specify how an engine should execute a particular graph.
- **Engine ID**: The 64-bit integer that identifies an engine throughout hipDNN. It's the FNV-1a hash of the engine name, so a name is enough to address an engine and an engine can't be renamed without changing its ID.
- **Engine Name**: The human-readable name of an engine, used for display and for name-based engine selection. It's an opaque, plugin-chosen string. hipDNN resolves it from the plugin that provides the engine, then from the engine details payload, then from the built-in registry in ``EngineNames.hpp``, and finally from a hexadecimal rendering of the engine ID. See :ref:`engine-names`.
- **Execution Context**: Runtime state and resources needed to execute a specific plan.
- **Execution Plan**: A compiled, ready-to-execute representation of an operation graph for a specific engine.
- **Plan Builder**: A component responsible for determining if an engine can handle a graph and constructing execution plans.
- **Plugin**: A dynamically loaded library that provides engine implementations via the hipDNN plugin API.
- **Workspace**: Temporary memory buffer required by an engine to execute operations.
