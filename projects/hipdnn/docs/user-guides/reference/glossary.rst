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
- **Execution Context**: Runtime state and resources needed to execute a specific plan.
- **Execution Plan**: A compiled, ready-to-execute representation of an operation graph for a specific engine.
- **Plan Builder**: A component responsible for determining if an engine can handle a graph and constructing execution plans.
- **Plugin**: A dynamically loaded library that provides engine implementations via the hipDNN plugin API.
- **Workspace**: Temporary memory buffer required by an engine to execute operations.
