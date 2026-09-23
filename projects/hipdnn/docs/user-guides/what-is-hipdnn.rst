.. meta::
   :description: hipDNN (Deep Neural Network) is a graph-based library providing improved performance for deep learning workloads with AMD GPUs.
   :keywords: hipDNN, ROCm, documentation

.. _what-is:

******************
What is hipDNN?
******************

hipDNN (Deep Neural Network) is a graph-based library that improves performance for deep learning workloads on AMD GPUs.

hipDNN uses operation graphs as an intermediate representation for describing computations, allowing different backend engines to optimize and execute them efficiently.

hipDNN provides an interface that follows established deep learning conventions and a plugin-based architecture, which allows advanced users to extend hipDNN without modifying the core library.

.. note::

  hipDNN is in beta. Running production workloads is not recommended.

hipDNN and the ROCm ecosystem
=============================

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Layer
     - Component
     - Function
   * - Framework
     - PyTorch/TensorFlow
     - The high-level AI tools developers use.
   * - Interface
     - **hipDNN**
     - The portable API for deep learning operations.
   * - Engine selection and execution
     - Engine provider plugins
     - Matches fused graph operations with engines for optimized performance.
   * - Runtime
     - HIP
     - The translation layer between C++ and the GPU hardware.
   * - Hardware
     - AMD Instinct™ / Radeon™
     - The physical GPU.

Features
========

- **Graph-based API**: Operations are expressed as computational graphs rather than individual function calls, enabling optimization opportunities.
- **Plugin architecture**: Backend kernel engines are implemented through plugins, allowing extensibility without modifying the core library.
- **Increased performance through fusion**: Multiple operations can be fused into single kernels for better performance.
- **Industry-standard API**: Provides a familiar interface that matches established deep learning library conventions.

Components
==========

- **Frontend**: A header-only C++ library that provides the industry-standard API for interacting with hipDNN. The frontend wraps the backend C API to provide a more user-friendly C++ interface.
- **Backend**: A shared library that provides a C API for hipDNN. The backend is the core component of hipDNN. It acts as a plugin loader and manager, connecting problems to engines (provided via plugins) that can solve them.
- **SDKs**: Header-only libraries that provide shared utilities and interfaces. hipDNN provides three SDKs: the Data SDK (graph schemas and data structures), Plugin SDK (plugin API and utilities), and Test SDK (testing utilities and CPU reference implementations).
- **Plugins**: Plugins provide additional operational support or performance improvements.

Engine provider plugins
=======================

hipDNN operations are implemented through provider plugins. Each provider plugin provides its own set of supported operations. See :ref:`plugin-support` for more information.
