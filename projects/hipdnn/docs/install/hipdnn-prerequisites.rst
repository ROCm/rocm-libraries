.. meta::
  :description: hipDNN installation prerequisites
  :keywords: hipDNN, ROCm, install, prerequisites

.. _prerequisites:

********************
hipDNN prerequisites
********************

Ensure your system has these requirements before installing hipDNN.

System requirements
===================

- An AMD GPU with ROCm support.
- Operating System:

  - **Linux**: Any distribution supported by `TheRock <https://github.com/ROCm/TheRock>`_ (such as Ubuntu 24).
  - **Windows**: Windows 11 (hipDNN has limited Windows support. See :ref:`windows` for more info).

Dependencies
============

Use the `prebuilt binaries and Docker files <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/dockerfiles/Dockerfile.ubuntu24>`_ for a consistent development environment with all dependencies pre-installed. 
This is the recommended approach for most users. For more details about these Docker images, see the `Docker README <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/dockerfiles/README.md>`_. 

.. note::

  Dockerfile development environments are not available for Windows. See :ref:`windows` for more info.

Required dependencies
---------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Dependency
     - Version
     - Description
   * - ROCm
     - Matching TheRock (ROCm version 7.0.0 or later)
     - AMD GPU programming stack (see `TheRock releases <https://github.com/ROCm/TheRock/releases>`_)
   * - CMake
     - 3.25.2 or later
     - Build system generator
   * - Ninja
     - 1.12.1 or later
     - Faster build system (recommended)
   * - C++ Compiler
     - C++17 compatible
     - hipDNN requires C++17 compatible AMD Clang (plugins using device code may require C++20)
   * - HIP
     - Matching TheRock
     - GPU programming interface (included with ROCm/TheRock)
   * - clang-format
     - 18.X
     - Code formatting tool
   * - clang-tidy 
     - 20.X
     - Static analysis tool
   * - LLVM Tools 
     - 20.X
     - LLVM tools for code_coverage, and ASAN enabled builds

Optional dependencies
---------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Dependency
     - Version
     - Description
   * - Docker
     - Latest
     - For a containerized build environment
   * - Python3
     - Latest
     - For test name validation
  
Third-party libraries
---------------------

These libraries are automatically managed by CMake (see `Dependencies.cmake <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/cmake/Dependencies.cmake>`_ for more information):

- `FlatBuffers <https://github.com/google/flatbuffers>`_: The serialization library
- `Google Test <https://github.com/google/googletest>`_: A unit testing framework
- `spdlog <https://github.com/gabime/spdlog>`_: The logging library