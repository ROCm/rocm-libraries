.. meta::
  :description: hipDNN installation prerequisites
  :keywords: hipDNN, ROCm, install, prerequisites

.. _prerequisites:

********************
hipDNN prerequisites
********************

Ensure your system has these requirements before installing hipDNN.

.. important::

  The hipDNN frontend API is distributed as a header-only library, requiring the development ROCm packages to be installed. The base ROCm install package does not include the hipDNN frontend API header files.
  For example, on Linux with a gfx950 GPU, install the `amdrocm-core-dev7.12-gfx950` package instead of the `amdrocm7.12-gfx950` package (`amdrocm-core-dev7.12-gfx950` includes the `amdrocm7.121-gfx950` package, so only the `amdrocm-core-dev7.12-gfx950` package needs to be installed).

System requirements
===================

- An AMD GPU with ROCm support. See `ROCm compatibility matrix <https://rocm.docs.amd.com/en/7.12.0/compatibility/compatibility-matrix.html>`
- Linux or Windows operating system. See `Install AMD ROCm <https://rocm.docs.amd.com/en/7.12.0/install/rocm.html>`

Dependencies
============

Required dependencies
---------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Dependency
     - Version
     - Description
   * - ROCm Development Install
     - ROCm version 7.12.0 or later
     - AMD GPU programming stack
   * - CMake
     - 3.25.2 or later
     - Build system generator
   * - Ninja (recommended)
     - 1.12.1 or later
     - Faster build system
   * - C++ Compiler
     - C++17 compatible
     - hipDNN API supports C++17

Install
=======

Follow the instructions at `Install AMD ROCm <https://rocm.docs.amd.com/en/7.12.0/install/rocm.html>` to install ROCm, including hipDNN.
