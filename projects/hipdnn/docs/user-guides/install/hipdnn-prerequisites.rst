.. meta::
  :description: Learn about the prerequisites and dependencies required to install hipDNN.
  :keywords: hipDNN, ROCm, install, prerequisites

.. _prerequisites:

*******************
hipDNN installation
*******************

The hipDNN frontend API is distributed as a header-only library; it requires a *development* ROCm package installation.
These development packages contain the ``dev`` prefix.

.. important::

  The base ROCm install package does *not* include the hipDNN frontend API header files. You must use the development ROCm packages.

  For example, on Linux with a gfx950 GPU, install the ``amdrocm-core-dev7.12-gfx950`` package instead of the ``amdrocm7.12-gfx950`` package (``amdrocm-core-dev7.12-gfx950`` includes the ``amdrocm7.12-gfx950`` package, so only the ``amdrocm-core-dev7.12-gfx950`` package needs to be installed).

System requirements
===================

- An AMD GPU with ROCm support (see the `ROCm compatibility matrix <https://advanced-micro-devices-rocm-internal--692.com.readthedocs.build/en/692/compatibility/compatibility-matrix.html?fam=instinct&gpu=mi355x&os=ubuntu&os-version=24.04&i=pkgman>`_)
- Linux or Windows operating system (see `Install AMD ROCm <https://advanced-micro-devices-rocm-internal--692.com.readthedocs.build/en/692/install/rocm.html?fam=instinct&gpu=mi355x&os=ubuntu&os-version=24.04&i=pkgman>`_)

Dependencies
============

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Dependency
     - Version
     - Description
   * - ROCm development installation
     - ROCm 7.12.0 or later
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

Install ROCm and hipDNN
=======================

Follow the instructions at `Install AMD ROCm <https://advanced-micro-devices-rocm-internal--692.com.readthedocs.build/en/692/install/rocm.html?fam=instinct&gpu=mi355x&os=ubuntu&os-version=24.04&i=pkgman>`_ to install ROCm, including hipDNN.
