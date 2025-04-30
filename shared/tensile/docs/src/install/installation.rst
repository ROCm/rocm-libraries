.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Tensile installation, GEMM, Tensor, tensor, Build Tensile, Run benchmarks

.. _installation:

*****************
Installation
*****************

This topic provides information required to install Tensile from source and run benchmarks.

.. _install-rocm:

Install ROCm
============

To begin, install ROCm for your platform. For installation instructions, refer to the `Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html>`_ or `Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/index.html>`_ installation guide.

.. tip::

   If using Bash, set ``PATH=/opt/rocm/bin/:$PATH`` in your ``~/.bashrc`` and refresh your shell using ``source ~/.bashrc``.
   Alternatively, export the path for your current shell session using ``export PATH=/opt/rocm/bin/:$PATH``.

Install OS dependencies
=========================

.. note::
   The steps below are for Ubuntu. For other distributions, use the appropriate package manager.

1. Install dependencies:

   .. code-block::

    apt-get install libyaml python3-yaml \
        libomp-dev libboost-program-options-dev libboost-filesystem-dev

2. Install one of the following, depending on your preferred Tensile data format. If both are installed, ``msgpack`` is preferred:

   .. code-block::

      apt-get install libmsgpack-dev    # If using the msgpack backend

      # OR

      apt-get install libtinfo-dev      # If using the YAML backend

3. Install build tools. For additional installation methods for the latest versions of CMake, see the `CMake installation <https://cliutils.gitlab.io/modern-cmake/chapters/intro/installing.html>`_ page.

   .. code-block::

      apt-get install build-essential cmake

Install Tensile from source
============================

To install Tensile from source, it is recommended to create a virtual environment first:

.. code-block:: bash

  python3 -m venv .venv
  source .venv/bin/activate

Then, you can install Tensile using pip or git.

Option 1: Install with pip
---------------------------

.. code-block:: bash

  pip3 install git+https://github.com/ROCmSoftwarePlatform/Tensile.git@develop


Option 2: Install with git
----------------------------

.. code-block:: bash

  git clone git@github.com:ROCm/Tensile.git && cd Tensile
  pip3 install .

You can now run Tensile's Python applications.

Running benchmark
===================

To run a benchmark, pass a tuning config to the ``Tensile`` program located in ``Tensile/bin``.

For demonstration purposes, the sample tuning file available in ``Tensile/Configs/rocblas_sgemm_example.yaml`` is used.
The sample tuning file allows you to specify the target architecture for which the benchmark will generate a library.
To find your device architecture, run:

.. code-block:: bash

   rocminfo | grep gfx

Specify the device architecture in the sample tuning file using ``ArchitectureName:``. Based on the device architecture, use ``ArchitectureName: "gfx90a"`` or ``ArchitectureName: "gfx1030"``.

You can now run benchmarks using Tensile. From the top-level directory, run:

.. code-block:: bash

   mkdir build && cd build
   ../Tensile/bin/Tensile ../Tensile/Configs/rocblas_sgemm_example.yaml ./

After the benchmark completes, Tensile creates the following directories:

- **0_Build**: Contains a client executable. Use this to launch Tensile from a library viewpoint.
- **1_BenchmarkProblems**: Contains all the problem descriptions and executables generated during benchmarking. Use the ``run.sh`` script to reproduce results.
- **2_BenchmarkData**: Contains the raw performance results of all kernels in CSV and YAML formats.
- **3_LibraryLogic**: Contains the winning (optimal) kernel configurations in YAML format. Typically, rocBLAS takes the YAML files from this folder.
- **4_LibraryClient**: Contains the code objects, kernels, and library code. This is the output of running ``TensileCreateLibrary`` using the ``3_LibraryLogic`` directory as an input.

The client is built at the beginning of the build and cached for future builds if the output directory and client build files are unchanged. To use the client, run:

.. code-block:: shell

   ./0_Build/client/tensile_client -h
   ./0_Build/client/tensile_client --config-file=1_BenchmarkProblems/Cijk_Ailk_Bljk_SB_00/00_Final/source/ClientParameters.ini

.. note::

  The benchmarking module Tensile.py is written in Python3. The programs generate kernels and build all object files and C/C++ files used for benchmarking. Note that Tensile is **NOT** compatible with Python2.
