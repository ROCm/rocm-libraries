

===================
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
