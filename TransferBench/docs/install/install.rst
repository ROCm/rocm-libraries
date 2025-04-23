.. meta::
  :description: TransferBench is a utility to benchmark simultaneous transfers between user-specified devices (CPUs or GPUs)
  :keywords: Build TransferBench, Install TransferBench

.. _install-transferbench:

---------------------------
Installing TransferBench
---------------------------

This topic describes how to build TransferBench.

Prerequisite
---------------

* Install ROCm stack on the system to obtain :doc:`HIP runtime <hip:index>`
* Install ``libnuma`` on the system
* `Enable AMD IOMMU <https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#iommu-configuration-systems-with-256-cpu-threads>`_ and set to passthrough for AMD Instinct cards

Building TransferBench
------------------------

To build TransferBench using Makefile, use:

.. code-block:: bash

  make

To build TransferBench using CMake, use:

.. code-block:: bash

  mkdir build
  cd build
  CXX=/opt/rocm/bin/hipcc cmake ..
  make

.. note::

  If ROCm is installed in a folder other than ``/opt/rocm/``, set ``ROCM_PATH`` appropriately.

Building documentation
-----------------------

To build documentation locally, use:

.. code-block:: bash

  cd docs
  pip3 install -r ./sphinx/requirements.txt
  python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html

NVIDIA platform support
--------------------------

You can build TransferBench to run on NVIDIA platforms using native NVIDIA CUDA Compiler Driver (NVCC).

To build with native NVCC, use:

.. code-block:: bash

  make

TransferBench looks for NVCC in ``/usr/local/cuda`` by default. To modify the location of NVCC, use environment variable `CUDA_PATH`:

.. code-block:: bash

  CUDA_PATH=/usr/local/cuda make
