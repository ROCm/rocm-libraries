.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Troubleshoot Tensile, Tensile support, Tensile help, Tensile troubleshooting, Tensile issues
.. highlight:: none

.. _troubleshooting:

******************
Troubleshooting
******************

This topic provides information required to help programmers and users to resolve common issues in Tensile.

=============================
Missing toolchain components
=============================

.. code-block::

   FileNotFoundError: ``amdclang++`` either not found or not executable in any search path

This error implies that Tensile can't locate one or more binaries required for proper program execution. This includes compilers, assemblers, linkers, and bundlers.

.. note::

   On Linux, the default installation location is ``/opt/rocm``.

   On Windows, the default installation location is ``C:\\Program Files\\AMD\\ROCm\\X.Y``, where ``X.Y`` identifies the major and minor version of the current ROCm installation.
   When the :doc:`HIP SDK <hip:index>` is installed on Windows, the variable ``HIP_PATH`` is set to the installation location of the HIP SDK.

There are two possible causes for this error:

- ROCm is not installed on the system. To resolve this issue, install ROCm by following the instructions at :ref:`install-rocm`.
- ROCm is installed, but in a non-default location and the binaries can't be found in the system ``PATH``.
In this case, add the installation location to the ``ROCM_PATH`` on Linux, ``HIP_PATH`` on Windows, or the system ``PATH`` on either.

- On Linux, use:

.. code-block:: shell

   export ROCM_PATH=<path_to_rocm>

- On Windows PowerShell, use:

.. code-block:: shell

   $env:HIP_PATH = "<path_to_rocm>\bin"``.
