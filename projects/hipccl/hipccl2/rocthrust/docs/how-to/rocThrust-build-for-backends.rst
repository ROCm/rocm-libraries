.. meta::
  :description: Building rocThrust for different backends
  :keywords: rocThrust, ROCm, HIPSTDPAR, installation

*******************************************
Building rocThrust for different backends
*******************************************

API calls can run either on the device (GPU) system or on the host (CPU) system. The system on which computations are run depends on the execution policy used in the code, as well as the options that were set when rocThrust was :doc:`built and installed <../install/rocThrust-install-overview>`. 

Two build options are used to set the backend when the different execution policies are used. ``THRUST_DEVICE_SYSTEM`` sets the backend for the ``thrust::device`` execution policy and  ``THRUST_HOST_SYSTEM`` sets the backend for the ``thrust::host`` policy.

The options for ``THRUST_DEVICE_SYSTEM`` are:

.. list-table:: 
    :widths: 20 80
    :header-rows: 1

    * - Backend
      - Description

    * - ``HIP``
      - | `HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_ backend for device acceleration. 
        | Requires a HIP-aware clang compiler such as hipcc. 
        | Default setting.

    * - ``TBB``
      - | |oneTBB|_ backend. 
        | Parallelizes computations on the host using oneTBB with no device acceleration. 
        | Requires a compiler that supports oneTBB.

    * - ``OMP``
      - | |OMP|_ backend. 
        | Parallelizes computations on the host using OpenMP with no device acceleration. 
        | Requires a compiler that supports OpenMP.

    * - ``CPP``
      - | Uses the g++ or clang++ compiler, and the standard C++ library. 
        | Forces sequential computation on the host with no device acceleration.


.. note:: 

    rocThrust examples and benchmarks require device acceleration. rocThrust must be built with ``THRUST_DEVICE_SYSTEM=HIP`` to use its examples and benchmarks.

The options for ``THRUST_HOST_SYSTEM`` are:

.. list-table:: 
    :widths: 20 80
    :header-rows: 1

    * - Backend
      - Description 

    * - ``CPP``
      - | The standard C++ library. 
        | Uses the g++ or clang++ compiler for sequential computations on the host with no device acceleration. 
        | Default setting.

    * - ``OMP`` 
      - | OpenMP backend. 
        | Parallelizes host-side operations using OpenMP. 
        | Requires a compiler that supports OpenMP.

    * - ``TBB`` 
      - | oneTBB backend. 
        | Parallelizes host-side operations using oneTBB.
        | Requires a compiler that supports oneTBB.

.. note::

    If ``THRUST_DEVICE_SYSTEM`` is set to ``OMP``, ``TBB``, or ``CPP``, then ``THRUST_HOST_SYSTEM`` must be set to the same backend.

rocThrust will link to the rocPRIM libraries even when the ``thrust::host`` execution policy is used and ``THRUST_DEVICE_SYSTEM`` is set to ``OMP``, ``TBB``, or ``CPP``. For full host-side execution, without linking to rocPRIM, rocThrust must be built with ``LINK_HIP_DEVICE_LIBS=OFF``.  Setting ``LINK_HIP_DEVICE_LIBS=OFF`` at build time will prevent rocThrust from linking to the rocPRIM libraries.

When rocThrust is built with ``LINK_HIP_DEVICE_LIBS=OFF``, the ``thrust::device`` policy will be ignored and the API calls will run on the host device.

For example, to build rocThrust with no device acceleration, using only the g++ compiler:

.. code:: shell

    `ROCM_PATH=/opt/rocm CXX=g++ cmake -B build -DBUILD_BENCHMARK=OFF -DBUILD_TEST=OFF -DTHRUST_HOST_SYSTEM=CPP -DTHRUST_DEVICE_SYSTEM=CPP -DLINK_HIP_DEVICE_LIBS=OFF`
 
For more information about build options and how to set them, see :doc:`building rocThrust with CMake <../install/rocThrust-install-with-cmake>` and :doc:`building rocThrust with rmake <../install/rocThrust-rmake-install>`.


.. |reg| raw:: html

    &reg;

.. |oneTBB| replace:: Intel\ |reg| oneAPI Threading Building Blocks (oneTBB)
.. _oneTBB: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html

.. |OMP| replace:: OpenMP\ |reg| 
.. _OMP: https://www.openmp.org