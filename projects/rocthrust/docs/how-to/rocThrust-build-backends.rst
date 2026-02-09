
.. meta::
   :description: How to build rocThrust applications with different backends
   :keywords: rocThrust, ROCm, cmake, CUDA, TBB, OpenMP, CPP

*******************************************************
Building rocThrust applications on different backends
*******************************************************

Applications that use the rocThrust libraries can be built for different backends.

rocThrust supports CPP, OpenMP, and TBB on HOST and device. NVIDIA CUDA is supported on device only.

The ``THRUST_HOST_SYSTEM``, ``_THRUST_HOST_SYSTEM_NAMESPACE``, ``THRUST_DEVICE_SYSTEM``, and ``_THRUST_DEVICE_SYSTEM_NAMESPACE`` build options specify the backend to use.


.. list-table::
    :header-rows: 1
    :widths: 52 13 13 13 13

    *   -
        - CPP
        - OpenMP
        - TBB
        - CUDA

    *   - **THRUST_HOST_SYSTEM**
        - 1
        - 2
        - 3
        - not supported

    *   - **_THRUST_HOST_SYSTEM_NAMESPACE**
        - cpp
        - omp
        - tbb
        - not supported

    *   - **THRUST_DEVICE_SYSTEM**
        - 4
        - 2
        - 3
        - 1

    *   - **_THRUST_DEVICE_SYSTEM_NAMESPACE**
        - cpp
        - omp
        - tbb
        - cuda


If ``THRUST_HOST_SYSTEM`` and ``_THRUST_HOST_SYSTEM_NAMESPACE`` are omitted, the application will be built for the HOST CPP backend.

If ``THRUST_DEVICE_SYSTEM`` and ``_THRUST_DEVICE_SYSTEM_NAMESPACE`` are omitted, the application will be built for the device CUDA backend.

To build the application, create the ``build`` directory under the same directory as your source code and CMakeLists file, and change directory to ``build``:

.. code:: shell

    mkdir build
    cd build

Run cmake with the appropriate build options for your backend:

.. code:: shell

   CXX=hipcc  [-DTHRUST_HOST_SYSTEM={1|2|3}
                -D__THRUST_HOST_SYSTEM_NAMESPACE={cpp|omp|tbb}]
                [-DTHRUST_HOST_SYSTEM={1|2|3|4}
                -D__THRUST_HOST_SYSTEM_NAMESPACE={cuda|omp|tbb|cpp}]

Then run ``make`` to build.
