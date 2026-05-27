.. meta::
  :description: Run rocThrust unit tests on multiple GPUs
  :keywords: ROCm libraries, rocThrust, ROCm, CTest, benchmarks, testing

************************************************
Running rocThrust unit tests over multiple GPUs
************************************************

The `CTest resource allocation feature <https://cmake.org/cmake/help/latest/manual/ctest.1.html#resource-allocation>`__ can be used to distribute tests across multiple GPUs, accelerating testing when multiple GPUs of the same family are in a system. It can also be used to test multiple product families without having to set ``HIP_VISIBLE_DEVICES``.

.. note::

   CMake 3.18 or later is required.

When rocThrust is built with :doc:`cmake -DBUILD_TEST=ON <../install/source-build>`, the ``generate_resource_spec`` binary file is created. 


Use ``generate_resource_spec`` to create a resource specification file. The resource specification file is a JSON file that describes the GPU resources available on your system. For example:

.. code:: shell

    ./generate_resource_spec resources.json

To run tests in parallel, pass the resource specification file and the maximum number of tests to run in parallel to ``ctest``:

.. code:: shell

    ctest --resource-spec-file PATH_TO_RESOURCE_SPECIFICATION_FILE --parallel MAXIMUM_NUMBER_OF_PARALLEL_TESTS


To restrict tests to a single family of GPUs, use the ``AMDGPU_TEST_TARGETS`` option when :doc:`building rocThrust <../install/source-build>`.