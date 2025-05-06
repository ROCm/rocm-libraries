.. meta::
  :description: Using multiple GPUs for testing
  :keywords: rocThrust, ROCm, testing, ctest, multiple GPUs, resource-spec

***************************************************
How to run tests on multiple GPUs
***************************************************

To run tests on multiple GPUs, you can configure your tests using the ``AMDGPU_TEST_TARGETS`` option or you can use CTest resource allocation.

The ``AMDGPU_TEST_TARGETS`` CTest option lets you specify the families of GPUs on which you want to run your tests. For example, if you have two GPUs from the gfx900 family in your system, you can specify ``-DAMDGPU_TEST_TARGETS=gfx900`` when you configure your test to specify that you only want that family of GPUs to be tested. If you don't set ``AMDGPU_TEST_TARGETS``, the tests will be run on the default device in your system. 

You can use CTest resource allocation to run tests in a distributed manner across multiple GPUs and test multiple product families from one invocation. 

CTest resource allocation requires a resource specification file. You can generate a resource specification file using the ``GenerateResourceSpec.cmake`` utility script. 

After you have cloned the ``rocThrust`` repository and built rocThrust with the ``-DBUILD_TESTS=ON`` option, change directory to the ``build`` directory and run:

.. code:: shell

    ../cmake/GenerateResourceSpec.cmake

This will generate a ``resources.json`` file in the ``build`` directory. Use the ``resources.json`` file in your call to ``ctest``. 

For example, if you have two compatible GPUs in your system, run:

.. code:: shell

    ctest --resource-spec-file ./resources.json --parallel 2


.. note:: 

    CTest resource allocation requires CMake 3.16 or later.






