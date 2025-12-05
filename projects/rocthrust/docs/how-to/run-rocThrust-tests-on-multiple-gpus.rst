.. meta::
  :description: Using multiple GPUs for testing
  :keywords: rocThrust, ROCm, testing, ctest, multiple GPUs, resource-spec

***************************************************
How to run tests on multiple GPUs
***************************************************

To run tests on multiple GPUs, you can use the CTest resource allocation feature. The feature requires two inputs:

  * the resource specification file which describes the resources available on the system, and
  * the ``RESOURCE_GROUPS`` property of tests, which describes the resources required by each individual test.

You can generate a resource specification file using the ``GenerateResourceSpec.cmake`` utility script. After you have cloned the ``rocThrust`` repository and built rocThrust with the ``-DBUILD_TESTS=ON`` option, change directory to the ``build`` directory and run:

.. code:: shell

    cmake -P ../cmake/GenerateResourceSpec.cmake

This will generate a ``resources.json`` file in the ``build`` directory.

When building rocThrust with the ``-DBUILD_TESTS=ON`` option, CMake has already added the default ``RESOURCE_GROUPS`` property for each test, which refers to the default GPU resource ``gpus`` in the generated ``resources.json`` file. Then use the ``--resource-spec-file`` option in your call to ``ctest`` with the specified number of jobs:

.. code:: shell

    ctest --resource-spec-file ./resources.json --parallel <number-of-jobs>

The tests will be run in a distributed manner across all the available GPUs in parallel. Note that the specified number of jobs can be independent to the number of GPUs on the system.

Alternatively, you can configure your tests using the ``AMDGPU_TEST_TARGETS`` CMake option. This option lets you specify the families of GPUs on which you want to run your tests. For example, if you have two GPUs from the ``gfx900`` family in your system, you can specify ``-DAMDGPU_TEST_TARGETS=gfx900`` to indicate that you only want that family of GPUs to be tested. This option will define the ``RESOURCE_GROUPS`` property to use the ``gfx900`` family of GPUs for the tests. If you don't set ``AMDGPU_TEST_TARGETS``, the tests will be run on the default GPU resource ``gpus`` as described above.


.. note::

    CTest resource allocation requires CMake 3.16 or later.
