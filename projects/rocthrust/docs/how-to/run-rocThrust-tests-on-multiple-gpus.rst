.. meta::
  :description: Using multiple GPUs for testing
  :keywords: rocThrust, ROCm, testing, ctest, multiple GPUs, resource-spec

***************************************************
How to run tests on multiple GPUs
***************************************************

To run tests on multiple GPUs, CTest provides the resource allocation feature. The feature requires two inputs:

  * the resource specification file which describes the resources available on the system, and
  * the ``RESOURCE_GROUPS`` property of tests, which describes the resources required by each individual test.

When you build rocThrust with the ``-DBUILD_TESTS=ON`` option, an executable named ``generate_resource_spec`` will be built and running it will generate the resource specification file. For example, in your build folder

.. code:: shell

    ./generate_resource_spec resources.json

will generate a resource specification file named ``resources.json``. This file describes the GPUs available on your system. In addition, CMake also defines the ``RESOURCE_GROUPS`` property for each test that refers to the default GPU resource ``gpus`` in ``resources.json``. Then when your run ``ctest`` with specified number of jobs:

.. code:: shell

    ctest --resource-spec-file ./resources.json --parallel <number-of-jobs>

tests will be distributed across the available GPUs and run concurently up to the specified number of jobs. Note that the specified number of jobs can be independent to the number of GPUs on the system. CTest will schedule tests in a way that running them simultaneously does not oversubscribe the GPUs.

Alternatively, you can configure your tests using the ``AMDGPU_TEST_TARGETS`` CMake option. This option lets you specify the families of GPUs on which you want to run your tests. For example, if you have two GPUs from the ``gfx900`` family in your system, you can specify ``-DAMDGPU_TEST_TARGETS=gfx900`` to indicate that you only want that family of GPUs to be tested. This option will define the ``RESOURCE_GROUPS`` property to use the ``gfx900`` resources in ``resources.json``. If you don't set ``AMDGPU_TEST_TARGETS``, the tests will be run on the default GPU resource ``gpus`` as described above.


.. note::

    CTest resource allocation requires CMake 3.16 or later.
