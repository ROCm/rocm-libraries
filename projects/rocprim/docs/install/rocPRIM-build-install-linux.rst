  .. meta::
    :description: Install rocPRIM on Linux
    :keywords: install, rocPRIM, AMD, ROCm, source code, cmake, Linux

********************************************************************
Installing rocPRIM on Linux 
********************************************************************

rocPRIM is installed on Linux using CMake. CMake is also used to build rocPRIM examples, tests, and benchmarks.

Create the ``build`` directory under the :doc:`cloned <./rocPRIM-install-overview>` ``rocprim`` directory. Change directory to ``build``.

.. code:: shell
    
    mkdir build
    cd build

Set the ``CXX`` environment variable to ``hipcc``:

.. code:: shell

    export CXX=hipcc

You can build and install the rocPRIM library without any examples, tests, or benchmarks by running ``cmake`` followed by ``make install``:

.. code::

    cmake ../.
    make install

Use the appropriate CMake directive: 

* ``BUILD_TEST``: Set to ``ON`` to build the CTests. ``OFF`` by default.
* ``BUILD_EXAMPLE``: Set to ``ON`` to build examples. ``OFF`` by default.
* ``BUILD_DOCS``: Set to ``ON`` to build a local copy of the rocPRIM documentation. ``OFF`` by default.
* ``BUILD_BENCHMARK``: Set to ``ON`` to build benchmarking tests. ``OFF`` by default.
* ``BENCHMARK_CONFIG_TUNING``: Set to ``ON`` to find the best kernel configuration parameters for benchmarking. Turning this on might increase compilation time significantly. ``OFF`` by default. 
* ``BENCHMARK_USE_AMDSMI``: Set to ``ON`` to let benchmarks use AMD SMI to output more GPU statistics. ``OFF`` by default.
* ``AMDGPU_TARGETS``: Set this to build the library, examples, tests, examples, and benchmarks for specific architecture targets. When not set, the examples, tests, and benchmarks are built for gfx803, gfx900:xnack-, gfx906:xnack-, gfx908:xnack-, gfx90a:xnack-, gfx90a:xnack+, gfx942;gfx950, gfx1030, gfx1100, gfx1101, gfx1102, gfx1151, gfx1200, and gfx1201 architectures. The list of targets must be separated by a semicolon (``;``).
* ``AMDGPU_TEST_TARGETS``: Set this to build tests for a subset of the architectures specified by ``AMDGPU_TARGETS``. When set, copies of the same test will be generated for each of the architectures listed. These tests can be run using ``ctest -R "TARGET_ARCHITECTURE"``. The list of targets must be separated by a semicolon (``;``).
* ``USE_SYSTEM_LIB``: Set to ``ON`` to use the installed ``ROCm`` libraries when building the tests. Off by default. For this option to take effect, ``BUILD_TEST`` must be ``ON``.
* ``ONLY_INSTALL``: Set to ``ON`` to ignore any example, test, or benchmark build instructions. ``OFF`` by default.

 Run ``make`` after ``cmake`` to build the examples, tests, and benchmarks, then run ``make install``. For example, to build tests run:

.. code:: 

    export CXX=hipcc
    cmake -DBUILD_TEST=ON ../.
    make
    sudo make install
