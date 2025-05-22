  .. meta::
    :description: Install rocPRIM on Linux
    :keywords: install, rocPRIM, AMD, ROCm, source code, cmake, Linux

********************************************************************
Installing rocPRIM on Linux 
********************************************************************

rocPRIM is installed on Linux using CMake. CMake is also used to build rocPRIM examples, tests, and benchmarks.

The rocPRIM source code is available from the `rocPRIM GitHub Repository <https://github.com/ROCm/rocPRIM>`_. 

Use a branch that matches the version of ROCm installed on your system.

After cloning rocPRIM, create the ``build`` directory under the ``rocPRIM`` directory. Change directory to ``build``.

.. code:: shell
    
    cd rocPRIM
    mkdir build
    cd build

Set the ``CXX`` environment variable to ``hipcc``:

.. code:: shell

    export CXX=hipcc

You can build and install the rocPRIM library without any examples, tests, or benchmarks by running ``cmake`` followed by ``make install``:

.. code::

    cmake ../.
    make install

To build examples, tests, or benchmarks, use the appropriate CMake directive: 

* ``BUILD_TEST``: Set to ``ON`` to build the CTests. ``OFF`` by default.
* ``BUILD_EXAMPLE``: Set to ``ON`` to build examples. ``OFF`` by default.
* ``BUILD_BENCHMARK``: Set to ``ON`` to build benchmarking tests. ``OFF`` by default.
* ``BUILD_DOCS``: Set to ``ON`` to build a local copy of the rocPRIM documentation. ``OFF`` by default.
* ``AMDGPU_TARGETS``: Set this to a specific architecture target or set of architecture targets. When not set, examples, tests, and benchmarks are built for gfx803, gfx900, gfx906, and gfx908 architectures. The list of targets must be separated by a semicolon (``;``).
* ``ONLY_INSTALL``: Set to ``ON`` to ignore any example, test, or benchmark build instructions. ``OFF`` by default.

Run ``make`` after ``cmake`` to build the examples, tests, and benchmarks, then run ``make install``. For example, to build tests run:

.. code:: 

    export CXX=hipcc
    cmake -DBUILD_TEST=ON ../.
    make
    sudo make install
