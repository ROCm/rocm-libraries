  .. meta::
    :description: Install rocThrust on Windows
    :keywords: install, rocThrust, AMD, ROCm, Windows

.. _install-with-cmake:

************************************************
Installing rocPRIM on Windows
************************************************

rocPRIM is installed on Windows using the ``rmake.py`` Python script. ``rmake.py`` is also used to build rocPRIM examples, tests, and benchmarks.

In the :doc:`cloned <./rocPRIM-install-overview>` ``rocprim`` directory, run ``rmake.py -i`` to install rocPRIM to ``C:\hipSDK\include\``:

.. code:: shell

    cd rocPRIM

    python3 rmake.py -i

Use the ``-c`` option to build the examples, tests, and benchmarks:

    python3 rmake.py -c

You can also build Microsoft Visual Studio projects for the examples, tests, and benchmarks.

Change directory to the ``example``, ``test``, or ``benchmark`` directory, and create the ``build`` directory. For example:

.. code:: shell

    cd benchmark
    mkdir build

Change directory to the ``build`` directory, and run ``cmake``:

.. code:: shell

    cd build
    cmake ../.

The Visual Studio projects and solutions will be created in the ``build`` directory.
  
