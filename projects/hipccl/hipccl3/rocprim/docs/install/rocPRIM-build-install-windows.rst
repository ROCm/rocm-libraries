  .. meta::
    :description: Install rocThrust on Windows
    :keywords: install, rocThrust, AMD, ROCm, Windows

.. _install-with-cmake:

************************************************
Installing rocPRIM on Windows
************************************************

rocPRIM is installed on Windows using the ``rmake.py`` Python script. ``rmake.py`` is also used to build rocPRIM examples, tests, and benchmarks.

The rocPRIM source code is available from the `rocPRIM GitHub Repository <https://github.com/ROCm/rocPRIM>`_. 

Use a branch that matches the version of HIP SDK for Windows installed on your system.

Change directory to the cloned ``rocPRIM`` directory and run ``rmake.py -i`` to install rocPRIM to ``C:\hipSDK\include\``:

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
  
