.. meta::
  :description: Build and install rocThrust with rmake.py
  :keywords: install, building, rocThrust, AMD, ROCm, source code, installation script, Windows

********************************************************************
Building and installing rocThrust on Windows with rmake.py
********************************************************************

You can use ``rmake.py`` to build and install rocThrust on Windows. You can also use `CMake <./rocThrust-install-with-cmake.html>`_ if you want more build and installation options.

:doc:`Clone the rocThrust project <./rocThrust-install-overview>`. ``rmake.py`` will be located in the ``rocthrust`` root directory.

To build and install rocThrust with ``rmake.py``, run:

.. code:: shell

    python rmake.py -i

This command also downloads `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_ and installs it in ``C:\hipSDK``.

The ``-c`` option builds all clients, including the unit tests:

.. code:: shell

    python rmake.py -c

To see a complete list of ``rmake.py`` options, run:

.. code-block:: shell

    python rmake.py --help
