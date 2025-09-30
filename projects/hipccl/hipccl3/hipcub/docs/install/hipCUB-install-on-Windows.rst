.. meta::
  :description: Build and install hipCUB with rmake.py
  :keywords: install, building, hipCUB, AMD, ROCm, source code, installation script, Windows

********************************************************************
Building and installing hipCUB on Windows
********************************************************************

You can use ``rmake.py`` to build and install hipCUB on Microsoft Windows. You can also use `CMake <./hipCUB-install-with-cmake.html>`_ if you want more build and installation options. 

:doc:`Clone the hipCUB project <./hipCUB-install-overview>`. ``rmake.py`` will be located in the ``hipcub`` root directory. 

To build and install hipCUB, run:

.. code:: shell

    python rmake.py -i

This command also downloads `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_ and installs it in ``C:\hipSDK``.

The ``-c`` option builds all clients, including the unit tests:

.. code:: shell

    python rmake.py -c

To see a complete list of ``rmake.py`` options, run:

.. code-block:: shell

    python rmake.py --help

 