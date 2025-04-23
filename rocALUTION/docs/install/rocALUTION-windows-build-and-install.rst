.. meta::
   :description: Building and installing rocALUTION on Windows
   :keywords: rocALUTION, ROCm, library, API, tool, Windows, installation, building, HIP SDK

.. _windows-installation:

*********************************
Installing rocALUTION on Windows
*********************************

rocALUTION can be installed along with HIP SDK for Windows as a single-node, accelerator-enabled library. To configure rocALUTION to use a different configuration such as a multi-node configuration, it must be built from source code. 

HIP SDK for Windows, as well as the following components, must be installed before building rocALUTION:

* `rocBLAS <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html>`_
* `rocSPARSE <https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/index.html>`_
* `rocRAND <https://rocm.docs.amd.com/projects/rocRAND/en/latest/index.html>`_
* `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_

See :doc:`rocALUTION prerequisites <./rocALUTION-prerequisites>` for the full list of requirements.

The rocALUTION source code is available from `https://github.com/ROCmSoftwarePlatform/rocALUTION <https://github.com/ROCmSoftwarePlatform/rocALUTION>`_. Use the version of the source code that corresponds to the installed version of HIP SDK for Windows.

To determine which version of HIP SDK for Windows is installed, run:

.. code:: shell

   hipcc --version

.. note::

   If ``hipcc`` can't be found on your system, add ``%HIP_PATH%/bin`` to your ``PATH`` variable.
   

Use the ``rmake.py`` script to build and install rocALUTION.

To build the library without installing it, run ``rmake.py`` without any arguments:

.. code:: shell

   python3 rmake.py

The rocALUTION library files will be saved to ``build\release\include\rocalution``.

To build the library and install it, use the ``-i`` argument:

.. code:: shell

   python3 rmake.py -i

The library files will be built and installed in ``%HIP_PATH%\include\rocalution``. 

To build the library and its clients, and install the library files, use the ``-ci`` argument:

.. code:: shell 

   python3 rmake.py -ci

You can also omit the ``i`` argument and build the clients and the library without installing the library.
