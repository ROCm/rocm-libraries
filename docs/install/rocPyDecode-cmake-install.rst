.. meta::
  :description: Installing rocPyDecode using CMake
  :keywords: install, rocPyDecode, AMD, ROCm, CMake

********************************************************************
Building and installing rocPyDecode with CMake
********************************************************************

.. note::

    sudo access might be required to build and install with CMake.

rocPyDecode can be built using CMake. Once it has been built and installed, wheel and egg packages can be generated. See the :doc:`pip installation instructions <rocPyDecode-pip-install>` for information about how to generate wheel and egg files.

The rocPyDecode source code and its installation scripts are available from the `rocPyDecode GitHub Repository <https://github.com/ROCm/rocPyDecode>`_. 

The develop branch is the default rocPyDecode branch. The develop branch is intended for users who want to preview new features or contribute to the rocPyDecode code base. If you don't intend to preview new features or contribute to the codebase, clone the rocPyDecode branch that corresponds to your version of ROCm.

Before building and installing rocPyDecode, run `rocPyDecode-requirements.py <https://github.com/ROCm/rocPyDecode/blob/develop/rocPyDecode-requirements.py>`_ from the ``rocPyDecode`` root directory:

.. code:: shell

    python3 rocPyDecode-requirements.py

If you're installing rocPyDecode on Ubuntu 22.04, install libstdc++-12-dev:

.. code:: shell

  apt install libstdc++-12-dev

To build rocPyDecode, create a ``build`` directory in the rocPyDecode root directory:

.. code::

    mkdir build

Change directory to ``build``, then use ``cmake`` to generate a makefile.

.. code:: 

    cd build
    cmake ../

By default rocPyDecode will be built and installed for all versions of Python available on the system. In some instances, such as when rocPyDecode is being built and installed in a conda environment in a Docker container, there may be a hidden Python installation. rocPyDecode won't be built or installed for this version of Python unless the ``-DPYTHON_FOLDER_SUGGESTED=path_to_python_installation`` cmake directive is used. This will add the hidden Python installation to the list of targets. 

For example, this will add the version of Python installed at ``/opt/miniconda3/bin/python`` to the target list: 

.. code:: shell

    cmake -DPYTHON_FOLDER_SUGGESTED=/opt/miniconda3/bin/python3 ../


You can also build and install rocPyDecode for a specific Python version. Use the ``-DPYTHON_VERSION_SUGGESTED=version_num`` cmake directive to build and install only for the specified Python version.

For example, the following command will build rocPyDecode only for Python 3.12: 

.. code:: shell

    cmake -DPYTHON_VERSION_SUGGESTED=3.12  ../


.. note::

    ``PYTHON_VERSION_SUGGESTED`` and ``PYTHON_FOLDER_SUGGESTED`` are mutually exclusive and can't be used together.

Once the makefile has been generated, make and install rocPyDecode:

.. code::
  
    make -j8
    sudo make install

You can also generate deb files and zipped tar files for distribution:

.. code::

    make package


You can then install the deb packages with ``apt install``. For example:

.. code:: 

    sudo apt install ./rocpydecode_0.4.0-local_amd64.deb
    sudo apt install ./rocpydecode-test_0.4.0-local_amd64.deb


 
