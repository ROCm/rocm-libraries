.. meta::
   :description: Building and installing rocALUTION on Linux
   :keywords: rocALUTION, ROCm, library, API, tool, Linux, building, installing

.. _linux-installation:

************************************
Installing rocALUTION on Linux
************************************

rocALUTION can be installed along with ROCm as a single-node, accelerator-enabled library. To configure rocALUTION to use a different configuration such as a multi-node configuration, it must be be built from source code. 

ROCm must be installed before building rocALUTION. Use the version of rocALUTION that corresponds to the installed version of ROCm.

The rocALUTION source code is available from `https://github.com/ROCm/rocALUTION <https://github.com/ROCm/rocALUTION>`_.

rocALUTION depends on the following ROCm components:

* `rocBLAS <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html>`_
* `rocSPARSE <https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/index.html>`_
* `rocRAND <https://rocm.docs.amd.com/projects/rocRAND/en/latest/index.html>`_
* `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_

Ensure that these components are installed before installing rocALUTION. See their respective documentation for installation instructions.

For multi-node configurations, download and install `OpenMP <https://www.openmp.org/>`_ and `MPI <https://www.mcs.anl.gov/research/projects/mpi/>`_.

Create a ``build`` directory in the ``rocALUTION`` root directory. Change directory to the ``build`` directory:

.. code:: shell

  mkdir build
  cd build

Use CMake to generate a makefile. The ``ROCM_PATH`` directive is required. Set it to point to the ROCm installation location.

The following optional directives can also be set:

* ``SUPPORT_HIP``: Build rocALUTION with HIP support. ``ON`` by default. 
* ``SUPPORT_OMP``: Build rocALUTION with OpenMP support. ``ON`` by default.
* ``SUPPORT_MPI``: Build rocALUTION with MPI. Set this to ``ON`` for multi-node support. ``OFF`` by default.
* ``BUILD_SHARED_LIBS``: Build rocALUTION as shared library. ``ON`` by default. ``ON`` is the recommended configuration.
* ``BUILD_EXAMPLES``: Build rocALUTION examples. ``ON`` by default.

For example, to build rocALUTION with MPI support, run this command:

.. code:: shell

  cmake .. -DSUPPORT_MPI=ON -DROCM_PATH=/opt/rocm/

Use ``make`` to build rocALUTION and ``make install`` to install the rocALUTION library under the ROCm installation directory.
 
.. code:: shell

  make
  make install


