 .. meta::
  :description: rocAL building and installing
  :keywords: rocAL, ROCm, API, documentation

.. _install:

********************************************************************
Building and installing rocAL from source code
********************************************************************

Before building and installing rocAL, ensure ROCm has been installed with the `AMDGPU installer <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html>`_ and the ``rocm`` usecase.

The rocAL source code is available from `https://github.com/ROCmSoftwarePlatform/rocAL <https://github.com/ROCmSoftwarePlatform/rocAL>`_. Use the rocAL version that corresponds to the installed version of ROCm.


rocAL supports both the HIP and OpenCL backends. 

rocAL is installed in the ROCm installation directory by default. If rocAL for both HIP and OpenCL backends will be installed on the system, each version must be installed in its own custom directory and not in the default directory. 


``rocAL_pybind`` is not supported on the OpenCL backend.

You can choose to use the |setup| setup script to install most :doc:`prerequisites <./rocAL-prerequisites>`


.. note::
  
  | TurboJPEG must be installed manually on SLES. 
  | To use FFmpeg on SLES and RedHat, the ``FFmpeg-dev`` package must be installed manually.


To build and install rocAL for the HIP backend, create the ``build_hip`` directory under the ``rocAL`` root directory. Change directory to ``build_hip``:

.. code:: shell
 
    mkdir build-hip
    cd build-hip

Use ``cmake`` to generate a makefile: 

.. code:: shell
  
    cmake ../

If rocAL will be built for both the HIP and OpenCL backends, use the ``-DCMAKE_INSTALL_PREFIX`` CMake directive to set the installation directory. For example:

.. code:: shell

    cmake -DCMAKE_INSTALL_PREFIX=/opt/hip_backend/


Run make:

.. code:: shell

    make 

Run ``cmake`` again to generate Python bindings for ``rocal_pybind`` then install:

.. code:: shell

  sudo cmake --build . --target PyPackageInstall
  sudo make install

The instructions to install rocAL for the OpenCL backend are similar to those for the HIP backend. Because OpenCL doesn't support ``rocal_pybind``, the second ``cmake`` command is omitted:

.. code:: shell

  mkdir build-ocl
  cd build-ocl
  cmake -DBACKEND=OPENCL ../
  make
  sudo make install

After the installation, the rocAL files will be installed under ``/opt/rocm/`` unless ``-DCMAKE_INSTALL_PREFIX`` was specified. If ``-DCMAKE_INSTALL_PREFIX`` was specified, the rocAL files will be installed under the specified directory.


To make and run the tests, use ``make test``.

.. |setup| replace:: ``rocAL-setup.py``
.. _openvx: https://github.com/ROCm/rocAL/blob/develop/rocAL-setup.py
