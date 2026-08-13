.. meta::
  :description: Building and installing ROCm Performance Primitives
  :keywords: rpp, ROCm Performance Primitives, ROCm, documentation, installing, building, source code

**************************************************************************
Building and installing ROCm Performance Primitives
**************************************************************************

ROCm Performance Primitives (RPP) supports the HIP backend running on `ROCm-supported AMD GPUs and accelerators <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_, and also supports a CPU-only backend.

Building the HIP backend requires a working ROCm installation. See :doc:`Installing RPP <./rpp-install>` for the supported installation paths.

Building with TheRock
=========================================================================

RPP is built and released as part of `TheRock <https://github.com/ROCm/TheRock>`_, and building through TheRock produces the same artifacts as the released packages. Follow the `TheRock build instructions <https://github.com/ROCm/TheRock/blob/main/README.md>`_ to fetch the sources and configure the build, then build the RPP subproject:

.. code:: shell

    cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=<your-gpu-family>
    ninja -C build rpp+dist

Building standalone
=========================================================================

RPP can also be built directly from its source directory against an existing ROCm installation.

Clone the source code from the `ROCm/rocm-libraries <https://github.com/ROCm/rocm-libraries>`_ monorepo, where RPP is located under ``projects/rpp``:

.. code:: shell

    git clone https://github.com/ROCm/rocm-libraries.git
    cd rocm-libraries/projects/rpp

Then use the following commands to build and install RPP:

.. tab-set::

  .. tab-item:: HIP

    .. code:: shell

        mkdir build-hip
        cd build-hip
        cmake ../
        make -j8
        sudo make install

  .. tab-item:: CPU-only

    .. code:: shell

        mkdir build-cpu
        cd build-cpu
        cmake -DBACKEND=CPU ../
        make -j8
        sudo make install
