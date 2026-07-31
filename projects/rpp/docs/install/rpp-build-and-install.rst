.. meta::
  :description: Building and installing ROCm Performance Primitives
  :keywords: rpp, ROCm Performance Primitives, ROCm, documentation, installing, building, source code

**************************************************************************
Building and installing ROCm Performance Primitives
**************************************************************************

ROCm Performance Primitives (RPP) supports the HIP backend running on `accelerators based on the CDNA architecture <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_, and supports CPU-only backends on CPUs that support PCIe™ atomics.

For the HIP backend, RPP requires ROCm installed with the `AMDGPU installer <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/amdgpu-installer-index.html>`_ and the ``rocm`` usecase:

.. code:: shell

    sudo amdgpu-install --usecase=rocm

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
