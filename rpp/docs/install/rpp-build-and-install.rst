.. meta::
  :description: Building and installing ROCm Performance Primitives 
  :keywords: rpp, ROCm Performance Primitives, ROCm, documentation, installing, building, source code

**************************************************************************
Building and installing ROCm Performance Primitives
**************************************************************************

ROCm Performance Primitives (RPP) supports HIP and OpenCL backends running on `accelerators based on the CDNA architecture <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_, and supports CPU-only backends on CPUs that support PCIe™ atomics.

On OpenCL and HIP backends, RPP requires ROCm installed with the `AMDGPU installer <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html>`_ and the ``rocm`` usecase:

.. code:: shell
    
    sudo amdgpu-install --usecase=rocm

Clone the source code from the `RPP GitHub repository <https://github.com/ROCm/rpp>`_, then use the following commands to build and install RPP:

.. tab-set::
 
  .. tab-item:: HIP

    .. code:: shell

        mkdir build-hip
        cd build-hip
        cmake ../
        make -j8
        sudo make install
  
  .. tab-item:: OpenCL

    .. code:: shell

        mkdir build-ocl
        cd build-ocl
        cmake -DBACKEND=OPENCL ../
        make -j8
        sudo make install

  .. tab-item:: CPU-only

    .. code:: shell

        mkdir build-cpu
        cd build-cpu
        cmake -DBACKEND=CPU ../
        make -j8
        sudo make install 
    


