.. meta::
  :description: Build and install MIGraphX
  :keywords: build, install, MIGraphX, AMD, ROCm, rbuild, development, contributing

********************************************************************
Building MIGraphX
********************************************************************

ROCm must be installed prior to building MIGraphX. See `ROCm installation for Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_ for information on ROCm installation on Linux.

.. note::
  
  This method for building MIGraphX requires using ``sudo``.

1. Install `rocm-cmake`, `pip3`, `rocblas`, and `miopen-hip`:

    .. code:: shell

        sudo apt install -y rocm-cmake python3-pip rocblas miopen-hip
    
2. Install `rbuild <https://github.com/RadeonOpenCompute/rbuild>`_:

    .. code:: shell

        pip3 install --prefix /usr/local https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
    

3. Build MIGraphX source code:

    .. code:: shell

        sudo rbuild build -d depend -B build -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')
