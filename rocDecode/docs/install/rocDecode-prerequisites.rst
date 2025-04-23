.. meta::
  :description: rocDecode Installation Prerequisites
  :keywords: install, rocDecode, AMD, ROCm, prerequisites, dependencies, requirements

********************************************************************
rocDecode prerequisites
********************************************************************

rocDecode requires ROCm 6.1 or later running on `accelerators based on the CDNA architecture <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_.

ROCm must be installed using the [AMDGPU installer](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html) with the ``rocm`` usecase:

.. code:: shell

  sudo amdgpu-install --usecase=rocm

rocDecode can be installed on the following Linux environments:
  
* Ubuntu 22.04, 24.04
* RHEL 8 or 9
* SLES: 15-SP5

The following prerequisites are installed by the package installer. If you are building and installing using the source code, use the `rocDecode-setup.py <https://github.com/ROCm/rocDecode/blob/develop/rocDecode-setup.py>`_ to install these prerequisites. 

.. note:: 

  To use the rocDecode samples, the ``rocdecode``, ``rocdecode-dev``, and ``rocdecode-test`` packages need to be installed.
  
  If you're installing using the rocDecode source code, the ``rocDecode-setup.py`` script must be run with ``--developer`` set to ``ON``.

* Libva-amdgpu-dev, an AMD implementation for Video Acceleration API (VA-API)
* AMD VA Drivers
* CMake version 3.10 or later
* AMD Clang++ Version 18.0.0 or later
* pkg-config
* FFmpeg runtime and headers
* libstdc++-12-dev for installations on Ubuntu 22.04 
