.. meta::
  :description: rocJPEG Installation Prerequisites
  :keywords: install, rocJPEG, AMD, ROCm, prerequisites, dependencies, requirements

********************************************************************
rocJPEG prerequisites
********************************************************************

rocJPEG requires ROCm 6.3 or later running on `accelerators based on the CDNA architecture <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_.

ROCm 6.3.0 or later must be installed before installing rocJPEG. See `Quick start installation guide <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html>`_ for detailed ROCm installation instructions.

rocJPEG can be installed on the following Linux environments:
  
* Ubuntu 22.04 or 24.04
* RHEL 8 or 9
* SLES: 15-SP5

The following prerequisites are installed by the package installer. If you are building and installing using the source code, use the `rocJPEG-setup.py <https://github.com/ROCm/rocJPEG/blob/develop/rocJPEG-setup.py>`_ setup script available in the rocJPEG GitHub repository to install these prerequisites. 

* AMD Clang++
* CMake version 3.10 or later
* Video Acceleration API (VA-API) - libva-amdgpu-dev is an AMD implementation for VA-API
* AMD VA Drivers
* libstdc++-12-dev for installations on Ubuntu 22.04 
