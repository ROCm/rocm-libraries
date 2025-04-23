.. meta::
  :description: rocPyDecode Installation Prerequisites
  :keywords: install, rocPyDecode, AMD, ROCm, prerequisites, dependencies, requirements

********************************************************************
rocPyDecode prerequisites
********************************************************************

rocPyDecode requires Ubuntu 22.04 or 24.04 with ROCm running on `accelerators based on the CDNA architecture <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_.

ROCm needs to be installed using the `AMDGPU installer <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html>`_ with the ``rocm`` usecase:

.. code:: shell

  sudo amdgpu-install --usecase=rocm
    
rocPyDecode has the following prerequisites:

* CMake 3.12 or later
* Python3 3.9 or later
* Python3 pip
* PyBind11
* rocDecode
* pkg-config
* FFmpeg runtime and headers
* DLPack
* NumPy, for running tests and samples

All prerequisites with the exception of NumPy are installed with the `rocPyDecode-requirements.py <https://github.com/ROCm/rocPyDecode/blob/develop/rocPyDecode-requirements.py>`_ script. NumPy can be installed with ``pip``, and libstdc++-12-dev can be installed with a Linux package installer.
