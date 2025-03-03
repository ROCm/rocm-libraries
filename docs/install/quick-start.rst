.. meta::
  :description: Install rocDecode
  :keywords: install, rocDecode, AMD, ROCm


Installation
*********************************


Prerequisites
===============

* GFortran version 7.5.0 or newer. For more information, refer to the `GFortran website. <https://fortran-lang.org/learn/os_setup/install_gfortran/>`_

Building and testing hipFORT from source
==========================================

1. Ensure you have ``gfortran``, ``git``, ``cmake``, and HIP installed.
2. Build, install, and test hipFORT from the source with the following commands: 

   .. code-block:: 

      git clone https://github.com/ROCm/hipfort.git
      cd hipfort
      cmake -S. -Bbuild -DHIPFORT_INSTALL_DIR=/tmp/hipfort -DBUILD_TESTING=ON
      make -C build
      make -C build check
