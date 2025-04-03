.. meta::
  :description: Quick start installation guide for hipFORT.
  :keywords: install, hipFORT, AMD, ROCm, quick start


Quick start installation guide
*********************************

This topic discusses how to quickly build hipFORT from source.

Prerequisites
===============

hipFORT requires GFortran version 7.5.0 or newer.
For more information, see the `GFortran website. <https://fortran-lang.org/learn/os_setup/install_gfortran/>`_

Building and testing hipFORT from source
==========================================

1. Ensure you have installed ``gfortran``, ``git``, ``cmake``, and :doc:`HIP <hip:index>`.
2. Build, install, and test hipFORT from source using the following commands:

   .. code-block:: shell

      git clone https://github.com/ROCm/hipfort.git
      cd hipfort
      cmake -S. -Bbuild -DHIPFORT_INSTALL_DIR=/tmp/hipfort -DBUILD_TESTING=ON
      make -C build
      make -C build check
