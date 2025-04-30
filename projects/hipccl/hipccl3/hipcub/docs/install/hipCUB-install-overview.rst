.. meta::
  :description: hipCUB installation overview 
  :keywords: install, hipCUB, AMD, ROCm, installation, overview, general

*********************************
hipCUB installation overview 
*********************************

The hipCUB source code is available from the `hipCUB GitHub Repository <https://github.com/ROCmSoftwarePlatform/hipCUB>`_. 

The develop branch is the default branch. The develop branch is intended for users who want to preview new features or contribute to the hipCUB code base.

If you don't intend to contribute to the hipCUB code base and won't be previewing features, use a branch that matches the version of ROCm installed on your system.

hipCUB can be built and installed with |rmake|_ on Windows, or `CMake <./hipCUB-install-with-cmake.html>`_ on both Windows and Linux.

.. |install| replace:: ``install``
.. _install: ./rocThrust-install-script.html

.. |rmake| replace:: ``rmake.py`` 
.. _rmake: ./hipCUB-install-on-Windows.html

CMake provides the most flexibility in building and installing hipCUB.