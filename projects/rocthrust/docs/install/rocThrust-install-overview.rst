.. meta::
  :description: rocThrust installation overview
  :keywords: install, rocThrust, AMD, ROCm, installation, overview, general

*********************************
rocThrust installation overview
*********************************

The rocThrust source code is available from the `ROCm libraries GitHub Repository <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocthrust>`_. Use sparse checkout when cloning the rocThrust project:

.. code::

  git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
  cd rocm-libraries
  git sparse-checkout init --cone
  git sparse-checkout set projects/rocthrust

Then use ``git checkout`` to check out the branch you need.

The develop branch is intended for users who want to preview new features or contribute to the rocThrust code base.

If you don't intend to contribute to the rocThrust code base and won't be previewing features, use a branch that matches the version of ROCm installed on your system.

rocThrust can be built and installed with |install|_ on Linux, |rmake|_ on Windows, or `CMake <./rocThrust-install-with-cmake.html>`_ on Windows and Linux.

.. |install| replace:: ``install``
.. _install: ./rocThrust-install-script.html

.. |rmake| replace:: ``rmake.py``
.. _rmake: ./rocThrust-rmake-install.html

CMake provides the most flexibility in building and installing rocThrust.
