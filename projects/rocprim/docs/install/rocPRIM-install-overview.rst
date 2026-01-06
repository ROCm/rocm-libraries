.. meta::
  :description: rocPRIM installation overview 
  :keywords: install, rocPRIM, AMD, ROCm, installation, overview, general

*********************************
rocPRIM installation overview 
*********************************

The rocPRIM source code is available from the `ROCm libraries GitHub Repository <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocprim>`_. Use sparse checkout when cloning the rocPRIM project:

.. code::

  git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
  cd rocm-libraries
  git sparse-checkout init --cone
  git sparse-checkout set projects/rocprim

Then use ``git checkout`` to check out the branch you need.

The develop branch is intended for users who want to preview new features or contribute to the rocPRIM code base.

If you don't intend to contribute to the rocPRIM code base and won't be previewing features, use a branch that matches the version of ROCm installed on your system.

rocPRIM can be built and installed on :doc:`Linux <./rocPRIM-build-install-linux>` or :doc:`Windows <./rocPRIM-build-install-windows>`.
