.. meta::
  :description: Installing ROCm Performance Primitives
  :keywords: rpp, ROCm Performance Primitives, ROCm, documentation, installing

.. _installation:

***********
Install RPP
***********

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`./rpp-build`.

.. _install-rocm:

Install the ROCm Core SDK
=========================

RPP requires the ROCm Core SDK on Linux for HIP backends. For the most
complete installation on Linux, we recommend that developers use the
``amdrocm-core-sdk`` meta package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including RPP.
See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.
