.. meta::
  :description: Installing ROCm Performance Primitives
  :keywords: rpp, ROCm Performance Primitives, ROCm, documentation, installing

********************************************************************
Installing ROCm Performance Primitives
********************************************************************

ROCm Performance Primitives (RPP) supports HIP and CPU-only backends. The CPU backend is also referred to as the HOST backend.

Starting with ROCm 10.1, RPP is built and delivered as part of `TheRock <https://github.com/ROCm/TheRock>`_, the unified ROCm build system. Earlier standalone RPP releases were delivered with ROCm 7.2.x and prior.

.. note::

    RPP packages built by TheRock are currently built with audio augmentations disabled (``RPP_AUDIO_SUPPORT=OFF``). Build :doc:`from source <./rpp-build-and-install>` to enable audio augmentation support.

A :doc:`package installer <./rpp-install-with-installer>` is available for installing either only the RPP runtime, or the RPP runtime and development packages.

RPP can also be :doc:`built from source <./rpp-build-and-install>`.

After installing RPP, :doc:`verify the installation using the test suite <./rpp-verify-install>`.
