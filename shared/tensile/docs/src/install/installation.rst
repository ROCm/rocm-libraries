.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Tensile installation, GEMM, Tensor, tensor, Build Tensile, Run benchmarks

.. _installation:

**************************
Build Tensile from source
**************************

This topic provides information required to install Tensile from source and run benchmarks.

.. _install-rocm:

Install ROCm
============

To begin, install ROCm for your platform. For installation instructions, refer to the `Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html>`_ or `Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/index.html>`_ installation guide.

.. note::

   If using Bash, set ``PATH=/opt/rocm/bin/:$PATH`` in your ``~/.bashrc`` and refresh your shell using ``source ~/.bashrc``.
   Alternatively, export the path for your current shell session using ``export PATH=/opt/rocm/bin/:$PATH``.

Install OS dependencies
=========================

.. note::

   The following steps are for Ubuntu. For other distributions, use the appropriate package manager.

1. Install dependencies:

   .. code-block::

    apt-get install libyaml-dev python3-yaml libomp-dev

2. Install one of the following, depending on your preferred Tensile data format. If both are installed, ``msgpack`` is preferred:

   .. code-block::

      apt-get install libmsgpack-dev    # If using the msgpack backend

      # OR

      apt-get install libtinfo-dev      # If using the YAML backend

3. Install build tools. For additional installation methods for the latest versions of CMake, see the `CMake installation <https://cliutils.gitlab.io/modern-cmake/installing/>`_ page.

   .. code-block::

      apt-get install build-essential cmake

Build Tensile from source
===========================

First, fetch Tensile standalone with git sparse checkout to avoid cloning all of rocm-libraries.

.. code-block:: bash

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set shared/tensile
   git checkout develop # or the branch you are starting from
   cd shared/tensile

Then, install Tensile from source in a virtual environment:

.. code-block:: bash

  python3 -m venv .venv
  source .venv/bin/activate
  pip3 install .

You can now run Tensile's Python applications, such as ``Tensile``, ``TensileCreateLibrary``, and others.
