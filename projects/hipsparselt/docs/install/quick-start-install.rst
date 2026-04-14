.. meta::
   :description: Quick-start: Installing hipSPARSELt on Linux
   :keywords: hipSPARSELt, ROCm, install, Linux, quick-start

.. _install-linux-quick:

****************************************************************
Quick start hipSPARSELt installation guide
****************************************************************

The root directory of the `hipSPARSELt project <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparselt>`_
within the `rocm-libraries GitHub <https://github.com/ROCm/rocm-libraries>`_ contains the
helper bash script ``install.sh`` for building and installing hipSPARSELt on Ubuntu with a single command. The
script only accepts a few options and hardcodes configuration that can be specified by invoking
CMake directly. However, it's a great way to get started quickly and can serve as an example of how to build
and install hipSPARSELt. Some commands require ``sudo`` access, so the script might prompt you for a password.

.. code-block:: bash

   # Run install.sh script
   # Command line options:
   #   -h|--help            - prints help message
   #   -i|--install         - install after build
   #   -d|--dependencies    - install build dependencies
   #   -c|--clients         - build library clients too (combines with -i & -d)
   #   -g|--debug           - build with debug flag
   #   -k|--relwithdebinfo  - build with RelWithDebInfo

   ./install.sh -dc

For more detailed installation instructions, see :ref:`install-linux`.
