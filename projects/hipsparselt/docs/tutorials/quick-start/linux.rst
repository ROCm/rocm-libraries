.. meta::
   :description: Quick-start: Installing hipSPARSELt on Linux
   :keywords: hipSPARSELt, ROCm, install, Linux, quick-start

.. _install-linux-quick:

****************************************************************
Quick-start installation (Ubuntu)
****************************************************************

The root of the
`hipSPARSELt GitHub repository <https://github.com/ROCm/hipSPARSELt>`_ has a
helper bash script `install.sh` to build and install hipSPARSELt on Ubuntu with a single command. It
doesn't take a lot of options and hard-codes configuration that can be specified through invoking
CMake directly, but it's a great way to get started quickly and can serve as an example of how to build
and install. A few commands in the script need sudo access, so it may prompt you for a password.

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

For a more in-depth installation guide, see :ref:`install-linux`.
