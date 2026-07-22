.. meta::
   :description: installation instructions for the hipTensor library
   :keywords: hipTensor, ROCm, library, API, tool, installation

.. _installation:

*****************
Install hipTensor
*****************

hipTensor is supported on AMD Instinct and Radeon GPUs by ROCm. See the `ROCm compatibility matrix
<https://rocm.docs.amd.com/en/docs-7.14.0/compatibility/compatibility-matrix.html>`__ for support information.

Before installing hipTensor, make sure your system meets the ROCm hardware,
software, and driver requirements. For instructions, see `Install AMD ROCm
<https://rocm.docs.amd.com/en/latest/install/rocm.html>`_.

Install hipTensor
-----------------

Package manager
^^^^^^^^^^^^^^^

1. `Install AMD ROCm <https://rocm.docs.amd.com/en/latest/install/rocm.html>`_.
   Remember to complete the `ROCm installation prerequisites
   <https://rocm.docs.amd.com/en/latest/install/rocm.html#prerequisites>`_ to
   install dependencies and configure GPU access permissions.

2. Install hipTensor for your GPU target. Replace ``<gfx-target>`` with your
   GPU architecture (for example, ``gfx942``, ``gfx950``, ``gfx1100``).

   .. tab-set::

      .. tab-item:: Ubuntu / Debian

         .. code-block:: bash

            sudo apt install amdrocm-hiptensor7.14-<gfx-target> amdrocm-hiptensor-dev7.14

      .. tab-item:: RHEL / Oracle Linux / Rocky Linux

         .. code-block:: bash

            sudo dnf install amdrocm-hiptensor7.14-<gfx-target> amdrocm-hiptensor-devel7.14

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-hiptensor7.14-<gfx-target> amdrocm-hiptensor-devel7.14

Tarball
^^^^^^^

The standard ROCm tarball installation includes hipTensor. No additional steps
are required. For details on ROCm tarball installation, refer to `Install AMD
ROCm <https://rocm.docs.amd.com/en/latest/install/rocm.html>`_ and select the
Tarball installation method.

Uninstall hipTensor
-------------------

Package manager
^^^^^^^^^^^^^^^

Replace ``<gfx-target>`` with the GPU architecture used during installation.

.. tab-set::

   .. tab-item:: Ubuntu / Debian

      .. code-block:: bash

         sudo apt autoremove amdrocm-hiptensor7.14-<gfx-target> amdrocm-hiptensor-dev7.14

   .. tab-item:: RHEL / Oracle Linux / Rocky Linux

      .. code-block:: bash

         sudo dnf remove amdrocm-hiptensor7.14-<gfx-target> amdrocm-hiptensor-devel7.14

   .. tab-item:: SLES

      .. code-block:: bash

         sudo zypper remove amdrocm-hiptensor7.14-<gfx-target> amdrocm-hiptensor-devel7.14

Tarball
^^^^^^^

The standard ROCm uninstallation process removes hipTensor. No additional steps
are required. Refer to the `Uninstalling
<https://rocm.docs.amd.com/en/latest/install/rocm.html#uninstalling>`_ section
and select the Tarball installation method.
