.. meta::
  :description: Installing MIGraphX using the package installer
  :keywords: install, MIGraphX, AMD, ROCm, package installer

********************************************************************
Installing MIGraphX
********************************************************************

ROCm must be installed before installing MIGraphX. See `ROCm installation for Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_ for information on how to install ROCm on Linux.

Installing MIGraphX using the package installer is sufficient for users who want to use the MIGraphX API.

If you want to develop for MIGraphX and contribute to the source code, see `Building MIGraphX <./building_migraphx.html>`_ and `Developing for MIGraphX <../dev/contributing-to-migraphx.html>`_

The package installer will install all the prerequisites needed for MIGraphX.

Use the following command to install MIGraphX: 

  .. code:: shell
  
    sudo apt update && sudo apt install -y migraphx
