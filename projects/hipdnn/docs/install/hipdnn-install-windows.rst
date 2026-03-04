.. meta::
  :description: hipDNN installation on Windows 
  :keywords: hipDNN, ROCm, install, Windows

.. _windows:

******************************
hipDNN installation on Windows
******************************

This guide demonstrates how to install hipDNN on Windows systems. Before you begin, ensure the required dependencies are installed on your system as outlined in :ref:`prerequisites`.
hipDNN supports Windows 10 and Windows 11 (recommended).

.. warning::

   Some GPU functionality and HIP-related tests are not currently supported on Windows.

.. note::

   Refer to the :ref:`llvm` section for approaches on how to manage the multiple Clang toolchain versions required for hipDNN.


To do a standalone build of hipDNN, you will need to set up a number of prerequisites.
The standalone build of hipDNN requires a subset of the full environment required for building TheRock. 

Refer to `TheRock Windows Support <https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md>`_ for a full Windows 11 build environment setup for TheRock (but do *not* perform a build of TheRock as this is generally not necessary for building hipDNN standalone).

Install hipDNN
==============

Follow these steps to manually install hipDNN for your Windows system.

.. tip:: 
   
   An automated PowerShell script is available to perform the steps outlined below. This script is provided as a convenience and may not work in all environments. 
   `Review the script <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/scripts/windows/windows_build_setup.ps1>`_ before running it to ensure it meets your needs.

Install Chocolatey Package Manager
----------------------------------

Use `Chocolatey <https://community.chocolatey.org/>`_ to streamline the environment setup. The Chocolatey client, ``choco``, is used in these instructions.

Install utilities
-----------------

These third-party tools are required to build hipDNN:

- Git (installed with both git and unix tools available on the windows ``PATH``)
- Visual Studio 2022 with C++ workload (easy way to get Windows libraries)
- CMake 3.25.2+
- Ninja
- Python 3

Using Chocolatey, install any of the missing required dependencies using an Administrative Command Prompt (or PowerShell):

.. code:: bash
   
   choco install visualstudio2022buildtools -y --params "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.VC.CMake.Project --add Microsoft.VisualStudio.Component.VC.ATL --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
   choco install git.install -y --params "'/GitAndUnixToolsOnPath'"
   choco install cmake --version=3.31.0 -y
   choco install ninja -y
   choco install python -y

Enable Windows 10 long paths
-----------------------------

A detailed description and instructions for enabling long paths on Windows 10+ are available `in the Windows App Development docs <https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later>`_.

From the Windows docs:

   In the Windows API (with some exceptions discussed in the following paragraphs), the maximum length for a path is MAX_PATH, which is defined as 260 characters. 
   A local path is structured in the following order: drive letter, colon, backslash, name components separated by backslashes, and a terminating null character. 
   For example, the maximum path on drive D is ``"D:\some 256-character path string<NUL>"`` where ``"<NUL>"`` represents the invisible terminating null character for the current system codepage. 
   (The characters ``<` `>`` are used here for visual clarity and cannot be part of a valid path string.)
   For example, you may hit this limitation if you are cloning a git repo that has long file names into a folder that itself has a long name.
   Starting in Windows 10, version 1607, MAX_PATH limitations have been removed from many common Win32 file and directory functions.
   The registry value ``HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem LongPathsEnabled`` (Type: ``REG_DWORD``) must exist and be set to ``1``. 
   The registry value will not be reloaded during the lifetime of the process. In order for all apps on the system to recognize the value, a reboot might be required because some processes may have started before the key was set.
   The following Administrative PowerShell command can be used to set this registry value:

   .. code:: PowerShell
   
      New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force


Enable Windows 10 symlinks
--------------------------

These instructions are summarized from these sources:

- `Enabling Symlinks on Windows <https://portal.perforce.com/s/article/3472>`_ 
- `Git symbolic links in Windows <https://stackoverflow.com/questions/5917249/git-symbolic-links-in-windows/59761201#59761201>`_  
- `Create symbolic links <https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/security-policy-settings/create-symbolic-links>`_

Verify that you can create symlinks by running ``mklink`` from a command window:

.. code:: bash
   
   echo "test" > mklinktest.txt
   mklink linkedfile.txt mklinktest.txt
   symbolic link created for link.txt <<===>> ExistingFile.txt

If you don't have the ability to create symlinks, you'll see:

.. code:: bash

   echo "test" > mklinktest.txt
   mklink linkedfile.txt mklinktest.txt
   You do not have sufficient privilege to perform this operation.

You can enable symlink creation by enabling `Developer Mode <https://www.wikihow.com/Enable-Developer-Mode-in-Windows-10>`_ in Windows 10/11:

- Windows 10: Go to **Settings** --> **Update & Security** --> **For Developers** --> **Developer Mode** --> Toggle **On** --> Click **Yes** to confirm.
- Windows 11: Go to **Settings** --> **System** --> **For Developers** --> **Developer Mode** --> Toggle **On**.

Restart your computer for the settings to take effect.

Configure Git
-------------

With ``long-paths`` and symlinks enabled, enable symlink and ``long-path`` support in Git:

.. code:: bash

   git config --global core.symlinks true
   git config --global core.longpaths true

.. tip:: 
   
   You can use ``git config --show-scope --show-origin core.symlinks`` and ``git config --show-scope --show-origin core.longpaths`` to determine what the current active git configuration is, and where that setting is configured.

Install Clang Toolchain
-----------------------

Though TheRock toolchain is used to build hipDNN, utilities such as clang-format are provided by Clang.

Download and unzip a recent ``20.X.X`` version of the `Clang Toolchain <https://github.com/llvm/llvm-project/releases?q=20>`_.

Unzip it to a path with no spaces. For example, after being unzipped to ``C:\dist\clang``, the bin folder will be located at ``C:\dist\clang\bin``.

Install ROCm SDK
----------------

Run ``amdgpu-arch.exe`` from the Clang release you just downloaded to find the GPU architecture you're using, and then record the result (it'll be something like ``gfx1103``).

For example:

.. code:: bash

   c:\dist\clang\bin\amdgpu-arch
   gfx1103

You can use the table at `TheROCK releases <https://github.com/ROCm/TheRock/blob/main/RELEASES.md#index-page-listing>`_ or the `CMake source <https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake#L43>`_ to determine the GFX Family. 
For example, the GPU architecture ``gfx1103`` is GFX Family ``gfx110X-all``.

Download a recent nightly Windows build tarball of TheRock ROCm SDK for your GFX Family from `ROCm SDK nightly tarballs <https://therock-nightly-tarball.s3.amazonaws.com/index.html>`_. 
For example: for ``gfx110X-all``, the most recent tarball available (at the time of this writing) is ``therock-dist-windows-gfx110X-all-7.10.0a20251103.tar.gz``.

Complete instructions and alternate methods for installing TheRock are available at `TheROCK releases <https://github.com/ROCm/TheRock/blob/main/RELEASES.md>`_.

.. note::

   If a nightly tarball is not available for your GFX Family, you  may be able to `build TheRock from source <https://github.com/ROCm/TheRock/tree/main#building-from-source>`_ or follow the `hipDNN Roadmap <https://github.com/ROCm/TheRock/blob/main/ROADMAP.md>`_ for more details.

Unzip the downloaded tarball to a path with no spaces. For example, after unzipped to ``C:\dist\therock`` the bin folder will be located at ``c:\dist\therock\bin``.

Set up environment variables
----------------------------

1. Add TheRock ``bin`` folder to the system ``PATH`` so that applications can find and load the DLL files. For example:

   .. code:: bash

      set PATH=C:\dist\therock\bin;%PATH%
   
   It isn't necessary to add the Clang toolchain to your system ``PATH`` to perform the build as these can be specified using the :ref:`llvm` option to cmake.

   The AMD toolchain should be discovered automatically. If not, refer to the :ref:`rocm-path` for additional ways to locate the toolchain.

2. Set the ``HIP_PLATFORM`` environment varilable:

   .. code:: bash

      set HIP_PLATFORM=amd
   

3. Optionally, you can set Ninja as the default generator so that ``-g Ninja`` doesn't need to be explicitly added to the ``cmake`` command line:
   
   .. code:: bash

      set CMAKE_GENERATOR=Ninja
   
Use ``hipconfig`` to check that TheRock is installed correctly and the ``PATH`` is setup correctly. The output from the command will show the detected ROCm path and ROCm clang path (``c:\dist\therock\`` will be replaced with the folder TheRock was installed to on your system). 
This command requires that TheRock ``bin`` directory is in your system ``PATH``.

.. code:: bash

   hipconfig -rocmpath -n --hipclangpath
   c:\dist\therock
   c:\dist\therock\lib\llvm\bin

Here's an example CMake configuration command (including ``ROCM_CMAKE_PATH`` for completeness, even though it isn't required when using the default value):

.. code:: cmake

   projects\hipdnn\build>set CMAKE_GENERATOR=Ninja
   projects\hipdnn\build>cmake -DGPU_TARGETS=gfx1103 -DROCM_CMAKE_PATH=c:/dist/therock -DCMAKE_PROGRAM_PATH=c:/dist/clang/bin ..
   -- Building for: Ninja
   -- Using ROCm Clang compilers from c:/dist/therock/lib/llvm/bin


.. note::

   When generating the project, be sure to set ``GPU_TARGETS`` to your GPU as auto-detection isn't currently on Windows. For example, ``cmake -DGPU_TARGETS=gfx1103 ..`` (replacing ``gfx1103`` with your GPU).


Clone the repository and build hipDNN
-------------------------------------

1. Clone the rocm-libraries repository with ``git sparse-checkout``:

   .. code:: bash

      git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
      cd rocm-libraries
      git sparse-checkout init --cone
      git sparse-checkout set projects/hipdnn dnn-providers/miopen-provider
      git checkout develop # or the branch you are starting from

2. Build hipDNN:

   .. code:: bash

      cd rocm-libraries/projects/hipdnn
      mkdir build && cd build

      # Configure with Ninja (recommended)
      cmake -GNinja ..

      # Build and run all tests
      # Note that some tests may take several minutes to complete
      ninja check

   Refer to the :ref:`target` section below for info on additional build targets.

3. Install hipDNN.

   Set the install path with a build configuration. Refer to the :ref:`configure` section to determine which build configuration suits your workflow. 

   .. code:: bash

      sudo ninja install

Notes
~~~~~

* Do *not* open the ``x64 Native Tools Command Prompt for VS 2022`` as this will interfere with the ROCm SDK and Clang toolchain.
* Do *not* set ``ROCM_PATH`` in your environment as this will interfere with toolchain detection. If used, specify it using the ``-DROCM_PATH=`` option to cmake.
* When generating the project, CMake will warn about a ``clang-format`` or ``clang-tidy`` mismatch. That can be resolved by installing the missing version of the toolchain to a parallel directory and setting the :ref:`llvm` variable accordingly.
* Generating the project files may take longer than on Linux, but should complete within a few minutes.
* You may want to limit the number of threads used by Ninja when building hipDNN so that your computer is not bogged-down by the build. You can use the ``ninja -j`` option to set the number of threads to something smaller than the number of threads available on your CPU.
* To reduce build time, the ``-DENABLE_CLANG_TIDY=OFF`` option can be used to disable ``clang-tidy`` check during development. Similarly the ``-DENABLE_CLANG_FORMAT=OFF`` option can be used to disable ``clang-format``.

