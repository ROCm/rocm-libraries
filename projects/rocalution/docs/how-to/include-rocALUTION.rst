.. meta::
   :description: Building and installing rocALUTION on Windows
   :keywords: rocALUTION, ROCm, library, API, tool, Windows, installation, building, HIP SDK

*****************************************
Using rocALUTION as a prebuilt packages
*****************************************

rocALUTION can be installed on Windows 11 or Windows 10 using the AMD HIP SDK installer.

The simplest way to use rocALUTION in your code is to use ``CMake`` that requires you to add the SDK installation location to your
`DCMAKE_PREFIX_PATH`. Note that you need to use quotes as the path contains a space, e.g.,

.. code:: shell

    -DCMAKE_PREFIX_PATH="C:\Program Files\AMD\ROCm\5.5"


After CMake configuration, in your ``CMakeLists.txt`` use:

.. code:: shell

    find_package(rocalution)

    target_link_libraries( your_exe PRIVATE roc::rocalution )

Once rocALUTION is installed, you can find ``rocalution.hpp`` in the HIP SDK ``\\include\\rocalution``
directory. Use only the installed file in the user application if needed.
You must include ``rocalution.hpp`` header file in the user code to make calls
into rocALUTION, so that the rocALUTION import library and dynamic link library become the respective link-time and run-time
dependencies for the user application.
