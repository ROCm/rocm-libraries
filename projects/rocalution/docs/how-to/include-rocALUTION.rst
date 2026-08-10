.. meta::
   :description: Use rocALUTION as a prebuilt package
   :keywords: rocALUTION, ROCm, library, API, tool, Linux, install, CMake, HIP SDK

******************************************
Use rocALUTION as a prebuilt package
******************************************

You can install rocALUTION from the ROCm software stack on Linux.
Install it with your distribution package manager or as part of a ROCm installation.

The simplest way to use rocALUTION in your code is with CMake.
Add the ROCm installation location to ``CMAKE_PREFIX_PATH``:

.. code-block:: shell

   -DCMAKE_PREFIX_PATH=/opt/rocm

In your ``CMakeLists.txt``:

.. code-block:: cmake

   find_package(rocalution)
   target_link_libraries(your_exe PRIVATE roc::rocalution)

After you install rocALUTION, include ``rocalution.hpp`` in your source code.
The rocALUTION shared library becomes a link-time and run-time dependency
for your application.
