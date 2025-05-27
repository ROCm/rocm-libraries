.. meta::
  :description: Using rocThrust in a CMake project
  :keywords: rocThrust, ROCm, cmake, find_package

*******************************************
How to use rocThrust in a CMake project
*******************************************

To use rocThrust in your own project, add the following lines to your CMakeLists file:

.. code::  

    # On ROCm rocThrust requires rocPRIM
    find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")

    # "/opt/rocm" - default install prefix
    find_package(rocthrust REQUIRED CONFIG PATHS "/opt/rocm/rocthrust")

    [...]

    # include rocThrust headers and roc::rocprim_hip target
    target_link_libraries(<your_target> roc::rocthrust)

