.. meta::
  :description: Using hipThreads in a CMake project
  :keywords: hipThreads, ROCm, cmake, find_package, link, include

.. _use-in-a-project:

*******************************************
Using hipThreads in a CMake project
*******************************************

To add hipThreads to your CMake project, add the following lines to your ``CMakeLists.txt`` file:

.. code-block:: cmake

   find_package(hipthreads REQUIRED)

   [...]

   target_link_libraries(target_name hipthreads::hipthreads)


Include the hipThreads headers in your code:

.. code-block:: cpp

   #include <hip/thread>
   #include <hip/mutex>
   #include <hip/condition_variable>


If hipThreads was installed in its default location under the ROCm installation, make sure that your ``ROCM_PATH`` environment variable is pointing to your ROCm root directory before building your application.

If hipThreads wasn't installed in the default location, build with the ``CMAKE_PREFIX_PATH`` CMake option pointing to the hipThreads root directory.

The GPU architecture of the consumer project must match the architecture hipThreads was built or packaged for. A mismatch produces undefined device-symbol errors at runtime.

On Windows, CMake doesn't auto-detect the GPU architecture. You must set ``CMAKE_HIP_ARCHITECTURES`` explicitly.

On Linux, CMake auto-detects the GPU on the local machine. That detection doesn't guarantee a match with a prebuilt ``amdrocm-threads`` package. Package names can include an ``llvm_target`` suffix such as ``gfx942``.
