.. meta::
  :description: Using hipThreads in a CMake project
  :keywords: hipThreads, ROCm, cmake, find_package, link, include

.. _use-in-a-project:

*******************************************
How to use hipThreads in a CMake project
*******************************************

After :doc:`installing hipThreads <../install/install>`, add the following lines to your ``CMakeLists.txt`` to find, include, and link the library:

.. code-block:: cmake

    find_package(hipthreads REQUIRED)

    # ...

    target_link_libraries(my_target hipthreads::hipthreads)

The ``hipthreads::hipthreads`` target carries the public include directories, so the headers are available without any extra ``target_include_directories()`` call.
Include the umbrella headers you need from your sources:

.. code-block:: cpp

   #include <hip/thread>
   #include <hip/mutex>
   #include <hip/condition_variable>

If hipThreads isn't installed under ``$ROCM_PATH``, point CMake at it by adding ``-DCMAKE_PREFIX_PATH=hipthreads_install_prefix`` to your configure command.

The ``examples/`` directory in the repository contains complete, standalone CMake projects.
These include SAXPY, a ray tracer, sparse matrix multiply, and a llama3.c port.
Each example is split into ``stepN-*`` directories that port the code from CPU ``std::thread`` to hipThreads.
The hipThreads steps use ``find_package(hipthreads)`` as in the preceding ``CMakeLists.txt`` snippet, so they're a good starting point for a new project.
The earlier baseline steps are CPU-only and link ``Threads::Threads`` instead.
