.. meta::
  :description: Using HIPSTDPAR with rocThrust
  :keywords: rocThrust, ROCm, HIPSTDPAR, installation

*******************************************
Using HIPSTDPAR
*******************************************

The HIPSTDPAR header files contain overloads of the C++ Standard Library parallel algorithms that offload the parallel algorithms to AMD accelerators and GPUs.

.. note::

    HIPSTDPAR is installed with rocThrust and can only be installed through rocThrust.

To use the HIPSTDPAR headers, compile your code with the ``--hipstdpar`` flag. For more information about the effects of compiling with ``--hipstdpar``, see `C++ Standard Parallelism Offload Support: Compiler And Runtime <https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/clang/html/HIPSupport.html#c-standard-parallelism-offload-support-compiler-and-runtime>`_.

Both the AMD fork of LLVM and the upstream LLVM support offloading parallel algorithms.

Tests for validating HIPSTDPAR implementations are enabled when rocThrust is built with ``BUILD_HIPSTDPAR_TEST=ON``. 

HIPSTDPAR requires rocThrust, `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_, and `TBB <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html>`_.


.. note:: 

    If you're using libstdc++ 9 or libstdc++ 10, your application might fail to compile due to incompatibilities between legacy TBB and oneTBB. See the `oneAPI Threading Building Blocks Release Notes <https://www.intel.com/content/www/us/en/developer/articles/release-notes/intel-oneapi-threading-building-blocks-release-notes.html>`_ for more information.
