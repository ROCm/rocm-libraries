.. meta::
  :description: How to use hipFORT
  :keywords: fortran, hipFORT, hipfc, compiler, AMD, ROCm, usage guide

***********************************
Using hipFORT in your application
***********************************

The following topic provides instructions and tips for using hipFORT.

Fortran interfaces
===================

hipFORT provides interfaces to the following HIP and ROCm-only libraries:

*  **HIP**:
  
   *  HIP runtime
   *  hipBLAS
   *  hipSPARSE
   *  hipFFT
   *  hipRAND
   *  hipSOLVER

*  **ROCm-only**:

   *  rocBLAS
   *  rocSPARSE
   *  rocFFT
   *  rocRAND
   *  rocSOLVER

.. note:: 

   hipSOLVER interfaces only work with AMD GPUs.

While the HIP-based interfaces and libraries let you write portable code for both AMD and CUDA devices,
the ROCm-only libraries can only be used with AMD devices.

The available interfaces depend on which Fortran compiler was used to compile the hipFORT modules and libraries.
The interfaces use the ``iso_c_binding`` module, so the minimum requirement is a Fortran compiler that supports
the Fortran 2003 standard (`f2003`). These interfaces typically require passing ``type(c_ptr)`` variables
and the number of bytes to memory management. Some examples include ``hipMalloc`` and math library routines like ``hipblasDGEMM``.

If your compiler can understand the Fortran 2008 (`f2008`) code constructs in the hipFORT source and test files,
additional interfaces are compiled into the hipFORT modules and libraries. 
These interfaces take Fortran (array) variables, the number of elements instead of ``type(c_ptr)`` variables,
and the number of bytes, respectively. Therefore, they reduce the chance of introducing compile-time and runtime errors
into your code and make it easier to read.

.. note:: 

   If you plan to use the `f2008` interfaces, GFortran version 7.5.0 or newer is recommended.
   Problems can occur with older versions.

Examples
--------

To see some examples for the `f2003` and `f2008` interfaces, see the :doc:`hipFORT samples <../tutorials/examples>`.

Supported HIP and ROCm APIs
---------------------------

The current set of hipFORT interfaces is derived from ROCm version 4.5.0. The following tables list the supported APIs:

* :doc:`HIP API <../doxygen/html/md_input_supported_api_hip>`
* :doc:`hipBLAS API <../doxygen/html/md_input_supported_api_hipblas>` 
* :doc:`hipFFT API <../doxygen/html/md_input_supported_api_hipfft>` 
* :doc:`hipRAND API <../doxygen/html/md_input_supported_api_hiprand>`
* :doc:`hipSOLVER API <../doxygen/html/md_input_supported_api_hipsolver>`
* :doc:`hipSPARSE API <../doxygen/html/md_input_supported_api_hipsparse>`
* :doc:`rocBLAS API <../doxygen/html/md_input_supported_api_rocblas>`
* :doc:`rocFFT API <../doxygen/html/md_input_supported_api_rocfft>`
* :doc:`rocRAND API <../doxygen/html/md_input_supported_api_rocrand>`
* :doc:`rocSOLVER API <../doxygen/html/md_input_supported_api_rocsolver>`
* :doc:`rocSPARSE API <../doxygen/html/md_input_supported_api_rocsparse>`

.. note:: 

   Use the **Search** function from the hipFORT table of contents to get more information on the arguments for an interface.

