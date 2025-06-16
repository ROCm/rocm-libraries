.. meta::
   :description: Porting from CUDA to hipSPARSELt
   :keywords: hipSPARSELt, ROCm, porting from CUDA, porting

.. _porting:

**********************************************************************
Porting from NVIDIA CUDA to hipSPARSELt
**********************************************************************

The hipSPARSELt interface is compatible with cuSPARSELt APIs. Porting a CUDA application that
originally calls the cuSPARSELt API to an application that calls the hipSPARSELt API should be relatively
straightforward.

For example, the hipSPARSELt matrix multiplication interface in the API is:

.. code-block:: c

   hipsparseStatus_t hipsparseLtMatmul(const hipsparseLtHandle_t*     handle,
                                       const hipsparseLtMatmulPlan_t* plan,
                                       const void*                    alpha,
                                       const void*                    d_A,
                                       const void*                    d_B,
                                       const void*                    beta,
                                       const void*                    d_C,
                                       void*                          d_D,
                                       void*                          workspace,
                                       hipStream_t*                   streams,
                                       int32_t                        numStreams);

hipSPARSELt assumes matrices ``A``, ``B``, ``C``, ``D``, and ``workspace`` are allocated in the GPU memory space and filled with
data. Users are responsible for copying data to and from the host and device memory.
