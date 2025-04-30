.. meta::
   :description: Porting from CUDA to hipSPARSELt
   :keywords: hipSPARSELt, ROCm, porting from CUDA, porting

.. _porting:

**********************************************************************
 Porting from CUDA to hipSPARSELt
**********************************************************************

The hipSPARSELt interface is compatible with cuSPARSELt APIs. Porting a CUDA application that
originally calls the cuSPARSELt API to an application that calls the hipSPARSELt API should be relatively
straightforward.

For example, the hipSPARSELt matrix multiplication interface is:

Matrix multiplication API

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

hipSPARSELt assumes matrix A, B, C, D and workspace are allocated in GPU memory space filled with
data. Users are responsible for copying data from/to the host and device memory.
