/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/
#ifndef HIPTENSOR_API_HPP
#define HIPTENSOR_API_HPP

#include "hiptensor_types.hpp"
#include "internal/hiptensor_utility.hpp"

//! @brief Allocates an instance of hiptensorHandle_t on the heap and updates the handle pointer
//!
//! @details Creates hipTensor handle for the associated device.
//! In order for the  hipTensor library to use a different device, set the new
//! device to be used by calling hipInit(0) and then create another hipTensor
//! handle, which will be associated with the new device, by calling
//! hiptensorCreate().
//! @param[out] handle Pointer to hiptensorHandle_t pointer
//! @returns HIPTENSOR_STATUS_SUCCESS on success and an error code otherwise
hiptensorStatus_t hiptensorCreate(hiptensorHandle_t** handle);

//! @brief De-allocates the instance of hiptensorHandle_t
//! @param[out] handle Pointer to hiptensorHandle_t
//! @returns HIPTENSOR_STATUS_SUCCESS on success and an error code otherwise
hiptensorStatus_t hiptensorDestroy(hiptensorHandle_t* handle);

//! @brief Initializes a tensor descriptor
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[out] desc Pointer to the allocated tensor descriptor object.
//! @param[in] numModes Number of modes.
//! @param[in] lens Extent of each mode(lengths) (must be larger than zero).
//! @param[in] strides stride[i] denotes the displacement (stride) between two consecutive
//! elements in the ith-mode. If stride is NULL, generalized packed column-major memory
//! layout is assumed (i.e., the strides increase monotonically from left to right).
//! @param[in] dataType Data type of the stored entries.
//! @param[in] unaryOp Unary operator that will be applied to the tensor.
//! @retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully.
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
hiptensorStatus_t hiptensorInitTensorDescriptor(const hiptensorHandle_t*     handle,
                                                hiptensorTensorDescriptor_t* desc,
                                                const uint32_t               numModes,
                                                const int64_t                lens[],
                                                const int64_t                strides[],
                                                hipDataType                  dataType,
                                                hiptensorOperator_t          unaryOp);

//! @brief Returns the description string for an error code
//! @param[in] error Error code to convert to string.
//! @retval the error string.
const char* hiptensorGetErrorString(const hiptensorStatus_t error);

//! @brief Tensor permutation
//! @details This function computes the permuation operation:
//! \f[
//! B_{\Pi^B(i_0,i_1,...,i_n)} = \alpha \Psi(A_{\Pi^A(i_0,i_1,...,i_n)})
//! \f]
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[in] alpha Scaling factor for A of the type typeScalar. Pointer to the host memory.
//! If alpha is zero, A is not read and the corresponding unary operator is not applied.
//! @param[in] A Multi-mode tensor of type typeA with nmodeA modes. Pointer to the GPU-accessible memory.
//! @param[in] descA A descriptor that holds information about the data type, modes, and strides of A.
//! @param[in] modeA Array of size descA->numModes that holds the names of the modes of A.
//! @param[in,out] B Multi-mode tensor of type typeB with nmodeB modes. Pointer to the GPU-accessible memory.
//! @param[in] descB A descriptor that holds information about the data type, modes, and strides of B.
//! @param[in] modeB Array of size descB->numModes that holds the names of the modes of B
//! @param[in] typeScalar data type of alpha
//! @param[in] stream HIP stream to perform all operations.
//! @retval HIPTENSOR_STATUS_NOT_SUPPORTED if the combination of data types or operations is not supported
//! @retval HIPTENSOR_STATUS_INVALID_VALUE if tensor dimensions or modes have an illegal value
//! @retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully without error
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
hiptensorStatus_t hiptensorPermutation(const hiptensorHandle_t*           handle,
                                       const void*                        alpha,
                                       const void*                        A,
                                       const hiptensorTensorDescriptor_t* descA,
                                       const int32_t                      modeA[],
                                       void*                              B,
                                       const hiptensorTensorDescriptor_t* descB,
                                       const int32_t                      modeB[],
                                       const hipDataType                  typeScalar,
                                       const hipStream_t                  stream);

//! @brief Performs an element-wise tensor operation on two input tensors.
//!
//! @details This function computes the element-wise operation:
//! \f[
//! D_{\Pi^C(i_0,i_1,...,i_n)} = \Phi_{AC}(\alpha \Psi_A(A_{\Pi^A(i_0,i_1,...,i_n)}), \gamma \Psi_C(C_{\Pi^C(i_0,i_1,...,i_n)}))
//! \f]
//! where:
//!   - \f$D\f$ is the output tensor.
//!   - \f$A\f$ and \f$C\f$ are the input tensors.
//!   - \f$\alpha\f$ and \f$\gamma\f$ are scalar scaling factors.
//!   - \f$\Psi_A\f$ and \f$\Psi_C\f$ are unary operators (applied only if \f$\alpha\f$ and \f$\gamma\f$ are non-zero).
//!   - \f$\Phi_{AC}\f$ is a binary element-wise operator.
//!   - \f$\Pi^A\f$ and \f$\Pi^C\f$ represent mode permutations.
//!
//! @param[in] handle Opaque handle to the hipTensor library context.
//! @param[in] alpha Scaling factor for tensor A (host memory).
//! @param[in] A Input tensor A (GPU memory).
//! @param[in] descA Descriptor for tensor A, including data type, modes, and strides.
//! @param[in] modeA Array of mode names for tensor A (host memory).
//! @param[in] gamma Scaling factor for tensor C (host memory).
//! @param[in] C Input tensor C (GPU memory).
//! @param[in] descC Descriptor for tensor C, including data type, modes, and strides.
//! @param[in] modeC Array of mode names for tensor C (host memory).
//! @param[out] D Output tensor D (GPU memory).
//! @param[in] descD Descriptor for tensor D (must match descC).
//! @param[in] modeD Array of mode names for tensor D (host memory).
//! @param[in] opAC Element-wise binary operator \f$\Phi_{AC}\f$. 
//! @param[in] typeScalar Scalar data type for intermediate computations.
//! @param[in] stream stream for execution.
//! @return HIPTENSOR_STATUS_NOT_SUPPORTED if data type or operation combination is unsupported.
//! @return HIPTENSOR_STATUS_INVALID_VALUE if tensor dimensions or modes are invalid.
//! @return HIPTENSOR_STATUS_SUCCESS if the operation completes successfully.
//! @return HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
hiptensorStatus_t hiptensorElementwiseBinary(const hiptensorHandle_t*           handle,
                                             const void*                        alpha,
                                             const void*                        A,
                                             const hiptensorTensorDescriptor_t* descA,
                                             const int32_t                      modeA[],
                                             const void*                        gamma,
                                             const void*                        C,
                                             const hiptensorTensorDescriptor_t* descC,
                                             const int32_t                      modeC[],
                                             void*                              D,
                                             const hiptensorTensorDescriptor_t* descD,
                                             const int32_t                      modeD[],
                                             hiptensorOperator_t                opAC,
                                             hipDataType                        typeScalar,
                                             hipStream_t                        stream);

//! @brief Performs an element-wise tensor operation with three input tensors.
//!
//! @details This function computes the element-wise operation:
//! \f[ D_{\Pi^C(i_0,i_1,...,i_n)} = \Phi_{ABC}(\Phi_{AB}(\alpha \Psi_A(A_{\Pi^A(i_0,i_1,...,i_n)}), \beta \Psi_B(B_{\Pi^B(i_0,i_1,...,i_n)})), \gamma \Psi_C(C_{\Pi^C(i_0,i_1,...,i_n)})) \f]
//!
//! Tensor modes can appear in any order, providing flexibility. However, the following restrictions apply:
//!   - Modes present in \f$A\f$ or \f$B\f$ must also be present in the output tensor \f$D\f$. Modes only in inputs would imply contraction, which is handled by hiptensorContraction or hiptensorReduction.
//!   - Each mode can appear at most once in each tensor.
//!
//! @param[in] handle Opaque handle to the hipTensor library context.
//! @param[in] alpha Scaling factor for tensor \f$A\f$ (host memory).
//! @param[in] A Input tensor \f$A\f$ (GPU memory).
//! @param[in] descA Descriptor for tensor \f$A\f$, including data type, modes, and strides.
//! @param[in] modeA Array of mode names for tensor \f$A\f$ (host memory).
//! @param[in] beta Scaling factor for tensor \f$B\f$ (host memory).
//! @param[in] B Input tensor \f$B\f$ (GPU memory).
//! @param[in] descB Descriptor for tensor \f$B\f$, including data type, modes, and strides.
//! @param[in] modeB Array of mode names for tensor \f$B\f$ (host memory).
//! @param[in] gamma Scaling factor for tensor \f$C\f$ (host memory).
//! @param[in] C Input tensor \f$C\f$ (GPU memory).
//! @param[in] descC Descriptor for tensor \f$C\f$, including data type, modes, and strides.
//! @param[in] modeC Array of mode names for tensor \f$C\f$ (host memory).
//! @param[out] D Output tensor \f$D\f$ (GPU memory). May alias input tensors if memory layouts match.
//! @param[in] descD Descriptor for tensor \f$D\f$ (must match descC).
//! @param[in] modeD Array of mode names for tensor \f$D\f$ (host memory).
//! @param[in] opAB Element-wise binary operator \f$\Phi_{AB}\f$.
//! @param[in] opABC Element-wise binary operator \f$\Phi_{ABC}\f$.
//! @param[in] typeScalar Data type for scalars alpha, beta, and gamma, and for intermediate computations.
//! @param[in] stream stream for execution.
//! @return HIPTENSOR_STATUS_SUCCESS if the operation completes successfully.
//! @return HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
//! @return HIPTENSOR_STATUS_INVALID_VALUE if input data is invalid.
//! @return HIPTENSOR_STATUS_ARCH_MISMATCH if the device is not ready or the architecture is unsupported.
hiptensorStatus_t hiptensorElementwiseTrinary(const hiptensorHandle_t*           handle,
                                              const void*                        alpha,
                                              const void*                        A,
                                              const hiptensorTensorDescriptor_t* descA,
                                              const int32_t                      modeA[],
                                              const void*                        beta,
                                              const void*                        B,
                                              const hiptensorTensorDescriptor_t* descB,
                                              const int32_t                      modeB[],
                                              const void*                        gamma,
                                              const void*                        C,
                                              const hiptensorTensorDescriptor_t* descC,
                                              const int32_t                      modeC[],
                                              void*                              D,
                                              const hiptensorTensorDescriptor_t* descD,
                                              const int32_t                      modeD[],
                                              hiptensorOperator_t                opAB,
                                              hiptensorOperator_t                opABC,
                                              hipDataType                        typeScalar,
                                              const hipStream_t                  stream);

//! @brief Computes the alignment requirement for a given pointer and descriptor.
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[in] ptr Pointer to the respective tensor data.
//! @param[in] desc Tensor descriptor for ptr data.
//! @param[out] alignmentRequirement Largest alignment requirement that ptr can fulfill (in bytes).
//! @retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully.
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
//! @retval HIPTENSOR_STATUS_INVALID_VALUE  if the unsupported parameter is passed.
hiptensorStatus_t hiptensorGetAlignmentRequirement(const hiptensorHandle_t*           handle,
                                                   const void*                        ptr,
                                                   const hiptensorTensorDescriptor_t* desc,
                                                   uint32_t* alignmentRequirement);

//! @brief Initializes a contraction descriptor for the tensor contraction problem.
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[out] desc Tensor contraction problem descriptor.
//! @param[in] descA A descriptor that holds information about tensor A.
//! @param[in] modeA Array with 'nmodeA' entries that represent the modes of A.
//! @param[in] alignmentRequirementA Alignment reqirement for A's pointer (in bytes);
//! @param[in] descB A descriptor that holds information about tensor B.
//! @param[in] modeB Array with 'nmodeB' entries that represent the modes of B.
//! @param[in] alignmentRequirementB Alignment reqirement for B's pointer (in bytes);
//! @param[in] modeC Array with 'nmodeC' entries that represent the modes of C.
//! @param[in] descC A descriptor that holds information about tensor C.
//! @param[in] alignmentRequirementC Alignment requirement for C's pointer (in bytes);
//! @param[in] modeD Array with 'nmodeD' entries that represent the modes of D (must be identical to modeC).
//! @param[in] descD A descriptor that holds information about tensor D (must be identical to descC).
//! @param[in] alignmentRequirementD Alignment requirement for D's pointer (in bytes);
//! @param[in] typeCompute Datatype for the intermediate computation  T = A * B.
//! @retval HIPTENSOR_STATUS_SUCCESS Successful completion of the operation.
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle or tensor descriptors are not initialized.
hiptensorStatus_t hiptensorInitContractionDescriptor(const hiptensorHandle_t*           handle,
                                                     hiptensorContractionDescriptor_t*  desc,
                                                     const hiptensorTensorDescriptor_t* descA,
                                                     const int32_t                      modeA[],
                                                     const uint32_t alignmentRequirementA,
                                                     const hiptensorTensorDescriptor_t* descB,
                                                     const int32_t                      modeB[],
                                                     const uint32_t alignmentRequirementB,
                                                     const hiptensorTensorDescriptor_t* descC,
                                                     const int32_t                      modeC[],
                                                     const uint32_t alignmentRequirementC,
                                                     const hiptensorTensorDescriptor_t* descD,
                                                     const int32_t                      modeD[],
                                                     const uint32_t         alignmentRequirementD,
                                                     hiptensorComputeType_t typeCompute);

//! @brief Narrows down the candidates for the contraction problem.
//! @details This function gives the user finer control over the candidates that
//! the subsequent call to @ref hiptensorInitContractionPlan is allowed to
//! evaluate. Currently, the backend provides few set of algorithms(DEFAULT).
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[out] find Narrowed set of candidates for the contraction problem.
//! @param[in] algo Allows users to select a specific algorithm.
//! @retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully.
//! @retval HIPTENSOR_STATUS_NOT_SUPPORTED If a specified algorithm is not supported
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle or find is not initialized.
hiptensorStatus_t hiptensorInitContractionFind(const hiptensorHandle_t*    handle,
                                               hiptensorContractionFind_t* find,
                                               const hiptensorAlgo_t       algo);

//! @brief Computes the size of workspace for a given tensor contraction
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[in] desc Tensor contraction descriptor.
//! @param[in] find Narrowed set of candidates for the contraction problem.
//! @param[in] pref Preference to choose the workspace size.
//! @param[out] workspaceSize Size of the workspace (in bytes).
//! @retval HIPTENSOR_STATUS_SUCCESS Successful completion of the operation.
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
//! @retval HIPTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
hiptensorStatus_t hiptensorContractionGetWorkspaceSize(const hiptensorHandle_t* handle,
                                                       const hiptensorContractionDescriptor_t* desc,
                                                       const hiptensorContractionFind_t*       find,
                                                       const hiptensorWorksizePreference_t     pref,
                                                       uint64_t* workspaceSize);

//! @brief Initializes the contraction plan for a given tensor contraction problem
//! @details This function creates a contraction plan for the problem by applying
//! hipTensor's heuristics to select a candidate. The creaated plan can be reused
//! multiple times for the same tensor contraction problem. The plan is created for
//! the active HIP device.
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[out] plan Opaque handle holding the contraction plan (i.e.,
//! the algorithm that will be executed, its runtime parameters for the given
//! tensor contraction problem).
//! @param[in] desc Tensor contraction descriptor.
//! @param[in] find Narrows down the candidates for the contraction problem.
//! @param[in] workspaceSize Available workspace size (in bytes).
//! @retval HIPTENSOR_STATUS_SUCCESS If a viable candidate has been found.
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle or find or desc is not
//! initialized.
hiptensorStatus_t hiptensorInitContractionPlan(const hiptensorHandle_t*                handle,
                                               hiptensorContractionPlan_t*             plan,
                                               const hiptensorContractionDescriptor_t* desc,
                                               const hiptensorContractionFind_t*       find,
                                               const uint64_t workspaceSize);

//! @brief Computes the tensor contraction \f[ D = alpha * A * B + beta * C \f]
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! HIP Device associated with the handle must be same/active at the time,0
//! the plan was created.
//! @param[in] plan Opaque handle holding the contraction plan (i.e.,
//! the algorithm that will be executed, its runtime parameters for the given
//! tensor contraction problem).
//! @param[in] alpha Scaling parameter for A*B of data type 'typeCompute'.
//! @param[in] A Pointer to A's data in device memory.
//! @param[in] B Pointer to B's data in device memory.
//! @param[in] beta Scaling parameter for C of data type 'typeCompute'.
//! @param[in] C Pointer to C's data in device memory.
//! @param[out] D Pointer to D's data in device memory.
//! @param[out] workspace Workspace pointer in device memory
//! @param[in] workspaceSize Available workspace size.
//! @param[in] stream HIP stream to perform all operations.
//! @retval HIPTENSOR_STATUS_SUCCESS Successful completion of the operation.
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle or pointers are not
//! initialized.
//! @retval HIPTENSOR_STATUS_CK_ERROR if some unknown composable_kernel (CK)
//! error has occurred (e.g., no instance supported by inputs).
hiptensorStatus_t hiptensorContraction(const hiptensorHandle_t*          handle,
                                       const hiptensorContractionPlan_t* plan,
                                       const void*                       alpha,
                                       const void*                       A,
                                       const void*                       B,
                                       const void*                       beta,
                                       const void*                       C,
                                       void*                             D,
                                       void*                             workspace,
                                       uint64_t                          workspaceSize,
                                       hipStream_t                       stream);

//! @brief Implements a tensor reduction of the form \f[ D = alpha * opReduce(opA(A)) + beta * opC(C) \f]
//!
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[in] alpha Scaling for A; its data type is determined by 'typeCompute'. Pointer to the host memory.
//! @param[in] A Pointer to the data corresponding to A in device memory. Pointer to the GPU-accessible memory.
//! @param[in] descA A descriptor that holds the information about the data type, modes and strides of A.
//! @param[in] modeA Array with 'nmodeA' entries that represent the modes of A. modeA[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to hiptensorInitTensorDescriptor. Modes that only appear in modeA but not in modeC are reduced (contracted).
//! @param[in] beta Scaling for C; its data type is determined by 'typeCompute'. Pointer to the host memory.
//! @param[in] C Pointer to the data corresponding to C in device memory. Pointer to the GPU-accessible memory.
//! @param[in] descC A descriptor that holds the information about the data type, modes and strides of C.
//! @param[in] modeC Array with 'nmodeC' entries that represent the modes of C. modeC[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to hiptensorInitTensorDescriptor.
//! @param[out] D Pointer to the data corresponding to C in device memory. Pointer to the GPU-accessible memory.
//! @param[in] descD Must be identical to descC for now.
//! @param[in] modeD Must be identical to modeC for now.
//! @param[in] opReduce binary operator used to reduce elements of A.
//! @param[in] typeCompute All arithmetic is performed using this data type (i.e., it affects the accuracy and performance).
//! @param[out] workspace Scratchpad (device) memory; the workspace must be aligned to 128 bytes.
//! @param[in] workspaceSize Please use hiptensorReductionGetWorkspaceSize() to query the required workspace.
//!            While lower values, including zero, are valid, they may lead to grossly suboptimal performance.
//! @param[in] stream The stream in which all the computation is performed.
//! @retval HIPTENSOR_STATUS_NOT_SUPPORTED if operation is not supported.
//! @retval HIPTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
//! @retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully.
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.

hiptensorStatus_t hiptensorReduction(const hiptensorHandle_t*           handle,
                                     const void*                        alpha,
                                     const void*                        A,
                                     const hiptensorTensorDescriptor_t* descA,
                                     const int32_t                      modeA[],
                                     const void*                        beta,
                                     const void*                        C,
                                     const hiptensorTensorDescriptor_t* descC,
                                     const int32_t                      modeC[],
                                     void*                              D,
                                     const hiptensorTensorDescriptor_t* descD,
                                     const int32_t                      modeD[],
                                     hiptensorOperator_t                opReduce,
                                     hiptensorComputeType_t             typeCompute,
                                     void*                              workspace,
                                     uint64_t                           workspaceSize,
                                     hipStream_t                        stream);

//! @brief Determines the required workspaceSize for a given tensor reduction (see \ref hiptensorReduction)
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[in] A same as in hiptensorReduction
//! @param[in] descA same as in hiptensorReduction
//! @param[in] modeA same as in hiptensorReduction
//! @param[in] C same as in hiptensorReduction
//! @param[in] descC same as in hiptensorReduction
//! @param[in] modeC same as in hiptensorReduction
//! @param[in] D same as in hiptensorReduction
//! @param[in] descD same as in hiptensorReduction
//! @param[in] modeD same as in hiptensorReduction
//! @param[in] opReduce same as in hiptensorReduction
//! @param[in] typeCompute same as in hiptensorReduction
//! @param[out] workspaceSize The workspace size (in bytes) that is required for the given tensor reduction.
//! @retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully.
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
//! @retval HIPTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
hiptensorStatus_t hiptensorReductionGetWorkspaceSize(const hiptensorHandle_t*           handle,
                                                     const void*                        A,
                                                     const hiptensorTensorDescriptor_t* descA,
                                                     const int32_t                      modeA[],
                                                     const void*                        C,
                                                     const hiptensorTensorDescriptor_t* descC,
                                                     const int32_t                      modeC[],
                                                     const void*                        D,
                                                     const hiptensorTensorDescriptor_t* descD,
                                                     const int32_t                      modeD[],
                                                     hiptensorOperator_t                opReduce,
                                                     hiptensorComputeType_t             typeCompute,
                                                     uint64_t* workspaceSize);

//! @brief Registers a callback function that will be invoked by logger calls.
//! @param[in] callback This parameter is the callback function pointer provided to the logger.
//! @retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
//! @retval HIPTENSOR_STATUS_INVALID_VALUE if the given callback is invalid.
hiptensorStatus_t hiptensorLoggerSetCallback(hiptensorLoggerCallback_t callback);

//! @brief Registers a file output stream to redirect logging output to.
//! @note File stream must be open and writable in text mode.
//! @param[in] file This parameter is a file stream pointer provided to the logger.
//! @retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
//! @retval HIPTENSOR_STATUS_IO_ERROR if the output file is not valid (defaults back to stdout).
hiptensorStatus_t hiptensorLoggerSetFile(FILE* file);

//! @brief Redirects log output to a file given by the user.
//! @param[in] logFile This parameter is a file name (relative to binary) or full path to redirect logger output.
//! @retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
//! @retval HIPTENSOR_STATUS_IO_ERROR if the output file is not valid (defaults back to stdout).
hiptensorStatus_t hiptensorLoggerOpenFile(const char* logFile);

//! @brief User-specified logging level. Logs in other contexts will not be recorded.
//! @param[in] level This parameter is the logging level to be enforced.
//! @retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
//! @retval HIPTENSOR_STATUS_INVALID_VALUE if the given log level is invalid.
hiptensorStatus_t hiptensorLoggerSetLevel(hiptensorLogLevel_t level);

//! @brief User-specified logging mask. A mask may be a binary OR combination of
//! several log levels together. Logs in other contexts will not be recorded.
//! @param[in] mask This parameter is the logging mask to be enforced.
//! @retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
//! @retval HIPTENSOR_STATUS_INVALID_VALUE if the given log mask is invalid.
hiptensorStatus_t hiptensorLoggerSetMask(int32_t mask);

//! @brief Disables logging.
//! @retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
hiptensorStatus_t hiptensorLoggerForceDisable();

//! @brief Query HIP runtime version.
//! @retval -1 if the operation failed.
//! @retval Integer HIP runtime version if the operation succeeded.
int hiptensorGetHiprtVersion();

#endif // HIPTENSOR_API_HPP
