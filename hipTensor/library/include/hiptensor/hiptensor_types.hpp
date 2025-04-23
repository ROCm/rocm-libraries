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
#ifndef HIPTENSOR_TYPES_HPP
#define HIPTENSOR_TYPES_HPP

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include <hip/hip_common.h>
#include <hip/library_types.h>

//! @brief hipTensor status type enumeration
//! @details The type is used to indicate the resulting status of hipTensor library function calls
typedef enum
{
    //! The operation is successful.
    HIPTENSOR_STATUS_SUCCESS = 0,
    //! The handle was not initialized.
    HIPTENSOR_STATUS_NOT_INITIALIZED = 1,
    //! Resource allocation failed inside the hipTensor library.
    HIPTENSOR_STATUS_ALLOC_FAILED = 3,
    //! Invalid value or parameter was passed to the function (indicates a user error).
    HIPTENSOR_STATUS_INVALID_VALUE = 7,
    //! Indicates that the target architecure is not supported, or the device is not ready.
    HIPTENSOR_STATUS_ARCH_MISMATCH = 8,
    //! Indicates the failure of a GPU program or a kernel, which can be caused by multiple reasons.
    HIPTENSOR_STATUS_EXECUTION_FAILED = 13,
    //! An internal error has occurred.
    HIPTENSOR_STATUS_INTERNAL_ERROR = 14,
    //! The requested operation is not supported.
    HIPTENSOR_STATUS_NOT_SUPPORTED = 15,
    //! A call to Composable Kernels did not succeed.
    HIPTENSOR_STATUS_CK_ERROR = 17,
    //! Unknown hipTensor error has occurred.
    HIPTENSOR_STATUS_HIP_ERROR = 18,
    //! The provided workspace was insufficient.
    HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19,
    //! Indicates that the driver version is insufficient.
    HIPTENSOR_STATUS_INSUFFICIENT_DRIVER = 20,
    //! Indicates an error related to file I/O.
    HIPTENSOR_STATUS_IO_ERROR = 21,

} hiptensorStatus_t;

//! @brief hipTensor compute type enumeration
typedef enum
{
    //! Single precision floating point
    HIPTENSOR_COMPUTE_32F = (1U << 2U),
    //! Double precision floating point
    HIPTENSOR_COMPUTE_64F = (1U << 4U),
    //! Half precision floating point
    HIPTENSOR_COMPUTE_16F = (1U << 0U),
    //! Brain float half precision floating point
    HIPTENSOR_COMPUTE_16BF = (1U << 10U),
    //! Complex single precision floating point
    HIPTENSOR_COMPUTE_C32F = (1U << 11U),
    //! Complex double precision floating point
    HIPTENSOR_COMPUTE_C64F = (1U << 12U),
    //! No type
    HIPTENSOR_COMPUTE_NONE = 0,

    // @cond
    //! <Following types to be added (TBA)>
    HIPTENSOR_COMPUTE_8U  = (1U << 6U),
    HIPTENSOR_COMPUTE_8I  = (1U << 8U),
    HIPTENSOR_COMPUTE_32U = (1U << 7U),
    HIPTENSOR_COMPUTE_32I = (1U << 9U),
    // @endcond

} hiptensorComputeType_t;

//! @brief Element-wise operations
typedef enum
{
    HIPTENSOR_OP_IDENTITY = 1, ///< Identity operator (i.e., elements are not changed)
    HIPTENSOR_OP_SQRT     = 2, ///< Square root
    HIPTENSOR_OP_RELU     = 8, ///< Rectified linear unit
    HIPTENSOR_OP_CONJ     = 9, ///< Complex conjugate
    HIPTENSOR_OP_RCP      = 10, ///< Reciprocal
    HIPTENSOR_OP_SIGMOID  = 11, ///< y=1/(1+exp(-x))
    HIPTENSOR_OP_TANH     = 12, ///< y=tanh(x)
    HIPTENSOR_OP_EXP      = 22, ///< Exponentiation.
    HIPTENSOR_OP_LOG      = 23, ///< Log (base e).
    HIPTENSOR_OP_ABS      = 24, ///< Absolute value.
    HIPTENSOR_OP_NEG      = 25, ///< Negation.
    HIPTENSOR_OP_SIN      = 26, ///< Sine.
    HIPTENSOR_OP_COS      = 27, ///< Cosine.
    HIPTENSOR_OP_TAN      = 28, ///< Tangent.
    HIPTENSOR_OP_SINH     = 29, ///< Hyperbolic sine.
    HIPTENSOR_OP_COSH     = 30, ///< Hyperbolic cosine.
    HIPTENSOR_OP_ASIN     = 31, ///< Inverse sine.
    HIPTENSOR_OP_ACOS     = 32, ///< Inverse cosine.
    HIPTENSOR_OP_ATAN     = 33, ///< Inverse tangent.
    HIPTENSOR_OP_ASINH    = 34, ///< Inverse hyperbolic sine.
    HIPTENSOR_OP_ACOSH    = 35, ///< Inverse hyperbolic cosine.
    HIPTENSOR_OP_ATANH    = 36, ///< Inverse hyperbolic tangent.
    HIPTENSOR_OP_CEIL     = 37, ///< Ceiling.
    HIPTENSOR_OP_FLOOR    = 38, ///< Floor.

    /* Binary */
    HIPTENSOR_OP_ADD = 3, ///< Addition of two elements
    HIPTENSOR_OP_MUL = 5, ///< Multiplication of two elements
    HIPTENSOR_OP_MAX = 6, ///< Maximum of two elements
    HIPTENSOR_OP_MIN = 7, ///< Minimum of two elements

    HIPTENSOR_OP_UNKNOWN = 126, ///< reserved for internal use only)
} hiptensorOperator_t;

//! @brief Tensor contraction kernel selection algorithm
typedef enum
{
    //! Uses novel actor-critic selection model
    HIPTENSOR_ALGO_ACTOR_CRITIC = -8,
    //! Lets the internal heuristic choose
    HIPTENSOR_ALGO_DEFAULT = -1,
    //! Uses the more accurate and time-consuming model
    HIPTENSOR_ALGO_DEFAULT_PATIENT = -6,

} hiptensorAlgo_t;

//! @brief Workspace size selection
typedef enum
{
    //! At least one algorithm will be available
    HIPTENSOR_WORKSPACE_MIN = 1,
    //! The most suitable algorithm will be available
    HIPTENSOR_WORKSPACE_RECOMMENDED = 2,
    //! All algorithms will be available
    HIPTENSOR_WORKSPACE_MAX = 3,

} hiptensorWorksizePreference_t;

//! @brief Logging context
//! @details The logger output of certain contexts maybe constrained to these levels
typedef enum
{
    //! No logging
    HIPTENSOR_LOG_LEVEL_OFF = 0,
    //! Log errors
    HIPTENSOR_LOG_LEVEL_ERROR = 1,
    //! Log performance messages
    HIPTENSOR_LOG_LEVEL_PERF_TRACE = 2,
    //! Log performance hints
    HIPTENSOR_LOG_LEVEL_PERF_HINT = 4,
    //! Log selection messages
    HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE = 8,
    //! Log a trace of API calls
    HIPTENSOR_LOG_LEVEL_API_TRACE = 16,

} hiptensorLogLevel_t;

//! @brief hipTensor's library context
struct hiptensorHandle_t
{
    int64_t fields[512];
};

//! @brief Structure representing a tensor descriptor
//!
//! Represents a descriptor for the tensor with the given properties of
//! data type, lengths, strides and element-wise unary operation.
//! Constructed with hiptensorInitTensorDescriptor() function.
struct hiptensorTensorDescriptor_t
{
    //! Data type of the tensors enum selection
    hipDataType mType;
    //! Lengths of the tensor
    std::vector<std::size_t> mLengths;
    //! Strides of the tensor
    std::vector<std::size_t> mStrides;
    //! Unary operator applied to the tensor
    hiptensorOperator_t mUnaryOp;
};

//! @brief Structure representing a tensor contraction descriptor
//!
//! Represents contraction descriptor with the given properties of internal
//! contraction op (either scale or bilinear), the internal compute type,
//! as well as all of the input tensor descriptors, their alignment requirements
//! and modes.
//! Constructed with hiptensorInitContractionDescriptor() function.
struct hiptensorContractionDescriptor_t
{
    //! Enum that differentiates the internal contraction operation
    int32_t mContractionOpId;
    //! Compute type for the contraction
    hiptensorComputeType_t mComputeType;
    //! Cache of tensor descriptors
    std::vector<hiptensorTensorDescriptor_t> mTensorDesc;
    //! Cache of alignment requirements
    std::vector<uint32_t> mAlignmentReq;
    //! Tensor modes
    std::vector<std::vector<int32_t>> mTensorMode;
};

//! @brief hipTensor structure representing the contraction selection algorithm and candidates.
struct hiptensorContractionFind_t
{
    //! Id of the selection algorithm
    hiptensorAlgo_t mSelectionAlgorithm;
    //! A vector of the solver candidates
    std::vector<void*> mCandidates;
};

//! @brief hipTensor structure representing a contraction plan.
//! Constructed with the hiptensorInitContractionPlan() function.
struct hiptensorContractionPlan_t
{
    //! Final solution candidate
    void* mSolution;
    //! Contraction parameters
    hiptensorContractionDescriptor_t mContractionDesc;
};

//! @brief Logging callback
//! The specified callback is invoked whenever logging is enabled and a message is generated.
//! @param logContext The logging context enum
//! @param funcName A string holding the function name where the logging message was generated
//! @param msg A string holding the logging message
typedef void (*hiptensorLoggerCallback_t)(int32_t     logContext,
                                          const char* funcName,
                                          const char* msg);

#endif // HIPTENSOR_TYPES_HPP
