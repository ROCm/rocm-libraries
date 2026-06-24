#include <atomic>

#include "host_tensor_executors.hpp"
#include "rpp_cpu_simd_math.hpp"

// Bitwise operation structures that encapsulate both scalar and SIMD operations
template <typename T>
struct BitwiseAnd {
    static inline void scalar_op(T* dst, T* src1, T* src2) {
        *dst = *src1 & *src2;
    }
    static inline void simd_op(__m256i& a, __m256i& b) {
        a = _mm256_and_si256(a, b);
    }
};

template <typename T>
struct BitwiseOr {
    static inline void scalar_op(T* dst, T* src1, T* src2) {
        *dst = *src1 | *src2;
    }
    static inline void simd_op(__m256i& a, __m256i& b) {
        a = _mm256_or_si256(a, b);
    }
};

template <typename T>
struct BitwiseXor {
    static inline void scalar_op(T* dst, T* src1, T* src2) {
        *dst = *src1 ^ *src2;
    }
    static inline void simd_op(__m256i& a, __m256i& b) {
        a = _mm256_xor_si256(a, b);
    }
};

// Helper functions for broadcasting for different datatypes (8 bit, 16 bit and 32 bit)
inline __m256i simd_set1_val(Rpp8u& val) {
    return _mm256_set1_epi8(val);
}
inline __m256i simd_set1_val(Rpp16u& val) {
    return _mm256_set1_epi16(val);
}
inline __m256i simd_set1_val(Rpp32u& val) {
    return _mm256_set1_epi32(val);
}

// Computes ND tensor bitwise operations recursively
template <typename T, typename Operation>
inline void tensor_binary_bitwise_op_recursive(T* src1, T* src2, Rpp32u* src1Strides,
                                               Rpp32u* src2Strides, T* dst, Rpp32u* dstStrides,
                                               Rpp32u* dstShape, Rpp32u nDim) {
    if (!nDim)
        Operation::scalar_op(dst, src1, src2);
    else {
        for (int i = 0; i < *dstShape; i++) {
            tensor_binary_bitwise_op_recursive<T, Operation>(src1, src2, src1Strides + 1,
                                                             src2Strides + 1, dst, dstStrides + 1,
                                                             dstShape + 1, nDim - 1);
            dst += *(dstStrides);
            src1 += *(src1Strides);
            src2 += *(src2Strides);
        }
    }
}

template <typename T, typename Operation>
RppStatus tensor_binary_bitwise_op_host_tensor(T* srcPtr1, T* srcPtr2,
                                               RpptGenericDescPtr srcPtr1GenericDescPtr,
                                               RpptGenericDescPtr srcPtr2GenericDescPtr, T* dstPtr,
                                               RpptGenericDescPtr dstGenericDescPtr,
                                               RpptBroadcastMode broadcastMode,
                                               Rpp32u vectorIncrement, Rpp32u* srcPtr1roiTensor,
                                               Rpp32u* srcPtr2roiTensor, rpp::Handle& handle) {
    Rpp32u src1NDim =
        srcPtr1GenericDescPtr->numDims - 1;  // Omitting batchSize here to get tensor dimension
    Rpp32u src2NDim =
        srcPtr2GenericDescPtr->numDims - 1;  // Omitting batchSize here to get tensor dimension
    Rpp32u dstDim = src1NDim > src2NDim
                        ? src1NDim
                        : src2NDim;  // Destination dimension set to maximum of the input dimensions
    Rpp32u minDim = src1NDim < src2NDim ? src1NDim : src2NDim;  // Minimum of input dimensions

    // Overall dimension compatibility check for the entire batch
    for (int test = 0; test < minDim; test++)
        if (srcPtr1GenericDescPtr->dims[src1NDim - test] !=
            srcPtr2GenericDescPtr->dims[src2NDim - test])
            if ((srcPtr1GenericDescPtr->dims[src1NDim - test] != 1) &&
                (srcPtr2GenericDescPtr->dims[src2NDim - test] != 1))
                return RPP_ERROR_INVALID_DIM_LENGTHS;

    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    std::atomic<RppStatus> broadcastCompatStatus{RPP_SUCCESS};

    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < batchSize; batchCount++) {
        Rpp32u* src1roi = srcPtr1roiTensor + batchCount * src1NDim * 2;
        Rpp32u* src1Begin = src1roi;
        Rpp32u* src1Dims = src1Begin + src1NDim;

        Rpp32u* src2roi = srcPtr2roiTensor + batchCount * src2NDim * 2;
        Rpp32u* src2Begin = src2roi;
        Rpp32u* src2Dims = src2Begin + src2NDim;

        Rpp32u* src1Strides = srcPtr1GenericDescPtr->strides;
        Rpp32u* src2Strides = srcPtr2GenericDescPtr->strides;
        Rpp32u* dstStrides = dstGenericDescPtr->strides;

        Rpp32u *length, *src1length, *src2length, *src1ValidStrides, *src2ValidStrides,
            *dstValidStrides;

        if (broadcastMode == RPP_BROADCAST_ENABLE) {
            // Dimensions and Strides based on individual sample ROIs, used for broadcasting
            // purposes
            Rpp32u src1BroadcastDims[RPPT_MAX_DIMS_SAMPLE], src2BroadcastDims[RPPT_MAX_DIMS_SAMPLE],
                dstBroadcastDims[RPPT_MAX_DIMS_SAMPLE];
            Rpp32u src1BroadcastStrides[RPPT_MAX_DIMS_SAMPLE],
                src2BroadcastStrides[RPPT_MAX_DIMS_SAMPLE],
                dstBroadcastStrides[RPPT_MAX_DIMS_SAMPLE];

            bool incompatibleDims = false;

            // Copy ROI limits and Strides to individual sample strides and dims until minDim
            for (int i = 0; i < minDim; i++) {
                Rpp32u curIndex = RPPT_MAX_DIMS_SAMPLE - i - 1;
                src1BroadcastDims[curIndex] = src1Dims[src1NDim - i - 1];
                src2BroadcastDims[curIndex] = src2Dims[src2NDim - i - 1];
                src1BroadcastStrides[curIndex] = src1Strides[src1NDim - i];
                src2BroadcastStrides[curIndex] = src2Strides[src2NDim - i];
                dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
                // Check compatibility of dimension i.e check for equal shape or one of the input
                // dims to be 1
                if ((src1BroadcastDims[curIndex] != src2BroadcastDims[curIndex]) &&
                    (src1BroadcastDims[curIndex] != 1) && (src2BroadcastDims[curIndex] != 1))
                    incompatibleDims = true;
                dstBroadcastDims[curIndex] =
                    std::max(src1BroadcastDims[curIndex], src2BroadcastDims[curIndex]);
            }

            // Dimension compatibility failure case
            if (incompatibleDims == true) {
                broadcastCompatStatus.store(RPP_ERROR_INVALID_DIM_LENGTHS,
                                            std::memory_order_relaxed);
                continue;
            }

            // Handle cases of mismatching num dims
            if (src1NDim < src2NDim)
                for (int i = minDim; i < dstDim; i++) {
                    Rpp32u curIndex = RPPT_MAX_DIMS_SAMPLE - i - 1;
                    src1BroadcastDims[curIndex] = 1;
                    src2BroadcastDims[curIndex] = src2Dims[src2NDim - i];
                    dstBroadcastDims[curIndex] = src2Dims[src2NDim - i];
                    // Setting stride to zero will allow for repetition of values operated required
                    // for broadcasting
                    src1BroadcastStrides[curIndex] = 0;
                    src2BroadcastStrides[curIndex] = src2Strides[src2NDim - i];
                    dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
                }
            else if (src1NDim > src2NDim)
                for (int i = minDim; i < dstDim; i++) {
                    Rpp32u curIndex = RPPT_MAX_DIMS_SAMPLE - i - 1;
                    src2BroadcastDims[curIndex] = 1;
                    src1BroadcastDims[curIndex] = src1Dims[src1NDim - i];
                    dstBroadcastDims[curIndex] = src1Dims[src1NDim - i];
                    src1BroadcastStrides[curIndex] = src1Strides[src1NDim - i];
                    // Setting stride to zero will allow for repetition of values operated required
                    // for broadcasting
                    src2BroadcastStrides[curIndex] = 0;
                    dstBroadcastStrides[curIndex] = dstStrides[dstDim - i];
                }

            // Source strides for sample set to zero if corresponding axis shape = 1 for
            // broadcasting purposes Setting stride to zero will allow for repetition of values
            // operated required for broadcasting
            for (int i = 0; i < minDim; i++) {
                if ((src1BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] !=
                     dstBroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i]) &&
                    (src1BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] == 1)) {
                    src1BroadcastStrides[RPPT_MAX_DIMS_SAMPLE - 1 - i] = 0;
                }
                if ((src2BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] !=
                     dstBroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i]) &&
                    (src2BroadcastDims[RPPT_MAX_DIMS_SAMPLE - 1 - i] == 1)) {
                    src2BroadcastStrides[RPPT_MAX_DIMS_SAMPLE - 1 - i] = 0;
                }
            }

            Rpp32u testOffset = RPPT_MAX_DIMS_SAMPLE - dstDim;

            // Shift dims and strides by offset to process only valid values
            length = dstBroadcastDims + testOffset;
            src1length = src1BroadcastDims + testOffset;
            src2length = src2BroadcastDims + testOffset;

            src1ValidStrides = src1BroadcastStrides + testOffset;
            src2ValidStrides = src2BroadcastStrides + testOffset;
            dstValidStrides = dstBroadcastStrides + testOffset;
        } else {
            // Both length for dest and srclength set to src1Dims because in non broadcast case,
            // source and destination dimensions are expected to be the same
            length = src1Dims;
            src1length = src1Dims;
            src2length = src2Dims;

            src1ValidStrides = srcPtr1GenericDescPtr->strides + 1;
            src2ValidStrides = srcPtr2GenericDescPtr->strides + 1;
            dstValidStrides = dstGenericDescPtr->strides + 1;
        }

        T* srcPtrTemp1 = srcPtr1 + batchCount * srcPtr1GenericDescPtr->strides[0];
        T* srcPtrTemp2 = srcPtr2 + batchCount * srcPtr2GenericDescPtr->strides[0];

        for (int i = 0; i < src1NDim; i++)
            srcPtrTemp1 += src1Begin[i] * srcPtr1GenericDescPtr->strides[i + 1];

        for (int i = 0; i < src2NDim; i++)
            srcPtrTemp2 += src2Begin[i] * srcPtr2GenericDescPtr->strides[i + 1];

        T* dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u alignMask = vectorIncrement - 1;

        // For nDim = 1, 2, 3 cases handled when the lowest axis shape = 1 and != 1 for efficient
        // processing
        if (dstDim == 1) {
            Rpp32u alignedLength = length[0] & ~alignMask;
            Rpp32u src1shape = src1length[0];
            Rpp32u src2shape = src2length[0];
            Rpp32u vectorLoopCount = 0;
            if (src1shape == 1) {
#if __AVX2__
                __m256i p1 = simd_set1_val(srcPtrTemp1[0]);  // simd broadcast
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256i p2 = _mm256_loadu_si256((const __m256i*)srcPtrTemp2);  // simd load
                    Operation::simd_op(p2, p1);                                    // simd op
                    _mm256_storeu_si256((__m256i*)dstPtrTemp, p2);                 // simd store
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < length[0]; vectorLoopCount++) {
                    Operation::scalar_op(dstPtrTemp, srcPtrTemp1, srcPtrTemp2);
                    srcPtrTemp2++;
                    dstPtrTemp++;
                }
            } else if (src2shape == 1) {
#if __AVX2__
                __m256i p2 = simd_set1_val(srcPtrTemp2[0]);  // simd broadcast
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256i p1 = _mm256_loadu_si256((const __m256i*)srcPtrTemp1);  // simd load
                    Operation::simd_op(p1, p2);                                    // simd op
                    _mm256_storeu_si256((__m256i*)dstPtrTemp, p1);                 // simd store
                    srcPtrTemp1 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < length[0]; vectorLoopCount++) {
                    Operation::scalar_op(dstPtrTemp, srcPtrTemp1, srcPtrTemp2);
                    srcPtrTemp1++;
                    dstPtrTemp++;
                }
            } else {
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256i p1 = _mm256_loadu_si256((const __m256i*)srcPtrTemp1);  // simd load
                    __m256i p2 = _mm256_loadu_si256((const __m256i*)srcPtrTemp2);  // simd load
                    Operation::simd_op(p1, p2);                                    // simd op
                    _mm256_storeu_si256((__m256i*)dstPtrTemp, p1);                 // simd store
                    srcPtrTemp1 += vectorIncrement;
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < length[0]; vectorLoopCount++) {
                    Operation::scalar_op(dstPtrTemp, srcPtrTemp1, srcPtrTemp2);
                    srcPtrTemp1++;
                    srcPtrTemp2++;
                    dstPtrTemp++;
                }
            }
        } else if (dstDim == 2) {
            Rpp32u alignedLength = length[1] & ~alignMask;
            Rpp32u src1shape = src1length[1];
            Rpp32u src2shape = src2length[1];
            if (src1shape == 1) {
                for (int i = 0; i < length[0]; i++) {
                    T* srcPtrElem1 = srcPtrTemp1;
                    T* srcPtrElem2 = srcPtrTemp2;
                    T* dstPtrElem = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    __m256i p1 = simd_set1_val(srcPtrElem1[0]);  // simd broadcast
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                        __m256i p2 = _mm256_loadu_si256((const __m256i*)srcPtrElem2);  // simd load
                        Operation::simd_op(p2, p1);                                    // simd op
                        _mm256_storeu_si256((__m256i*)dstPtrElem, p2);                 // simd store
                        srcPtrElem2 += vectorIncrement;
                        dstPtrElem += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++) {
                        Operation::scalar_op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                        srcPtrElem2++;
                        dstPtrElem++;
                    }
                    srcPtrTemp1 += src1ValidStrides[0];
                    srcPtrTemp2 += src2ValidStrides[0];
                    dstPtrTemp += dstValidStrides[0];
                }
            } else if (src2shape == 1) {
                for (int i = 0; i < length[0]; i++) {
                    T* srcPtrElem1 = srcPtrTemp1;
                    T* srcPtrElem2 = srcPtrTemp2;
                    T* dstPtrElem = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    __m256i p2 = simd_set1_val(srcPtrElem2[0]);  // simd broadcast
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                        __m256i p1 = _mm256_loadu_si256((const __m256i*)srcPtrElem1);  // simd load
                        Operation::simd_op(p1, p2);                                    // simd op
                        _mm256_storeu_si256((__m256i*)dstPtrElem, p1);                 // simd store
                        srcPtrElem1 += vectorIncrement;
                        dstPtrElem += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++) {
                        Operation::scalar_op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                        srcPtrElem1++;
                        dstPtrElem++;
                    }
                    srcPtrTemp1 += src1ValidStrides[0];
                    srcPtrTemp2 += src2ValidStrides[0];
                    dstPtrTemp += dstValidStrides[0];
                }
            } else {
                for (int i = 0; i < length[0]; i++) {
                    T* srcPtrElem1 = srcPtrTemp1;
                    T* srcPtrElem2 = srcPtrTemp2;
                    T* dstPtrElem = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                        __m256i p1 = _mm256_loadu_si256((const __m256i*)srcPtrElem1);  // simd load
                        __m256i p2 = _mm256_loadu_si256((const __m256i*)srcPtrElem2);  // simd load
                        Operation::simd_op(p1, p2);                                    // simd op
                        _mm256_storeu_si256((__m256i*)dstPtrElem, p1);                 // simd store
                        srcPtrElem1 += vectorIncrement;
                        srcPtrElem2 += vectorIncrement;
                        dstPtrElem += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++) {
                        Operation::scalar_op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                        srcPtrElem1++;
                        srcPtrElem2++;
                        dstPtrElem++;
                    }
                    srcPtrTemp1 += src1ValidStrides[0];
                    srcPtrTemp2 += src2ValidStrides[0];
                    dstPtrTemp += dstValidStrides[0];
                }
            }
        } else if (dstDim == 3) {
            Rpp32u alignedLength = length[2] & ~alignMask;
            Rpp32u src1shape = src1length[2];
            Rpp32u src2shape = src2length[2];
            if (src1shape == 1) {
                for (int i = 0; i < length[0]; i++) {
                    T* srcPtrOuter1 = srcPtrTemp1;
                    T* srcPtrOuter2 = srcPtrTemp2;
                    T* dstPtrOuter = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++) {
                        T* srcPtrElem1 = srcPtrOuter1;
                        T* srcPtrElem2 = srcPtrOuter2;
                        T* dstPtrElem = dstPtrOuter;

                        int vectorLoopCount = 0;
#if __AVX2__
                        __m256i p1 = simd_set1_val(srcPtrElem1[0]);  // simd broadcast
                        for (; vectorLoopCount < alignedLength;
                             vectorLoopCount += vectorIncrement) {
                            __m256i p2 =
                                _mm256_loadu_si256((const __m256i*)srcPtrElem2);  // simd load
                            Operation::simd_op(p2, p1);                           // simd op
                            _mm256_storeu_si256((__m256i*)dstPtrElem, p2);        // simd store
                            srcPtrElem2 += vectorIncrement;
                            dstPtrElem += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++) {
                            Operation::scalar_op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                            srcPtrElem2++;
                            dstPtrElem++;
                        }

                        srcPtrOuter1 += src1ValidStrides[1];
                        srcPtrOuter2 += src2ValidStrides[1];
                        dstPtrOuter += dstValidStrides[1];
                    }

                    srcPtrTemp1 += src1ValidStrides[0];
                    srcPtrTemp2 += src2ValidStrides[0];
                    dstPtrTemp += dstValidStrides[0];
                }
            } else if (src2shape == 1) {
                for (int i = 0; i < length[0]; i++) {
                    T* srcPtrOuter1 = srcPtrTemp1;
                    T* srcPtrOuter2 = srcPtrTemp2;
                    T* dstPtrOuter = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++) {
                        T* srcPtrElem1 = srcPtrOuter1;
                        T* srcPtrElem2 = srcPtrOuter2;
                        T* dstPtrElem = dstPtrOuter;

                        int vectorLoopCount = 0;
#if __AVX2__
                        __m256i p2 = simd_set1_val(srcPtrElem2[0]);  // simd broadcast
                        for (; vectorLoopCount < alignedLength;
                             vectorLoopCount += vectorIncrement) {
                            __m256i p1 =
                                _mm256_loadu_si256((const __m256i*)srcPtrElem1);  // simd load
                            Operation::simd_op(p1, p2);                           // simd op
                            _mm256_storeu_si256((__m256i*)dstPtrElem, p1);        // simd store
                            srcPtrElem1 += vectorIncrement;
                            dstPtrElem += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++) {
                            Operation::scalar_op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                            srcPtrElem1++;
                            dstPtrElem++;
                        }

                        srcPtrOuter1 += src1ValidStrides[1];
                        srcPtrOuter2 += src2ValidStrides[1];
                        dstPtrOuter += dstValidStrides[1];
                    }

                    srcPtrTemp1 += src1ValidStrides[0];
                    srcPtrTemp2 += src2ValidStrides[0];
                    dstPtrTemp += dstValidStrides[0];
                }
            } else {
                for (int i = 0; i < length[0]; i++) {
                    T* srcPtrOuter1 = srcPtrTemp1;
                    T* srcPtrOuter2 = srcPtrTemp2;
                    T* dstPtrOuter = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++) {
                        T* srcPtrElem1 = srcPtrOuter1;
                        T* srcPtrElem2 = srcPtrOuter2;
                        T* dstPtrElem = dstPtrOuter;

                        int vectorLoopCount = 0;
#if __AVX2__
                        for (; vectorLoopCount < alignedLength;
                             vectorLoopCount += vectorIncrement) {
                            __m256i p1 =
                                _mm256_loadu_si256((const __m256i*)srcPtrElem1);  // simd load
                            __m256i p2 =
                                _mm256_loadu_si256((const __m256i*)srcPtrElem2);  // simd load
                            Operation::simd_op(p1, p2);                           // simd op
                            _mm256_storeu_si256((__m256i*)dstPtrElem, p1);        // simd store
                            srcPtrElem1 += vectorIncrement;
                            srcPtrElem2 += vectorIncrement;
                            dstPtrElem += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++) {
                            Operation::scalar_op(dstPtrElem, srcPtrElem1, srcPtrElem2);
                            srcPtrElem1++;
                            srcPtrElem2++;
                            dstPtrElem++;
                        }

                        srcPtrOuter1 += src1ValidStrides[1];
                        srcPtrOuter2 += src2ValidStrides[1];
                        dstPtrOuter += dstValidStrides[1];
                    }

                    srcPtrTemp1 += src1ValidStrides[0];
                    srcPtrTemp2 += src2ValidStrides[0];
                    dstPtrTemp += dstValidStrides[0];
                }
            }
        } else
            tensor_binary_bitwise_op_recursive<T, Operation>(
                srcPtrTemp1, srcPtrTemp2, src1ValidStrides, src2ValidStrides, dstPtrTemp,
                dstValidStrides, length, dstDim);
    }

    RppStatus compatSt = broadcastCompatStatus.load();
    if (compatSt != RPP_SUCCESS) return compatSt;
    return RPP_SUCCESS;
}

// Dispatcher function that dispatches the calls to the appropriate templated function based on the
// datatype and operation
template <typename T>
RppStatus tensor_binary_bitwise_op_dispatch_host_tensor(
    T* srcPtr1, T* srcPtr2, RpptGenericDescPtr srcPtr1GenericDescPtr,
    RpptGenericDescPtr srcPtr2GenericDescPtr, T* dstPtr, RpptGenericDescPtr dstGenericDescPtr,
    RpptBitwiseOp tensorOp, RpptBroadcastMode broadcastMode, Rpp32u* srcPtr1roiTensor,
    Rpp32u* srcPtr2roiTensor, rpp::Handle& handle) {
    int vectorIncrement = 32;  // Vector Increment for U8/I8 datatype
    if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) ||
        (srcPtr1GenericDescPtr->dataType == RpptDataType::I16))
        vectorIncrement = 16;  // Vector Increment for U16/I16 datatype
    else if ((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) ||
             (srcPtr1GenericDescPtr->dataType == RpptDataType::I32))
        vectorIncrement = 8;  // Vector Increment for U32/I32 datatype

    switch (tensorOp) {
        case RPP_TENSOR_OP_AND:
            return tensor_binary_bitwise_op_host_tensor<T, BitwiseAnd<T>>(
                srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr,
                dstGenericDescPtr, broadcastMode, vectorIncrement, srcPtr1roiTensor,
                srcPtr2roiTensor, handle);

        case RPP_TENSOR_OP_OR:
            return tensor_binary_bitwise_op_host_tensor<T, BitwiseOr<T>>(
                srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr,
                dstGenericDescPtr, broadcastMode, vectorIncrement, srcPtr1roiTensor,
                srcPtr2roiTensor, handle);

        case RPP_TENSOR_OP_XOR:
            return tensor_binary_bitwise_op_host_tensor<T, BitwiseXor<T>>(
                srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr,
                dstGenericDescPtr, broadcastMode, vectorIncrement, srcPtr1roiTensor,
                srcPtr2roiTensor, handle);
        default:
            return RPP_ERROR_NOT_IMPLEMENTED;
    }
}

template RppStatus tensor_binary_bitwise_op_dispatch_host_tensor<Rpp8u>(
    Rpp8u*, Rpp8u*, RpptGenericDescPtr, RpptGenericDescPtr, Rpp8u*, RpptGenericDescPtr,
    RpptBitwiseOp, RpptBroadcastMode, Rpp32u*, Rpp32u*, rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_host_tensor<Rpp16u>(
    Rpp16u*, Rpp16u*, RpptGenericDescPtr, RpptGenericDescPtr, Rpp16u*, RpptGenericDescPtr,
    RpptBitwiseOp, RpptBroadcastMode, Rpp32u*, Rpp32u*, rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_host_tensor<Rpp32u>(
    Rpp32u*, Rpp32u*, RpptGenericDescPtr, RpptGenericDescPtr, Rpp32u*, RpptGenericDescPtr,
    RpptBitwiseOp, RpptBroadcastMode, Rpp32u*, Rpp32u*, rpp::Handle&);
