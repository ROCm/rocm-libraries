#include "hip_tensor_executors.hpp"
#include "rpp_hip_math.hpp"
#include <atomic>
#include <omp.h>

// -------------------- Set 1 - vector helper kernels --------------------

// Structures to Dispatch load/store functions for uchar/ushort/uint types
template <typename T> struct BitwiseLoadStoreExecute;

template<> struct BitwiseLoadStoreExecute<uchar>
{
    using VectorType = d_uchar8;

    __device__ __forceinline__ static void rpp_hip_load8(uchar *src, uchar *dst) { rpp_hip_load8_to_uchar8(src, dst); }
    __device__ __forceinline__ static void rpp_hip_pack_and_store8(uchar *dst, d_uchar8 *dst_u8) { rpp_hip_pack_uchar8_and_store8(dst, dst_u8); };
};

template<> struct BitwiseLoadStoreExecute<ushort>
{
    using VectorType = d_ushort8;

    __device__ __forceinline__ static void rpp_hip_load8(ushort *src, ushort *dst) { rpp_hip_load8_to_ushort8(src, dst); }
    __device__ __forceinline__ static void rpp_hip_pack_and_store8(ushort *dst, d_ushort8 *dst_u8) { rpp_hip_pack_ushort8_and_store8(dst, dst_u8); };
};

template<> struct BitwiseLoadStoreExecute<uint>
{
    using VectorType = d_uint8;

    __device__ __forceinline__ static void rpp_hip_load8(uint *src, uint *dst) { rpp_hip_load8_to_uint8(src, dst); }
    __device__ __forceinline__ static void rpp_hip_pack_and_store8(uint *dst, d_uint8 *dst_u8) { rpp_hip_pack_uint8_and_store8(dst, dst_u8); };
};

// Structures used to dispatch execution of bitwise operations (AND, OR, XOR)
template<typename VectorType, typename Operation> struct BitwiseOperationExecute;

template<typename VectorType> struct BitwiseOperationExecute<VectorType, BitwiseOr>
{
    __device__ __forceinline__ static void rpp_hip_math_bitwiseOp8(VectorType *a, VectorType *b, VectorType *c)
    {
        rpp_hip_math_bitwise_op8<BitwiseOr>(a, b, c);
    }
};

template<typename VectorType> struct BitwiseOperationExecute<VectorType, BitwiseXor>
{
    __device__ __forceinline__ static void rpp_hip_math_bitwiseOp8(VectorType *a, VectorType *b, VectorType *c)
    {
        rpp_hip_math_bitwise_op8<BitwiseXor>(a, b, c);
    }
};

template<typename VectorType> struct BitwiseOperationExecute<VectorType, BitwiseAnd>
{
    __device__ __forceinline__ static void rpp_hip_math_bitwiseOp8(VectorType *a, VectorType *b, VectorType *c)
    {
        rpp_hip_math_bitwise_op8<BitwiseAnd>(a, b, c);
    }
};

// -------------------- Set 2 - bitwise operation kernels --------------------

template <typename T, typename Operation>
__global__ void tensor_or_tensor_1d_hip_tensor(T *srcPtr1,
                                               T *srcPtr2,
                                               uint* srcStrides1,
                                               uint* srcStrides2,
                                               uint *src1BeginOffsets,
                                               uint *src2BeginOffsets,
                                               T *dstPtr,
                                               uint* dstStrides,
                                               uint *dstDims,
                                               Operation op)
{
    uint id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8; // width
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

    using VectorType = typename BitwiseLoadStoreExecute<T>::VectorType;

    uint* dstSampleDims = dstDims + id_z * RPPT_MAX_DIMS;
    uint* src1SampleStrides = srcStrides1 + id_z * RPPT_MAX_DIMS;
    uint* src2SampleStrides = srcStrides2 + id_z * RPPT_MAX_DIMS;
    uint* dstSampleStrides = dstStrides + id_z * RPPT_MAX_DIMS;

    if(id_x >= dstSampleDims[0])
        return;

    uint numRows = dstSampleDims[0] - id_x;

    if(numRows >= 8)
    {
        uint srcBaseIdx1 = (id_z * src1SampleStrides[0]) + src1BeginOffsets[id_z];
        uint srcBaseIdx2 = (id_z * src2SampleStrides[0]) + src2BeginOffsets[id_z];

        uint dstBaseIdx = (id_z * dstSampleStrides[0]) + id_x;

        T srcArr1[8], srcArr2[8];

        #pragma unroll
        for(int i1 = 0; i1 < 8; i1++)
        {
            uint srcIdx1 = srcBaseIdx1 + (id_x * src1SampleStrides[1]);
            uint srcIdx2 = srcBaseIdx2 + (id_x * src2SampleStrides[1]);
            srcArr1[i1] = srcPtr1[srcIdx1];
            srcArr2[i1] = srcPtr2[srcIdx2];
            id_x++;
        }

        VectorType dst_vec8;
        BitwiseOperationExecute<VectorType, Operation>::rpp_hip_math_bitwiseOp8((VectorType*)srcArr1, (VectorType*)srcArr2, &dst_vec8);
        BitwiseLoadStoreExecute<T>::rpp_hip_pack_and_store8(dstPtr + dstBaseIdx, &dst_vec8);
    }
    else
    {
        uint srcBaseIdx1 = (id_z * src1SampleStrides[0]) + src1BeginOffsets[id_z];
        uint srcBaseIdx2 = (id_z * src2SampleStrides[0]) + src2BeginOffsets[id_z];

        uint dstBaseIdx = (id_z * dstSampleStrides[0]);

        for(int i1 = 0; i1 < numRows; i1++)
        {
            uint srcIdx1 = srcBaseIdx1 + (id_x * src1SampleStrides[1]);
            uint srcIdx2 = srcBaseIdx2 + (id_x * src2SampleStrides[1]);
            uint dstIdx = dstBaseIdx + id_x;

            id_x++;
            dstPtr[dstIdx] = Operation::op(srcPtr1[srcIdx1], srcPtr2[srcIdx2]);
        }
    }
}

template <typename T, typename Operation>
__global__ void tensor_or_tensor_non_broadcast_1d_hip_tensor(T *src1Ptr,
                                                             T *src2Ptr,
                                                             T *dstPtr,
                                                             uint strides,
                                                             uint *roiTensor1,
                                                             uint *roiTensor2,
                                                             Operation op)
{
    uint id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    using VectorType = typename BitwiseLoadStoreExecute<T>::VectorType;

    uint *roi1 = &roiTensor1[id_z * 4];
    uint beginX1 = roi1[0];

    uint *roi2 = &roiTensor2[id_z * 4];
    uint beginX2 = roi2[0];
    uint width = roi2[1];

    if(id_x >= width)
        return;

    uint srcIdx1 = (id_z * strides) + id_x + beginX1;
    uint srcIdx2 = (id_z * strides) + id_x + beginX2;
    uint dstIdx = (id_z * strides) + id_x;

    uint remaining = width - id_x;

    if (remaining >= 8)
    {
        VectorType src1_vec8, src2_vec8, dst_vec8;
        BitwiseLoadStoreExecute<T>::rpp_hip_load8(src1Ptr + srcIdx1, (T*)&src1_vec8);
        BitwiseLoadStoreExecute<T>::rpp_hip_load8(src2Ptr + srcIdx2, (T*)&src2_vec8);
        BitwiseOperationExecute<VectorType, Operation>::rpp_hip_math_bitwiseOp8(&src1_vec8, &src2_vec8, &dst_vec8);
        BitwiseLoadStoreExecute<T>::rpp_hip_pack_and_store8(dstPtr + dstIdx, &dst_vec8);
    }
    else
    {
        // Tail handling for the last partial (width % 8) elements
        for (uint i = 0; i < remaining; ++i)
        {
            T v1 = src1Ptr[srcIdx1 + i];
            T v2 = src2Ptr[srcIdx2 + i];
            dstPtr[dstIdx + i] = Operation::op(v1, v2);
        }
    }
}

template <typename T, typename Operation>
__global__ void tensor_or_tensor_2d_hip_tensor(T *srcPtr1,
                                               T *srcPtr2,
                                               uint* srcStrides1,
                                               uint* srcStrides2,
                                               uint *src1BeginOffsets,
                                               uint *src2BeginOffsets,
                                               T *dstPtr,
                                               uint *dstStrides,
                                               uint *dstDims,
                                               Operation op)
{
    uint id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8; // width
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // height
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

    using VectorType = typename BitwiseLoadStoreExecute<T>::VectorType;

    uint* dstSampleDims = dstDims + id_z * RPPT_MAX_DIMS;
    uint* src1SampleStrides = srcStrides1 + id_z * RPPT_MAX_DIMS;
    uint* src2SampleStrides = srcStrides2 + id_z * RPPT_MAX_DIMS;
    uint* dstSampleStrides = dstStrides + id_z * RPPT_MAX_DIMS;

    if(id_x >= dstSampleDims[1] || id_y >= dstSampleDims[0])
        return;

    uint numRows = dstSampleDims[1] - id_x;

    if(numRows >= 8)
    {
        uint srcBaseIdx1 = (id_z * src1SampleStrides[0]) + ((id_y) * src1SampleStrides[1]) + src1BeginOffsets[id_z];
        uint srcBaseIdx2 = (id_z * src2SampleStrides[0]) + ((id_y) * src2SampleStrides[1]) + src2BeginOffsets[id_z];

        uint dstBaseIdx = (id_z * dstSampleStrides[0]) + (id_y * dstSampleStrides[1]) + id_x;

        T srcArr1[8], srcArr2[8];

        #pragma unroll
        for(int i1 = 0; i1 < 8; i1++)
        {
            uint srcIdx1 = srcBaseIdx1 + (id_x * src1SampleStrides[2]);
            uint srcIdx2 = srcBaseIdx2 + (id_x * src2SampleStrides[2]);
            srcArr1[i1] = srcPtr1[srcIdx1];
            srcArr2[i1] = srcPtr2[srcIdx2];
            id_x++;
        }

        VectorType dst_vec8;
        BitwiseOperationExecute<VectorType, Operation>::rpp_hip_math_bitwiseOp8((VectorType*)srcArr1, (VectorType*)srcArr2, &dst_vec8);
        BitwiseLoadStoreExecute<T>::rpp_hip_pack_and_store8(dstPtr + dstBaseIdx, &dst_vec8);
    }
    else
    {
        uint srcBaseIdx1 = (id_z * src1SampleStrides[0]) + ((id_y) * src1SampleStrides[1]) + src1BeginOffsets[id_z];
        uint srcBaseIdx2 = (id_z * src2SampleStrides[0]) + ((id_y) * src2SampleStrides[1]) + src2BeginOffsets[id_z];

        uint dstBaseIdx = (id_z * dstSampleStrides[0]) + (id_y * dstSampleStrides[1]);

        for(int i1 = 0; i1 < numRows; i1++)
        {
            uint srcIdx1 = srcBaseIdx1 + (id_x * src1SampleStrides[2]);
            uint srcIdx2 = srcBaseIdx2 + (id_x * src2SampleStrides[2]);
            uint dstIdx = dstBaseIdx + id_x;

            id_x++;
            dstPtr[dstIdx] = Operation::op(srcPtr1[srcIdx1], srcPtr2[srcIdx2]);
        }
    }
}

template <typename T, typename Operation>
__global__ void tensor_or_tensor_non_broadcast_2d_hip_tensor(T *src1Ptr,
                                                             T *src2Ptr,
                                                             T *dstPtr,
                                                             uint2 stridesNH,
                                                             uint *roiTensor1,
                                                             uint *roiTensor2,
                                                             Operation op)
{
    uint id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8; // width
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;       // height
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;       // batchsize

    using VectorType = typename BitwiseLoadStoreExecute<T>::VectorType;

    uint *roi1 = &roiTensor1[id_z * 4];
    uint beginY1 = roi1[0];
    uint beginX1 = roi1[1];

    uint *roi2 = &roiTensor2[id_z * 4];
    uint beginY2 = roi2[0];
    uint beginX2 = roi2[1];
    uint height = roi2[2];
    uint width = roi2[3];

    if(id_x >= width || id_y >= height)
        return;

    uint remaining = width - id_x;

    // Vectorized path: safe to load/store 8 elements
    if (remaining >= 8)
    {
        uint srcIdx1 = (id_z * stridesNH.x) + ((id_y + beginY1) * stridesNH.y) + id_x + beginX1;
        uint srcIdx2 = (id_z * stridesNH.x) + ((id_y + beginY2) * stridesNH.y) + id_x + beginX2;
        uint dstIdx  = (id_z * stridesNH.x) + (id_y * stridesNH.y) + id_x;

        VectorType src1_vec8, src2_vec8, dst_vec8;
        BitwiseLoadStoreExecute<T>::rpp_hip_load8(src1Ptr + srcIdx1, (T*)&src1_vec8);
        BitwiseLoadStoreExecute<T>::rpp_hip_load8(src2Ptr + srcIdx2, (T*)&src2_vec8);
        BitwiseOperationExecute<VectorType, Operation>::rpp_hip_math_bitwiseOp8(&src1_vec8, &src2_vec8, &dst_vec8);
        BitwiseLoadStoreExecute<T>::rpp_hip_pack_and_store8(dstPtr + dstIdx, &dst_vec8);
    }
    // Remainder path: handle final 1..7 elements safely
    else
    {
        uint srcIdx1 = (id_z * stridesNH.x) + ((id_y + beginY1) * stridesNH.y) + id_x + beginX1;
        uint srcIdx2 = (id_z * stridesNH.x) + ((id_y + beginY2) * stridesNH.y) + id_x + beginX2;
        uint dstIdx  = (id_z * stridesNH.x) + (id_y * stridesNH.y) + id_x;

        for (uint i = 0; i < remaining; ++i)
        {
            dstPtr[dstIdx + i] = Operation::op(src1Ptr[srcIdx1 + i], src2Ptr[srcIdx2 + i]);
        }
    }
}

template <typename T, typename Operation>
__global__ void tensor_or_tensor_3d_hip_tensor(T *srcPtr1,
                                               T *srcPtr2,
                                               uint* srcStrides1,
                                               uint* srcStrides2,
                                               uint src1BeginOffset,
                                               uint src2BeginOffset,
                                               T *dstPtr,
                                               uint* dstStrides,
                                               uint *dstDims,
                                               Operation op)
{
    uint id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8; // lengthX
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // lengthY
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // lengthZ

    using VectorType = typename BitwiseLoadStoreExecute<T>::VectorType;

    if(id_x >= dstDims[2] || id_y >= dstDims[1] || id_z >= dstDims[0])
        return;

    uint numRows = dstDims[2] - id_x;

    if(numRows >= 8)
    {
        uint srcBaseIdx1 = (id_z * srcStrides1[1]) + ((id_y) * srcStrides1[2]) + src1BeginOffset;
        uint srcBaseIdx2 = (id_z * srcStrides2[1]) + ((id_y) * srcStrides2[2]) + src2BeginOffset;

        uint dstBaseIdx = (id_z * dstStrides[1]) + (id_y * dstStrides[2]) + id_x;

        T srcArr1[8], srcArr2[8];

        #pragma unroll
        for(int i1 = 0; i1 < 8; i1++)
        {
            uint srcIdx1 = srcBaseIdx1 + (id_x * srcStrides1[3]);
            uint srcIdx2 = srcBaseIdx2 + (id_x * srcStrides2[3]);
            srcArr1[i1] = srcPtr1[srcIdx1];
            srcArr2[i1] = srcPtr2[srcIdx2];
            id_x++;
        }

        VectorType dst_vec8;
        BitwiseOperationExecute<VectorType, Operation>::rpp_hip_math_bitwiseOp8((VectorType*)srcArr1, (VectorType*)srcArr2, &dst_vec8);
        BitwiseLoadStoreExecute<T>::rpp_hip_pack_and_store8(dstPtr + dstBaseIdx, &dst_vec8);
    }
    else
    {
        uint srcBaseIdx1 = (id_z * srcStrides1[1]) + ((id_y) * srcStrides1[2]) + src1BeginOffset;
        uint srcBaseIdx2 = (id_z * srcStrides2[1]) + ((id_y) * srcStrides2[2]) + src2BeginOffset;

        uint dstBaseIdx = (id_z * dstStrides[1]) + (id_y * dstStrides[2]);

        for(int i1 = 0; i1 < numRows; i1++)
        {
            uint srcIdx1 = srcBaseIdx1 + (id_x * srcStrides1[3]);
            uint srcIdx2 = srcBaseIdx2 + (id_x * srcStrides2[3]);
            uint dstIdx = dstBaseIdx + id_x;

            id_x++;
            dstPtr[dstIdx] = Operation::op(srcPtr1[srcIdx1], srcPtr2[srcIdx2]);
        }
    }
}

template <typename T, typename Operation>
__global__ void tensor_or_tensor_non_broadcast_3d_hip_tensor(T *src1Ptr,
                                                             T *src2Ptr,
                                                             T *dstPtr,
                                                             uint2 stridesDH,
                                                             uint *roiTensor1,
                                                             uint *roiTensor2,
                                                             Operation op)
{
    uint id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8; // lengthX
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;       // lengthY
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;       // lengthZ

    using VectorType = typename BitwiseLoadStoreExecute<T>::VectorType;

    uint *roi1 = roiTensor1;
    uint beginZ1 = roi1[0];
    uint beginY1 = roi1[1];
    uint beginX1 = roi1[2];

    uint *roi2 = roiTensor2;
    uint beginZ2 = roi2[0];
    uint beginY2 = roi2[1];
    uint beginX2 = roi2[2];
    uint lengthZ2 = roi2[3];
    uint lengthY2 = roi2[4];
    uint lengthX2 = roi2[5];

    if(id_x >= lengthX2 || id_y >= lengthY2 || id_z >= lengthZ2)
        return;

    uint srcIdx1 = ((id_z + beginZ1) * stridesDH.x) + ((id_y + beginY1) * stridesDH.y) + id_x + beginX1;
    uint srcIdx2 = ((id_z + beginZ2) * stridesDH.x) + ((id_y + beginY2) * stridesDH.y) + id_x + beginX2;
    uint dstIdx = (id_z * stridesDH.x) + (id_y * stridesDH.y) + id_x;

    // Use vectorized path only when a full 8-element vector is within bounds.
    if(id_x + 7 < lengthX2)
    {
        VectorType src1_vec8, src2_vec8, dst_vec8;
        BitwiseLoadStoreExecute<T>::rpp_hip_load8(src1Ptr + srcIdx1, (T *)&src1_vec8);
        BitwiseLoadStoreExecute<T>::rpp_hip_load8(src2Ptr + srcIdx2, (T *)&src2_vec8);
        BitwiseOperationExecute<VectorType, Operation>::rpp_hip_math_bitwiseOp8(&src1_vec8, &src2_vec8, &dst_vec8);
        BitwiseLoadStoreExecute<T>::rpp_hip_pack_and_store8(dstPtr + dstIdx, &dst_vec8);
    }
    else
    {
        // Tail processing for remaining (less than 8) elements along X.
        uint tail = lengthX2 - id_x;
        for(uint i = 0; i < tail; ++i)
        {
            dstPtr[dstIdx + i] = Operation::op(src1Ptr[srcIdx1 + i], src2Ptr[srcIdx2 + i]);
        }
    }
}

template <typename T, typename Operation>
__global__ void tensor_or_tensor_nd_hip_tensor(T *srcPtr1,
                                               T *srcPtr2,
                                               uint *srcStrides1,
                                               uint *srcStrides2,
                                               uint *src1BeginOffsets,
                                               uint *src2BeginOffsets,
                                               uint numDims,
                                               T *dstPtr,
                                               uint *dstStrides,
                                               uint *dstDims,
                                               Operation op)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

    uint* dstSampleDims = dstDims + id_z * RPPT_MAX_DIMS;
    uint* src1SampleStrides = srcStrides1 + id_z * RPPT_MAX_DIMS;
    uint* src2SampleStrides = srcStrides2 + id_z * RPPT_MAX_DIMS;
    uint* dstSampleStrides = dstStrides + id_z * RPPT_MAX_DIMS;
    if(id_x >= dstSampleStrides[0])
        return;

    uint dstIdx = id_x + id_z * dstSampleStrides[0];
    uint srcIdx1 = id_z * src1SampleStrides[0];
    uint srcIdx2 = id_z * src2SampleStrides[0];

    for(int i = numDims - 1; i >= 0; i--)
    {
        int index = id_x % dstSampleDims[i];
        srcIdx1 = srcIdx1 + (index * src1SampleStrides[i + 1]);
        srcIdx2 = srcIdx2 + (index * src2SampleStrides[i + 1]);
        id_x = id_x / dstSampleDims[i];
    }

    srcIdx1 += src1BeginOffsets[id_z];
    srcIdx2 += src2BeginOffsets[id_z];

    dstPtr[dstIdx] = Operation::op(srcPtr1[srcIdx1], srcPtr2[srcIdx2]);
}

template <typename T, typename Operation>
__global__ void tensor_or_tensor_non_broadcast_nd_hip_tensor(T *src1Ptr,
                                                             T *src2Ptr,
                                                             uint* src1Dims,
                                                             uint numDims,
                                                             T *dstPtr,
                                                             uint *strides,
                                                             uint *roiTensor1,
                                                             uint *roiTensor2,
                                                             Operation op)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

    using VectorType = typename BitwiseLoadStoreExecute<T>::VectorType;

    // Total number of elements per tensor (flattened length)
    uint totalElements = *strides;

    if(id_x >= totalElements)
        return;

    uint remaining = totalElements - id_x;

    if (remaining >= 8)
    {
        uint *roi1 = roiTensor1 + id_z * numDims * 2;
        uint *begin1 = roi1;
        uint *roi2 = roiTensor2 + id_z * numDims * 2;
        uint *begin2 = roi2;
        uint *length = &roi2[numDims];
        uint dstIdx = (id_z * totalElements);
        uint srcIdx1 = (id_z * totalElements);
        uint srcIdx2 = (id_z * totalElements);
        uint *dimStrides = strides + 1;
        uint coords[RPPT_MAX_DIMS];

        for(int i = 0; i < numDims; i++)
        {
            coords[i] = (id_x / dimStrides[i]) % src1Dims[i];
            if(coords[i] >= length[i])
                return;
        }

        for(int i = 0; i < numDims; i++)
        {
            dstIdx += (coords[i] * dimStrides[i]);
            srcIdx1 += (begin1[i] + (coords[i] * dimStrides[i]));
            srcIdx2 += (begin2[i] + (coords[i] * dimStrides[i]));
        }

        VectorType src1_vec8, src2_vec8, dst_vec8;
        BitwiseLoadStoreExecute<T>::rpp_hip_load8(src1Ptr + srcIdx1, (T*)&src1_vec8);
        BitwiseLoadStoreExecute<T>::rpp_hip_load8(src2Ptr + srcIdx2, (T*)&src2_vec8);
        BitwiseOperationExecute<VectorType, Operation>::rpp_hip_math_bitwiseOp8(&src1_vec8, &src2_vec8, &dst_vec8);
        BitwiseLoadStoreExecute<T>::rpp_hip_pack_and_store8(dstPtr + dstIdx, &dst_vec8);
    }
    else
    {
        // Tail handling for the final (remaining < 8) elements to avoid out-of-bounds access
        uint *roi1 = roiTensor1 + id_z * numDims * 2;
        uint *begin1 = roi1;
        uint *roi2 = roiTensor2 + id_z * numDims * 2;
        uint *begin2 = roi2;
        uint *length = &roi2[numDims];

        // Base indices for this batch
        uint dstBaseIdx   = id_z * totalElements;
        uint srcBaseIdx1  = id_z * totalElements;
        uint srcBaseIdx2  = id_z * totalElements;

        // Strides for each dimension start at strides[1]
        uint *dimStrides = strides + 1;

        for (uint e = 0; e < remaining; e++)
        {
            uint flatIdx = static_cast<uint>(id_x) + e;

            uint dstIdx = dstBaseIdx;
            uint srcIdx1 = srcBaseIdx1;
            uint srcIdx2 = srcBaseIdx2;
            bool skipElement = false;

            // Compute coordinates and indices for this element
            for (int i = 0; i < numDims; i++)
            {
                uint coord = (flatIdx / dimStrides[i]) % src1Dims[i];
                if (coord >= length[i])
                {
                    skipElement = true;
                    break;
                }

                dstIdx  += coord * dimStrides[i];
                srcIdx1 += begin1[i] + coord * dimStrides[i];
                srcIdx2 += begin2[i] + coord * dimStrides[i];
            }

            if (skipElement)
                continue;

            dstPtr[dstIdx] = Operation::op(src1Ptr[srcIdx1], src2Ptr[srcIdx2]);
        }
    }
}

// -------------------- Set 3 - executor kernels --------------------

// Contains kernel launches to the broadcast version, can be used for the non broadcast cases also
template <typename T, typename Operation>
RppStatus hip_exec_tensor_binary_bitwise_generic_tensor(T *srcPtr1,
                                                        T *srcPtr2,
                                                        RpptGenericDescPtr srcGenericDescPtr1,
                                                        RpptGenericDescPtr srcGenericDescPtr2,
                                                        T *dstPtr,
                                                        RpptGenericDescPtr dstGenericDescPtr,
                                                        Operation op,
                                                        uint *roiTensor1,
                                                        uint *roiTensor2,
                                                        rpp::Handle& handle)
{
    Rpp32u batchSize = dstGenericDescPtr->dims[0]; // Number of samples in batch
    Rpp32u src1NDim = srcGenericDescPtr1->numDims - 1; // Omitting batchSize here to get tensor dimension
    Rpp32u src2NDim = srcGenericDescPtr2->numDims - 1; // Omitting batchSize here to get tensor dimension
    Rpp32u dstDim = src1NDim > src2NDim ? src1NDim : src2NDim; // Destination dimension set to maximum of the input dimensions
    Rpp32u minDim = src1NDim < src2NDim ? src1NDim : src2NDim; // Minimum of input dimensions
    Rpp32u *src1Strides = srcGenericDescPtr1->strides;
    Rpp32u *src2Strides = srcGenericDescPtr2->strides;
    Rpp32u *dstStrides = dstGenericDescPtr->strides;

    for(int i = 0; i < minDim; i++)
    {
        if(srcGenericDescPtr1->dims[src1NDim - i] != srcGenericDescPtr2->dims[src2NDim - i])
        {
            if((srcGenericDescPtr1->dims[src1NDim - i] != 1) && (srcGenericDescPtr2->dims[src2NDim - i] != 1))
                return RPP_ERROR_INVALID_DIM_LENGTHS;
        }
    }

    // Allocate pinned buffers for broadcast dims/strides - Strides and Dims for each sample in batch
    Rpp32u *src1BroadcastDims = reinterpret_cast<Rpp32u *>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
    Rpp32u *src2BroadcastDims = src1BroadcastDims + (batchSize * RPPT_MAX_DIMS);
    Rpp32u *dstBroadcastDims = src2BroadcastDims + (batchSize * RPPT_MAX_DIMS);
    Rpp32u *src1BeginOffsets = dstBroadcastDims + (batchSize * RPPT_MAX_DIMS);
    Rpp32u *src2BeginOffsets = src1BeginOffsets + batchSize;

    Rpp32u *src1BroadcastStrides = src2BeginOffsets + batchSize;
    Rpp32u *src2BroadcastStrides = src1BroadcastStrides + (batchSize * RPPT_MAX_DIMS);
    Rpp32u *dstBroadcastStrides = src2BroadcastStrides + (batchSize * RPPT_MAX_DIMS);

    std::atomic<RppStatus> broadcastCompatStatus{RPP_SUCCESS};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(batchSize)
    for(int i = 0; i < batchSize; i++)
    {
        bool incompatibleDims = false;

        Rpp32u *src1roi = roiTensor1 + i * src1NDim * 2;
        Rpp32u *src1Begin = src1roi;
        Rpp32u *src1Dims = src1Begin + src1NDim;

        Rpp32u *src2roi = roiTensor2 + i * src2NDim * 2;
        Rpp32u *src2Begin = src2roi;
        Rpp32u *src2Dims = src2Begin + src2NDim;

        Rpp32u *src1SampleDims = src1BroadcastDims + i * RPPT_MAX_DIMS;
        Rpp32u *src2SampleDims = src2BroadcastDims + i * RPPT_MAX_DIMS;
        Rpp32u *dstSampleDims  = dstBroadcastDims  + i * RPPT_MAX_DIMS;

        Rpp32u *src1SampleStrides = src1BroadcastStrides + i * RPPT_MAX_DIMS;
        Rpp32u *src2SampleStrides = src2BroadcastStrides + i * RPPT_MAX_DIMS;
        Rpp32u *dstSampleStrides  = dstBroadcastStrides  + i * RPPT_MAX_DIMS;

        // Copy the first stride i.e stride for traversing entire sample
        src1SampleStrides[0] = src1Strides[0];
        src2SampleStrides[0] = src2Strides[0];
        dstSampleStrides[0]  = dstStrides[0];

        src1BeginOffsets[i] = 0;
        src2BeginOffsets[i] = 0;

        // Calculate offsets for left-padding to align trailing axes
        int src1Offset = dstDim - src1NDim;
        int src2Offset = dstDim - src2NDim;

        std::fill_n(src1SampleDims, dstDim, 1);
        std::fill_n(src2SampleDims, dstDim, 1);
        std::fill_n(src1SampleStrides + 1, dstDim, 0);
        std::fill_n(src2SampleStrides + 1, dstDim, 0);

        // Copy ROI limits and Strides to individual sample arrays with proper alignment
        memcpy(src1SampleDims + src1Offset, src1Dims, src1NDim * sizeof(Rpp32u));
        memcpy(src2SampleDims + src2Offset, src2Dims, src2NDim * sizeof(Rpp32u));
        memcpy(src1SampleStrides + 1 + src1Offset, src1Strides + 1, src1NDim * sizeof(Rpp32u));
        memcpy(src2SampleStrides + 1 + src2Offset, src2Strides + 1, src2NDim * sizeof(Rpp32u));
        memcpy(dstSampleStrides + 1, dstStrides + 1, dstDim * sizeof(Rpp32u));

        // Compute begin offsets based on ROIs & check incompatibility of dimensions and compute dstSampleDims
        for(int j = 0; j < dstDim; j++)
        {
            if((src1SampleDims[j] != src2SampleDims[j]) && (src1SampleDims[j] != 1) && (src2SampleDims[j] != 1))
                incompatibleDims = true;

            dstSampleDims[j] = std::max(src1SampleDims[j], src2SampleDims[j]);
        }

        if(incompatibleDims == true)
        {
            broadcastCompatStatus.store(RPP_ERROR_INVALID_DIM_LENGTHS, std::memory_order_relaxed);
            continue;
        }

        // Compute begin offsets for src1 and src2
        for(int j = 0; j < src1NDim; j++)
            src1BeginOffsets[i] += src1Begin[j] * src1Strides[j + 1];
        for(int j = 0; j < src2NDim; j++)
            src2BeginOffsets[i] += src2Begin[j] * src2Strides[j + 1];

        // Source strides for sample set to zero if corresponding axis shape = 1 for broadcasting purposes
        // Setting stride to zero will allow for repetition of values operated required for broadcasting
        for(int j = 0; j < dstDim; j++)
        {
            if((src1SampleDims[j] != dstSampleDims[j]) && (src1SampleDims[j] == 1))
                src1SampleStrides[j + 1] = 0;
            if((src2SampleDims[j] != dstSampleDims[j]) && (src2SampleDims[j] == 1))
                src2SampleStrides[j + 1] = 0;
        }
    }

    RppStatus compatSt = broadcastCompatStatus.load();
    if (compatSt != RPP_SUCCESS)
        return compatSt;

    // Allocate device memory for HIP kernel inputs - Strides and Dims for each sample in batch
    Rpp32u *d_dstBroadcastDims, *d_src1BeginOffsets, *d_src2BeginOffsets;
    Rpp32u *d_src1BroadcastStrides, *d_src2BroadcastStrides, *d_dstBroadcastStrides;

    d_dstBroadcastDims = reinterpret_cast<Rpp32u *>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
    d_src1BeginOffsets = d_dstBroadcastDims + (batchSize * RPPT_MAX_DIMS);
    d_src2BeginOffsets = d_src1BeginOffsets + batchSize;
    d_src1BroadcastStrides = d_src2BeginOffsets + batchSize;
    d_src2BroadcastStrides = d_src1BroadcastStrides + (batchSize * RPPT_MAX_DIMS);
    d_dstBroadcastStrides = d_src2BroadcastStrides + (batchSize * RPPT_MAX_DIMS);

    // Copy to device
    RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(d_dstBroadcastDims, dstBroadcastDims, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u), hipMemcpyHostToDevice, handle.GetStream()));
    RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(d_src1BeginOffsets, src1BeginOffsets, batchSize * sizeof(Rpp32u), hipMemcpyHostToDevice, handle.GetStream()));
    RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(d_src2BeginOffsets, src2BeginOffsets, batchSize * sizeof(Rpp32u), hipMemcpyHostToDevice, handle.GetStream()));
    RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(d_src1BroadcastStrides, src1BroadcastStrides, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u), hipMemcpyHostToDevice, handle.GetStream()));
    RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(d_src2BroadcastStrides, src2BroadcastStrides, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u), hipMemcpyHostToDevice, handle.GetStream()));
    RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(d_dstBroadcastStrides, dstBroadcastStrides,  batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u), hipMemcpyHostToDevice, handle.GetStream()));

    // based on number of dimensions call the corresponding kernel
    if(dstDim == 1)
    {
        // NW
        int globalThreads_x = dstGenericDescPtr->dims[1];
        int globalThreads_y = 1;
        int globalThreads_z = dstGenericDescPtr->dims[0];
        hipLaunchKernelGGL(tensor_or_tensor_1d_hip_tensor,
                        dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                        dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                        0,
                        handle.GetStream(),
                        srcPtr1,
                        srcPtr2,
                        d_src1BroadcastStrides,
                        d_src2BroadcastStrides,
                        d_src1BeginOffsets,
                        d_src2BeginOffsets,
                        dstPtr,
                        d_dstBroadcastStrides,
                        d_dstBroadcastDims,
                        op);
    }
    else if(dstDim == 2)
    {
        // NHW
        int globalThreads_x = dstGenericDescPtr->dims[2];
        int globalThreads_y = dstGenericDescPtr->dims[1];
        int globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(tensor_or_tensor_2d_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           d_src1BroadcastStrides,
                           d_src2BroadcastStrides,
                           d_src1BeginOffsets,
                           d_src2BeginOffsets,
                           dstPtr,
                           d_dstBroadcastStrides,
                           d_dstBroadcastDims,
                           op);
    }
    else if(dstDim == 3)
    {
        // NDHW
        int globalThreads_x = dstGenericDescPtr->dims[3];
        int globalThreads_y = dstGenericDescPtr->dims[2];
        int globalThreads_z = dstGenericDescPtr->dims[1];

        for(int batchCount = 0; batchCount < batchSize; batchCount++)
        {
            hipLaunchKernelGGL(tensor_or_tensor_3d_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1 + (batchCount * srcGenericDescPtr1->strides[0]),
                               srcPtr2 + (batchCount * srcGenericDescPtr2->strides[0]),
                               d_src1BroadcastStrides + batchCount * RPPT_MAX_DIMS,
                               d_src2BroadcastStrides + batchCount * RPPT_MAX_DIMS,
                               src1BeginOffsets[batchCount],
                               src2BeginOffsets[batchCount],
                               dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                               d_dstBroadcastStrides + batchCount * RPPT_MAX_DIMS,
                               d_dstBroadcastDims + batchCount * RPPT_MAX_DIMS,
                               op);
        }
    }
    else
    {
        // interpret the input as 1D tensor
        int globalThreads_x = dstGenericDescPtr->strides[0];
        int globalThreads_y = 1;
        int globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(tensor_or_tensor_nd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                           dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           d_src1BroadcastStrides,
                           d_src2BroadcastStrides,
                           d_src1BeginOffsets,
                           d_src2BeginOffsets,
                           dstGenericDescPtr->numDims - 1,
                           dstPtr,
                           d_dstBroadcastStrides,
                           d_dstBroadcastDims,
                           op);
    }

    return RPP_SUCCESS;
}

// Contains kernel launches specific to the non broadcast version, cannot be used for the broadcast cases
template <typename T, typename Operation>
RppStatus hip_exec_tensor_non_broadcast_binary_bitwise_generic_tensor(T *srcPtr1,
                                                                      T *srcPtr2,
                                                                      RpptGenericDescPtr srcGenericDescPtr1,
                                                                      RpptGenericDescPtr srcGenericDescPtr2,
                                                                      T *dstPtr,
                                                                      RpptGenericDescPtr dstGenericDescPtr,
                                                                      Operation op,
                                                                      uint *roiTensor1,
                                                                      uint *roiTensor2,
                                                                      rpp::Handle& handle)
{
    Rpp32u numDims = srcGenericDescPtr1->numDims - 1; // exclude batchsize from input dims
    // based on number of dimensions call the corresponding kernel
    if(numDims == 1)
    {

        // NW
        int globalThreads_x = (dstGenericDescPtr->dims[1] + 7) >> 3;
        int globalThreads_y = 1;
        int globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(tensor_or_tensor_non_broadcast_1d_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           dstPtr,
                           dstGenericDescPtr->strides[0],
                           roiTensor1,
                           roiTensor2,
                           op);
    }
    else if(numDims == 2)
    {
        // NHW
        int globalThreads_x = (dstGenericDescPtr->dims[2] + 7) >> 3;
        int globalThreads_y = dstGenericDescPtr->dims[1];
        int globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(tensor_or_tensor_non_broadcast_2d_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           dstPtr,
                           make_uint2(dstGenericDescPtr->strides[0], dstGenericDescPtr->strides[1]),
                           roiTensor1,
                           roiTensor2,
                           op);
    }
    else if(numDims == 3)
    {
        // NDHW
        int globalThreads_x = (dstGenericDescPtr->dims[3] + 7) >> 3;
        int globalThreads_y = dstGenericDescPtr->dims[2];
        int globalThreads_z = dstGenericDescPtr->dims[1];

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            hipLaunchKernelGGL(tensor_or_tensor_non_broadcast_3d_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1 + (batchCount * srcGenericDescPtr1->strides[0]),
                               srcPtr2 + (batchCount * srcGenericDescPtr2->strides[0]),
                               dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                               make_uint2(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2]),
                               &roiTensor1[batchCount * 6],
                               &roiTensor2[batchCount * 6],
                               op);
        }
    }
    else
    {
        // interpret the input as 1D tensor
        int globalThreads_x = dstGenericDescPtr->strides[0];
        int globalThreads_y = 1;
        int globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(tensor_or_tensor_non_broadcast_nd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           srcGenericDescPtr1->dims + 1,
                           srcGenericDescPtr1->numDims - 1,
                           dstPtr,
                           dstGenericDescPtr->strides,
                           roiTensor1,
                           roiTensor2,
                           op);
    }

    return RPP_SUCCESS;
}

// Dispatcher function that dispatches the calls to the appropriate templated function based on the datatype and operation
template<typename T>
RppStatus tensor_binary_bitwise_op_dispatch_gpu_tensor(T *srcPtr1,
                                                       T *srcPtr2,
                                                       RpptGenericDescPtr srcPtr1GenericDescPtr,
                                                       RpptGenericDescPtr srcPtr2GenericDescPtr,
                                                       T *dstPtr,
                                                       RpptGenericDescPtr dstGenericDescPtr,
                                                       RpptBitwiseOp tensorOp,
                                                       RpptBroadcastMode broadcastMode,
                                                       Rpp32u *srcPtr1roiTensor,
                                                       Rpp32u *srcPtr2roiTensor,
                                                       rpp::Handle& handle)
{
    if(broadcastMode == RPP_BROADCAST_ENABLE)
    {
        switch(tensorOp)
        {
            case RPP_TENSOR_OP_AND:
                hip_exec_tensor_binary_bitwise_generic_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, BitwiseAnd(), srcPtr1roiTensor, srcPtr2roiTensor, handle);
                break;
            case RPP_TENSOR_OP_OR:
                hip_exec_tensor_binary_bitwise_generic_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, BitwiseOr(), srcPtr1roiTensor, srcPtr2roiTensor, handle);
                break;
            case RPP_TENSOR_OP_XOR:
                hip_exec_tensor_binary_bitwise_generic_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, BitwiseXor(), srcPtr1roiTensor, srcPtr2roiTensor, handle);
                break;
        }
    }
    else
    {
        switch(tensorOp)
        {
            case RPP_TENSOR_OP_AND:
                hip_exec_tensor_non_broadcast_binary_bitwise_generic_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, BitwiseAnd(), srcPtr1roiTensor, srcPtr2roiTensor, handle);
                break;
            case RPP_TENSOR_OP_OR:
                hip_exec_tensor_non_broadcast_binary_bitwise_generic_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, BitwiseOr(), srcPtr1roiTensor, srcPtr2roiTensor, handle);
                break;
            case RPP_TENSOR_OP_XOR:
                hip_exec_tensor_non_broadcast_binary_bitwise_generic_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, BitwiseXor(), srcPtr1roiTensor, srcPtr2roiTensor, handle);
                break;
        }
    }

    return RPP_SUCCESS;
}

template RppStatus tensor_binary_bitwise_op_dispatch_gpu_tensor<Rpp8u>(Rpp8u*,
                                                                       Rpp8u*,
                                                                       RpptGenericDescPtr,
                                                                       RpptGenericDescPtr,
                                                                       Rpp8u*,
                                                                       RpptGenericDescPtr,
                                                                       RpptBitwiseOp,
                                                                       RpptBroadcastMode,
                                                                       Rpp32u*,
                                                                       Rpp32u*,
                                                                       rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_gpu_tensor<Rpp16u>(Rpp16u*,
                                                                        Rpp16u*,
                                                                        RpptGenericDescPtr,
                                                                        RpptGenericDescPtr,
                                                                        Rpp16u*,
                                                                        RpptGenericDescPtr,
                                                                        RpptBitwiseOp,
                                                                        RpptBroadcastMode,
                                                                        Rpp32u*,
                                                                        Rpp32u*,
                                                                        rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_gpu_tensor<Rpp32u>(Rpp32u*,
                                                                        Rpp32u*,
                                                                        RpptGenericDescPtr,
                                                                        RpptGenericDescPtr,
                                                                        Rpp32u*,
                                                                        RpptGenericDescPtr,
                                                                        RpptBitwiseOp,
                                                                        RpptBroadcastMode,
                                                                        Rpp32u*,
                                                                        Rpp32u*,
                                                                        rpp::Handle&);
