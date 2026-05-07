// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/utility/amd_swmma.hpp"
#include "ck/library/utility/fill.hpp"

namespace ck {
namespace swmma_op_util {

template <typename DataType>
void dump_tensor(Tensor<DataType> mat)
{
    size_t row = mat.GetLengths()[0];
    size_t col = mat.GetLengths()[1];
    std::cout << "mat [ " << std::endl;
    for(uint32_t i = 0; i < row; i++)
    {
        std::cout << "    [";
        for(uint32_t j = 0; j < col; j++)
        {
            std::vector<std::size_t> idx({i, j});
            std::cout << ck::type_convert<float>(mat(idx)) << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}

template <typename T, index_t vec_size, index_t kMultiplier, typename = void>
struct SWMMAVecType
{
    static_assert(sizeof(T) == 0, "VecType is not specialized for this type");
};

template <typename T, index_t vec_size, index_t kMultiplier>
struct SWMMAVecType<T,
                    vec_size,
                    kMultiplier,
                    ck::enable_if_t<ck::is_same_v<T, ck::half_t> || ck::is_same_v<T, ck::bhalf_t>>>
{
    static constexpr bool layoutTransform = false;
    static constexpr int ToIntDim         = 2;

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<T, D>;
    }

    using VecT  = vector_type<T, kMultiplier * vec_size>;
    using ViewT = vector_type<T, 2>;
};

template <typename T, index_t vec_size, index_t kMultiplier>
struct SWMMAVecType<T,
                    vec_size,
                    kMultiplier,
                    ck::enable_if_t<ck::is_same_v<T, ck::f8_t> || ck::is_same_v<T, ck::bf8_t>>>
{
    static constexpr bool layoutTransform = true;
    static constexpr int ToIntDim         = 4;

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<D, ck::f8_t> || ck::is_same_v<D, ck::bf8_t>;
    }

    using VecT  = vector_type<T, kMultiplier * vec_size>;
    using ViewT = vector_type<T, 4>;
};

template <typename T, index_t vec_size, index_t kMultiplier>
struct SWMMAVecType<T,
                    vec_size,
                    kMultiplier,
                    ck::enable_if_t<ck::is_same_v<T, int8_t> || ck::is_same_v<T, uint8_t>>>
{
    static constexpr bool layoutTransform = true;
    static constexpr int ToIntDim         = 4;
    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<D, int8_t> || ck::is_same_v<D, uint8_t>;
    }

    using VecT  = vector_type<T, kMultiplier * vec_size>;
    using ViewT = vector_type<T, 4>;
};

template <typename T, index_t vec_size, index_t kMultiplier>
struct SWMMAVecType<T, vec_size, kMultiplier, ck::enable_if_t<ck::is_same_v<T, ck::int4_t>>>
{
    static constexpr bool layoutTransform = true;
    static constexpr int ToIntDim         = 8;
    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<T, D>;
    }

    using VecT  = vector_type<int32_t, kMultiplier * vec_size / 8>;
    using ViewT = vector_type<int32_t, 1>;
};

#define call_swmma_branch(src0, src1, dst)                                                    \
    intrin_swmma_##dst##_16x16_##src0##src1##_w32<16, 16, true, kMultiplier, SparseSel>::Run( \
        reg_a, reg_b, reg_d.GetVectorTypeReference(Number<0>{}), index_val)

#define call_swmma_branch_with_neg(src0, src1, dst, neg0, neg1)                                 \
    intrin_swmma_##dst##_16x16_##src0##src1##_w32<16,                                           \
                                                  16,                                           \
                                                  neg0,                                         \
                                                  neg1,                                         \
                                                  true,                                         \
                                                  kMultiplier,                                  \
                                                  SparseSel>::Run(reg_a,                        \
                                                                  reg_b,                        \
                                                                  reg_d.GetVectorTypeReference( \
                                                                      Number<0>{}),             \
                                                                  index_val)

#define call_swmma_branch_with_neg_with_acc(src0, src1, acc, dst, neg_a, neg_b) \
    intrin_swmma_##dst##acc##_16x16_##src0##src1##_w32<                         \
        16,                                                                     \
        16,                                                                     \
        neg_a,                                                                  \
        neg_b,                                                                  \
        true,                                                                   \
        kMultiplier,                                                            \
        SparseSel>::Run(reg_a,                                                  \
                        reg_b,                                                  \
                        reg_c.GetVectorTypeReference(Number<0>{}),              \
                        reg_d.GetVectorTypeReference(Number<0>{}),              \
                        index_val)

#define call_swmma_branch_instrinsic(                                                      \
    src0_type, src0_fmt, src1_type, src1_fmt, dst0_type, dst0_fmt, dst1_type, dst1_fmt)    \
    if constexpr(ck::is_same_v<srcAType, src0_type> && ck::is_same_v<srcBType, src1_type>) \
    {                                                                                      \
        if constexpr(ck::is_same_v<accType, dst0_type>)                                    \
        {                                                                                  \
            call_swmma_branch(src0_fmt, src1_fmt, dst0_fmt);                               \
        }                                                                                  \
        else if constexpr(ck::is_same_v<accType, dst1_type>)                               \
        {                                                                                  \
            call_swmma_branch(src0_fmt, src1_fmt, dst1_fmt);                               \
        }                                                                                  \
    }

#define call_swmma_branch_instrinsic_with_boolean(src0_type,                               \
                                                  src0_fmt,                                \
                                                  src1_type,                               \
                                                  src1_fmt,                                \
                                                  dst0_type,                               \
                                                  dst0_fmt,                                \
                                                  dst1_type,                               \
                                                  dst1_fmt,                                \
                                                  neg0,                                    \
                                                  neg1)                                    \
    if constexpr(ck::is_same_v<srcAType, src0_type> && ck::is_same_v<srcBType, src1_type>) \
    {                                                                                      \
        if constexpr(ck::is_same_v<accType, dst0_type>)                                    \
        {                                                                                  \
            call_swmma_branch_with_neg(src0_fmt, src1_fmt, dst0_fmt, neg0, neg1);          \
        }                                                                                  \
        else if constexpr(ck::is_same_v<accType, dst1_type>)                               \
        {                                                                                  \
            call_swmma_branch_with_neg(src0_fmt, src1_fmt, dst1_fmt, neg0, neg1);          \
        }                                                                                  \
    }

#define call_swmma_branch_instrinsic_with_boolean_with_acc(                                     \
    src0_type, src0_fmt, src1_type, src1_fmt, acc_type, acc_fmt, dst_type, dst_fmt, neg0, neg1) \
    if constexpr(ck::is_same_v<srcAType, src0_type> && ck::is_same_v<srcBType, src1_type>)      \
    {                                                                                           \
        call_swmma_branch_with_neg_with_acc(src0_fmt, src1_fmt, acc_fmt, dst_fmt, neg0, neg1);  \
    }

template <typename srcAType,
          typename srcBType,
          typename accType,
          typename dstType,
          index_t kMultiplier,
          bool SparseSel>
__device__ void builtin_swmma_naive_selector(
    const typename SWMMAVecType<srcAType, 8, kMultiplier>::VecT::type& reg_a,
    const typename SWMMAVecType<srcBType, 16, kMultiplier>::VecT::type& reg_b,
    const StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, accType, 1, 8, true>& reg_c,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, dstType, 1, 8, true>& reg_d,
    index_t index_val)
{
    if constexpr(ck::is_same_v<accType, dstType>)
    {
        call_swmma_branch_instrinsic(half_t, f16, half_t, f16, float, f32, ck::half_t, f16);
        call_swmma_branch_instrinsic(bhalf_t, bf16, bhalf_t, bf16, float, f32, ck::bhalf_t, bf16);
        call_swmma_branch_instrinsic(f8_t, f8, f8_t, f8, float, f32, ck::half_t, f16);
        call_swmma_branch_instrinsic(bf8_t, bf8, bf8_t, bf8, float, f32, ck::half_t, f16);
        call_swmma_branch_instrinsic(f8_t, f8, bf8_t, bf8, float, f32, ck::half_t, f16);
        call_swmma_branch_instrinsic(bf8_t, bf8, f8_t, f8, float, f32, ck::half_t, f16);
        call_swmma_branch_instrinsic_with_boolean(
            int8_t, iu8, int8_t, iu8, int32_t, i32, float, f32, true, true);
        call_swmma_branch_instrinsic_with_boolean(
            int8_t, iu8, uint8_t, iu8, int32_t, i32, float, f32, true, false);
        call_swmma_branch_instrinsic_with_boolean(
            uint8_t, iu8, int8_t, iu8, int32_t, i32, float, f32, false, true);
        call_swmma_branch_instrinsic_with_boolean(
            uint8_t, iu8, uint8_t, iu8, int32_t, i32, float, f32, false, false);
        call_swmma_branch_instrinsic_with_boolean(
            int4_t, iu4, int4_t, iu4, int32_t, i32, float, f32, true, true);
    }
    else
    {
        call_swmma_branch_instrinsic_with_boolean_with_acc(
            int8_t, iu8, int8_t, iu8, int32_t, i32 /*acc*/, float, f32 /*dst*/, true, true);
        call_swmma_branch_instrinsic_with_boolean_with_acc(
            int4_t, iu4, int4_t, iu4, int32_t, i32 /*acc*/, float, f32 /*dst*/, true, true);
    }
}

template <typename src1_t,
          typename src2_t,
          typename acc_t,
          typename dst_t,
          index_t M,
          index_t N,
          index_t K,
          index_t kMultiplier,
          bool SparseSel>
__global__ void matmul(const src1_t* a, const int32_t* a_index, const src2_t* b, dst_t* c)
{
    using srcA_traits = SWMMAVecType<src1_t, 8, kMultiplier>;
    using srcB_traits = SWMMAVecType<src2_t, 16, kMultiplier>;
    static_assert(srcA_traits::template is_compatible<src2_t>(),
                  "the data format for srcA and srcB is unsupported in gfx13");
    // view to int32_t
    using srcA_cast_type = srcA_traits::ViewT::type;
    using srcB_cast_type = srcB_traits::ViewT::type;

    using srcA_vec = srcA_traits::VecT;
    using srcB_vec = srcB_traits::VecT;

    // this means how many elements in src1_t data type will be loaded
    constexpr int a_src_dim  = 8 * kMultiplier;
    constexpr int b_src_dim  = 16 * kMultiplier;
    constexpr int a_row_size = 16 * kMultiplier;
    constexpr int b_row_size = 32 * kMultiplier;
    constexpr int view_dim   = srcA_traits::ToIntDim;
    // this means how many elements in int type will be loaded
    constexpr int a_iteration  = a_src_dim / view_dim;
    constexpr int b_iteration  = b_src_dim / view_dim;
    constexpr int a_src_stride = a_row_size / view_dim;
    constexpr int b_src_stride = b_row_size / view_dim;
    static constexpr auto I0   = Number<0>{};

    srcA_vec a_frag = {};
    srcB_vec b_frag = {};
    using acc_vec   = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, 8, true>;
    acc_vec acc_thread_buf_;

    using dst_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, dst_t, 1, 8, true>;
    dst_vec dst_thread_buf_;

    const srcA_cast_type* vgpr_a_ptr = reinterpret_cast<const srcA_cast_type*>(a);
    const srcB_cast_type* vgpr_b_ptr = reinterpret_cast<const srcB_cast_type*>(b);
    const int lane                   = threadIdx.x;
    // this is int32's offset; because a and b's element number is different because of sparsity.
    int a_start_idx = (lane >> 1) * a_src_stride + (lane & 1);
    int b_start_idx = (lane >> 1) * b_src_stride + (lane & 1);

    static_for<0, a_iteration, 1>{}([&](auto ele) {
        int index                                     = a_start_idx + (ele << 1);
        a_frag.template AsType<srcA_cast_type>()(ele) = vgpr_a_ptr[index];
    });

    static_for<0, b_iteration, 1>{}([&](auto ele) {
        int index                                     = b_start_idx + (ele << 1);
        b_frag.template AsType<srcB_cast_type>()(ele) = vgpr_b_ptr[index];
    });
    int32_t index_val = 0;
    if constexpr(kMultiplier == 1)
    {
        if constexpr(SparseSel == false)
        {
            if((lane & 1) == 0) // even lane
            {
                index_val = a_index[lane >> 1];
            }
        }
        else
        {
            if((lane & 1) == 1) // odd lane
            {
                index_val = a_index[lane >> 1];
            }
        }
    }
    else
    {
        index_val = a_index[lane];
    }

// currently only support gfx13
#if defined(__gfx13__)
    using srcA_vec_type = srcA_vec::type;
    using srcB_vec_type = srcB_vec::type;
    builtin_swmma_naive_selector<src1_t, src2_t, acc_t, dst_t, kMultiplier, SparseSel>(
        a_frag.template AsType<srcA_vec_type>()(I0),
        b_frag.template AsType<srcB_vec_type>()(I0),
        acc_thread_buf_,
        dst_thread_buf_,
        index_val);
#else
    ignore = acc_thread_buf_;
    ignore = dst_thread_buf_;
    ignore = index_val;
    ignore = I0;
#endif

    if constexpr(srcA_traits::layoutTransform)
    {
        static_for<0, 8, 1>{}([&](auto ele) {
            const int col = lane >> 1;
            const int row = ((ele & 4) << 1) + (ele & 3) + ((lane & 1) << 2);
            // store results from unpacked c_thread_buf_ output
            c[16 * row + col] = dst_thread_buf_[Number<ele>{}];
        });
    }
    else
    {
        static_for<0, 8, 1>{}([&](auto ele) {
            const int col = lane >> 1;
            const int row = ((ele & 6) << 1) + (ele & 1) + ((lane & 1) << 1);
            // store results from unpacked c_thread_buf_ output
            c[16 * row + col] = dst_thread_buf_[Number<ele>{}];
        });
    }
}

struct GemmParams
{
    GemmParams() : M(16), N(16), K(32), StrideA(32), StrideB(32), StrideC(16), alpha(1), beta(0) {}

    ck::index_t M;
    ck::index_t N;
    ck::index_t K;

    ck::index_t StrideA;
    ck::index_t StrideB;
    ck::index_t StrideC;

    float alpha;
    float beta;
};

template <typename GemmInstance,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void RunHostGEMM(const Tensor<ADataType>& A,
                 const Tensor<BDataType>& B,
                 Tensor<CDataType>& C,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
{
    auto ref_gemm     = GemmInstance{};
    auto ref_invoker  = ref_gemm.MakeInvoker();
    auto ref_argument = ref_gemm.MakeArgument(A, B, C, a_element_op, b_element_op, c_element_op);

    ref_invoker.Run(ref_argument);
}

template <typename KernelType, typename ADataType, typename BDataType, typename CDataType>
bool RunDeviceGEMM(KernelType kernel,
                   const Tensor<ADataType>& A,
                   const Tensor<int32_t>& A_index,
                   const Tensor<BDataType>& B,
                   Tensor<CDataType>& C)
{
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * A.mDesc.GetElementSpaceSize());
    DeviceMem a_index_device_buffer(sizeof(int32_t) * A_index.mDesc.GetElementSpaceSize());
    DeviceMem b_n_k_device_buf(sizeof(BDataType) * B.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * C.mDesc.GetElementSpaceSize());

    if constexpr(ck::is_same_v<ck::int4_t, ADataType> && ck::is_same_v<ck::int4_t, BDataType>)
    {
        std::vector<uint8_t> A_packed;
        std::vector<uint8_t> B_packed;
        A_packed.resize(A.mData.size());
        B_packed.resize(B.mData.size());
        for(size_t i = 0; i < A.mData.size(); i += 2)
        {
            uint8_t val0    = (A.mData[i] & 0xf);
            uint8_t val1    = (A.mData[i + 1] & 0xf);
            A_packed[i / 2] = val0 | (val1 << 4);
        }
        for(size_t i = 0; i < B.mData.size(); i += 2)
        {
            uint8_t val0    = (B.mData[i] & 0xf);
            uint8_t val1    = (B.mData[i + 1] & 0xf);
            B_packed[i / 2] = val0 | (val1 << 4);
        }
        a_m_k_device_buf.ToDevice(A_packed.data());
        b_n_k_device_buf.ToDevice(B_packed.data());
    }
    else
    {
        a_m_k_device_buf.ToDevice(A.data());
        b_n_k_device_buf.ToDevice(B.data());
    }

    a_index_device_buffer.ToDevice(A_index.data());
    kernel<<<1, 32>>>(static_cast<const ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                      static_cast<const int32_t*>(a_index_device_buffer.GetDeviceBuffer()),
                      static_cast<const BDataType*>(b_n_k_device_buf.GetDeviceBuffer()),
                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()));
    c_m_n_device_buf.FromDevice(C.mData.data());

    return true;
}

template <typename DeviceSwmmaKernel,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GPUAccDataType,
          typename CPUAccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          index_t CAccNum,
          index_t M,
          index_t N,
          index_t K>
struct TestSwmma
{
    auto PrepareGemmTensor(const ck::swmma_op_util::GemmParams& params)
    {
        auto f_host_tensor_descriptor =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({stride, 1}));
                }
                else
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({1, stride}));
                }
            };

        Tensor<ADataType> a_m_k(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<BDataType> b_n_k(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<CDataType> c_m_n_host_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_device_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        // only support ALayout is RowMajor
        // compressed_m_k is the same as A and delete 0 from it;
        Tensor<ADataType> a_compressed_m_k(
            f_host_tensor_descriptor(params.M, params.K / 2, params.StrideA / 2, ALayout{}));
        Tensor<int32_t> index_m_k(
            f_host_tensor_descriptor(params.M, params.K / 2, params.StrideA / 2, ALayout{}));
        Tensor<int32_t> index_compressed_m_k(
            f_host_tensor_descriptor(params.M,
                                     ck::math::integer_divide_ceil(params.K, 32),
                                     ck::math::integer_divide_ceil(params.StrideA, 32),
                                     ALayout{}));
        auto f_generate_tensor_value = [](auto& tensor, auto type, auto min_value, auto max_value) {
            using dataType = decltype(type);
            tensor.GenerateTensorValue(GeneratorTensor_2<dataType>{min_value, max_value});
        };

        // a always has no 0 zero elements
        f_generate_tensor_value(a_m_k, ADataType{}, 1, 10);
        f_generate_tensor_value(b_n_k, BDataType{}, -2, 2);
        ck::utils::TransformIntoStructuralSparsity<ADataType>{}(a_m_k);
        compressSparseMatrix(a_m_k, a_compressed_m_k, index_m_k, index_compressed_m_k);
        return std::make_tuple(a_m_k,
                               b_n_k,
                               a_compressed_m_k,
                               index_compressed_m_k,
                               c_m_n_host_result,
                               c_m_n_device_result);
    }

    template <typename T>
    void compressSparseMatrix(const Tensor<T>& a,
                              Tensor<T>& a_compressed,
                              Tensor<int32_t>& index,
                              Tensor<int32_t>& index_compressed)
    {
        size_t row = a.GetLengths()[0];
        size_t col = a.GetLengths()[1];
        for(uint32_t i = 0; i < row; i++)
        {
            // get nonzero elements and store them in a_compressed
            // get nonzero index and store them in index
            uint32_t nonzero_j = 0;
            for(uint32_t j = 0; j < col; j += 4)
            {
                for(uint32_t k = 0; k < 4; k++)
                {
                    std::vector<std::size_t> a_idx({i, j + k});
                    if(abs(float(a(a_idx))) > 1e-5)
                    {
                        std::vector<std::size_t> nonzero_idx({i, nonzero_j});
                        a_compressed(nonzero_idx) = a(a_idx);
                        index(nonzero_idx)        = k;
                        nonzero_j++;
                    }
                }
            }
            // get nonzero index and pack them into index_compressed
            uint32_t compressed_index_j = 0;
            for(uint32_t j = 0; j < (col >> 1); j += 16)
            {
                int32_t packed_value = 0;
                for(int k = 0; k < 16; k++)
                {
                    std::vector<std::size_t> a_idx({i, j + k});
                    packed_value |= (index(a_idx) & 0x3) << (2 * k);
                }
                std::vector<std::size_t> compressed_index_idx({i, compressed_index_j});
                index_compressed(compressed_index_idx) = packed_value;
                compressed_index_j++;
            }
        }
    }

    auto operator()(const DeviceSwmmaKernel& swmma_kernel)
    {
        std::cout << "ALayout = " << ALayout{}.name << ", BLayout = " << BLayout{}.name
                  << ", CLayout = " << CLayout{}.name << std::endl;

        // Arrange
        ck::swmma_op_util::GemmParams params;
        params.M       = M;
        params.N       = N;
        params.K       = K;
        params.StrideA = K; // M K
        params.StrideB = K; // K N
        params.StrideC = N; // M N

        auto host_tensors = PrepareGemmTensor(params);

        const Tensor<ADataType>& a              = std::get<0>(host_tensors);
        const Tensor<BDataType>& b              = std::get<1>(host_tensors);
        const Tensor<ADataType>& a_compressed   = std::get<2>(host_tensors);
        const Tensor<int32_t>& index_compressed = std::get<3>(host_tensors);
        Tensor<CDataType>& c_host               = std::get<4>(host_tensors);
        Tensor<CDataType>& c_device             = std::get<5>(host_tensors);

        for(auto& t : index_compressed.mData)
            std::cout << std::hex << t << " ";
        std::cout << std::dec; // Reset to decimal format
        std::cout << std::endl;
        auto a_element_op = AElementwiseOperation{};
        auto b_element_op = BElementwiseOperation{};
        auto c_element_op = CElementwiseOperation{};

        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                      BDataType,
                                                      CDataType,
                                                      CPUAccDataType,
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation>;
        ck::swmma_op_util::RunHostGEMM<ReferenceGemmInstance>(
            a, b, c_host, a_element_op, b_element_op, c_element_op);

        // Act
        bool is_supported = ck::swmma_op_util::RunDeviceGEMM(
            swmma_kernel, a_compressed, index_compressed, b, c_device);

        if(is_supported)
        {
            // Assert
            bool res = false;
            if(std::is_same<CDataType, float>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, ck::half_t>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, ck::bhalf_t>::value)
            {
                // 0.5 Pixel Error Tolerance is introduced by Accumulator difference.
                // BF16 WMMA Accumulator is in BF16 Type while On Host-side Accumulator is Float.
                res = ck::utils::check_err(
                    c_device.mData, c_host.mData, "Error: Incorrect results!", 0, 1.0);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, int8_t>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, int32_t>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, double>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else
            {
                std::cout << "UNSUPPORTED CDataType" << std::endl;
            }

            return res;
        }
        else
        {
            return true;
        }
    }
};

} // namespace swmma_op_util
} // namespace ck
