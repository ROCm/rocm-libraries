// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/utility/amd_wmma.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace wmma_op_util {

#if defined(CK_USE_GFX1250)

#define CK_WMMA_CALL_INTRIN_1(dst_fmt, src0_fmt, size)              \
    intrin_wmma_##dst_fmt##_16x16x##size##_##src0_fmt<16, 16>::Run( \
        reg_a, reg_b, reg_d.GetVectorTypeReference(Number<0>{}))

#define CK_WMMA_CALL_INTRIN_2(dst_fmt, src0_fmt, src1_fmt, size)              \
    intrin_wmma_##dst_fmt##_16x16x##size##_##src0_fmt##src1_fmt<16, 16>::Run( \
        reg_a, reg_b, reg_d.GetVectorTypeReference(Number<0>{}))

#define CK_WMMA_CALL_INTRIN_3(dst_fmt, acc_fmt, src0_fmt, size)              \
    intrin_wmma_##dst_fmt##acc_fmt##_16x16x##size##_##src0_fmt<16, 16>::Run( \
        reg_a,                                                               \
        reg_b,                                                               \
        reg_c.GetVectorTypeReference(Number<0>{}),                           \
        reg_d.GetVectorTypeReference(Number<0>{}))

#define CK_WMMA_CALL_INTRIN_4(dst_fmt, src0_fmt, neg_a, neg_b, size)              \
    intrin_wmma_##dst_fmt##_16x16x##size##_##src0_fmt<16, 16, neg_a, neg_b>::Run( \
        reg_a, reg_b, reg_d.GetVectorTypeReference(Number<0>{}))

template <typename T, index_t kValue, typename = void>
struct WMMAVecType
{
    static_assert(sizeof(T) == 0, "VecType is not specialized for this type");
};

// fp64 specialization
template <typename T, index_t kValue>
struct WMMAVecType<T, kValue, ck::enable_if_t<ck::is_same_v<T, double>>>
{
    static constexpr bool layoutTransform = false;
    static constexpr int ToIntDim         = 1;

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<T, D>;
    }

    using VecT                = vector_type<T, kValue / 2>;
    using ViewT               = vector_type<T, 1>;
    static constexpr int size = kValue / 2;
};

// fp32 specialization
template <typename T, index_t kValue>
struct WMMAVecType<T, kValue, ck::enable_if_t<ck::is_same_v<T, float>>>
{
    static constexpr bool layoutTransform = false;
    static constexpr int ToIntDim         = 1;

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<T, D>;
    }

    using VecT                = vector_type<T, kValue / 2>;
    using ViewT               = vector_type<T, 1>;
    static constexpr int size = kValue / 2;
};

// fp16 specialization
template <typename T, index_t kValue>
struct WMMAVecType<T,
                   kValue,
                   ck::enable_if_t<ck::is_same_v<T, ck::half_t> || ck::is_same_v<T, ck::bhalf_t>>>
{
    static constexpr bool layoutTransform = false;
    static constexpr int ToIntDim         = 2;

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<T, D>;
    }

    using VecT                = vector_type<T, kValue / 2>;
    using ViewT               = vector_type<T, 2>;
    static constexpr int size = kValue / 2;
};

// f8 ocp or bf8 ocp specialization
template <typename T, index_t kValue>
struct WMMAVecType<
    T,
    kValue,
    ck::enable_if_t<ck::is_same_v<T, ck::f8_ocp_t> || ck::is_same_v<T, ck::bf8_ocp_t>>>
{
    static constexpr bool layoutTransform = true;
    static constexpr int ToIntDim         = 4;

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<D, ck::f8_t> || ck::is_same_v<D, ck::bf8_t>;
    }

    using VecT                = vector_type<typename T::data_type, kValue / 2>;
    using ViewT               = vector_type<typename T::data_type, 4>;
    static constexpr int size = kValue / 2;
};

// f8 fnuz or bf8 fnuz specialization
template <typename T, index_t kValue>
struct WMMAVecType<
    T,
    kValue,
    ck::enable_if_t<ck::is_same_v<T, ck::f8_fnuz_t> || ck::is_same_v<T, ck::bf8_fnuz_t>>>
{
    static constexpr bool layoutTransform = true;
    static constexpr int ToIntDim         = 4;

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<D, ck::f8_t> || ck::is_same_v<D, ck::bf8_t>;
    }

    using VecT                = vector_type<T, kValue / 2>;
    using ViewT               = vector_type<T, 4>;
    static constexpr int size = kValue / 2;
};

// int8 specialization
template <typename T, index_t kValue>
struct WMMAVecType<T, kValue, ck::enable_if_t<ck::is_same_v<T, int8_t>>>
{
    static constexpr bool layoutTransform = true;
    static constexpr int ToIntDim         = 4;
    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<T, D>;
    }

    using VecT                = vector_type<T, kValue / 2>;
    using ViewT               = vector_type<T, 4>;
    static constexpr int size = kValue / 2;
};

// gfx1250 builtin_wmma_naive_selector
template <typename srcAType, typename srcBType, typename dstType, typename accType, index_t kValue>
__device__ void builtin_wmma_naive_selector(
    const typename WMMAVecType<srcAType, kValue>::VecT::type& reg_a,
    const typename WMMAVecType<srcBType, kValue>::VecT::type& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, accType, 1, 8, true>& reg_c,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, dstType, 1, 8, true>& reg_d)
{
    constexpr int size = kValue;

    // if accType and dstType the same
    if constexpr(std::is_same_v<accType, dstType>)
    {
        if constexpr((ck::is_same_v<srcAType, ck::bf8_t> || ck::is_same_v<srcAType, ck::f8_t>) &&
                     (ck::is_same_v<srcBType, ck::bf8_t> || ck::is_same_v<srcBType, ck::f8_t>))
        {
            if constexpr(ck::is_same_v<dstType, ck::half_t>)
            {
                if constexpr(ck::is_same_v<srcAType, ck::bf8_t> &&
                             ck::is_same_v<srcBType, ck::bf8_t>)
                {
                    if constexpr(size == 64)
                        CK_WMMA_CALL_INTRIN_2(f16, bf8, bf8, 64);
                    else if constexpr(size == 128)
                        CK_WMMA_CALL_INTRIN_2(f16, bf8, bf8, 128);
                    else
                        static_assert(false);
                }
                else if constexpr(ck::is_same_v<srcAType, ck::bf8_t> &&
                                  ck::is_same_v<srcBType, ck::f8_t>)
                {
                    if constexpr(size == 64)
                        CK_WMMA_CALL_INTRIN_2(f16, bf8, f8, 64);
                    else if constexpr(size == 128)
                        CK_WMMA_CALL_INTRIN_2(f16, bf8, f8, 128);
                    else
                        static_assert(false);
                }
                else if constexpr(ck::is_same_v<srcAType, ck::f8_t> &&
                                  ck::is_same_v<srcBType, ck::bf8_t>)
                {
                    if constexpr(size == 64)
                        CK_WMMA_CALL_INTRIN_2(f16, f8, bf8, 64);
                    else if constexpr(size == 128)
                        CK_WMMA_CALL_INTRIN_2(f16, f8, bf8, 128);
                    else
                        static_assert(false);
                }
                else if constexpr(ck::is_same_v<srcAType, ck::f8_t> &&
                                  ck::is_same_v<srcBType, ck::f8_t>)
                {
                    if constexpr(size == 64)
                        CK_WMMA_CALL_INTRIN_2(f16, f8, f8, 64);
                    else if constexpr(size == 128)
                        CK_WMMA_CALL_INTRIN_2(f16, f8, f8, 128);
                    else
                        static_assert(false);
                }
                else
                {
                    static_assert(false);
                }
            }
            else if constexpr(ck::is_same_v<dstType, float>)
            {
                if constexpr(ck::is_same_v<srcAType, ck::bf8_t> &&
                             ck::is_same_v<srcBType, ck::bf8_t>)
                {
                    if constexpr(size == 64)
                        CK_WMMA_CALL_INTRIN_2(f32, bf8, bf8, 64);
                    else if constexpr(size == 128)
                        CK_WMMA_CALL_INTRIN_2(f32, bf8, bf8, 128);
                    else
                        static_assert(false);
                }
                else if constexpr(ck::is_same_v<srcAType, ck::bf8_t> &&
                                  ck::is_same_v<srcBType, ck::f8_t>)
                {
                    if constexpr(size == 64)
                        CK_WMMA_CALL_INTRIN_2(f32, bf8, f8, 64);
                    else if constexpr(size == 128)
                        CK_WMMA_CALL_INTRIN_2(f32, bf8, f8, 128);
                    else
                        static_assert(false);
                }
                else if constexpr(ck::is_same_v<srcAType, ck::f8_t> &&
                                  ck::is_same_v<srcBType, ck::bf8_t>)
                {
                    if constexpr(size == 64)
                        CK_WMMA_CALL_INTRIN_2(f32, f8, bf8, 64);
                    else if constexpr(size == 128)
                        CK_WMMA_CALL_INTRIN_2(f32, f8, bf8, 128);
                    else
                        static_assert(false);
                }
                else if constexpr(ck::is_same_v<srcAType, ck::f8_t> &&
                                  ck::is_same_v<srcBType, ck::f8_t>)
                {
                    if constexpr(size == 64)
                        CK_WMMA_CALL_INTRIN_2(f32, f8, f8, 64);
                    else if constexpr(size == 128)
                        CK_WMMA_CALL_INTRIN_2(f32, f8, f8, 128);
                    else
                        static_assert(false);
                }
                else
                {
                    static_assert(false);
                }
            }
        }
        else
        { // not fp8 or bf8
            if constexpr(std::is_same_v<srcAType, ck::half_t> &&
                         std::is_same_v<srcBType, ck::half_t> &&
                         std::is_same_v<dstType, ck::half_t>)
            {
                CK_WMMA_CALL_INTRIN_1(f16, f16, 32);
            }
            else if constexpr(std::is_same_v<srcAType, ck::half_t> &&
                              std::is_same_v<srcBType, ck::half_t> &&
                              std::is_same_v<dstType, float>)
            {
                CK_WMMA_CALL_INTRIN_1(f32, f16, 32);
            }
            else if constexpr(std::is_same_v<srcAType, ck::bhalf_t> &&
                              std::is_same_v<srcBType, ck::bhalf_t> &&
                              std::is_same_v<dstType, float>)
            {
                CK_WMMA_CALL_INTRIN_1(f32, bf16, 32);
            }
            else if constexpr(std::is_same_v<srcAType, ck::bhalf_t> &&
                              std::is_same_v<srcBType, ck::bhalf_t> &&
                              std::is_same_v<dstType, ck::bhalf_t>)
            {
                CK_WMMA_CALL_INTRIN_1(bf16, bf16, 32);
            }
            else if constexpr(std::is_same_v<srcAType, int8_t> &&
                              std::is_same_v<srcBType, int8_t> && std::is_same_v<dstType, int32_t>)
            {
                CK_WMMA_CALL_INTRIN_4(i32, iu8, true, true, 64);
            }
            else if constexpr(std::is_same_v<srcAType, float> && std::is_same_v<srcBType, float> &&
                              std::is_same_v<dstType, float>)
            {
                CK_WMMA_CALL_INTRIN_1(f32, f32, 4);
            }
            else if constexpr(std::is_same_v<srcAType, double> &&
                              std::is_same_v<srcBType, double> && std::is_same_v<dstType, double>)
            {
                CK_WMMA_CALL_INTRIN_1(f64, f64, 4);
            }
            else
            {
                static_assert(false);
            }
        }
    }
    else if constexpr(!std::is_same_v<accType, dstType>)
    {
        if constexpr(std::is_same_v<accType, float> && std::is_same_v<dstType, ck::bhalf_t>)
        {
            CK_WMMA_CALL_INTRIN_3(bf16, f32, bf16, 32);
        }
        else
        {
            static_assert(false);
        }
    }
    else
    {
        static_assert(false);
    }
}

template <typename srcA_t, typename srcB_t, typename dst_t, typename acc_t, ck::index_t kValue>
__global__ void matmul_gfx1250(const srcA_t* a, const srcB_t* b, dst_t* c)
{
    static_assert(WMMAVecType<srcA_t, kValue>::template is_compatible<srcB_t>(),
                  "the data format for srcA and srcB is unsupported in gfx1250");
    using srcA_cast_T    = WMMAVecType<srcA_t, kValue>::ViewT;
    using srcB_cast_T    = WMMAVecType<srcB_t, kValue>::ViewT;
    using srcA_cast_type = typename srcA_cast_T::type;
    using srcB_cast_type = typename srcB_cast_T::type;

    using srcA_vec      = typename WMMAVecType<srcA_t, kValue>::VecT;
    using srcB_vec      = typename WMMAVecType<srcB_t, kValue>::VecT;
    using srcA_vec_type = srcA_vec::type;
    using srcB_vec_type = srcB_vec::type;

    const srcA_cast_type* a_ptr = reinterpret_cast<const srcA_cast_type*>(a);
    const srcB_cast_type* b_ptr = reinterpret_cast<const srcB_cast_type*>(b);

    srcA_vec a_frag = {};
    srcB_vec b_frag = {};

    srcA_vec a_temp = {};
    srcB_vec b_temp = {};

    using acc_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, 8, true>;
    acc_vec acc_thread_buf_;

    using dst_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, dst_t, 1, 8, true>;
    dst_vec dst_thread_buf_;

    // Num elements per 32B packed chunk
    constexpr int ToIntDim = WMMAVecType<srcA_t, kValue>::ToIntDim;

    // to int dim is 1 for float, 2 for half; base dim assumption is 16
    constexpr int SRC_DIM = WMMAVecType<srcA_t, kValue>::size / ToIntDim;

    // 2 threads per a row
    constexpr int ROW_SIZE = 2 * SRC_DIM;

    // 16 is base dim assumption, 2 is for a input and b input both
    constexpr int LDS_DIM = 2 * 16 * ROW_SIZE;

    constexpr int LDS_B_START = LDS_DIM / 2;

    constexpr int QUADRANT_SIZE = ROW_SIZE / 4;

    __shared__ srcA_cast_type p_shared[LDS_DIM];

    // strongly-type compile time index value of 0 for template containers
    static constexpr auto I0 = Number<0>{};

    // use a directly, as lds_shared allocated with A; A better be bigger than B if mixed
    const srcA_cast_type* local_a_ptr = p_shared;

    // get pointer as B type given LDS allocated with A
    const srcB_cast_type* local_b_ptr = reinterpret_cast<const srcB_cast_type*>(p_shared);

    const int lIdx    = threadIdx.x;
    const int lane    = lIdx % 32; // wave size
    const int lowHigh = lane / 16;

    bool use_QUADS = true;

    if(use_QUADS == true)
    {

        // load A to registers using QUADRANTS
        static_for<0, QUADRANT_SIZE, 1>{}([&](auto ele) {
            int i      = ele;
            int j      = ele + QUADRANT_SIZE * 2;
            int rowIdx = lane % 16;

            int offset1 = (rowIdx * ROW_SIZE) + (i + (lowHigh * QUADRANT_SIZE));
            int offset2 = (rowIdx * ROW_SIZE) + (j + (lowHigh * QUADRANT_SIZE));

            a_temp.template AsType<srcA_cast_type>()(ele) = a_ptr[offset1];
            a_temp.template AsType<srcA_cast_type>()(Number<ele + QUADRANT_SIZE>{}) =
                a_ptr[offset2];
        });

        // load B to registers using QUADRANTS
        static_for<0, QUADRANT_SIZE, 1>{}([&](auto ele) {
            int i      = ele;
            int j      = ele + QUADRANT_SIZE * 2;
            int rowIdx = lane % 16;

            int offset1 = (rowIdx * ROW_SIZE) + (i + (lowHigh * QUADRANT_SIZE));
            int offset2 = (rowIdx * ROW_SIZE) + (j + (lowHigh * QUADRANT_SIZE));

            b_temp.template AsType<srcB_cast_type>()(ele) = b_ptr[offset1];
            b_temp.template AsType<srcB_cast_type>()(Number<ele + QUADRANT_SIZE>{}) =
                b_ptr[offset2];
        });

        // Load A into LDS with quadrants
        constexpr int BLOCK_SIZE = 4 * QUADRANT_SIZE;
        static_for<0, QUADRANT_SIZE, 1>{}([&](auto ele) {
            int rowIdx = lIdx % 16;
            int hi     = lIdx / 16;

            // Each thread gets a block based on rowIdx and hi
            int base = rowIdx * BLOCK_SIZE + hi * QUADRANT_SIZE;

            // Write first quadrant
            p_shared[base + ele] = a_temp.template AsType<srcA_cast_type>()(ele);

            // Write second quadrant (offset by 2 quad sizes)
            p_shared[base + ele + QUADRANT_SIZE * 2] =
                a_temp.template AsType<srcA_cast_type>()(Number<ele + QUADRANT_SIZE>{});
        });

        // Load B into LDS with quadrants -- OK
        static_for<0, QUADRANT_SIZE, 1>{}([&](auto ele) {
            int rowIdx = lIdx % 16;
            int hi     = lIdx / 16;

            // Each thread gets a block based on rowIdx and hi
            int base = rowIdx * BLOCK_SIZE + hi * QUADRANT_SIZE + LDS_B_START;

            // Write first quadrant
            int idx1       = base + ele;
            p_shared[idx1] = b_temp.template AsType<srcB_cast_type>()(ele);

            // Write second quadrant (offset by 2 quad sizes)
            int idx2 = base + ele + QUADRANT_SIZE * 2;
            p_shared[idx2] =
                b_temp.template AsType<srcB_cast_type>()(Number<ele + QUADRANT_SIZE>{});
        });

        __syncthreads();

        // Construct a_frag and b_frag for WMMA call -- OK
        static_for<0, QUADRANT_SIZE, 1>{}([&](auto ele) {
            int rowIdx = lIdx % 16;
            int hi     = lIdx / 16;

            // base for a
            int base_a = rowIdx * BLOCK_SIZE + hi * QUADRANT_SIZE;
            int base_b = rowIdx * BLOCK_SIZE + hi * QUADRANT_SIZE + LDS_B_START;

            int idx1_a = base_a + ele;
            int idx2_a = base_a + ele + QUADRANT_SIZE * 2;
            int idx1_b = base_b + ele;
            int idx2_b = base_b + ele + QUADRANT_SIZE * 2;

            // index for first quadrant access
            a_frag.template AsType<srcA_cast_type>()(ele) = local_a_ptr[idx1_a];
            b_frag.template AsType<srcB_cast_type>()(ele) = local_b_ptr[idx1_b];

            // index for second quadrant access
            a_frag.template AsType<srcA_cast_type>()(Number<ele + QUADRANT_SIZE>{}) =
                local_a_ptr[idx2_a];
            b_frag.template AsType<srcB_cast_type>()(Number<ele + QUADRANT_SIZE>{}) =
                local_b_ptr[idx2_b];
        });
    }
    else // Don't use quads
    {
        // Load A to registers without quadrants
        static_for<0, SRC_DIM, 1>{}([&](auto ele) {
            int i      = ele;
            int rowIdx = lane % 16;

            int offset1 = (rowIdx * ROW_SIZE) + (i + (lowHigh * SRC_DIM));

            a_temp.template AsType<srcA_cast_type>()(ele) = a_ptr[offset1];
        });

        // Load B to registers without quadrants
        static_for<0, SRC_DIM, 1>{}([&](auto ele) {
            int i      = ele;
            int rowIdx = lane % 16;

            int offset1 = (rowIdx * ROW_SIZE) + (i + (lowHigh * SRC_DIM));

            b_temp.template AsType<srcB_cast_type>()(ele) = b_ptr[offset1];
        });

        // Load A into LDS without quadrants
        static_for<0, SRC_DIM, 1>{}([&](auto ele) {
            int rowIdx = lIdx % 16;
            int hi     = lIdx / 16;

            // Each thread gets a block based on rowIdx and hi
            int base = rowIdx * ROW_SIZE + hi * SRC_DIM;

            // Write first quadrant
            p_shared[base + ele] = a_temp.template AsType<srcA_cast_type>()(ele);
        });

        // Load B into LDS without quadrants
        static_for<0, SRC_DIM, 1>{}([&](auto ele) {
            int rowIdx = lIdx % 16;
            int hi     = lIdx / 16;

            // Each thread gets a block based on rowIdx and hi
            int base = rowIdx * ROW_SIZE + hi * SRC_DIM + LDS_B_START;

            // Write first quadrant
            int idx1       = base + ele;
            p_shared[idx1] = b_temp.template AsType<srcB_cast_type>()(ele);
        });

        static_for<0, SRC_DIM, 1>{}([&](auto ele) {
            int rowIdx = lIdx % 16;
            int hi     = lIdx / 16;

            // base for a
            int base_a = rowIdx * ROW_SIZE + hi * SRC_DIM;
            int base_b = rowIdx * ROW_SIZE + hi * SRC_DIM + LDS_B_START;

            int idx1_a = base_a + ele;
            int idx1_b = base_b + ele;

            // index for first quadrant access
            a_frag.template AsType<srcA_cast_type>()(ele) = local_a_ptr[idx1_a];
            b_frag.template AsType<srcB_cast_type>()(ele) = local_b_ptr[idx1_b];
        });
    }

    __syncthreads();

    // Call the WMMA intrinsic selector
    builtin_wmma_naive_selector<srcA_t, srcB_t, dst_t, acc_t, kValue>(
        a_frag.template AsType<srcA_vec_type>()(I0),
        b_frag.template AsType<srcB_vec_type>()(I0),
        acc_thread_buf_,
        dst_thread_buf_);

    static_for<0, 8, 1>{}([&](auto ele) {
        int lowHi         = lIdx / 16;
        int col           = lIdx % 16;
        int row           = (lowHi) * 8 + static_cast<int>(ele);
        c[col + 16 * row] = dst_thread_buf_[Number<ele>{}];
    });
}

template <typename srcA_t, typename srcB_t, typename dst_t, typename acc_t, ck::index_t kValue>
__global__ void matmul_swizzle_a_gfx1250(const srcA_t* a, const srcB_t* b, dst_t* c)
{
    static_assert(WMMAVecType<srcA_t, kValue>::template is_compatible<srcB_t>(),
                  "the data format for srcA and srcB is unsupported in gfx1250");
    using srcA_cast_T    = WMMAVecType<srcA_t, kValue>::ViewT;
    using srcB_cast_T    = WMMAVecType<srcB_t, kValue>::ViewT;
    using srcA_cast_type = typename srcA_cast_T::type;
    using srcB_cast_type = typename srcB_cast_T::type;

    using srcA_vec      = typename WMMAVecType<srcA_t, kValue>::VecT;
    using srcB_vec      = typename WMMAVecType<srcB_t, kValue>::VecT;
    using srcA_vec_type = srcA_vec::type;
    using srcB_vec_type = srcB_vec::type;

    const srcA_cast_type* a_ptr = reinterpret_cast<const srcA_cast_type*>(a);
    const srcB_cast_type* b_ptr = reinterpret_cast<const srcB_cast_type*>(b);

    srcA_vec a_frag = {};
    srcB_vec b_frag = {};

    using acc_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, 8, true>;
    acc_vec acc_thread_buf_;

    using dst_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, dst_t, 1, 8, true>;
    dst_vec dst_thread_buf_;

    // Num elements per 32B packed chunk
    constexpr int ToIntDim = WMMAVecType<srcA_t, kValue>::ToIntDim;

    constexpr int SRC_DIM = WMMAVecType<srcA_t, kValue>::size / ToIntDim;

    // 2 threads per a row
    constexpr int ROW_SIZE = 2 * SRC_DIM;

    constexpr int QUADRANT_SIZE = ROW_SIZE / 4;

    // strongly-type compile time index value of 0 for template containers
    static constexpr auto I0 = Number<0>{};

    const int lIdx = threadIdx.x;

    bool use_QUADS = true;

    if(use_QUADS == true)
    {
        constexpr int BLOCK_SIZE = 4 * QUADRANT_SIZE;
        static_for<0, QUADRANT_SIZE, 1>{}([&](auto ele) {
            int rowIdx = lIdx % 16;
            int hi     = lIdx / 16;

            // base for a
            int base_a = rowIdx * BLOCK_SIZE + hi * QUADRANT_SIZE;
            int base_b = rowIdx * BLOCK_SIZE + hi * QUADRANT_SIZE;

            int idx1_a = base_a + ele;
            int idx2_a = base_a + ele + QUADRANT_SIZE * 2;
            int idx1_b = base_b + ele;
            int idx2_b = base_b + ele + QUADRANT_SIZE * 2;

            // index for first quadrant access
            a_frag.template AsType<srcA_cast_type>()(ele) = a_ptr[idx1_a];
            b_frag.template AsType<srcB_cast_type>()(ele) = b_ptr[idx1_b];

            // index for second quadrant access
            a_frag.template AsType<srcA_cast_type>()(Number<ele + QUADRANT_SIZE>{}) = a_ptr[idx2_a];
            b_frag.template AsType<srcB_cast_type>()(Number<ele + QUADRANT_SIZE>{}) = b_ptr[idx2_b];
        });
    }
    else
    {
        static_for<0, SRC_DIM, 1>{}([&](auto ele) {
            int rowIdx = lIdx % 16;
            int hi     = lIdx / 16;

            // base for a
            int base = rowIdx * ROW_SIZE + hi * SRC_DIM;
            int idx1 = base + ele;

            // index for first quadrant access
            a_frag.template AsType<srcA_cast_type>()(ele) = a_ptr[idx1];
            b_frag.template AsType<srcB_cast_type>()(ele) = b_ptr[idx1];
        });
    }

    __syncthreads();

    builtin_wmma_naive_selector<srcA_t, srcB_t, dst_t, acc_t, kValue>(
        a_frag.template AsType<srcA_vec_type>()(I0),
        b_frag.template AsType<srcB_vec_type>()(I0),
        acc_thread_buf_,
        dst_thread_buf_);

    static_for<0, 8, 1>{}([&](auto ele) {
        int lowHi = lIdx / 16;
        int col   = lIdx % 16;
        int row   = (lowHi) * 8 + static_cast<int>(ele);
        c[col + 16 * row] =
            dst_thread_buf_[Number<ele>{}]; // idea each thread contiguous along column
    });
}

#else
template <typename src_vec, typename acc_vec>
__device__ void builtin_wmma_naive_selector(const src_vec&, const src_vec&, acc_vec&)
{
}

template <>
__device__ void
builtin_wmma_naive_selector<half16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>>(
    const half16_t& reg_a,
    const half16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>& reg_c)
{
    intrin_wmma_f32_16x16x16_f16_w32<16, 16>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<bhalf16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>>(
    const bhalf16_t& reg_a,
    const bhalf16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>& reg_c)
{
    intrin_wmma_f32_16x16x16_bf16_w32<16, 16>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<half16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, half_t, 1, 16, true>>(
    const half16_t& reg_a,
    const half16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, half_t, 1, 16, true>& reg_c)
{
    intrin_wmma_f16_16x16x16_f16_w32<16, 16, 0>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void builtin_wmma_naive_selector<
    bhalf16_t,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, bhalf_t, 1, 16, true>>(
    const bhalf16_t& reg_a,
    const bhalf16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, bhalf_t, 1, 16, true>& reg_c)
{
    intrin_wmma_bf16_16x16x16_bf16_w32<16, 16, 0>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<int8x16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>>(
    const int8x16_t& reg_a,
    const int8x16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>& reg_c)
{
    intrin_wmma_i32_16x16x16_iu8_w32<16, 16, true, true, false>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<half8_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>>(
    const half8_t& reg_a,
    const half8_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>& reg_c)
{
    intrin_wmma_f32_16x16x16_f16_w32<16, 16>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<bhalf8_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>>(
    const bhalf8_t& reg_a,
    const bhalf8_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>& reg_c)
{
    intrin_wmma_f32_16x16x16_bf16_w32<16, 16>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<half8_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, half_t, 1, 8, true>>(
    const half8_t& reg_a,
    const half8_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, half_t, 1, 8, true>& reg_c)
{
    intrin_wmma_f16_16x16x16_f16_w32<16, 16, 0 /*OpSel*/>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<bhalf8_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, bhalf_t, 1, 8, true>>(
    const bhalf8_t& reg_a,
    const bhalf8_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, bhalf_t, 1, 8, true>& reg_c)
{
    intrin_wmma_bf16_16x16x16_bf16_w32<16, 16, 0 /*OpSel*/>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<int8x8_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>>(
    const int8x8_t& reg_a,
    const int8x8_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>& reg_c)
{
    intrin_wmma_i32_16x16x16_iu8_w32<16, 16, true, true, false>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

// WMMAVecType is used in gfx13
template <typename T, index_t kMultiplier, typename = void>
struct WMMAVecType
{
    static_assert(sizeof(T) == 0, "VecType is not specialized for this type");
};

template <typename T, index_t kMultiplier>
struct WMMAVecType<T,
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

    using VecT  = vector_type<T, kMultiplier * 8>;
    using ViewT = vector_type<T, 2>;
};

template <typename T, index_t kMultiplier>
struct WMMAVecType<
    T,
    kMultiplier,
    ck::enable_if_t<ck::is_same_v<T, ck::f8_ocp_t> || ck::is_same_v<T, ck::bf8_ocp_t>>>
{
    static constexpr bool layoutTransform = true;
    static constexpr int ToIntDim         = 4;

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<D, ck::f8_t> || ck::is_same_v<D, ck::bf8_t>;
    }

    using VecT  = vector_type<typename T::data_type, kMultiplier * 8>;
    using ViewT = vector_type<typename T::data_type, 4>;
};

template <typename T, index_t kMultiplier>
struct WMMAVecType<
    T,
    kMultiplier,
    ck::enable_if_t<ck::is_same_v<T, ck::f8_fnuz_t> || ck::is_same_v<T, ck::bf8_fnuz_t>>>
{
    static constexpr bool layoutTransform = true;
    static constexpr int ToIntDim         = 4;

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<D, ck::f8_t> || ck::is_same_v<D, ck::bf8_t>;
    }

    using VecT  = vector_type<T, kMultiplier * 8>;
    using ViewT = vector_type<T, 4>;
};

template <typename T, index_t kMultiplier>
struct WMMAVecType<T, kMultiplier, ck::enable_if_t<ck::is_same_v<T, int8_t>>>
{
    static constexpr bool layoutTransform = true;
    static constexpr int ToIntDim         = 4;
    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<T, D>;
    }

    using VecT  = vector_type<T, kMultiplier * 8>;
    using ViewT = vector_type<T, 4>;
};

template <typename T, index_t kMultiplier>
struct WMMAVecType<T, kMultiplier, ck::enable_if_t<ck::is_same_v<T, ck::int4_t>>>
{
    static constexpr bool layoutTransform = true;
    static constexpr int ToIntDim         = 8;
    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<T, D>;
    }

    using VecT  = vector_type<int32_t, kMultiplier>;
    using ViewT = vector_type<int32_t, 1>;
};

// gfx13 uses the below builtin_wmma_naive_selector
// clang-format off
#define call_intrinsic_wmma_func0(src0, src1, dst)                                 \
    intrin_wmma_##dst##_16x16_##src0##src1##_w32<16, 16, false, kMultiplier>::Run( \
        reg_a, reg_b, reg_d.GetVectorTypeReference(Number<0>{}))

#define call_intrinsic_wmma_func1(src0, src1, acc, dst, neg_a, neg_b)                                 \
    intrin_wmma_##dst##acc##_16x16_##src0##src1##_w32<16, 16, neg_a, neg_b, false, kMultiplier>::Run( \
            reg_a,                                                                                    \
            reg_b,                                                                                    \
            reg_c.GetVectorTypeReference(Number<0>{}),                                                \
            reg_d.GetVectorTypeReference(Number<0>{}))

#define call_intrinsic_wmma_func_with_option(src0, src1, dst, neg_a, neg_b)                      \
    intrin_wmma_##dst##_16x16_##src0##src1##_w32<16, 16, neg_a, neg_b, false, kMultiplier>::Run( \
        reg_a, reg_b, reg_d.GetVectorTypeReference(Number<0>{}))

#define call_intrinsic_wmma_branch_with_option(src0_type,                                     \
                                               src0_fmt,                                      \
                                               src1_type,                                     \
                                               src1_fmt,                                      \
                                               dst0_type,                                     \
                                               dst0_fmt,                                      \
                                               dst1_type,                                     \
                                               dst1_fmt,                                      \
                                               neg_a,                                         \
                                               neg_b)                                         \
    if constexpr(ck::is_same_v<srcAType, src0_type> && ck::is_same_v<srcBType, src1_type>)    \
    {                                                                                         \
        if constexpr(ck::is_same_v<accType, dst0_type>)                                       \
        {                                                                                     \
            call_intrinsic_wmma_func_with_option(src0_fmt, src1_fmt, dst0_fmt, neg_a, neg_b); \
        }                                                                                     \
        else if constexpr(ck::is_same_v<accType, dst1_type>)                                  \
        {                                                                                     \
            call_intrinsic_wmma_func_with_option(src0_fmt, src1_fmt, dst1_fmt, neg_a, neg_b); \
        }                                                                                     \
    }

#define call_intrinsic_wmma_branch(                                                        \
    src0_type, src0_fmt, src1_type, src1_fmt, dst0_type, dst0_fmt, dst1_type, dst1_fmt)    \
    if constexpr(ck::is_same_v<srcAType, src0_type> && ck::is_same_v<srcBType, src1_type>) \
    {                                                                                      \
        if constexpr(ck::is_same_v<accType, dst0_type>)                                    \
        {                                                                                  \
            call_intrinsic_wmma_func0(src0_fmt, src1_fmt, dst0_fmt);                       \
        }                                                                                  \
        else if constexpr(ck::is_same_v<accType, dst1_type>)                               \
        {                                                                                  \
            call_intrinsic_wmma_func0(src0_fmt, src1_fmt, dst1_fmt);                       \
        }                                                                                  \
    }
// clang-format on
// this selector is only used in V_WMMA_F32I32_16X16_IU8 and V_WMMA_F32I32_16X16_IU4
template <typename srcAType,
          typename srcBType,
          typename accType,
          typename dstType,
          index_t kMultiplier>
__device__ void builtin_wmma_naive_selector(
    const typename WMMAVecType<srcAType, kMultiplier>::VecT::type& reg_a,
    const typename WMMAVecType<srcBType, kMultiplier>::VecT::type& reg_b,
    const StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, accType, 1, 8, true>& reg_c,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, dstType, 1, 8, true>& reg_d)
{
    if constexpr(ck::is_same_v<accType, dstType>)
    {
        call_intrinsic_wmma_branch(half_t, f16, half_t, f16, float, f32, ck::half_t, f16);
        call_intrinsic_wmma_branch(bhalf_t, bf16, bhalf_t, bf16, float, f32, ck::bhalf_t, bf16);
        call_intrinsic_wmma_branch_with_option(
            int8_t, iu8, int8_t, iu8, int32_t, i32, float, f32, true, true);
        call_intrinsic_wmma_branch(ck::f8_t, f8, ck::f8_t, f8, float, f32, ck::half_t, f16);
        call_intrinsic_wmma_branch(ck::f8_t, f8, ck::bf8_t, bf8, float, f32, ck::half_t, f16);
        call_intrinsic_wmma_branch(ck::bf8_t, bf8, ck::f8_t, f8, float, f32, ck::half_t, f16);
        call_intrinsic_wmma_branch(ck::bf8_t, bf8, ck::bf8_t, bf8, float, f32, ck::half_t, f16);
        call_intrinsic_wmma_branch_with_option(
            int4_t, iu4, int4_t, iu4, int32_t, i32, float, f32, true, true);
    }
    else
    {
        if constexpr(ck::is_same_v<srcAType, int8_t> && ck::is_same_v<srcBType, int8_t>)
        {
            call_intrinsic_wmma_func1(iu8, iu8, i32, f32, true, true);
        }
        else if constexpr(ck::is_same_v<srcAType, ck::int4_t> &&
                          ck::is_same_v<srcBType, ck::int4_t>)
        {
            call_intrinsic_wmma_func1(iu4, iu4, i32, f32, true, true);
        }
    }
}

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
__device__ void
builtin_wmma_naive_selector<int4x16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>>(
    const int4x16_t& reg_a,
    const int4x16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>& reg_c)
{
    intrin_wmma_i32_16x16x16_iu4_w32<16, 16, true, true, false>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}
#endif

#endif

template <typename src_t, typename dst_t, typename acc_t, index_t acc_num>
__global__ void matmul(const src_t* a, const src_t* b, dst_t* c)
{
#if defined(__gfx11__)
    __shared__ src_t p_shared[16 * 16 * 2];
    const int lIdx = threadIdx.x;
    // a and b fragments are stored in 8 VGPRs each, in packed format, so 16 elements each for a and
    // b a_frag will store one column of the 16x16 matrix tile b_frag will store one row of the
    // 16x16 matrix tile
    using src_vec  = typename vector_type<src_t, 16>::type;
    src_vec a_frag = {};
    src_vec b_frag = {};

    src_vec a_temp = {};
    src_vec b_temp = {};
    // initialize c fragment to 0
    using acc_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, acc_num, true>;
    acc_vec c_thread_buf_;

    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in gfx11
    // see https://atlvsp3.amd.com/sp3_gfx11_5_instructions.pdf page 482
    // TODO: remove this dependency in gfx12 https://ontrack-internal.amd.com/browse/DEGFXSP3-101
    const int lane    = lIdx % 16;
    const int lane_lo = lIdx / 2;
    const int lane_hi = lIdx % 2;
    for(int ele = 0; ele < 8; ++ele)
    {
        a_temp[ele] = a[8 * lane_hi + 16 * lane_lo + ele];
    }

    for(int ele = 0; ele < 8; ++ele)
    {
        b_temp[ele] = b[8 * lane_hi + 16 * lane_lo + ele];
    }

    __syncthreads();

    for(int ele = 0; ele < 8; ++ele)
    {
        p_shared[8 * 16 * lane_hi + 8 * lane_lo + ele] = a_temp[ele];
    }

    for(int ele = 0; ele < 8; ++ele)
    {
        p_shared[8 * 16 * lane_hi + 8 * lane_lo + ele + 16 * 16] = b_temp[ele];
    }

    __syncthreads();

    for(int ele = 0; ele < 16; ++ele)
    {
        b_frag[ele] = p_shared[(ele / 8) * 16 * 8 + 8 * lane + ele % 8 + 16 * 16];
    }
    // follow origin design
    for(int ele = 0; ele < 16; ++ele)
    {
        a_frag[ele] = p_shared[(ele / 8) * 16 * 8 + 8 * lane + ele % 8];
    }

    __syncthreads();

    // sync threads, similar to mma_sync
    // __syncthreads();
    builtin_wmma_naive_selector<src_vec, acc_vec>(a_frag, b_frag, c_thread_buf_);
    // since only fp16_fp32 asm wmma implemented for experiment purpose, restrict test case to fp16
    // when enable this ck::amd_assembly_wmma_f32_16x16x16_f16_w32(a_frag, b_frag,
    // c_thread_buf_.GetVectorTypeReference(Number<0>{}).template AsType<float8_t>()(Number<0>{}));
    __syncthreads();
    // wait for results, similar to mma_sync
    static_for<0, 8, 1>{}([&](auto ele) {
        const int r = ele * 2 + (lIdx / 16);
        // store results from unpacked c_thread_buf_ output
        c[16 * r + lane] = ck::type_convert<dst_t>(c_thread_buf_[Number<ele * acc_num / 8>{}]);
    });
#elif defined(__gfx120__)
    __shared__ src_t p_shared[16 * 16 * 2];
    const int lIdx = threadIdx.x;

    using src_vec  = typename vector_type<src_t, 8>::type;
    src_vec a_frag = {};
    src_vec b_frag = {};

    src_vec a_temp = {};
    src_vec b_temp = {};
    // initialize c fragment to 0
    using acc_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, acc_num, true>;
    acc_vec c_thread_buf_ = {};

    const int lane = lIdx % 16;
    const int blk  = lIdx / 16;
    // Row major
    for(int ele = 0; ele < 8; ++ele)
    {
        b_temp[ele] = b[16 * lane + 8 * blk + ele];
    }

    // Colum major
    // const int offset_m = (((lane & 1) << 3) | (lane >> 1));
    for(int ele = 0; ele < 8; ++ele)
    {
        a_temp[ele] = a[16 * lane + 8 * blk + ele];
    }

    __syncthreads();

    for(int ele = 0; ele < 8; ++ele)
    {
        p_shared[8 * lIdx + ele] = a_temp[ele];
    }

    for(int ele = 0; ele < 8; ++ele)
    {
        p_shared[8 * lIdx + ele + 16 * 16] = b_temp[ele];
    }

    asm volatile("\
    s_wait_dscnt 0x0 \n \
    s_barrier_signal -1 \n \
    s_barrier_wait -1 \
    " ::);

    for(int ele = 0; ele < 8; ++ele)
    {
        b_frag[ele] = p_shared[8 * lIdx + ele + 16 * 16];
    }
    // follow origin design
    for(int ele = 0; ele < 8; ++ele)
    {
        a_frag[ele] = p_shared[8 * lIdx + ele];
    }

    asm volatile("\
    s_wait_dscnt 0x0 \n \
    s_barrier_signal -1 \n \
    s_barrier_wait -1 \
    " ::);

    // sync threads, similar to mma_sync
    //__syncthreads();
    builtin_wmma_naive_selector<src_vec, acc_vec>(a_frag, b_frag, c_thread_buf_);
    // since only fp16_fp32 asm wmma implemented for experiment purpose, restrict test case to fp16
    // when enable this ck::amd_assembly_wmma_f32_16x16x16_f16_w32(a_frag, b_frag,
    // c_thread_buf_.GetVectorTypeReference(Number<0>{}).template AsType<float8_t>()(Number<0>{}));
    __syncthreads();

    // wait for results, similar to mma_sync
    static_for<0, 8, 1>{}([&](auto ele) {
        const int r = ele + (lIdx / 16) * 8;
        // store results from unpacked c_thread_buf_ output
        c[16 * r + lane] = ck::type_convert<dst_t>(c_thread_buf_[Number<ele>{}]);
    });

#else
    ignore = a;
    ignore = b;
    ignore = c;
#endif
}

template <typename src_t, typename dst_t, typename acc_t, index_t acc_num>
__global__ void matmul_swizzle_a(const src_t* a, const src_t* b, dst_t* c)
{
#if defined(__gfx11__)
    const int lIdx = threadIdx.x;

    using src_vec  = typename vector_type<src_t, 16>::type;
    src_vec a_frag = {};
    src_vec b_frag = {};
    using acc_vec  = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, acc_num, true>;
    acc_vec c_thread_buf_;

    const int lane = lIdx % 16;

    for(int ele = 0; ele < 16; ++ele)
    {
        b_frag[ele] = b[16 * lane + ele];
    }

    const int offset_m = (((lane & 1) << 3) | (lane >> 1));
    for(int ele = 0; ele < 16; ++ele)
    {
        a_frag[ele] = a[16 * offset_m + ele];
    }

    __syncthreads();
    builtin_wmma_naive_selector<src_vec, acc_vec>(a_frag, b_frag, c_thread_buf_);
    __syncthreads();

    static_for<0, 8, 1>{}([&](auto ele) {
        const int blk = lIdx / 16;
        const int r   = ele;
        c[16 * 8 * blk + 16 * r + lane] =
            ck::type_convert<dst_t>(c_thread_buf_[Number<ele * acc_num / 8>{}]);
    });
#elif defined(__gfx120__)
    const int lIdx = threadIdx.x;

    using src_vec  = typename vector_type<src_t, 8>::type;
    src_vec a_frag = {};
    src_vec b_frag = {};
    using acc_vec  = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, acc_num, true>;
    acc_vec c_thread_buf_;

    const int lane = lIdx % 16;
    const int blk  = lIdx / 16;
    // Row major
    for(int ele = 0; ele < 8; ++ele)
    {
        b_frag[ele] = b[16 * lane + 8 * blk + ele];
    }

    // Colum major
    // const int offset_m = (((lane & 1) << 3) | (lane >> 1));
    for(int ele = 0; ele < 8; ++ele)
    {
        a_frag[ele] = a[16 * lane + 8 * blk + ele];
    }

    __syncthreads();
    builtin_wmma_naive_selector<src_vec, acc_vec>(a_frag, b_frag, c_thread_buf_);
    __syncthreads();

    // Colum major -> Row major
    static_for<0, 8, 1>{}([&](auto ele) {
        const int r = ele + (lIdx / 16) * 8;
        // store results from unpacked c_thread_buf_ output
        c[16 * r + lane] = ck::type_convert<dst_t>(c_thread_buf_[Number<ele>{}]);
    });
#else
    ignore = a;
    ignore = b;
    ignore = c;
#endif
}

template <typename srcA_t, typename srcB_t, typename dst_t, typename acc_t, index_t kMultiplier>
__global__ void matmul_with_kMultiplier(const srcA_t* a, const srcB_t* b, dst_t* c)
{
#ifdef __gfx13__
    static_assert(WMMAVecType<srcA_t, kMultiplier>::template is_compatible<srcB_t>(),
                  "the data format for srcA and srcB is unsupported in gfx13");
    using srcA_cast_T    = WMMAVecType<srcA_t, kMultiplier>::ViewT; // view to int32_t
    using srcB_cast_T    = WMMAVecType<srcB_t, kMultiplier>::ViewT;
    using srcA_cast_type = srcA_cast_T::type;
    using srcB_cast_type = srcB_cast_T::type;

    // the below is used to send to intrisic
    using srcA_vec      = WMMAVecType<srcA_t, kMultiplier>::VecT;
    using srcB_vec      = WMMAVecType<srcB_t, kMultiplier>::VecT;
    using srcA_vec_type = srcA_vec::type;
    using srcB_vec_type = srcB_vec::type;

    // cast to int32_t
    const srcA_cast_type* a_ptr = reinterpret_cast<const srcA_cast_type*>(a);
    // cast to same type as A because of the data type of lds
    const srcA_cast_type* b_ptr = reinterpret_cast<const srcA_cast_type*>(b);

    // ToIntDim is used to check how many elements to merge into one int32_t type
    constexpr int ToIntDim    = WMMAVecType<srcA_t, kMultiplier>::ToIntDim;
    constexpr int LDS_DIM     = 16 * 16 * kMultiplier * 2 / ToIntDim;
    constexpr int LDS_B_START = LDS_DIM / 2;
    __shared__ srcA_cast_type p_shared[LDS_DIM];

    const int lIdx = threadIdx.x;
    constexpr int SRC_DIM =
        8 / ToIntDim * kMultiplier; // view as int32_t, how many int32_t need to load per thread
    constexpr int ROW_SIZE = SRC_DIM * 2;

    srcA_vec a_frag = {};
    srcB_vec b_frag = {};

    srcA_vec a_temp = {};
    srcA_vec b_temp = {};
    // initialize acc and dst fragment to 0
    using acc_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, 8, true>;
    acc_vec acc_thread_buf_;

    using dst_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, dst_t, 1, 8, true>;
    dst_vec dst_thread_buf_;

    const int lane = lIdx % 2;
    const int blk  = lIdx / 2;

    static_for<0, SRC_DIM, 1>{}([&](auto ele) {
        a_temp.template AsType<srcA_cast_type>()(ele) =
            a_ptr[ROW_SIZE * blk + SRC_DIM * lane + ele];
    });

    static_for<0, SRC_DIM, 1>{}([&](auto ele) {
        b_temp.template AsType<srcA_cast_type>()(ele) =
            b_ptr[ROW_SIZE * blk + SRC_DIM * lane + ele];
    });

    static_for<0, SRC_DIM, 1>{}([&](auto ele) {
        p_shared[SRC_DIM * lIdx + ele] = a_temp.template AsType<srcA_cast_type>()(ele);
    });

    static_for<0, SRC_DIM, 1>{}([&](auto ele) {
        p_shared[SRC_DIM * lIdx + ele + LDS_B_START] =
            b_temp.template AsType<srcA_cast_type>()(ele);
    });

    asm volatile("\
    s_wait_dscnt 0x0 \n \
    s_barrier_signal -1 \n \
    s_barrier_wait -1 \
    " ::);

    static constexpr auto I0          = Number<0>{};
    const srcA_cast_type* local_a_ptr = p_shared;
    const srcB_cast_type* local_b_ptr =
        reinterpret_cast<const srcB_cast_type*>(p_shared + LDS_B_START);
    int start_idx = ((lIdx >> 1) * ROW_SIZE) + (lIdx & 1); // this is int32_t's offset
    static_for<0, SRC_DIM, 1>{}([&](auto ele) {
        int index                                     = start_idx + (ele << 1);
        a_frag.template AsType<srcA_cast_type>()(ele) = local_a_ptr[index];
        b_frag.template AsType<srcB_cast_type>()(ele) = local_b_ptr[index];
    });

    asm volatile("\
    s_wait_dscnt 0x0 \n \
    s_barrier_signal -1 \n \
    s_barrier_wait -1 \
    " ::);

    builtin_wmma_naive_selector<srcA_t, srcB_t, acc_t, dst_t, kMultiplier>(
        a_frag.template AsType<srcA_vec_type>()(I0),
        b_frag.template AsType<srcB_vec_type>()(I0),
        acc_thread_buf_,
        dst_thread_buf_);
    if constexpr(WMMAVecType<srcA_t, kMultiplier>::layoutTransform)
    {
        static_for<0, 8, 1>{}([&](auto ele) {
            const int col = lIdx >> 1;
            const int row = ((ele & 4) << 1) + (ele & 3) + ((lIdx & 1) << 2);
            // store results from unpacked c_thread_buf_ output
            c[16 * row + col] = dst_thread_buf_[Number<ele>{}];
        });
    }
    else
    {
        static_for<0, 8, 1>{}([&](auto ele) {
            const int col = lIdx >> 1;
            const int row = ((ele & 6) << 1) + (ele & 1) + ((lIdx & 1) << 1);
            // store results from unpacked c_thread_buf_ output
            c[16 * row + col] = dst_thread_buf_[Number<ele>{}];
        });
    }
#else
    ignore = a;
    ignore = b;
    ignore = c;
#endif
}

template <typename srcA_t, typename srcB_t, typename dst_t, typename acc_t, index_t kMultiplier>
__global__ void matmul_swizzle_a_with_kMultiplier(const srcA_t* a, const srcB_t* b, dst_t* c)
{
#ifdef __gfx13__
    static_assert(WMMAVecType<srcA_t, kMultiplier>::template is_compatible<srcB_t>(),
                  "the data format for srcA and srcB is unsupported in gfx13");
    const int lIdx = threadIdx.x;
    // for original data type each thread load 8 * kMultiplier elements
    constexpr int SRC_DIM = 8 * kMultiplier;
    // each row two threads will load data
    constexpr int ROW_SIZE       = 16 * kMultiplier;
    constexpr int VIEW_DIM       = WMMAVecType<srcA_t, kMultiplier>::ToIntDim;
    constexpr int ITERATION      = SRC_DIM / VIEW_DIM;
    constexpr int SRC_DIM_STRIDE = ROW_SIZE / VIEW_DIM;
    static constexpr auto I0     = Number<0>{};

    using srcA_vec       = WMMAVecType<srcA_t, kMultiplier>::VecT;
    using srcB_vec       = WMMAVecType<srcB_t, kMultiplier>::VecT;
    using srcA_cast_T    = WMMAVecType<srcA_t, kMultiplier>::ViewT; // view to int32_t
    using srcB_cast_T    = WMMAVecType<srcB_t, kMultiplier>::ViewT;
    using srcA_cast_type = srcA_cast_T::type;
    using srcB_cast_type = srcB_cast_T::type;

    srcA_vec a_frag = {};
    srcB_vec b_frag = {};

    //  initialize c fragment to 0
    using acc_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, 8, true>;
    acc_vec acc_thread_buf_;

    using dst_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, dst_t, 1, 8, true>;
    dst_vec dst_thread_buf_;

    const srcA_cast_type* vgpr_a_ptr = reinterpret_cast<const srcA_cast_type*>(a);
    const srcB_cast_type* vgpr_b_ptr = reinterpret_cast<const srcB_cast_type*>(b);

    int start_idx = ((lIdx >> 1) * SRC_DIM_STRIDE) + (lIdx & 1); // this is int32_t's offset

    static_for<0, ITERATION, 1>{}([&](auto ele) {
        int index                                     = start_idx + (ele << 1);
        a_frag.template AsType<srcA_cast_type>()(ele) = vgpr_a_ptr[index];
        b_frag.template AsType<srcB_cast_type>()(ele) = vgpr_b_ptr[index];
    });

    builtin_wmma_naive_selector<srcA_t, srcB_t, acc_t, dst_t, kMultiplier>(
        a_frag.template AsType<typename srcA_vec::type>()(I0),
        b_frag.template AsType<typename srcB_vec::type>()(I0),
        acc_thread_buf_,
        dst_thread_buf_);

    if constexpr(WMMAVecType<srcA_t, kMultiplier>::layoutTransform)
    {
        static_for<0, 8, 1>{}([&](auto ele) {
            const int col = lIdx >> 1;
            const int row = ((ele & 4) << 1) + (ele & 3) + ((lIdx & 1) << 2);
            // store results from unpacked c_thread_buf_ output
            c[16 * row + col] = dst_thread_buf_[Number<ele>{}];
        });
    }
    else
    {
        static_for<0, 8, 1>{}([&](auto ele) {
            const int col = lIdx >> 1;
            const int row = ((ele & 6) << 1) + (ele & 1) + ((lIdx & 1) << 1);
            // store results from unpacked c_thread_buf_ output
            c[16 * row + col] = dst_thread_buf_[Number<ele>{}];
        });
    }
#else
    ignore = a;
    ignore = b;
    ignore = c;
#endif
}

template <typename src0_t,
          typename src1_t,
          ck::index_t AScaleSel,
          ck::index_t BScaleSel,
          typename dst_t,
          typename acc_t>
__global__ void matmul_mixedfp(const typename src0_t::type_t* a,
                               const typename src1_t::type_t* b,
                               const int32_t* a_block_scale,
                               const int32_t* b_block_scale,
                               dst_t* c)
{
    const int lIdx = threadIdx.x;

    using src0_vec  = typename src0_t::vec_t;
    using src1_vec  = typename src1_t::vec_t;
    src0_vec a_frag = {};
    src1_vec b_frag = {};
    using acc_vec   = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, 8, true>;
    acc_vec c_thread_buf_;
    // cast to int32
    const int32_t* vgpr_a_ptr = reinterpret_cast<const int32_t*>(a);
    const int32_t* vgpr_b_ptr = reinterpret_cast<const int32_t*>(b);
    int a_start_idx =
        ((lIdx >> 1) * src0_t::dwords_per_wmmak) + (lIdx & 1); // this is int32_t's offset
    for(int ele = 0; ele < src0_t::vec_size; ele++)
    {
        int index   = a_start_idx + (ele << 1);
        a_frag[ele] = vgpr_a_ptr[index];
    }
    int b_start_idx =
        ((lIdx >> 1) * src1_t::dwords_per_wmmak) + (lIdx & 1); // this is int32_t's offset
    for(int ele = 0; ele < src1_t::vec_size; ele++)
    {
        int index   = b_start_idx + (ele << 1);
        b_frag[ele] = vgpr_b_ptr[index];
    }
    const int32_t a_scale = a_block_scale[lIdx];
    const int32_t b_scale = b_block_scale[lIdx];

    intrin_wmma_f32_16x16_f8f6f4_w32<16, 16, src0_t, src1_t, AScaleSel, BScaleSel, false>::Run(
        a_frag, b_frag, a_scale, b_scale, c_thread_buf_.GetVectorTypeReference(Number<0>{}));
    //// Colum major -> Row major
    static_for<0, 8, 1>{}([&](auto ele) {
        const int col = lIdx >> 1;
        const int row = ((ele & 4) << 1) + (ele & 3) + ((lIdx & 1) << 2);
        // store results from unpacked c_thread_buf_ output
        c[16 * row + col] = ck::type_convert<dst_t>(c_thread_buf_[Number<ele>{}]);
    });
}

struct GemmParams
{
    GemmParams() : M(16), N(16), K(16), StrideA(16), StrideB(16), StrideC(16), alpha(1), beta(0) {}
    GemmParams(ck::index_t m_,
               ck::index_t n_,
               ck::index_t k_,
               ck::index_t strideA_,
               ck::index_t strideB_,
               ck::index_t strideC_,
               float alpha_,
               float beta_)
        : M(m_),
          N(n_),
          K(k_),
          StrideA(strideA_),
          StrideB(strideB_),
          StrideC(strideC_),
          alpha(alpha_),
          beta(beta_)
    {
    }

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

template <typename GemmInstance,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void RunHostMixedTypeGEMM(const Tensor<ADataType>& A,
                          const Tensor<BDataType>& B,
                          const Tensor<int32_t>& a_block_scale,
                          const Tensor<int32_t>& b_block_scale,
                          Tensor<CDataType>& C,
                          AElementwiseOperation a_element_op,
                          BElementwiseOperation b_element_op,
                          CElementwiseOperation c_element_op)
{
#if defined(__gfx13__)
    auto ref_gemm     = GemmInstance{};
    auto ref_invoker  = ref_gemm.MakeInvoker();
    auto ref_argument = ref_gemm.MakeArgument(
        A, B, a_block_scale, b_block_scale, C, a_element_op, b_element_op, c_element_op);

    ref_invoker.Run(ref_argument);
#else
    ignore = A;
    ignore = B;
    ignore = a_block_scale;
    ignore = b_block_scale;
    ignore = C;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
#endif
}

template <typename KernelType, typename ADataType, typename BDataType, typename CDataType>
bool RunDeviceGEMM(KernelType kernel,
                   const Tensor<ADataType>& A,
                   const Tensor<BDataType>& B,
                   Tensor<CDataType>& C)
{
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * A.mDesc.GetElementSpaceSize());
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
        a_m_k_device_buf.ToDevice(A.mData.data());
        b_n_k_device_buf.ToDevice(B.mData.data());
    }

    kernel<<<1, 32>>>(static_cast<const ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                      static_cast<const BDataType*>(b_n_k_device_buf.GetDeviceBuffer()),
                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()));
    hipError_t err = hipGetLastError();
    if(err != hipSuccess)
    {
        std::cerr << "HIP kernel launch error: " << hipGetErrorString(err) << std::endl;
    }
    c_m_n_device_buf.FromDevice(C.mData.data());

    return true;
}

template <typename KernelType,
          typename ADataType,
          typename BDataType,
          typename DeviceAType,
          typename DeviceBType,
          typename CDataType>
bool RunMixedTypeDeviceGEMM(KernelType kernel,
                            const Tensor<ADataType>& A,
                            const Tensor<BDataType>& B,
                            const Tensor<int32_t>& a_block_scale,
                            const Tensor<int32_t>& b_block_scale,
                            Tensor<CDataType>& C,
                            DeviceAType,
                            DeviceBType)
{
#if defined(__gfx13__)
    DeviceMem a_m_k_device_buf(sizeof(typename DeviceAType::type_t) *
                               A.mDesc.GetElementSpaceSize());
    DeviceMem b_n_k_device_buf(sizeof(typename DeviceBType::type_t) *
                               B.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * C.mDesc.GetElementSpaceSize());
    DeviceMem a_block_scale_device_buf(sizeof(int32_t) * a_block_scale.mDesc.GetElementSpaceSize());
    DeviceMem b_block_scale_device_buf(sizeof(int32_t) * b_block_scale.mDesc.GetElementSpaceSize());

    auto converted_a_vec = ck::convert_utils<DeviceAType>(A.mData);
    auto converted_b_vec = ck::convert_utils<DeviceBType>(B.mData);
    a_m_k_device_buf.ToDevice(converted_a_vec.data());
    b_n_k_device_buf.ToDevice(converted_b_vec.data());
    a_block_scale_device_buf.ToDevice(a_block_scale.data());
    b_block_scale_device_buf.ToDevice(b_block_scale.data());
    kernel<<<1, 32>>>(
        static_cast<const typename DeviceAType::type_t*>(a_m_k_device_buf.GetDeviceBuffer()),
        static_cast<const typename DeviceBType::type_t*>(b_n_k_device_buf.GetDeviceBuffer()),
        static_cast<int32_t*>(a_block_scale_device_buf.GetDeviceBuffer()),
        static_cast<int32_t*>(b_block_scale_device_buf.GetDeviceBuffer()),
        static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()));
    c_m_n_device_buf.FromDevice(C.mData.data());

    return true;
#else
    ignore = kernel;
    ignore = A;
    ignore = B;
    ignore = a_block_scale;
    ignore = b_block_scale;
    ignore = C;
    return true;
#endif
}

template <typename DeviceWmma,
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
          index_t KMultiplier = 1>
struct TestWmma
{
    auto PrepareGemmTensor(const ck::wmma_op_util::GemmParams& params)
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

        auto f_generate_tensor_value = [](auto& tensor, auto type) {
            using dataType = decltype(type);

            tensor.GenerateTensorValue(GeneratorTensor_2<dataType>{-5, 5});
        };

        f_generate_tensor_value(a_m_k, ADataType{});
        f_generate_tensor_value(b_n_k, BDataType{});

        return std::make_tuple(a_m_k, b_n_k, c_m_n_host_result, c_m_n_device_result);
    }
    template <typename DataType>
    void dump_tensor(Tensor<DataType> mat)
    {
        std::cout << "mat [ " << std::endl;

        auto len = mat.GetLengths();
        for(uint32_t i = 0; i < len[0]; i++)
        {
            std::cout << "    [";
            for(uint32_t j = 0; j < len[1]; j++)
            {
                std::vector<std::size_t> idx({i, j});
                std::cout << ck::type_convert<float>(mat(idx)) << ", ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }

    auto operator()(const DeviceWmma& wmma_kernel)
    {
        std::cout << "ALayout = " << ALayout{}.name << ", BLayout = " << BLayout{}.name
                  << ", CLayout = " << CLayout{}.name << std::endl;

        // Arrange
        ck::wmma_op_util::GemmParams params;
        params.M = 16;
        params.N = 16;
        if constexpr(KMultiplier <= 4)
        {
            params.K       = 16 * KMultiplier;
            params.StrideA = 16 * KMultiplier;
            params.StrideB = 16 * KMultiplier;
        }
        else
        {
            constexpr auto kValue = KMultiplier;
            params.K              = kValue;
            params.StrideA        = kValue;
            params.StrideB        = kValue;
        }
        params.StrideC = 16;

        auto host_tensors = PrepareGemmTensor(params);

        const Tensor<ADataType>& a  = std::get<0>(host_tensors);
        const Tensor<BDataType>& b  = std::get<1>(host_tensors);
        Tensor<CDataType>& c_host   = std::get<2>(host_tensors);
        Tensor<CDataType>& c_device = std::get<3>(host_tensors);

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
        ck::wmma_op_util::RunHostGEMM<ReferenceGemmInstance>(
            a, b, c_host, a_element_op, b_element_op, c_element_op);

        // Act
        bool is_supported =
            (ck::is_gfx11_supported() || ck::is_gfx12_supported() || ck::is_gfx13_supported()) &&
            ck::wmma_op_util::RunDeviceGEMM(wmma_kernel, a, b, c_device);

        //   dump_tensor(a);
        //   dump_tensor(b);
        //   dump_tensor(c_device);
        //   dump_tensor(c_host);

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
                // BF16 WMMA Accumulator is in BF16 Type while On Host-side Accumulator is
                // Float.
                res = ck::utils::check_err(
                    c_device.mData, c_host.mData, "Error: Incorrect results!", 0, 1.0);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, int8_t>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, double>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, int32_t>::value)
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

template <typename DeviceWmma,
          typename AGPUDataType,
          typename BGPUDataType,
          typename ACPUDataType,
          typename BCPUDataType,
          typename CDataType,
          typename GPUAccDataType,
          typename CPUAccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          ck::index_t AScaleSel,
          ck::index_t BScaleSel>
struct TestMixedFPWmma
{
    auto PrepareGemmTensor(const ck::wmma_op_util::GemmParams& params)
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
        Tensor<ACPUDataType> a_m_k(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<BCPUDataType> b_n_k(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<CDataType> c_m_n_host_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_device_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        Tensor<int32_t> a_matrix_scale(
            f_host_tensor_descriptor(32, 1, 1, ck::tensor_layout::gemm::RowMajor{}));

        Tensor<int32_t> b_matrix_scale(
            f_host_tensor_descriptor(32, 1, 1, ck::tensor_layout::gemm::RowMajor{}));

        auto f_generate_tensor_value = [](auto& tensor, auto type) {
            using dataType = decltype(type);
            // because fp4 cannot generate the accurate value for (+/-)5.f; so the range is shrinked
            tensor.GenerateTensorValue(GeneratorTensor_2<dataType>{-4, 4});
        };

        auto f_generate_scale_value = [](auto& tensor) {
            tensor.GenerateTensorValue(GeneratorTensor_2<int32_t>{0x7f80816f, 0x7f808170});
        };

        f_generate_tensor_value(a_m_k, ACPUDataType{});
        f_generate_tensor_value(b_n_k, BCPUDataType{});
        f_generate_scale_value(a_matrix_scale);
        f_generate_scale_value(b_matrix_scale);

        return std::make_tuple(
            a_m_k, b_n_k, a_matrix_scale, b_matrix_scale, c_m_n_host_result, c_m_n_device_result);
    }
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

    auto operator()(const DeviceWmma& wmma_kernel,
                    ck::wmma_op_util::GemmParams params = GemmParams{})
    {
        std::cout << "ALayout = " << ALayout{}.name << ", BLayout = " << BLayout{}.name
                  << ", CLayout = " << CLayout{}.name << std::endl;
        // Arrange
        // ck::wmma_op_util::GemmParams params;
        // params.M       = 16;
        // params.N       = 16;
        // params.K       = 16;
        // params.StrideA = 16;
        // params.StrideB = 16;
        // params.StrideC = 16;

        auto host_tensors = PrepareGemmTensor(params);

        const Tensor<ACPUDataType>& a        = std::get<0>(host_tensors);
        const Tensor<BCPUDataType>& b        = std::get<1>(host_tensors);
        const Tensor<int32_t>& a_block_scale = std::get<2>(host_tensors);
        const Tensor<int32_t>& b_block_scale = std::get<3>(host_tensors);
        Tensor<CDataType>& c_host            = std::get<4>(host_tensors);
        Tensor<CDataType>& c_device          = std::get<5>(host_tensors);

        auto a_element_op = AElementwiseOperation{};
        auto b_element_op = BElementwiseOperation{};
        auto c_element_op = CElementwiseOperation{};

        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceScaleBlockGemm<ACPUDataType,
                                                                BCPUDataType,
                                                                AGPUDataType,
                                                                BGPUDataType,
                                                                AScaleSel,
                                                                BScaleSel,
                                                                CDataType,
                                                                CPUAccDataType,
                                                                AElementwiseOperation,
                                                                BElementwiseOperation,
                                                                CElementwiseOperation>;
        ck::wmma_op_util::RunHostMixedTypeGEMM<ReferenceGemmInstance>(
            a, b, a_block_scale, b_block_scale, c_host, a_element_op, b_element_op, c_element_op);

        // Act
        bool is_supported = ck::wmma_op_util::RunMixedTypeDeviceGEMM(wmma_kernel,
                                                                     a,
                                                                     b,
                                                                     a_block_scale,
                                                                     b_block_scale,
                                                                     c_device,
                                                                     AGPUDataType{},
                                                                     BGPUDataType{});

        // dump_tensor(a);
        // dump_tensor(b);
        // dump_tensor(c_device);
        // dump_tensor(c_host);

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
            else if(std::is_same<CDataType, double>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, int32_t>::value)
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

} // namespace wmma_op_util
} // namespace ck
