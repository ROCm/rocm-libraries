// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/dtype_complex.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "complex_test_data.hpp"

namespace {

using ck::complex_t;
using ck::index_t;
using ck_complex_test::same_bits;
using ck_complex_test::special_values;

// Values in [-4, 4) with different real and imaginary parts, so that a lost or swapped part shows.
template <typename T>
std::vector<complex_t<T>> make_data(std::size_t count, unsigned seed)
{
    std::vector<complex_t<T>> v(count);
    for(auto& x : v)
    {
        seed      = seed * 1664525u + 1013904223u;
        const T r = static_cast<T>(static_cast<int>(seed >> 8) % 8192) / T{1024};
        seed      = seed * 1664525u + 1013904223u;
        const T i = static_cast<T>(static_cast<int>(seed >> 8) % 8192) / T{1024};
        x         = complex_t<T>{r - T{4}, i - T{4}};
    }
    return v;
}

template <typename T>
long count_changed(const std::vector<complex_t<T>>& y, const std::vector<complex_t<T>>& x)
{
    long bad = 0;
    for(std::size_t i = 0; i < x.size(); ++i)
    {
        bad += same_bits(y[i], x[i]) ? 0 : 1;
    }
    return bad;
}

template <typename T>
class ComplexLoadStore : public ::testing::Test
{
};

using ComponentTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ComplexLoadStore, ComponentTypes);

// Each thread loads Vec elements with the buffer load and stores them with the buffer store.
template <typename T, index_t Vec>
__global__ void copy_kernel(const complex_t<T>* src, complex_t<T>* dst, index_t space)
{
    const index_t offset = static_cast<index_t>(threadIdx.x) * Vec;
    const auto loaded    = ck::amd_buffer_load_invalid_element_return_zero<complex_t<T>, Vec>(
        src, offset, true, space);
    ck::amd_buffer_store<complex_t<T>, Vec>(loaded, dst, offset, true, space);
}

// Copies x on the device, Vec elements per thread, and returns the copy.
template <typename T, index_t Vec>
std::vector<complex_t<T>> copy_on_device(const std::vector<complex_t<T>>& x)
{
    using Elem          = complex_t<T>;
    const index_t space = static_cast<index_t>(x.size());
    ck::DeviceMem x_dev(sizeof(Elem) * x.size()), y_dev(sizeof(Elem) * x.size());
    x_dev.ToDevice(x.data());
    y_dev.SetZero();
    hipLaunchKernelGGL((copy_kernel<T, Vec>),
                       dim3(1),
                       dim3(space / Vec),
                       0,
                       nullptr,
                       static_cast<const Elem*>(x_dev.GetDeviceBuffer()),
                       static_cast<Elem*>(y_dev.GetDeviceBuffer()),
                       space);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
    std::vector<Elem> y(x.size());
    y_dev.FromDevice(y.data());
    return y;
}

template <typename T, index_t Vec>
void check_copy()
{
    const auto x = make_data<T>(64 * Vec, 1u);
    EXPECT_EQ(count_changed(copy_on_device<T, Vec>(x), x), 0) << "elements per vector: " << Vec;
}

// Up to 64 bytes per vector, the most CK's buffer functions move in one call (8 doubles):
// 8 complex_float_t or 4 complex_double_t. A buffer load moves at most 16 bytes, so the wider
// vectors take several.
TYPED_TEST(ComplexLoadStore, CopyVectorWidths)
{
    check_copy<TypeParam, 1>();
    check_copy<TypeParam, 2>();
    check_copy<TypeParam, 4>();
    if constexpr(ck::is_same_v<TypeParam, float>)
    {
        check_copy<TypeParam, 8>();
    }
}

// Every pair of special values as (real, imag), twice, so that two elements per vector divide it.
template <typename T>
std::vector<complex_t<T>> make_special_data()
{
    const auto v        = special_values<T>();
    const std::size_t m = v.size();
    std::vector<complex_t<T>> d(2 * m * m);
    for(std::size_t i = 0; i < d.size(); ++i)
    {
        d[i] = complex_t<T>{v[i % m], v[(i / m) % m]};
    }
    return d;
}

TYPED_TEST(ComplexLoadStore, CopySpecialValuesKeepsBits)
{
    const auto x = make_special_data<TypeParam>();
    EXPECT_EQ(count_changed(copy_on_device<TypeParam, 2>(x), x), 0);
}

// Every third vector is marked invalid while its offset stays inside the buffer, so only the
// functions themselves, not the hardware range check, can keep it out: an invalid load returns
// zero or the given value, an invalid store leaves memory unchanged. The zero load runs twice:
// once with a branch on the flag, once with the offset trick, where the flag moves the offset
// out of the buffer.
constexpr index_t kVectors = 64;

template <typename T, index_t Vec>
__global__ void buffer_masking_kernel(const complex_t<T>* src,
                                      complex_t<T>* zero_out,
                                      complex_t<T>* trick_out,
                                      complex_t<T>* value_out,
                                      complex_t<T>* store_out,
                                      complex_t<T> value)
{
    using Elem             = complex_t<T>;
    using Coherence        = ck::AmdBufferCoherenceEnum;
    const index_t t        = static_cast<index_t>(threadIdx.x);
    const index_t offset   = t * Vec;
    const bool valid       = t % 3 != 1;
    const index_t space    = kVectors * Vec;
    const auto zero_loaded = ck::
        amd_buffer_load_invalid_element_return_zero<Elem, Vec, Coherence::DefaultCoherence, false>(
            src, offset, valid, space);
    const auto trick_loaded = ck::
        amd_buffer_load_invalid_element_return_zero<Elem, Vec, Coherence::DefaultCoherence, true>(
            src, offset, valid, space);
    const auto value_loaded =
        ck::amd_buffer_load_invalid_element_return_customized_value<Elem, Vec>(
            src, offset, valid, space, value);
    const auto all_loaded =
        ck::amd_buffer_load_invalid_element_return_zero<Elem, Vec>(src, offset, true, space);
    ck::amd_buffer_store<Elem, Vec>(zero_loaded, zero_out, offset, true, space);
    ck::amd_buffer_store<Elem, Vec>(trick_loaded, trick_out, offset, true, space);
    ck::amd_buffer_store<Elem, Vec>(value_loaded, value_out, offset, true, space);
    ck::amd_buffer_store<Elem, Vec>(all_loaded, store_out, offset, valid, space);
}

template <typename T, index_t Vec>
void check_buffer_masking()
{
    using Elem              = complex_t<T>;
    const std::size_t count = static_cast<std::size_t>(kVectors) * Vec;
    const auto src          = make_data<T>(count, 10u);
    const Elem zero{T{0}, T{0}};
    const Elem value{T{0.5}, T{-0.25}};
    const Elem sentinel{T{-7.25}, T{9.5}};
    const std::vector<Elem> filled(count, sentinel);

    ck::DeviceMem src_dev(sizeof(Elem) * count), zero_dev(sizeof(Elem) * count),
        trick_dev(sizeof(Elem) * count), value_dev(sizeof(Elem) * count),
        store_dev(sizeof(Elem) * count);
    src_dev.ToDevice(src.data());
    store_dev.ToDevice(filled.data());
    hipLaunchKernelGGL((buffer_masking_kernel<T, Vec>),
                       dim3(1),
                       dim3(kVectors),
                       0,
                       nullptr,
                       static_cast<const Elem*>(src_dev.GetDeviceBuffer()),
                       static_cast<Elem*>(zero_dev.GetDeviceBuffer()),
                       static_cast<Elem*>(trick_dev.GetDeviceBuffer()),
                       static_cast<Elem*>(value_dev.GetDeviceBuffer()),
                       static_cast<Elem*>(store_dev.GetDeviceBuffer()),
                       value);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<Elem> zero_out(count), trick_out(count), value_out(count), store_out(count);
    zero_dev.FromDevice(zero_out.data());
    trick_dev.FromDevice(trick_out.data());
    value_dev.FromDevice(value_out.data());
    store_dev.FromDevice(store_out.data());
    long bad_zero = 0, bad_trick = 0, bad_value = 0, bad_store = 0;
    for(std::size_t i = 0; i < count; ++i)
    {
        const bool valid = (i / Vec) % 3 != 1;
        bad_zero += same_bits(zero_out[i], valid ? src[i] : zero) ? 0 : 1;
        bad_trick += same_bits(trick_out[i], valid ? src[i] : zero) ? 0 : 1;
        bad_value += same_bits(value_out[i], valid ? src[i] : value) ? 0 : 1;
        bad_store += same_bits(store_out[i], valid ? src[i] : sentinel) ? 0 : 1;
    }
    EXPECT_EQ(bad_zero, 0) << "zero load, elements per vector: " << Vec;
    EXPECT_EQ(bad_trick, 0) << "zero load with the offset trick, elements per vector: " << Vec;
    EXPECT_EQ(bad_value, 0) << "given-value load, elements per vector: " << Vec;
    EXPECT_EQ(bad_store, 0) << "masked store, elements per vector: " << Vec;
}

TYPED_TEST(ComplexLoadStore, BufferLoadStoreMasking)
{
    check_buffer_masking<TypeParam, 1>();
    check_buffer_masking<TypeParam, 2>();
}

// Kernels read and write through DynamicBuffer. For LDS, and for global memory with a 64-bit
// index, it does not use the buffer functions but plain loads and stores, and builds the value of
// an invalid element itself. Every third vector is invalid: it reads as zero, and a store to it is
// skipped.
template <typename T, index_t Vec, bool InLds>
__global__ void
dynamic_buffer_kernel(const complex_t<T>* src, complex_t<T>* got_out, complex_t<T>* set_out)
{
    using Elem               = complex_t<T>;
    using V                  = typename ck::vector_type<Elem, Vec>::type;
    const index_t t          = static_cast<index_t>(threadIdx.x);
    const index_t offset     = t * Vec;
    const bool valid         = t % 3 != 1;
    constexpr index_t kCount = kVectors * Vec;
    if constexpr(InLds)
    {
        __shared__ Elem in[kCount];
        __shared__ Elem out[kCount];
        for(index_t j = 0; j < Vec; ++j)
        {
            in[offset + j]  = src[offset + j];
            out[offset + j] = set_out[offset + j];
        }
        __syncthreads();
        auto in_buf  = ck::make_dynamic_buffer<ck::AddressSpaceEnum::Lds>(&in[0], kCount);
        auto out_buf = ck::make_dynamic_buffer<ck::AddressSpaceEnum::Lds>(&out[0], kCount);
        const V got  = in_buf.template Get<V>(offset, valid);
        out_buf.template Set<V>(offset, valid, in_buf.template Get<V>(offset, true));
        __syncthreads();
        for(index_t j = 0; j < Vec; ++j)
        {
            set_out[offset + j] = out[offset + j];
        }
        __builtin_memcpy(got_out + offset, &got, sizeof(V));
    }
    else
    {
        using Coherence = ck::AmdBufferCoherenceEnum;
        auto in_buf     = ck::make_dynamic_buffer<ck::AddressSpaceEnum::Global,
                                                  Coherence::DefaultCoherence,
                                                  ck::long_index_t>(src, kCount);
        auto out_buf    = ck::make_dynamic_buffer<ck::AddressSpaceEnum::Global,
                                                  Coherence::DefaultCoherence,
                                                  ck::long_index_t>(set_out, kCount);
        const V got     = in_buf.template Get<V>(offset, valid);
        out_buf.template Set<V>(offset, valid, in_buf.template Get<V>(offset, true));
        __builtin_memcpy(got_out + offset, &got, sizeof(V));
    }
}

template <typename T, index_t Vec, bool InLds>
void check_dynamic_buffer()
{
    using Elem              = complex_t<T>;
    const std::size_t count = static_cast<std::size_t>(kVectors) * Vec;
    const auto src          = make_data<T>(count, 11u);
    const Elem zero{T{0}, T{0}};
    const Elem sentinel{T{-7.25}, T{9.5}};
    const std::vector<Elem> filled(count, sentinel);

    ck::DeviceMem src_dev(sizeof(Elem) * count), got_dev(sizeof(Elem) * count),
        set_dev(sizeof(Elem) * count);
    src_dev.ToDevice(src.data());
    set_dev.ToDevice(filled.data());
    hipLaunchKernelGGL((dynamic_buffer_kernel<T, Vec, InLds>),
                       dim3(1),
                       dim3(kVectors),
                       0,
                       nullptr,
                       static_cast<const Elem*>(src_dev.GetDeviceBuffer()),
                       static_cast<Elem*>(got_dev.GetDeviceBuffer()),
                       static_cast<Elem*>(set_dev.GetDeviceBuffer()));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<Elem> got(count), set(count);
    got_dev.FromDevice(got.data());
    set_dev.FromDevice(set.data());
    long bad_get = 0, bad_set = 0;
    for(std::size_t i = 0; i < count; ++i)
    {
        const bool valid = (i / Vec) % 3 != 1;
        bad_get += same_bits(got[i], valid ? src[i] : zero) ? 0 : 1;
        bad_set += same_bits(set[i], valid ? src[i] : sentinel) ? 0 : 1;
    }
    const char* where = InLds ? "LDS" : "global memory, 64-bit index";
    EXPECT_EQ(bad_get, 0) << where << ", Get, elements per vector: " << Vec;
    EXPECT_EQ(bad_set, 0) << where << ", Set, elements per vector: " << Vec;
}

TYPED_TEST(ComplexLoadStore, DynamicBufferWithoutBufferFunctions)
{
    check_dynamic_buffer<TypeParam, 1, true>();
    check_dynamic_buffer<TypeParam, 2, true>();
    check_dynamic_buffer<TypeParam, 4, true>();
    check_dynamic_buffer<TypeParam, 1, false>();
    check_dynamic_buffer<TypeParam, 2, false>();
    check_dynamic_buffer<TypeParam, 4, false>();
}

} // namespace
