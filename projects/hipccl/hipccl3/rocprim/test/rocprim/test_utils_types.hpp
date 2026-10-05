// Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef TEST_TEST_UTILS_TYPES_HPP_
#define TEST_TEST_UTILS_TYPES_HPP_

#include "../../common/utils_custom_type.hpp"

// required rocprim headers
#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_store.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <type_traits>

template<
    class T,
    unsigned int WarpSize,
    unsigned int ItemsPerThread = 1
>
struct warp_params
{
    using type = T;
    static constexpr unsigned int warp_size = WarpSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
};

template<
    class T,
    class U,
    unsigned int BlockSize = 256U
>
struct block_params
{
    using input_type = T;
    using output_type = U;
    static constexpr unsigned int block_size = BlockSize;
};

template<
    class T,
    class U,
    unsigned int ItemsPerThread,
    bool ShouldBeVectorized
>
struct vector_params
{
    using type = T;
    using vector_type = U;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr bool should_be_vectorized = ShouldBeVectorized;
};

template<
    class Type,
    rocprim::block_load_method Load,
    rocprim::block_store_method Store,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
struct class_params
{
    using type = Type;
    static constexpr rocprim::block_load_method load_method = Load;
    static constexpr rocprim::block_store_method store_method = Store;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
};

// clang-format off
#define warp_param_type(type) \
   warp_params<type, 4U>,     \
   warp_params<type, 8U>,     \
   warp_params<type, 16U>,    \
   warp_params<type, 32U>,    \
   warp_params<type, 64U>,    \
   warp_params<type, 3U>,     \
   warp_params<type, 7U>,     \
   warp_params<type, 15U>,    \
   warp_params<type, 37U>,    \
   warp_params<type, 61U>

#define block_param_type(input_type, output_type) \
    block_params<input_type, output_type, 32U>,   \
    block_params<input_type, output_type, 64U>,   \
    block_params<input_type, output_type, 128U>,  \
    block_params<input_type, output_type, 192U>,  \
    block_params<input_type, output_type, 256U>,  \
    block_params<input_type, output_type, 129U>,  \
    block_params<input_type, output_type, 162U>,  \
    block_params<input_type, output_type, 255U>
// clang-format on

using WarpParamsIntegral
    = ::testing::Types<warp_param_type(int), warp_param_type(int8_t), warp_param_type(uint8_t)>;

using WarpParamsFloating = ::testing::Types<warp_param_type(float),
                                            warp_param_type(double),
                                            warp_param_type(rocprim::half),
                                            warp_param_type(rocprim::bfloat16)>;

// Separate sort params (only power of two warp sizes)
#define warp_sort_param_type(type, items_per_thread) \
   warp_params<type, 2U, items_per_thread>, \
   warp_params<type, 4U, items_per_thread>, \
   warp_params<type, 8U, items_per_thread>, \
   warp_params<type, 16U, items_per_thread>, \
   warp_params<type, 32U, items_per_thread>, \
   warp_params<type, 64U, items_per_thread>

using __custom_int2 = common::custom_type<int, int, true>;

using WarpSortParamsIntegral = ::testing::Types<warp_sort_param_type(int, 1),
                                                warp_sort_param_type(__custom_int2, 1),
                                                warp_sort_param_type(uint8_t, 1),
                                                warp_sort_param_type(int8_t, 1)>;

using WarpSortParamsFloating = ::testing::Types<warp_sort_param_type(float, 1),
                                                warp_sort_param_type(double, 1),
                                                warp_sort_param_type(rocprim::half, 1),
                                                warp_sort_param_type(rocprim::bfloat16, 1)>;

using WarpSortParamsIntegralMultiThread = ::testing::Types<warp_sort_param_type(int, 2),
                                                           warp_sort_param_type(__custom_int2, 2),
                                                           warp_sort_param_type(uint8_t, 2),
                                                           warp_sort_param_type(int8_t, 2),
                                                           warp_sort_param_type(int, 4),
                                                           warp_sort_param_type(__custom_int2, 4),
                                                           warp_sort_param_type(uint8_t, 4),
                                                           warp_sort_param_type(int8_t, 4),
                                                           warp_sort_param_type(int8_t, 8),
                                                           warp_sort_param_type(int8_t, 16)>;

using BlockParamsIntegral = ::testing::Types<block_param_type(int, __custom_int2),
                                             block_param_type(uint8_t, short),
                                             block_param_type(int8_t, float)>;

using BlockParamsIntegralExtended = ::testing::Types<block_param_type(int, __custom_int2),
                                                     block_param_type(uint8_t, short),
                                                     block_param_type(int8_t, float),
                                                     block_param_type(bool, rocprim::half)
#if ROCPRIM_HAS_INT128_SUPPORT
                                                         ,
                                                     block_param_type(rocprim::uint128_t, short),
                                                     block_param_type(rocprim::int128_t, float)
#endif
                                                     >;

using __custom_double2 = common::custom_type<double, double, true>;

using BlockParamsFloating
    = ::testing::Types<block_param_type(float, long),
                       block_param_type(double, __custom_double2),
                       block_param_type(rocprim::half, int),
                       block_param_type(rocprim::half, rocprim::half),
                       block_param_type(rocprim::bfloat16, int),
                       block_param_type(rocprim::bfloat16, rocprim::bfloat16)>;

using BlockDiscParamsIntegral = ::testing::Types<block_param_type(__custom_int2, int),
                                                 block_param_type(uint8_t, bool),
                                                 block_param_type(int8_t, bool)>;

using BlockDiscParamsFloating
    = ::testing::Types<block_param_type(float, long), block_param_type(double, unsigned int)>;

using BlockDiscParamsFloatingHalf
    = ::testing::Types<block_param_type(rocprim::half, int),
                       block_param_type(rocprim::half, rocprim::half),
                       block_param_type(rocprim::bfloat16, int),
                       block_param_type(rocprim::bfloat16, rocprim::bfloat16)>;

using BlockHistAtomicParamsIntegral = ::testing::Types<block_param_type(unsigned int, unsigned int),
                                                       block_param_type(int8_t, unsigned int),
                                                       block_param_type(uint8_t, unsigned int)>;

using BlockExchParamsFloating
    = ::testing::Types<block_param_type(float, long),
                       block_param_type(double, __custom_double2),
                       block_param_type(double, int8_t),
                       block_param_type(rocprim::half, rocprim::half),
                       block_param_type(rocprim::half, int16_t),
                       block_param_type(rocprim::bfloat16, rocprim::bfloat16)>;

using BlockHistAtomicParamsFloating
    = ::testing::Types<block_param_type(float, float),
                       block_param_type(float, unsigned int),
                       block_param_type(float, unsigned long long),
                       block_param_type(double, float),
                       block_param_type(double, unsigned long long),
                       block_param_type(rocprim::half, float),
                       block_param_type(rocprim::half, unsigned long long),
                       block_param_type(rocprim::bfloat16, float),
                       block_param_type(rocprim::bfloat16, unsigned long long)>;

using BlockHistSortParamsIntegral = ::testing::Types<block_param_type(int, uint8_t),
                                                     block_param_type(short, uint8_t),
                                                     block_param_type(uint8_t, uint8_t),
                                                     block_param_type(int, int8_t),
                                                     block_param_type(short, int8_t),
                                                     block_param_type(int8_t, int8_t)>;

using BlockHistSortParamsFloating
    = ::testing::Types<block_param_type(float, unsigned short),
                       block_param_type(float, unsigned int),
                       block_param_type(double, unsigned short),
                       block_param_type(double, unsigned int),
                       block_param_type(rocprim::half, unsigned short),
                       block_param_type(rocprim::half, unsigned int),
                       block_param_type(rocprim::bfloat16, unsigned short),
                       block_param_type(rocprim::bfloat16, unsigned int)>;

static constexpr size_t n_items = 7;
static constexpr unsigned int items[n_items] = {
    1, 2, 4, 5, 7, 15, 32
};

// Global utility defines
#define test_suite_type_def_helper(name, suffix) \
    template<class Params> \
    class name ## suffix : public ::testing::Test { \
    public: \
        using params = Params; \
    };

#define test_suite_type_def(name, suffix) test_suite_type_def_helper(name, suffix)

#define block_histo_test_suite_type_def_helper(name, suffix) \
    template<class Params> \
    class name ## suffix : public ::testing::Test { \
    public: \
        using type = typename Params::input_type; \
        using bin_type = typename Params::output_type; \
        static constexpr unsigned int block_size = Params::block_size; \
        static constexpr unsigned int bin_size = Params::block_size; \
    };

#define block_histo_test_suite_type_def(name, suffix) block_histo_test_suite_type_def_helper(name, suffix)

#define block_reduce_test_suite_type_def_helper(name, suffix) \
    template<class Params> \
    class name ## suffix : public ::testing::Test { \
    public: \
        using input_type = typename Params::input_type; \
        static constexpr unsigned int block_size = Params::block_size; \
    };

#define block_reduce_test_suite_type_def(name, suffix) block_reduce_test_suite_type_def_helper(name, suffix)

#define block_sort_test_suite_type_def_helper(name, suffix) \
    template<class Params> \
    class name ## suffix : public ::testing::Test { \
    public: \
        using key_type = typename Params::input_type; \
        using value_type = typename Params::output_type; \
        static constexpr unsigned int block_size = Params::block_size; \
    };

#define block_sort_test_suite_type_def(name, suffix) block_sort_test_suite_type_def_helper(name, suffix)

// Variadic so callers may pass either 3 arguments or 4 arguments with the 4th
// argument being a name generator forwarded to TYPED_TEST_SUITE's 3rd parameter.
#define typed_test_suite_def_helper(name, suffix, ...) TYPED_TEST_SUITE(name ## suffix, __VA_ARGS__)

#define typed_test_suite_def(name, suffix, ...) typed_test_suite_def_helper(name, suffix, __VA_ARGS__)

#define typed_test_def_helper(suite, suffix, name) TYPED_TEST(suite ## suffix, name)

#define typed_test_def(suite, suffix, name) typed_test_def_helper(suite, suffix, name)

// ----------------------------------------------------------------------------
// Default TYPED_TEST_SUITE naming uses the positional index of a type in its
// ::testing::Types list, which may silently point at a different type whenever
// the list is reordered. The generators below build a stable name from the
// parameter struct's own fields to make a TYPED_TEST_SUITE filterable by name.
// ----------------------------------------------------------------------------

template<class>
struct dependent_false : std::false_type
{};

// Maps every concrete type used in the typed test parameter lists to a unique,
// readable name.
template<class T>
inline std::string type_tag()
{
    if constexpr(std::is_same_v<T, int>) return "Int";
    else if constexpr(std::is_same_v<T, unsigned int>) return "Uint";
    else if constexpr(std::is_same_v<T, int8_t>) return "Int8";
    else if constexpr(std::is_same_v<T, uint8_t>) return "Uint8";
    else if constexpr(std::is_same_v<T, short>) return "Short";
    else if constexpr(std::is_same_v<T, unsigned short>) return "Ushort";
    else if constexpr(std::is_same_v<T, long>) return "Long";
    else if constexpr(std::is_same_v<T, unsigned long>) return "Ulong";
    else if constexpr(std::is_same_v<T, unsigned long long>) return "Uint64";
    else if constexpr(std::is_same_v<T, long long>) return "Int64";
    else if constexpr(std::is_same_v<T, char>) return "Char";
    else if constexpr(std::is_same_v<T, bool>) return "Bool";
    else if constexpr(std::is_same_v<T, float>) return "Float";
    else if constexpr(std::is_same_v<T, double>) return "Double";
    else if constexpr(std::is_same_v<T, rocprim::half>) return "Half";
    else if constexpr(std::is_same_v<T, rocprim::bfloat16>) return "Bfloat16";
    else if constexpr(std::is_same_v<T, float2>) return "Float2";
    else if constexpr(std::is_same_v<T, common::custom_type<int, int, true>>) return "CustomInt2";
    else if constexpr(std::is_same_v<T, common::custom_type<double, double, true>>)
        return "CustomDouble2";
    else if constexpr(std::is_same_v<T, common::custom_type<float, float, true>>)
        return "CustomFloat2";
    else if constexpr(std::is_same_v<T, common::custom_type<uint8_t, uint8_t, true>>)
        return "CustomUint8_2";
    else if constexpr(std::is_same_v<T, common::custom_type<short, short, true>>)
        return "CustomShort2";
    else if constexpr(std::is_same_v<T, common::custom_huge_type<1024, long long>>)
        return "CustomLarge";
    else if constexpr(std::is_same_v<T, common::custom_huge_type<1024, int>>)
        return "CustomHuge1024Int";
    else if constexpr(std::is_same_v<T, common::custom_type<long long, long long, true>>)
        return "CustomInt64x2";
    // custom_type<uint64_t,...> resolves to unsigned long on LP64 (Linux) and to
    // unsigned long long on LLP64 (Windows); both must yield the same name.
    else if constexpr(std::is_same_v<T, common::custom_type<unsigned long, unsigned long, true>>)
        return "CustomUint64x2";
    else if constexpr(std::is_same_v<
                          T,
                          common::custom_type<unsigned long long, unsigned long long, true>>)
        return "CustomUint64x2";
#if ROCPRIM_HAS_INT128_SUPPORT
    else if constexpr(std::is_same_v<T, rocprim::uint128_t>) return "Uint128";
    else if constexpr(std::is_same_v<T, rocprim::int128_t>) return "Int128";
#endif
    else
        static_assert(dependent_false<T>::value, "type_tag: add a case for this type");
}

// warp_params<T, WarpSize, ItemsPerThread> -> e.g. "Int_w32_i1".
struct warp_params_name_generator
{
    template<class Params>
    static std::string GetName(int)
    {
        return type_tag<typename Params::type>() + "_w" + std::to_string(Params::warp_size) + "_i"
               + std::to_string(Params::items_per_thread);
    }
};

// block_params<T, U, BlockSize> -> e.g. "Int_CustomInt2_b256".
struct block_params_name_generator
{
    template<class Params>
    static std::string GetName(int)
    {
        return type_tag<typename Params::input_type>() + "_"
               + type_tag<typename Params::output_type>() + "_b"
               + std::to_string(Params::block_size);
    }
};

// vector_params<T, U, ItemsPerThread, ShouldBeVectorized> -> e.g. "Int_Int_i4_vec".
struct vector_params_name_generator
{
    template<class Params>
    static std::string GetName(int)
    {
        return type_tag<typename Params::type>() + "_" + type_tag<typename Params::vector_type>()
               + "_i" + std::to_string(Params::items_per_thread)
               + (Params::should_be_vectorized ? "_vec" : "_novec");
    }
};

// class_params<T, Load, Store, BlockSize, ItemsPerThread> -> e.g. "Int_l0_s0_b256_i4".
struct class_params_name_generator
{
    template<class Params>
    static std::string GetName(int)
    {
        return type_tag<typename Params::type>() + "_l"
               + std::to_string(static_cast<int>(Params::load_method)) + "_s"
               + std::to_string(static_cast<int>(Params::store_method)) + "_b"
               + std::to_string(Params::block_size) + "_i"
               + std::to_string(Params::items_per_thread);
    }
};

#endif // TEST_TEST_UTILS_TYPES_HPP_
