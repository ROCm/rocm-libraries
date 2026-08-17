/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#pragma once

#include "c_style_pointer_cast.hpp"
#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"
#include "enable_if.hpp"
#include "generic_memory_space_atomic.hpp"

namespace ck
{

    // T may be scalar or vector
    // X may be scalar or vector
    // T and X have same scalar type
    // X contains multiple T
    template <AddressSpaceEnum BufferAddressSpace,
              typename T,
              typename InElementOp,
              typename ElementSpaceSize,
              bool                   InvalidElementUseNumericalZeroValue,
              AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
    struct HiptensorDynamicBuffer
    {
        using type = T;

        T*               p_data_;
        ElementSpaceSize element_space_size_;
        T                invalid_element_value_ = T{0};
        InElementOp      in_op_{};

        __device__ constexpr HiptensorDynamicBuffer(T* p_data, ElementSpaceSize element_space_size)
            : p_data_{p_data}
            , element_space_size_{element_space_size}
        {
        }

        __device__ constexpr HiptensorDynamicBuffer(T*               p_data,
                                                    InElementOp      in_op,
                                                    ElementSpaceSize element_space_size,
                                                    T                invalid_element_value)
            : p_data_{p_data}
            , element_space_size_{element_space_size}
            , in_op_(in_op)
            , invalid_element_value_{invalid_element_value}
        {
        }

        __device__ static constexpr AddressSpaceEnum GetAddressSpace()
        {
            return BufferAddressSpace;
        }

        __device__ constexpr const T& operator[](index_t i) const
        {
            return p_data_[i];
        }

        __device__ constexpr T& operator()(index_t i)
        {
            return p_data_[i];
        }

        template <typename X,
                  typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                             typename scalar_type<remove_cvref_t<T>>::type>::value
                                         || !is_native_type<X>(),
                                     bool>::type
                  = false>
        __device__ constexpr auto Get(index_t i, bool is_valid_element) const
        {
            // X contains multiple T
            constexpr index_t scalar_per_t_vector = scalar_type<remove_cvref_t<T>>::vector_size;

            constexpr index_t scalar_per_x_vector = scalar_type<remove_cvref_t<X>>::vector_size;

            static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                          "wrong! X should contain multiple T");

#if CK_USE_AMD_BUFFER_LOAD
            bool constexpr use_amd_buffer_addressing = true;
#else
            bool constexpr use_amd_buffer_addressing = false;
#endif

            if constexpr(GetAddressSpace() == AddressSpaceEnum::Global && use_amd_buffer_addressing)
            {
                constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

                if constexpr(InvalidElementUseNumericalZeroValue)
                {
                    auto x = amd_buffer_load_invalid_element_return_zero<remove_cvref_t<T>,
                                                                         t_per_x,
                                                                         coherence>(
                        p_data_, i, is_valid_element, element_space_size_);
                    auto y = x;
                    in_op_(y, x);
                    return is_valid_element ? y : x;
                }
                else
                {
                    auto x
                        = amd_buffer_load_invalid_element_return_customized_value<remove_cvref_t<T>,
                                                                                  t_per_x,
                                                                                  coherence>(
                            p_data_,
                            i,
                            is_valid_element,
                            element_space_size_,
                            invalid_element_value_);
                    auto y = x;
                    in_op_(y, x);
                    return is_valid_element ? y : x;
                }
            }
            else
            {
                if(is_valid_element)
                {
#if CK_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
                    X tmp;

                    __builtin_memcpy(&tmp, &(p_data_[i]), sizeof(X));

                    return tmp;
#else
                    return *c_style_pointer_cast<const X*>(&p_data_[i]);
#endif
                }
                else
                {
                    if constexpr(InvalidElementUseNumericalZeroValue)
                    {
                        return X{0};
                    }
                    else
                    {
                        return X{invalid_element_value_};
                    }
                }
            }
        }

        template <InMemoryDataOperationEnum Op,
                  typename X,
                  typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                             typename scalar_type<remove_cvref_t<T>>::type>::value,
                                     bool>::type
                  = false>
        __device__ void Update(index_t i, bool is_valid_element, const X& x)
        {
            if constexpr(Op == InMemoryDataOperationEnum::Set)
            {
                this->template Set<X>(i, is_valid_element, x);
            }
            else if constexpr(Op == InMemoryDataOperationEnum::AtomicAdd)
            {
                this->template AtomicAdd<X>(i, is_valid_element, x);
            }
            else if constexpr(Op == InMemoryDataOperationEnum::AtomicMax)
            {
                this->template AtomicMax<X>(i, is_valid_element, x);
            }
            else if constexpr(Op == InMemoryDataOperationEnum::Add)
            {
                auto tmp       = this->template Get<X>(i, is_valid_element);
                using scalar_t = typename scalar_type<remove_cvref_t<T>>::type;
                // handle bfloat addition
                if constexpr(is_same_v<scalar_t, bhalf_t>)
                {
                    if constexpr(is_scalar_type<X>::value)
                    {
                        // Scalar type
                        auto result
                            = type_convert<X>(type_convert<float>(x) + type_convert<float>(tmp));
                        this->template Set<X>(i, is_valid_element, result);
                    }
                    else
                    {
                        // Vector type
                        constexpr auto vector_size = scalar_type<remove_cvref_t<X>>::vector_size;
                        const vector_type<scalar_t, vector_size> a_vector{tmp};
                        const vector_type<scalar_t, vector_size> b_vector{x};
                        static_for<0, vector_size, 1>{}([&](auto idx) {
                            auto result = type_convert<scalar_t>(
                                type_convert<float>(a_vector.template AsType<scalar_t>()[idx])
                                + type_convert<float>(b_vector.template AsType<scalar_t>()[idx]));
                            this->template Set<scalar_t>(i + idx, is_valid_element, result);
                        });
                    }
                }
                else
                {
                    this->template Set<X>(i, is_valid_element, x + tmp);
                }
            }
        }

        template <typename DstBuffer, index_t NumElemsPerThread>
        __device__ void DirectCopyToLds(DstBuffer& dst_buf,
                                        index_t    src_offset,
                                        index_t    dst_offset,
                                        bool       is_valid_element) const
        {
            // Copy data from global to LDS memory using direct loads.
            static_assert(GetAddressSpace() == AddressSpaceEnum::Global,
                          "Source data must come from a global memory buffer.");
            static_assert(DstBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                          "Destination data must be stored in an LDS memory buffer.");

            amd_direct_load_global_to_lds<T, NumElemsPerThread>(p_data_,
                                                                src_offset,
                                                                dst_buf.p_data_,
                                                                dst_offset,
                                                                is_valid_element,
                                                                element_space_size_);
        }

        template <typename X,
                  typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                             typename scalar_type<remove_cvref_t<T>>::type>::value
                                         || !is_native_type<X>(),
                                     bool>::type
                  = false>
        __device__ void Set(index_t i, bool is_valid_element, const X& x)
        {
            // X contains multiple T
            constexpr index_t scalar_per_t_vector = scalar_type<remove_cvref_t<T>>::vector_size;

            constexpr index_t scalar_per_x_vector = scalar_type<remove_cvref_t<X>>::vector_size;

            static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                          "wrong! X should contain multiple T");

#if CK_USE_AMD_BUFFER_STORE
            bool constexpr use_amd_buffer_addressing = true;
#else
            bool constexpr use_amd_buffer_addressing      = false;
#endif

#if CK_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE
            bool constexpr workaround_int8_ds_write_issue = true;
#else
            bool constexpr workaround_int8_ds_write_issue = false;
#endif

            if constexpr(GetAddressSpace() == AddressSpaceEnum::Global && use_amd_buffer_addressing)
            {
                constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

                amd_buffer_store<remove_cvref_t<T>, t_per_x, coherence>(
                    x, p_data_, i, is_valid_element, element_space_size_);
            }
            else if constexpr(GetAddressSpace() == AddressSpaceEnum::Lds
                              && is_same<typename scalar_type<remove_cvref_t<T>>::type,
                                         int8_t>::value
                              && workaround_int8_ds_write_issue)
            {
                if(is_valid_element)
                {
                    // HACK: compiler would lower IR "store<i8, 16> address_space(3)" into inefficient
                    // ISA, so I try to let compiler emit IR "store<i32, 4>" which would be lower to
                    // ds_write_b128
                    // TODO: remove this after compiler fix
                    static_assert((is_same<remove_cvref_t<T>, int8_t>::value
                                   && is_same<remove_cvref_t<X>, int8_t>::value)
                                      || (is_same<remove_cvref_t<T>, int8_t>::value
                                          && is_same<remove_cvref_t<X>, int8x2_t>::value)
                                      || (is_same<remove_cvref_t<T>, int8_t>::value
                                          && is_same<remove_cvref_t<X>, int8x4_t>::value)
                                      || (is_same<remove_cvref_t<T>, int8_t>::value
                                          && is_same<remove_cvref_t<X>, int8x8_t>::value)
                                      || (is_same<remove_cvref_t<T>, int8_t>::value
                                          && is_same<remove_cvref_t<X>, int8x16_t>::value)
                                      || (is_same<remove_cvref_t<T>, int8x4_t>::value
                                          && is_same<remove_cvref_t<X>, int8x4_t>::value)
                                      || (is_same<remove_cvref_t<T>, int8x8_t>::value
                                          && is_same<remove_cvref_t<X>, int8x8_t>::value)
                                      || (is_same<remove_cvref_t<T>, int8x16_t>::value
                                          && is_same<remove_cvref_t<X>, int8x16_t>::value),
                                  "wrong! not implemented for this combination, please add "
                                  "implementation");

                    if constexpr(is_same<remove_cvref_t<T>, int8_t>::value
                                 && is_same<remove_cvref_t<X>, int8_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *c_style_pointer_cast<int8_t*>(&p_data_[i])
                            = *c_style_pointer_cast<const int8_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cvref_t<T>, int8_t>::value
                                      && is_same<remove_cvref_t<X>, int8x2_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *c_style_pointer_cast<int16_t*>(&p_data_[i])
                            = *c_style_pointer_cast<const int16_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cvref_t<T>, int8_t>::value
                                      && is_same<remove_cvref_t<X>, int8x4_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *c_style_pointer_cast<int32_t*>(&p_data_[i])
                            = *c_style_pointer_cast<const int32_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cvref_t<T>, int8_t>::value
                                      && is_same<remove_cvref_t<X>, int8x8_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *c_style_pointer_cast<int32x2_t*>(&p_data_[i])
                            = *c_style_pointer_cast<const int32x2_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cvref_t<T>, int8_t>::value
                                      && is_same<remove_cvref_t<X>, int8x16_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *c_style_pointer_cast<int32x4_t*>(&p_data_[i])
                            = *c_style_pointer_cast<const int32x4_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cvref_t<T>, int8x4_t>::value
                                      && is_same<remove_cvref_t<X>, int8x4_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *c_style_pointer_cast<int32_t*>(&p_data_[i])
                            = *c_style_pointer_cast<const int32_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cvref_t<T>, int8x8_t>::value
                                      && is_same<remove_cvref_t<X>, int8x8_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *c_style_pointer_cast<int32x2_t*>(&p_data_[i])
                            = *c_style_pointer_cast<const int32x2_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cvref_t<T>, int8x16_t>::value
                                      && is_same<remove_cvref_t<X>, int8x16_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *c_style_pointer_cast<int32x4_t*>(&p_data_[i])
                            = *c_style_pointer_cast<const int32x4_t*>(&x);
                    }
                }
            }
            else
            {
                if(is_valid_element)
                {
#if CK_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
                    X tmp = x;

                    __builtin_memcpy(&(p_data_[i]), &tmp, sizeof(X));
#else
                    *c_style_pointer_cast<X*>(&p_data_[i]) = x;
#endif
                }
            }
        }

        template <typename X,
                  typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                             typename scalar_type<remove_cvref_t<T>>::type>::value,
                                     bool>::type
                  = false>
        __device__ void AtomicAdd(index_t i, bool is_valid_element, const X& x)
        {
            using scalar_t = typename scalar_type<remove_cvref_t<T>>::type;

            // X contains multiple T
            constexpr index_t scalar_per_t_vector = scalar_type<remove_cvref_t<T>>::vector_size;

            constexpr index_t scalar_per_x_vector = scalar_type<remove_cvref_t<X>>::vector_size;

            static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                          "wrong! X should contain multiple T");

            static_assert(GetAddressSpace() == AddressSpaceEnum::Global, "only support global mem");

#if CK_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER && CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT
            bool constexpr use_amd_buffer_addressing
                = is_same_v<
                      remove_cvref_t<scalar_t>,
                      int32_t> || is_same_v<remove_cvref_t<scalar_t>, float> || (is_same_v<remove_cvref_t<scalar_t>, half_t> && scalar_per_x_vector % 2 == 0)
                  || (is_same_v<remove_cvref_t<scalar_t>, bhalf_t> && scalar_per_x_vector % 2 == 0);
#elif CK_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER && (!CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT)
            bool constexpr use_amd_buffer_addressing = is_same_v<remove_cvref_t<scalar_t>, int32_t>;
#elif(!CK_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER) && CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT
            bool constexpr use_amd_buffer_addressing
                = is_same_v<
                      remove_cvref_t<scalar_t>,
                      float> || (is_same_v<remove_cvref_t<scalar_t>, half_t> && scalar_per_x_vector % 2 == 0)
                  || (is_same_v<remove_cvref_t<scalar_t>, bhalf_t> && scalar_per_x_vector % 2 == 0);
#else
            bool constexpr use_amd_buffer_addressing = false;
#endif

            if constexpr(use_amd_buffer_addressing)
            {
                constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

                amd_buffer_atomic_add<remove_cvref_t<T>, t_per_x>(
                    x, p_data_, i, is_valid_element, element_space_size_);
            }
            else
            {
                if(is_valid_element)
                {
                    atomic_add<X>(c_style_pointer_cast<X*>(&p_data_[i]), x);
                }
            }
        }

        template <typename X,
                  typename enable_if<is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                             typename scalar_type<remove_cvref_t<T>>::type>::value,
                                     bool>::type
                  = false>
        __device__ void AtomicMax(index_t i, bool is_valid_element, const X& x)
        {
            // X contains multiple T
            constexpr index_t scalar_per_t_vector = scalar_type<remove_cvref_t<T>>::vector_size;

            constexpr index_t scalar_per_x_vector = scalar_type<remove_cvref_t<X>>::vector_size;

            static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                          "wrong! X should contain multiple T");

            static_assert(GetAddressSpace() == AddressSpaceEnum::Global, "only support global mem");

#if CK_USE_AMD_BUFFER_ATOMIC_MAX_FLOAT64
            using scalar_t = typename scalar_type<remove_cvref_t<T>>::type;
            bool constexpr use_amd_buffer_addressing = is_same_v<remove_cvref_t<scalar_t>, double>;
#else
            bool constexpr use_amd_buffer_addressing = false;
#endif

            if constexpr(use_amd_buffer_addressing)
            {
                constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

                amd_buffer_atomic_max<remove_cvref_t<T>, t_per_x>(
                    x, p_data_, i, is_valid_element, element_space_size_);
            }
            else if(is_valid_element)
            {
                atomic_max<X>(c_style_pointer_cast<X*>(&p_data_[i]), x);
            }
        }

        __device__ static constexpr bool IsStaticBuffer()
        {
            return false;
        }

        __device__ static constexpr bool IsDynamicBuffer()
        {
            return true;
        }
    };

    template <AddressSpaceEnum       BufferAddressSpace,
              AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence,
              typename T,
              typename InElementOp,
              typename ElementSpaceSize>
    __device__ constexpr auto
        make_hiptensor_dynamic_buffer(T* p, InElementOp in_op, ElementSpaceSize element_space_size)
    {
        return HiptensorDynamicBuffer<BufferAddressSpace,
                                      T,
                                      InElementOp,
                                      ElementSpaceSize,
                                      true,
                                      coherence>{p, in_op, element_space_size};
    }

    template <AddressSpaceEnum       BufferAddressSpace,
              AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence,
              typename T,
              typename InElementOp,
              typename ElementSpaceSize,
              typename X,
              typename enable_if<is_same<remove_cvref_t<T>, remove_cvref_t<X>>::value, bool>::type
              = false>
    __device__ constexpr auto make_hiptensor_dynamic_buffer(T*               p,
                                                            InElementOp      in_op,
                                                            ElementSpaceSize element_space_size,
                                                            X                invalid_element_value)
    {
        return HiptensorDynamicBuffer<BufferAddressSpace,
                                      T,
                                      InElementOp,
                                      ElementSpaceSize,
                                      false,
                                      coherence>{
            p, in_op, element_space_size, invalid_element_value};
    }

} // namespace ck
