/******************************************************************************.
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2025-2026, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIBCUB_ROCPRIM_DEVICE_DEVICE_TRANSFORM_HPP_
#define HIBCUB_ROCPRIM_DEVICE_DEVICE_TRANSFORM_HPP_

#include <hip/hip_runtime.h>

#include "../../../config.hpp"
#include "../../../tuple.hpp"
#include "../detail/util_env.hpp"

#include <rocprim/device/device_transform.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/discard_iterator.hpp>
#include <rocprim/iterator/zip_iterator.hpp>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{

struct TransformStableAddrsImpl
{
    template<typename... RandomAccessIteratorsIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp>
    HIPCUB_RUNTIME_FUNCTION
    hipError_t doit(hipcub::tuple<RandomAccessIteratorsIn...> inputs,
                    RandomAccessIteratorOut                   output,
                    NumItemsT                                 num_items,
                    TransformOp                               transform_op,
                    hipStream_t                               stream = nullptr)
    {
        auto input_it = rocprim::make_zip_iterator(inputs);

        return rocprim::transform(
            rocprim::counting_iterator<size_t>{},
            rocprim::discard_iterator{},
            num_items,
            [transform_op, input_it, output](size_t offset)
            {
                output[offset]
                    = std::move(rocprim::apply(transform_op, std::move(input_it[offset])));
                return rocprim::empty_type{};
            },
            stream);
    }
};

template<typename T>
struct constant_generator
{
    T value;

    ROCPRIM_DEVICE ROCPRIM_INLINE T operator()() const
    {
        return value;
    }
};
} // namespace detail

struct DeviceTransform
{
    template<typename... RandomAccessIteratorsIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp,
             typename EnvT = _HIPCUB_STD_EXEC::env<>>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Transform(hipcub::tuple<RandomAccessIteratorsIn...> inputs,
                                RandomAccessIteratorOut                   output,
                                NumItemsT                                 num_items,
                                TransformOp                               transform_op,
                                EnvT                                      env = {})
    {
        return detail::env_invoke(
            env,
            [&](void* /*d_temp_storage*/, size_t& /*temp_storage_bytes*/, auto stream)
            {
                return ::rocprim::transform(inputs,
                                            output,
                                            num_items,
                                            transform_op,
                                            stream,
                                            HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
            });
    }

    template<typename... RandomAccessIteratorsIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Transform([[maybe_unused]] void*                    d_temp_storage,
                                [[maybe_unused]] size_t&                  temp_storage_bytes,
                                hipcub::tuple<RandomAccessIteratorsIn...> inputs,
                                RandomAccessIteratorOut                   output,
                                NumItemsT                                 num_items,
                                TransformOp                               transform_op,
                                hipStream_t                               stream = nullptr)
    {
        return ::rocprim::transform(inputs,
                                    output,
                                    num_items,
                                    transform_op,
                                    stream,
                                    HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename RandomAccessIteratorIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp,
             typename EnvT = _HIPCUB_STD_EXEC::env<>>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Transform(RandomAccessIteratorIn  input,
                                RandomAccessIteratorOut output,
                                NumItemsT               num_items,
                                TransformOp             transform_op,
                                EnvT                    env = {})
    {
        return detail::env_invoke(
            env,
            [&](void* /*d_temp_storage*/, size_t& /*temp_storage_bytes*/, auto stream)
            {
                return ::rocprim::transform(input,
                                            output,
                                            num_items,
                                            transform_op,
                                            stream,
                                            HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
            });
    }

    template<typename RandomAccessIteratorIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t Transform([[maybe_unused]] void*   d_temp_storage,
                                [[maybe_unused]] size_t& temp_storage_bytes,
                                RandomAccessIteratorIn   input,
                                RandomAccessIteratorOut  output,
                                NumItemsT                num_items,
                                TransformOp              transform_op,
                                hipStream_t              stream = nullptr)
    {
        return ::rocprim::transform(input,
                                    output,
                                    num_items,
                                    transform_op,
                                    stream,
                                    HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename Generator,
             typename EnvT = _HIPCUB_STD_EXEC::env<>>
   HIPCUB_RUNTIME_FUNCTION
    static hipError_t Generate(RandomAccessIteratorOut output,
                               NumItemsT               num_items,
                               Generator               generator,
                               EnvT                    env = {})
    {
        return detail::env_invoke(
            env,
            [&](void* /*d_temp_storage*/, size_t& /*temp_storage_bytes*/, auto stream)
            { return ::rocprim::generate(output, num_items, generator, stream); });
    }

    template<typename RandomAccessIteratorOut, typename NumItemsT, typename Generator>
   HIPCUB_RUNTIME_FUNCTION
    static hipError_t Generate([[maybe_unused]] void*   d_temp_storage,
                               [[maybe_unused]] size_t& temp_storage_bytes,
                               RandomAccessIteratorOut  output,
                               NumItemsT                num_items,
                               Generator                generator,
                               hipStream_t              stream = nullptr)
    {
        return ::rocprim::generate(output, num_items, generator, stream);
    }

    template<typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename Value,
             typename EnvT = _HIPCUB_STD_EXEC::env<>>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t
        Fill(RandomAccessIteratorOut output, NumItemsT num_items, Value value, EnvT env = {})
    {
        return detail::env_invoke(
            env,
            [&](void* /*d_temp_storage*/, size_t& /*temp_storage_bytes*/, auto stream)
            {
                return ::rocprim::generate(output,
                                           num_items,
                                           detail::constant_generator<Value>{value},
                                           stream);
            });
    }

    template<typename RandomAccessIteratorOut, typename NumItemsT, typename Value>
  HIPCUB_RUNTIME_FUNCTION
    static hipError_t Fill([[maybe_unused]] void*   d_temp_storage,
                           [[maybe_unused]] size_t& temp_storage_bytes,
                           RandomAccessIteratorOut  output,
                           NumItemsT                num_items,
                           Value                    value,
                           hipStream_t              stream = nullptr)
    {
        return Generate(output, num_items, detail::constant_generator<Value>{value}, stream);
    }

    template<typename... RandomAccessIteratorsIn,
             typename RandomAccessIteratorOut,
             typename Predicate,
             typename NumItemsT,
             typename TransformOp,
             typename EnvT = _HIPCUB_STD_EXEC::env<>>
   HIPCUB_RUNTIME_FUNCTION
    static hipError_t TransformIf(hipcub::tuple<RandomAccessIteratorsIn...> inputs,
                                  RandomAccessIteratorOut                   output,
                                  NumItemsT                                 num_items,
                                  Predicate                                 predicate,
                                  TransformOp                               transform_op,
                                  EnvT                                      env = {})
    {
        return detail::env_invoke(
            env,
            [&](void* /*d_temp_storage*/, size_t& /*temp_storage_bytes*/, auto stream) {
                return ::rocprim::transform(inputs,
                                            output,
                                            num_items,
                                            predicate,
                                            transform_op,
                                            stream);
            });
    }

    template<typename... RandomAccessIteratorsIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename Predicate,
             typename TransformOp>
  HIPCUB_RUNTIME_FUNCTION
    static hipError_t TransformIf([[maybe_unused]] void*                    d_temp_storage,
                                  [[maybe_unused]] size_t&                  temp_storage_bytes,
                                  hipcub::tuple<RandomAccessIteratorsIn...> inputs,
                                  RandomAccessIteratorOut                   output,
                                  NumItemsT                                 num_items,
                                  Predicate                                 predicate,
                                  TransformOp                               transform_op,
                                  hipStream_t                               stream = nullptr)
    {
        return ::rocprim::transform(inputs, output, num_items, predicate, transform_op, stream);
    }

    template<typename RandomAccessIteratorsIn,
             typename RandomAccessIteratorOut,
             typename Predicate,
             typename NumItemsT,
             typename TransformOp,
             typename EnvT = _HIPCUB_STD_EXEC::env<>>
   HIPCUB_RUNTIME_FUNCTION
    static hipError_t TransformIf(RandomAccessIteratorsIn input,
                                  RandomAccessIteratorOut output,
                                  NumItemsT               num_items,
                                  Predicate               predicate,
                                  TransformOp             transform_op,
                                  EnvT                    env = {})
    {
        return detail::env_invoke(
            env,
            [&](void* /*d_temp_storage*/, size_t& /*temp_storage_bytes*/, auto stream) {
                return ::rocprim::transform(input,
                                            output,
                                            num_items,
                                            predicate,
                                            transform_op,
                                            stream);
            });
    }

    template<typename RandomAccessIteratorsIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename Predicate,
             typename TransformOp>
  HIPCUB_RUNTIME_FUNCTION
    static hipError_t TransformIf([[maybe_unused]] void*   d_temp_storage,
                                  [[maybe_unused]] size_t& temp_storage_bytes,
                                  RandomAccessIteratorsIn  input,
                                  RandomAccessIteratorOut  output,
                                  NumItemsT                num_items,
                                  Predicate                predicate,
                                  TransformOp              transform_op,
                                  hipStream_t              stream = nullptr)
    {
        return ::rocprim::transform(input, output, num_items, predicate, transform_op, stream);
    }

    template<typename... RandomAccessIteratorsIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp,
             typename EnvT = _HIPCUB_STD_EXEC::env<>>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t
        TransformStableArgumentAddresses(hipcub::tuple<RandomAccessIteratorsIn...> inputs,
                                         RandomAccessIteratorOut                   output,
                                         NumItemsT                                 num_items,
                                         TransformOp                               transform_op,
                                         EnvT                                      env = {})
    {
        return detail::env_invoke(
            env,
            [&](void* /*d_temp_storage*/, size_t& /*temp_storage_bytes*/, auto stream)
            {
                return ::rocprim::transform(inputs,
                                            output,
                                            num_items,
                                            transform_op,
                                            stream,
                                            HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
            });
    }

    template<typename... RandomAccessIteratorsIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t
        TransformStableArgumentAddresses([[maybe_unused]] void*   d_temp_storage,
                                         [[maybe_unused]] size_t& temp_storage_bytes,
                                         hipcub::tuple<RandomAccessIteratorsIn...> inputs,
                                         RandomAccessIteratorOut                   output,
                                         NumItemsT                                 num_items,
                                         TransformOp                               transform_op,
                                         hipStream_t                               stream = nullptr)
    {
        return detail::TransformStableAddrsImpl{}.doit(inputs,
                                                       output,
                                                       num_items,
                                                       transform_op,
                                                       stream);
    }

    template<typename RandomAccessIteratorIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp,
             typename EnvT = _HIPCUB_STD_EXEC::env<>>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t TransformStableArgumentAddresses(RandomAccessIteratorIn  input,
                                                       RandomAccessIteratorOut output,
                                                       NumItemsT               num_items,
                                                       TransformOp             transform_op,
                                                       EnvT                    env = {})
    {
        return detail::env_invoke(
            env,
            [&](void* /*d_temp_storage*/, size_t& /*temp_storage_bytes*/, auto stream)
            {
                return detail::TransformStableAddrsImpl{}.doit(rocprim::make_tuple(input),
                                                               output,
                                                               num_items,
                                                               transform_op,
                                                               stream);
            });
    }

    template<typename RandomAccessIteratorIn,
             typename RandomAccessIteratorOut,
             typename NumItemsT,
             typename TransformOp>
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t TransformStableArgumentAddresses([[maybe_unused]] void*   d_temp_storage,
                                                       [[maybe_unused]] size_t& temp_storage_bytes,
                                                       RandomAccessIteratorIn   input,
                                                       RandomAccessIteratorOut  output,
                                                       NumItemsT                num_items,
                                                       TransformOp              transform_op,
                                                       hipStream_t              stream = nullptr)
    {
        return detail::TransformStableAddrsImpl{}.doit(rocprim::make_tuple(input),
                                                       output,
                                                       num_items,
                                                       transform_op,
                                                       stream);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIBCUB_ROCPRIM_DEVICE_DEVICE_TRANSFORM_HPP_
