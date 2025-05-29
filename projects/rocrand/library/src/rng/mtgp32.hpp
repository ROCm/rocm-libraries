// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

/*
 * Copyright (c) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and Hiroshima
 * University.  All rights reserved.
 * Copyright (c) 2011 Mutsuo Saito, Makoto Matsumoto, Hiroshima
 * University and University of Tokyo.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *     * Neither the name of the Hiroshima University nor the names of
 *       its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ROCRAND_RNG_MTGP32_H_
#define ROCRAND_RNG_MTGP32_H_

// IWYU pragma: begin_keep
#include "config/mtgp32_config.hpp"
// IWYU pragma: end_keep

#include "common.hpp"
#include "config_types.hpp"
#include "distributions.hpp"
#include "generator_type.hpp"
#include "system.hpp"
#include "utils/cpp_utils.hpp"

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_mtgp32.h>
#include <rocrand/rocrand_mtgp32_11213.h>

#include <hip/hip_runtime.h>

#include <algorithm>

namespace rocrand_impl::host
{

struct mtgp32_device_engine : ::rocrand_device::mtgp32_engine
{
    // suppress warning about no initialization for __shared__ variables
    __host__ __device__ mtgp32_device_engine(){};

    __forceinline__ __host__ __device__
    unsigned int next()
    {
#ifdef __HIP_DEVICE_COMPILE__
        // all threads in block produce one value and advance the state by that many values
        return ::rocrand_device::mtgp32_engine::next();
#else
        // produce one value and advance the state by one value
        const unsigned int o = next_thread(0);
        m_state.offset       = (++m_state.offset) & MTGP_MASK;
        return o;
#endif
    }
};

template<class T, class Distribution, unsigned int BlockSize>
__host__ void generate(unsigned int (&input)[BlockSize][Distribution::input_width],
                       T (&output)[BlockSize][Distribution::output_width],
                       Distribution&         distribution,
                       mtgp32_device_engine& engine)
{
    for(unsigned int i = 0; i < Distribution::input_width; i++)
    {
        for(unsigned int j = 0; j < BlockSize; j++)
        {
            input[j][i] = engine.next();
        }
    }
    for(unsigned int j = 0; j < BlockSize; j++)
    {
        distribution(input[j], output[j]);
    }
}

template<class T, class Distribution>
__forceinline__ __device__
void generate(unsigned int (&input)[Distribution::input_width],
              T (&output)[Distribution::output_width],
              Distribution&         distribution,
              mtgp32_device_engine& engine)
{
    for(unsigned int i = 0; i < Distribution::input_width; i++)
    {
        input[i] = engine.next();
    }
    distribution(input, output);
}

template<class vec_type, class T, unsigned int output_width, unsigned int BlockSize>
__host__ void save_vec_n(vec_type* vec_data, T (&output)[BlockSize][output_width], size_t index)
{
    for(unsigned int j = 0; j < BlockSize; j++)
    {
        vec_data[index + j] = *reinterpret_cast<vec_type*>(output[j]);
    }
}

template<class vec_type, class T, unsigned int output_width>
__forceinline__ __device__
void save_vec_n(vec_type* vec_data, T (&output)[output_width], size_t index)
{
    vec_data[index] = *reinterpret_cast<vec_type*>(output);
}

template<class vec_type, class T, unsigned int output_width, unsigned int BlockSize>
__host__ void
    save_n(vec_type* vec_data, T (&output)[BlockSize][output_width], size_t index, size_t vec_n)
{
    for(unsigned int j = 0; j < BlockSize; j++)
    {
        if(index + j < vec_n)
        {
            vec_data[index + j] = *reinterpret_cast<vec_type*>(output[j]);
        }
    }
}

template<class vec_type, class T, unsigned int output_width>
__forceinline__ __device__
void save_n(vec_type* vec_data, T (&output)[output_width], size_t index, size_t vec_n)
{
    if(index < vec_n)
    {
        vec_data[index] = *reinterpret_cast<vec_type*>(output);
    }
}

template<class T, unsigned int output_width>
__forceinline__ __host__ __device__
void save_head_tail_impl(T (&output)[output_width],
                         size_t index,
                         T*     data,
                         size_t n,
                         size_t head_size,
                         size_t tail_size,
                         size_t vec_n_up)
{
    if(index == vec_n_up)
    {
        for(unsigned int o = 0; o < output_width; o++)
        {
            if(o < head_size)
            {
                data[o] = output[o];
            }
        }
    }

    if(index == vec_n_up + 1)
    {
        for(unsigned int o = 0; o < output_width; o++)
        {
            if(o < tail_size)
            {
                data[n - tail_size + o] = output[o];
            }
        }
    }
}

template<class T, unsigned int output_width, unsigned int BlockSize>
__host__ void save_head_tail(T (&output)[BlockSize][output_width],
                             size_t index,
                             T*     data,
                             size_t n,
                             size_t head_size,
                             size_t tail_size,
                             size_t vec_n_up)
{
    for(unsigned int j = 0; j < BlockSize; j++)
    {
        save_head_tail_impl(output[j], index + j, data, n, head_size, tail_size, vec_n_up);
    }
}

template<class T, unsigned int output_width>
__forceinline__ __device__
void save_head_tail(T (&output)[output_width],
                    size_t index,
                    T*     data,
                    size_t n,
                    size_t head_size,
                    size_t tail_size,
                    size_t vec_n_up)
{
    save_head_tail_impl(output, index, data, n, head_size, tail_size, vec_n_up);
}

template<class ConfigProvider, bool IsDynamic, class T, class Distribution>
__host__ __device__ __forceinline__ void generate_mtgp(dim3 block_idx,
                                       dim3 thread_idx,
                                       dim3 grid_dim,
                                       dim3 /*block_dim*/,
                                       mtgp32_device_engine* engines,
                                       T*                    data,
                                       const size_t          n,
                                       Distribution          distribution)
{
    static_assert(is_single_tile_config<ConfigProvider, T>(IsDynamic),
                  "This kernel should only be used with single tile configs");
    constexpr unsigned int BlockSize    = get_block_size<ConfigProvider, T>(IsDynamic);
    constexpr unsigned int input_width  = Distribution::input_width;
    constexpr unsigned int output_width = Distribution::output_width;

    using vec_type = aligned_vec_type<T, output_width>;

    const unsigned int engine_id = block_idx.x;
    const unsigned int stride    = grid_dim.x * BlockSize;
    size_t             index     = block_idx.x * BlockSize + thread_idx.x;

// Load device engine
#ifdef __HIP_DEVICE_COMPILE__
    __shared__
#endif
        mtgp32_device_engine engine;
    engine.copy(&engines[engine_id]);

#ifdef __HIP_DEVICE_COMPILE__
    unsigned int input[input_width];
    T            output[output_width];
#else
    // Due to the lock-step-like behavior of the device generator, the first value of a distribution
    //   for thread i is i, the next value is i + BlockSize, etc. Hence, all values must be cached for the host generator.
    unsigned int input[BlockSize][input_width];
    T            output[BlockSize][output_width];
#endif

    const uintptr_t uintptr   = reinterpret_cast<uintptr_t>(data);
    const size_t misalignment = (output_width - uintptr / sizeof(T) % output_width) % output_width;
    const unsigned int head_size    = cpp_utils::min(n, misalignment);
    const unsigned int tail_size = (n - head_size) % output_width;
    const size_t       vec_n     = (n - head_size) / output_width;

    // vec_n rounded up and down to the nearest multiple of BlockSize
    const size_t remainder_value = vec_n % BlockSize;
    const size_t vec_n_down      = vec_n - remainder_value;
    const size_t vec_n_up        = remainder_value == 0 ? vec_n_down : (vec_n_down + BlockSize);

    vec_type* vec_data = reinterpret_cast<vec_type*>(data + misalignment);
    // Generate and store all aligned vector multiples
    while(index < vec_n_down)
    {
        generate(input, output, distribution, engine);
        save_vec_n(vec_data, output, index);
        index += stride;
    }
    // Generate and store all aligned vector multiples for which not all threads participate in storing
    if(index < vec_n_up)
    {
        generate(input, output, distribution, engine);
        save_n(vec_data, output, index, vec_n);
        index += stride;
    }
    // Generate and store the remaining T that are not aligned to vec_type
    if(output_width > 1 && (head_size > 0 || tail_size > 0))
    {
        generate(input, output, distribution, engine);
        save_head_tail(output, index, data, n, head_size, tail_size, vec_n_up);
    }

    // Save engine with its state
    engines[engine_id].copy(&engine);
}

template<class System, class ConfigProvider>
class mtgp32_generator_template : public generator_impl_base
{
public:
    using base_type   = generator_impl_base;
    using engine_type = mtgp32_device_engine;
    using system_type = System;
    using poisson_distribution_manager_t
        = poisson_distribution_manager<DISCRETE_METHOD_ALIAS, system_type>;
    using poisson_distribution_t = typename poisson_distribution_manager_t::distribution_t;
    using poisson_approx_distribution_t =
        typename poisson_distribution_manager_t::approx_distribution_t;

    mtgp32_generator_template(unsigned long long seed   = 0,
                              unsigned long long offset = 0,
                              rocrand_ordering   order  = ROCRAND_ORDERING_PSEUDO_DEFAULT,
                              hipStream_t        stream = 0)
        : base_type(order, offset, stream), m_seed(seed)
    {}

    mtgp32_generator_template(const mtgp32_generator_template&) = delete;

    mtgp32_generator_template(mtgp32_generator_template&& other)
        : base_type(other)
        , m_engines_initialized(other.m_engines_initialized)
        , m_engines(other.m_engines)
        , m_engines_size(other.m_engines_size)
        , m_seed(other.m_seed)
        , m_poisson(std::move(other.m_poisson))
    {
        other.m_engines_initialized = false;
        other.m_engines             = nullptr;
    }

    mtgp32_generator_template& operator=(const mtgp32_generator_template&) = delete;

    mtgp32_generator_template& operator=(mtgp32_generator_template&& other)
    {
        *static_cast<base_type*>(this) = other;
        m_engines_initialized          = other.m_engines_initialized;
        m_engines                      = other.m_engines;
        m_engines_size                 = other.m_engines_size;
        m_seed                         = other.m_seed;
        m_poisson                      = std::move(other.m_poisson);

        other.m_engines_initialized = false;
        other.m_engines             = nullptr;

        return *this;
    }

    ~mtgp32_generator_template()
    {
        if(m_engines != nullptr)
        {
            system_type::free(m_engines);
            m_engines = nullptr;
        }
    }

    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_PSEUDO_MTGP32;
    }

    void reset() override final
    {
        m_engines_initialized = false;
    }

    /// Changes seed to \p seed and resets generator state.
    void set_seed(unsigned long long seed)
    {
        m_seed = seed;
        reset();
    }

    unsigned long long get_seed() const
    {
        return m_seed;
    }

    rocrand_status set_offset(unsigned long long offset)
    {
        // Can't set offset for MTGP32
        (void)offset;
        return ROCRAND_STATUS_TYPE_ERROR;
    }

    rocrand_status set_order(rocrand_ordering order)
    {
        if(!system_type::is_device() && order == ROCRAND_ORDERING_PSEUDO_DYNAMIC)
        {
            return ROCRAND_STATUS_OUT_OF_RANGE;
        }
        static constexpr std::array supported_orderings{
            ROCRAND_ORDERING_PSEUDO_DEFAULT,
            ROCRAND_ORDERING_PSEUDO_DYNAMIC,
            ROCRAND_ORDERING_PSEUDO_BEST,
            ROCRAND_ORDERING_PSEUDO_LEGACY,
        };
        if(std::find(supported_orderings.begin(), supported_orderings.end(), order)
           == supported_orderings.end())
        {
            return ROCRAND_STATUS_OUT_OF_RANGE;
        }
        m_order = order;
        reset();
        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_status set_stream(hipStream_t stream)
    {
        const rocrand_status status = m_poisson.set_stream(stream);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }
        base_type::set_stream(stream);
        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_status init()
    {
        if (m_engines_initialized)
        {
            return ROCRAND_STATUS_SUCCESS;
        }

        generator_config config;
        // Assuming that the config is the same for every type.
        hipError_t error
            = ConfigProvider::template host_config<unsigned int>(m_stream, m_order, config);
        if(error != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }
        m_engines_size = config.blocks;

        if (m_engines_size > mtgpdc_params_11213_num)
        {
            return ROCRAND_STATUS_ALLOCATION_FAILED;
        }

        rocrand_status status = system_type::alloc(&m_engines, m_engines_size);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        status = rocrand_make_state_mtgp32(m_engines,
                                           mtgp32dc_params_fast_11213,
                                           m_engines_size,
                                           m_seed);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return ROCRAND_STATUS_ALLOCATION_FAILED;
        }

        status = m_poisson.init();
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        m_engines_initialized = true;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T, class Distribution = uniform_distribution<T> >
    rocrand_status generate(T * data, size_t data_size,
                            Distribution distribution = Distribution())
    {
        rocrand_status status = init();
        if (status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        generator_config config;
        const hipError_t error = ConfigProvider::template host_config<T>(m_stream, m_order, config);
        if(error != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }

        if(data == nullptr)
        {
            return ROCRAND_STATUS_SUCCESS;
        }

        // The host generator uses a block of size one to emulate a device generator that uses a shared memory state
        const dim3 threads
            = std::is_same_v<system_type, system::device_system> ? config.threads : dim3(1);
        status
            = dynamic_dispatch(m_order,
                               [&, this](auto is_dynamic)
                               {
                                   return system_type::template launch<
                                       generate_mtgp<ConfigProvider, is_dynamic, T, Distribution>,
                                       ConfigProvider,
                                       T,
                                       is_dynamic>(dim3(config.blocks),
                                                   dim3(threads),
                                                   0,
                                                   m_stream,
                                                   m_engines,
                                                   data,
                                                   data_size,
                                                   distribution);
                               });

        // Check kernel status
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_status generate(unsigned long long* data, size_t data_size)
    {
        // Cannot generate 64-bit values with this generator.
        (void)data;
        (void)data_size;
        return ROCRAND_STATUS_TYPE_ERROR;
    }

    template<typename Distribution>
    rocrand_status generate(unsigned long long* data, size_t data_size, Distribution distribution)
    {
        // Cannot generate 64-bit values with this generator.
        (void)data;
        (void)data_size;
        (void)distribution;
        return ROCRAND_STATUS_TYPE_ERROR;
    }

    template<class T>
    rocrand_status generate_uniform(T * data, size_t data_size)
    {
        uniform_distribution<T> distribution;
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_normal(T * data, size_t data_size, T mean, T stddev)
    {
        normal_distribution<T> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_log_normal(T * data, size_t data_size, T mean, T stddev)
    {
        log_normal_distribution<T> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    rocrand_status generate_poisson(unsigned int * data, size_t data_size, double lambda)
    {
        // For an unknown reason, on CUDA, the initialization of the engines must precede
        // the initialization of the poisson distribution, otherwise spurious miscalculations
        // occur
        if(!m_engines_initialized)
        {
            const auto status = init();
            if(status != ROCRAND_STATUS_SUCCESS)
            {
                return status;
            }
        }
        auto result = m_poisson.get_distribution(lambda);
        if(auto* dis = std::get_if<poisson_distribution_t>(&result))
        {
            return generate(data, data_size, *dis);
        }
        if(auto* dis = std::get_if<poisson_approx_distribution_t>(&result))
        {
            return generate(data, data_size, *dis);
        }
        return std::get<rocrand_status>(result);
    }

private:
    bool         m_engines_initialized = false;
    engine_type* m_engines             = nullptr;
    unsigned int m_engines_size        = false;

    unsigned long long m_seed;

    // For caching of Poisson for consecutive generations with the same lambda
    poisson_distribution_manager_t m_poisson;

    // m_seed from base_type
    // m_offset from base_type
};

using mtgp32_generator
    = mtgp32_generator_template<system::device_system,
                                default_config_provider<ROCRAND_RNG_PSEUDO_MTGP32>>;
template<bool UseHostFunc>
using mtgp32_generator_host
    = mtgp32_generator_template<system::host_system<UseHostFunc>,
                                default_config_provider<ROCRAND_RNG_PSEUDO_MTGP32>>;

} // namespace rocrand_impl::host

#endif // ROCRAND_RNG_MTGP32_H_
