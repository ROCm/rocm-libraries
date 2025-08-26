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
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef ROCRAND_RNG_PHILOX4X32_10_H_
#define ROCRAND_RNG_PHILOX4X32_10_H_

// IWYU pragma: begin_keep
#include "config/philox4_32_10_config.hpp"
// IWYU pragma: end_keep

#include "common.hpp"
#include "config_types.hpp"
#include "distributions.hpp"
#include "generator_type.hpp"
#include "system.hpp"
#include "utils/cpp_utils.hpp"

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_philox4x32_10.h>

#include <hip/hip_runtime.h>

#include <algorithm>

namespace rocrand_impl::host
{
struct philox4x32_10_device_engine : public ::rocrand_device::philox4x32_10_engine
{
    typedef ::rocrand_device::philox4x32_10_engine base_type;
    typedef base_type::philox4x32_10_state         state_type;

    __forceinline__ philox4x32_10_device_engine() = default;

    __forceinline__ __device__ __host__
        philox4x32_10_device_engine(const unsigned long long seed,
                                    const unsigned long long subsequence,
                                    const unsigned long long offset)
        : base_type(seed, subsequence, offset)
    {}

    __forceinline__ __device__ __host__ uint4 next4_leap(unsigned int leap)
    {
        uint4 ret = m_state.result;
        if(m_state.substate > 0)
        {
            const uint4 next_counter = this->bump_counter(m_state.counter);
            const uint4 next         = this->ten_rounds(next_counter, m_state.key);
            ret                      = this->interleave(ret, next);
        }

        this->discard_state(leap);
        m_state.result = this->ten_rounds(m_state.counter, m_state.key);
        return ret;
    }

    // m_state from base class
};

template<typename T, typename Distribution>
__host__ __device__ __forceinline__ void generate_philox(dim3                        block_idx,
                                                         dim3                        thread_idx,
                                                         dim3                        grid_dim,
                                                         dim3                        block_dim,
                                                         philox4x32_10_device_engine engine,
                                                         T*                          data,
                                                         const size_t                n,
                                                         Distribution                distribution)
{
    constexpr unsigned int input_width  = Distribution::input_width;
    constexpr unsigned int output_width = Distribution::output_width;

    static_assert(4 % input_width == 0 && input_width <= 4, "Incorrect input_width");
    constexpr unsigned int output_per_thread = 4 / input_width;
    constexpr unsigned int full_output_width = output_per_thread * output_width;

    using vec_type = aligned_vec_type<T, output_per_thread * output_width>;

    const unsigned int thread_id = block_idx.x * block_dim.x + thread_idx.x;
    const unsigned int stride    = grid_dim.x * block_dim.x;

    unsigned int input[input_width];
    T            output[output_per_thread][output_width];

    const uintptr_t uintptr = reinterpret_cast<uintptr_t>(data);
    const size_t    misalignment
        = (full_output_width - uintptr / sizeof(T) % full_output_width) % full_output_width;
    const unsigned int head_size = cpp_utils::min(n, misalignment);
    const unsigned int tail_size = (n - head_size) % full_output_width;
    const size_t       vec_n     = (n - head_size) / full_output_width;

    const unsigned int engine_offset = 4 * thread_id + (thread_id == 0 ? 0 : head_size);
    engine.discard(engine_offset);

    // If data is not aligned by sizeof(vec_type)
    if(thread_id == 0 && head_size > 0)
    {
        for(unsigned int s = 0; s < output_per_thread; ++s)
        {
            if(s * output_width >= head_size)
            {
                break;
            }

            for(unsigned int i = 0; i < input_width; ++i)
            {
                input[i] = engine();
            }
            distribution(input, output[s]);

            for(unsigned int o = 0; o < output_width; ++o)
            {
                if(s * output_width + o < head_size)
                {
                    data[s * output_width + o] = output[s][o];
                }
            }
        }
    }

    // Save multiple values as one vec_type
    vec_type* vec_data = reinterpret_cast<vec_type*>(data + misalignment);
    size_t    index    = thread_id;
    while(index < vec_n)
    {
        const uint4        v     = engine.next4_leap(stride);
        const unsigned int vs[4] = {v.x, v.y, v.z, v.w};
        for(unsigned int s = 0; s < output_per_thread; s++)
        {
            for(unsigned int i = 0; i < input_width; i++)
            {
                input[i] = vs[s * input_width + i];
            }
            distribution(input, output[s]);
        }
        vec_data[index] = *reinterpret_cast<vec_type*>(output);
        // Next position
        index += stride;
    }

    // Check if we need to save tail.
    // Those numbers should be generated by the thread that would
    // save next vec_type.
    if(index == vec_n && tail_size > 0)
    {
        for(unsigned int s = 0; s < output_per_thread; ++s)
        {
            if(s * output_width >= tail_size)
            {
                break;
            }

            for(unsigned int i = 0; i < input_width; ++i)
            {
                input[i] = engine();
            }
            distribution(input, output[s]);

            for(unsigned int o = 0; o < output_width; ++o)
            {
                if(s * output_width + o < tail_size)
                {
                    data[n - tail_size + s * output_width + o] = output[s][o];
                }
            }
        }
    }
}

template<typename System, typename ConfigProvider>
class philox4x32_10_generator_template : public generator_impl_base
{
public:
    using base_type   = generator_impl_base;
    using engine_type = philox4x32_10_device_engine;
    using system_type = System;
    using poisson_distribution_manager_t
        = poisson_distribution_manager<DISCRETE_METHOD_ALIAS, system_type>;
    using poisson_distribution_t = typename poisson_distribution_manager_t::distribution_t;
    using poisson_approx_distribution_t =
        typename poisson_distribution_manager_t::approx_distribution_t;

    philox4x32_10_generator_template(unsigned long long seed   = 0,
                                     unsigned long long offset = 0,
                                     rocrand_ordering   order  = ROCRAND_ORDERING_PSEUDO_DEFAULT,
                                     hipStream_t        stream = 0)
        : base_type(order, offset, stream), m_seed(seed)
    {}

    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_PSEUDO_PHILOX4_32_10;
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
        if(m_engines_initialized)
            return ROCRAND_STATUS_SUCCESS;

        m_engine = engine_type{m_seed, 0, m_offset};

        rocrand_status status = m_poisson.init();
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        m_engines_initialized = true;
        return ROCRAND_STATUS_SUCCESS;
    }

    template<typename T, typename Distribution = uniform_distribution<T>>
    rocrand_status generate(T* data, size_t data_size, Distribution distribution = Distribution())
    {
        rocrand_status status = init();
        if(status != ROCRAND_STATUS_SUCCESS)
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

        status = dynamic_dispatch(
            m_order,
            [&, this](auto is_dynamic)
            {
                return system_type::template launch<generate_philox<T, Distribution>,
                                                    ConfigProvider,
                                                    T,
                                                    is_dynamic>(dim3(config.blocks),
                                                                dim3(config.threads),
                                                                0,
                                                                m_stream,
                                                                m_engine,
                                                                data,
                                                                data_size,
                                                                distribution);
            });
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        // Generating data_size values will use this many distributions
        const auto num_applied_generators = (data_size + Distribution::output_width - 1)
                                            / Distribution::output_width
                                            * Distribution::input_width;

        m_engine.discard(num_applied_generators);

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
    rocrand_status generate_uniform(T* data, size_t data_size)
    {
        uniform_distribution<T> distribution;
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_normal(T* data, size_t data_size, T mean, T stddev)
    {
        normal_distribution<T> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_log_normal(T* data, size_t data_size, T mean, T stddev)
    {
        log_normal_distribution<T> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    rocrand_status generate_poisson(unsigned int* data, size_t data_size, double lambda)
    {
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
    bool        m_engines_initialized = false;
    engine_type m_engine;

    unsigned long long m_seed;

    // For caching of Poisson for consecutive generations with the same lambda
    poisson_distribution_manager_t m_poisson;

    // m_seed from base_type
    // m_offset from base_type
};

using philox4x32_10_generator
    = philox4x32_10_generator_template<system::device_system,
                                       default_config_provider<ROCRAND_RNG_PSEUDO_PHILOX4_32_10>>;

template<bool UseHostFunc>
using philox4x32_10_generator_host
    = philox4x32_10_generator_template<system::host_system<UseHostFunc>,
                                       default_config_provider<ROCRAND_RNG_PSEUDO_PHILOX4_32_10>>;

} // namespace rocrand_impl::host

#endif // ROCRAND_RNG_PHILOX4X32_10_H_
