// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_RNG_THREEFRY_H_
#define ROCRAND_RNG_THREEFRY_H_

// IWYU pragma: begin_keep
#include "config/threefry2_32_20_config.hpp"
#include "config/threefry2_64_20_config.hpp"
#include "config/threefry4_32_20_config.hpp"
#include "config/threefry4_64_20_config.hpp"
// IWYU pragma: end_keep

#include "common.hpp"
#include "config_types.hpp"
#include "distributions.hpp"
#include "generator_type.hpp"
#include "system.hpp"
#include "utils/cpp_utils.hpp"

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_threefry2x32_20.h>
#include <rocrand/rocrand_threefry2x64_20.h>
#include <rocrand/rocrand_threefry4x32_20.h>
#include <rocrand/rocrand_threefry4x64_20.h>

#include <hip/hip_runtime.h>

#include <type_traits>

namespace rocrand_impl::host
{

template<class BaseType>
struct threefry_device_engine : public BaseType
{
    using base_type   = BaseType;
    using vector_type = typename base_type::state_vector_type;
    using scalar_type = cpp_utils::vector_element_t<vector_type>;
    using state_type  = typename base_type::state_type;
    static inline constexpr unsigned int vector_dim
        = static_cast<unsigned int>(cpp_utils::vector_size_v<vector_type>);

    __forceinline__ __device__ __host__ threefry_device_engine() {}

    __forceinline__ __device__ __host__ threefry_device_engine(const unsigned long long seed,
                                                               const unsigned long long subsequence,
                                                               const unsigned long long offset)
        : base_type(seed, subsequence, offset)
    {}

    __forceinline__ __device__ __host__ vector_type next_leap(unsigned int leap)
    {
        vector_type ret = this->m_state.result;
        if(this->m_state.substate > 0)
        {
            const vector_type next_counter = this->bump_counter(this->m_state.counter);
            const vector_type next         = this->threefry_rounds(next_counter, this->m_state.key);
            ret                            = this->interleave(ret, next);
        }

        this->discard_state(leap);
        this->m_state.result = this->threefry_rounds(this->m_state.counter, this->m_state.key);
        return ret;
    }

    // m_state from base class
};

template<class Engine, class T, class Distribution>
__host__ __device__ __forceinline__ void generate_threefry(dim3         block_idx,
                                                           dim3         thread_idx,
                                                           dim3         grid_dim,
                                                           dim3         block_dim,
                                                           Engine       engine,
                                                           T*           data,
                                                           const size_t n,
                                                           Distribution distribution)
{
    using engine_scalar_type = typename Engine::scalar_type;

    constexpr unsigned int input_width  = Distribution::input_width;
    constexpr unsigned int output_width = Distribution::output_width;
    constexpr unsigned int vector_dim   = Engine::vector_dim;

    static_assert(vector_dim % input_width == 0 && input_width <= vector_dim,
                  "Incorrect input_width");
    constexpr unsigned int output_per_thread = vector_dim / input_width;
    constexpr unsigned int full_output_width = output_per_thread * output_width;

    using vec_type = aligned_vec_type<T, output_per_thread * output_width>;

    const unsigned int thread_id = block_idx.x * block_dim.x + thread_idx.x;
    const unsigned int stride    = grid_dim.x * block_dim.x;

    engine_scalar_type input[input_width];
    T                  output[output_per_thread][output_width];

    const uintptr_t uintptr = reinterpret_cast<uintptr_t>(data);
    const size_t    misalignment
        = (full_output_width - uintptr / sizeof(T) % full_output_width) % full_output_width;
    const unsigned int head_size = cpp_utils::min(n, misalignment);
    const unsigned int tail_size = (n - head_size) % full_output_width;
    const size_t       vec_n     = (n - head_size) / full_output_width;

    const unsigned int engine_offset
        = vector_dim * thread_id + (thread_id == 0 ? 0 : head_size / output_width * input_width);
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
        const auto             v = engine.next_leap(stride);
        cpp_utils::vec_wrapper vs(v);
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

template<class System, class Engine, class ConfigProvider>
class threefry_generator_template : public generator_impl_base
{
public:
    using base_type   = generator_impl_base;
    using engine_type = Engine;
    using scalar_type = typename engine_type::scalar_type;
    using system_type = System;
    using poisson_distribution_manager_t
        = poisson_distribution_manager<DISCRETE_METHOD_ALIAS, system_type>;
    using poisson_distribution_t = typename poisson_distribution_manager_t::distribution_t;
    using poisson_approx_distribution_t =
        typename poisson_distribution_manager_t::approx_distribution_t;

    threefry_generator_template(unsigned long long seed   = 0,
                                unsigned long long offset = 0,
                                rocrand_ordering   order  = ROCRAND_ORDERING_PSEUDO_DEFAULT,
                                hipStream_t        stream = 0)
        : base_type(order, offset, stream), m_seed(seed)
    {}

    static constexpr rocrand_rng_type type()
    {
        if constexpr(engine_type::vector_dim == 2)
        {
            if constexpr(std::is_same_v<unsigned int, scalar_type>)
            {
                return ROCRAND_RNG_PSEUDO_THREEFRY2_32_20;
            }
            else
            {
                return ROCRAND_RNG_PSEUDO_THREEFRY2_64_20;
            }
        }
        else
        {
            if constexpr(std::is_same_v<unsigned int, scalar_type>)
            {
                return ROCRAND_RNG_PSEUDO_THREEFRY4_32_20;
            }
            else
            {
                return ROCRAND_RNG_PSEUDO_THREEFRY4_64_20;
            }
        }
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
        {
            return ROCRAND_STATUS_SUCCESS;
        }

        m_engine = engine_type{m_seed, 0, m_offset};

        rocrand_status status = m_poisson.init();
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        m_engines_initialized = true;
        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T, class Distribution = uniform_distribution<T, scalar_type>>
    rocrand_status generate(T* data, size_t data_size, Distribution distribution = Distribution())
    {
        if constexpr(std::is_same_v<T, unsigned long long int>
                     && std::is_same_v<scalar_type, unsigned int>)
        {
            // Cannot generate 64-bit values with this generator.
            return ROCRAND_STATUS_TYPE_ERROR;
        }
        else
        {
            rocrand_status status = init();
            if(status != ROCRAND_STATUS_SUCCESS)
            {
                return status;
            }

            generator_config                       config;
            const hipError_t                       error
                = ConfigProvider::template host_config<T>(m_stream, m_order, config);
            if(error != hipSuccess)
            {
                return ROCRAND_STATUS_INTERNAL_ERROR;
            }

            if(data == nullptr)
            {
                return ROCRAND_STATUS_SUCCESS;
            }

            status = dynamic_dispatch(m_order,
                                      [&, this](auto is_dynamic)
                                      {
                                          return system_type::template launch<
                                              generate_threefry<engine_type, T, Distribution>,
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

            // Check kernel status
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
    }

    rocrand_status generate(unsigned long long int* data, size_t data_size)
    {
        if constexpr(std::is_same_v<scalar_type, unsigned int>)
        {
            // Cannot generate 64-bit values with this generator.
            return ROCRAND_STATUS_TYPE_ERROR;
        }
        uniform_distribution<unsigned long long int, unsigned long long int> distribution;
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_uniform(T* data, size_t data_size)
    {
        uniform_distribution<T, scalar_type> distribution;
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_normal(T* data, size_t data_size, T mean, T stddev)
    {
        constexpr unsigned int input_width = normal_distribution_max_input_width<type(), T>;
        normal_distribution<T, scalar_type, input_width> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_log_normal(T* data, size_t data_size, T mean, T stddev)
    {
        constexpr unsigned int input_width = log_normal_distribution_max_input_width<type(), T>;
        log_normal_distribution<T, scalar_type, input_width> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_poisson(T* data, size_t data_size, double lambda)
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

template<>
constexpr inline unsigned int
    normal_distribution_max_input_width<ROCRAND_RNG_PSEUDO_THREEFRY2_32_20, double>
    = 2;

template<>
constexpr inline unsigned int
    log_normal_distribution_max_input_width<ROCRAND_RNG_PSEUDO_THREEFRY2_32_20, double>
    = 2;

using threefry2x32_20_generator
    = threefry_generator_template<system::device_system,
                                  threefry_device_engine<rocrand_device::threefry2x32_20_engine>,
                                  default_config_provider<ROCRAND_RNG_PSEUDO_THREEFRY2_32_20>>;

template<bool UseHostFunc>
using threefry2x32_20_generator_host
    = threefry_generator_template<system::host_system<UseHostFunc>,
                                  threefry_device_engine<rocrand_device::threefry2x32_20_engine>,
                                  default_config_provider<ROCRAND_RNG_PSEUDO_THREEFRY2_32_20>>;

using threefry2x64_20_generator
    = threefry_generator_template<system::device_system,
                                  threefry_device_engine<rocrand_device::threefry2x64_20_engine>,
                                  default_config_provider<ROCRAND_RNG_PSEUDO_THREEFRY2_64_20>>;

template<bool UseHostFunc>
using threefry2x64_20_generator_host
    = threefry_generator_template<system::host_system<UseHostFunc>,
                                  threefry_device_engine<rocrand_device::threefry2x64_20_engine>,
                                  default_config_provider<ROCRAND_RNG_PSEUDO_THREEFRY2_64_20>>;

using threefry4x32_20_generator
    = threefry_generator_template<system::device_system,
                                  threefry_device_engine<rocrand_device::threefry4x32_20_engine>,
                                  default_config_provider<ROCRAND_RNG_PSEUDO_THREEFRY4_32_20>>;

template<bool UseHostFunc>
using threefry4x32_20_generator_host
    = threefry_generator_template<system::host_system<UseHostFunc>,
                                  threefry_device_engine<rocrand_device::threefry4x32_20_engine>,
                                  default_config_provider<ROCRAND_RNG_PSEUDO_THREEFRY4_32_20>>;

using threefry4x64_20_generator
    = threefry_generator_template<system::device_system,
                                  threefry_device_engine<rocrand_device::threefry4x64_20_engine>,
                                  default_config_provider<ROCRAND_RNG_PSEUDO_THREEFRY4_64_20>>;

template<bool UseHostFunc>
using threefry4x64_20_generator_host
    = threefry_generator_template<system::host_system<UseHostFunc>,
                                  threefry_device_engine<rocrand_device::threefry4x64_20_engine>,
                                  default_config_provider<ROCRAND_RNG_PSEUDO_THREEFRY4_64_20>>;

} // namespace rocrand_impl::host

#endif // ROCRAND_RNG_THREEFRY2X32_20_H_
