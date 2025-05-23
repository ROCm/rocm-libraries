// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_RNG_MRG_H_
#define ROCRAND_RNG_MRG_H_

// IWYU pragma: begin_keep
#include "config/mrg31k3p_config.hpp"
#include "config/mrg32k3a_config.hpp"
// IWYU pragma: end_keep

#include "common.hpp"
#include "config_types.hpp"
#include "distributions.hpp"
#include "generator_type.hpp"
#include "system.hpp"
#include "utils/cpp_utils.hpp"

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_mrg31k3p.h>
#include <rocrand/rocrand_mrg32k3a.h>

#include <hip/hip_runtime.h>

#include <type_traits>

namespace rocrand_impl::host
{

template<class Engine>
__host__ __device__ void init_engines_mrg(dim3               block_idx,
                                          dim3               thread_idx,
                                          dim3               grid_dim,
                                          dim3               block_dim,
                                          Engine*            engines,
                                          const unsigned int start_engine_id,
                                          const unsigned int engines_size,
                                          unsigned long long seed,
                                          unsigned long long offset)
{
    (void)grid_dim;
    const unsigned int engine_id = block_idx.x * block_dim.x + thread_idx.x;
    if(engine_id < engines_size)
    {
        engines[engine_id]
            = Engine(seed, engine_id, offset + (engine_id < start_engine_id ? 1 : 0));
    }
}

template<class ConfigProvider, bool IsDynamic, class Engine, class T, class Distribution>
__host__ __device__ __forceinline__ void generate_mrg(dim3 block_idx,
                                      dim3 thread_idx,
                                      dim3 grid_dim,
                                      dim3 /*block_dim*/,
                                      Engine*            engines,
                                      const unsigned int start_engine_id,
                                      T*                 data,
                                      const size_t       n,
                                      Distribution       distribution)
{
    static_assert(is_single_tile_config<ConfigProvider, T>(IsDynamic),
                  "This kernel should only be used with single tile configs");
    constexpr unsigned int block_size   = get_block_size<ConfigProvider, T>(IsDynamic);
    constexpr unsigned int input_width  = Distribution::input_width;
    constexpr unsigned int output_width = Distribution::output_width;

    using vec_type = aligned_vec_type<T, output_width>;

    const unsigned int id     = block_idx.x * block_size + thread_idx.x;
    const unsigned int stride = grid_dim.x * block_size;

    const unsigned int engine_id = (id + start_engine_id) % stride;
    Engine             engine    = engines[engine_id];

    unsigned int input[input_width];
    T            output[output_width];

    const uintptr_t uintptr   = reinterpret_cast<uintptr_t>(data);
    const size_t misalignment = (output_width - uintptr / sizeof(T) % output_width) % output_width;
    const unsigned int head_size    = cpp_utils::min(n, misalignment);
    const unsigned int tail_size = (n - head_size) % output_width;
    const size_t       vec_n     = (n - head_size) / output_width;

    vec_type* vec_data = reinterpret_cast<vec_type*>(data + misalignment);
    size_t    index    = id;
    while(index < vec_n)
    {
        for(unsigned int i = 0; i < input_width; i++)
        {
            input[i] = engine();
        }
        distribution(input, output);

        vec_data[index] = *reinterpret_cast<vec_type*>(output);
        // Next position
        index += stride;
    }

    // Check if we need to save head and tail.
    // Those numbers should be generated by the thread that would
    // save next vec_type.
    if(output_width > 1 && index == vec_n)
    {
        // If data is not aligned by sizeof(vec_type)
        if(head_size > 0)
        {
            for(unsigned int i = 0; i < input_width; i++)
            {
                input[i] = engine();
            }
            distribution(input, output);

            for(unsigned int o = 0; o < output_width; o++)
            {
                if(o < head_size)
                {
                    data[o] = output[o];
                }
            }
        }

        if(tail_size > 0)
        {
            for(unsigned int i = 0; i < input_width; i++)
            {
                input[i] = engine();
            }
            distribution(input, output);

            for(unsigned int o = 0; o < output_width; o++)
            {
                if(o < tail_size)
                {
                    data[n - tail_size + o] = output[o];
                }
            }
        }
    }

    // Save engine with its state
    engines[engine_id] = engine;
}

template<typename System, typename Engine, typename ConfigProvider>
class mrg_generator_template : public generator_impl_base
{
public:
    using base_type   = generator_impl_base;
    using engine_type = Engine;
    using system_type = System;
    using poisson_distribution_manager_t
        = poisson_distribution_manager<DISCRETE_METHOD_ALIAS, system_type>;
    using poisson_distribution_t = typename poisson_distribution_manager_t::distribution_t;
    using poisson_approx_distribution_t =
        typename poisson_distribution_manager_t::approx_distribution_t;

    mrg_generator_template(unsigned long long seed   = 0,
                           unsigned long long offset = 0,
                           rocrand_ordering   order  = ROCRAND_ORDERING_PSEUDO_DEFAULT,
                           hipStream_t        stream = 0)
        : base_type(order, offset, stream), m_seed(seed)
    {
        if(m_seed == 0)
        {
            m_seed = get_default_seed();
        }
    }

    mrg_generator_template(const mrg_generator_template&) = delete;

    mrg_generator_template(mrg_generator_template&& other)
        : base_type(other)
        , m_engines_initialized(std::exchange(other.m_engines_initialized, false))
        , m_engines(std::exchange(other.m_engines, nullptr))
        , m_engines_size(other.m_engines_size)
        , m_start_engine_id(other.m_start_engine_id)
        , m_seed(other.m_seed)
        , m_poisson(std::move(other.m_poisson))
    {}

    mrg_generator_template& operator=(const mrg_generator_template&) = delete;

    mrg_generator_template& operator=(mrg_generator_template&& other)
    {
        *static_cast<base_type*>(this) = other;
        m_engines_initialized          = std::exchange(other.m_engines_initialized, false);
        m_engines                      = std::exchange(other.m_engines, nullptr);
        m_engines_size                 = other.m_engines_size;
        m_start_engine_id              = other.m_start_engine_id;
        m_seed                         = other.m_seed;
        m_poisson                      = std::move(other.m_poisson);

        return *this;
    }

    ~mrg_generator_template()
    {
        if(m_engines != nullptr)
        {
            system_type::free(m_engines);
        }
    }

    static constexpr rocrand_rng_type type()
    {
        if constexpr(std::is_same_v<engine_type, rocrand_device::mrg31k3p_engine>)
        {
            return ROCRAND_RNG_PSEUDO_MRG31K3P;
        }
        else if constexpr(std::is_same_v<engine_type, rocrand_device::mrg32k3a_engine>)
        {
            return ROCRAND_RNG_PSEUDO_MRG32K3A;
        }
    }

    void reset() override final
    {
        m_engines_initialized = false;
    }

    /// Changes seed to \p seed and resets generator state.
    ///
    /// New seed value should not be zero. If \p seed_value is equal
    /// zero, value \p ROCRAND_MRG31K3P_DEFAULT_SEED is used instead.
    void set_seed(unsigned long long seed)
    {
        m_seed = seed == 0 ? get_default_seed() : seed;
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
        {
            return ROCRAND_STATUS_SUCCESS;
        }

        hipError_t error
            = get_least_common_grid_size<ConfigProvider>(m_stream, m_order, m_engines_size);
        if(error != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }

        m_start_engine_id = m_offset % m_engines_size;

        if(m_engines != nullptr)
        {
            system_type::free(m_engines);
        }
        rocrand_status status = system_type::alloc(&m_engines, m_engines_size);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        constexpr unsigned int init_threads = ROCRAND_DEFAULT_MAX_BLOCK_SIZE;
        const unsigned int     init_blocks  = (m_engines_size + init_threads - 1) / init_threads;

        status = system_type::template launch<init_engines_mrg<engine_type>,
                                              static_block_size_config_provider<init_threads>>(
            dim3(init_blocks),
            dim3(init_threads),
            0,
            m_stream,
            m_engines,
            m_start_engine_id,
            m_engines_size,
            m_seed,
            m_offset / m_engines_size);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        status = m_poisson.init();
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        m_engines_initialized = true;
        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T, class Distribution = mrg_engine_uniform_distribution<T, engine_type>>
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
                return system_type::template launch<
                    generate_mrg<ConfigProvider, is_dynamic, engine_type, T, Distribution>,
                    ConfigProvider,
                    T,
                    is_dynamic>(dim3(config.blocks),
                                dim3(config.threads),
                                0,
                                m_stream,
                                m_engines,
                                m_start_engine_id,
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
        const auto touched_engines
            = (data_size + Distribution::output_width - 1) / Distribution::output_width;

        m_start_engine_id = (m_start_engine_id + touched_engines) % m_engines_size;

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
        mrg_engine_uniform_distribution<T, engine_type> distribution;
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_normal(T* data, size_t data_size, T mean, T stddev)
    {
        mrg_engine_normal_distribution<T, engine_type> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_log_normal(T* data, size_t data_size, T mean, T stddev)
    {
        mrg_engine_log_normal_distribution<T, engine_type> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    rocrand_status generate_poisson(unsigned int* data, size_t data_size, double lambda)
    {
        auto result = m_poisson.get_distribution(lambda);
        if(auto* dis = std::get_if<poisson_distribution_t>(&result))
        {
            mrg_engine_poisson_distribution<engine_type, poisson_distribution_t> mrg_dis(*dis);
            return generate(data, data_size, mrg_dis);
        }
        if(auto* dis = std::get_if<poisson_approx_distribution_t>(&result))
        {
            mrg_engine_poisson_distribution<engine_type, poisson_approx_distribution_t> mrg_dis(
                *dis);
            return generate(data, data_size, mrg_dis);
        }
        return std::get<rocrand_status>(result);
    }

private:
    constexpr static unsigned long long int get_default_seed()
    {
        if constexpr(std::is_same_v<engine_type, rocrand_device::mrg31k3p_engine>)
        {
            return ROCRAND_MRG31K3P_DEFAULT_SEED;
        }
        else if constexpr(std::is_same_v<engine_type, rocrand_device::mrg32k3a_engine>)
        {
            return ROCRAND_MRG32K3A_DEFAULT_SEED;
        }
    }

    bool         m_engines_initialized = false;
    engine_type* m_engines             = nullptr;
    unsigned int m_engines_size        = 0;
    unsigned int m_start_engine_id     = 0;

    unsigned long long m_seed;

    // For caching of Poisson for consecutive generations with the same lambda
    poisson_distribution_manager_t m_poisson;

    // m_seed from base_type
    // m_offset from base_type
};

using mrg31k3p_generator
    = mrg_generator_template<system::device_system,
                             rocrand_device::mrg31k3p_engine,
                             default_config_provider<ROCRAND_RNG_PSEUDO_MRG31K3P>>;

template<bool UseHostFunc>
using mrg31k3p_generator_host
    = mrg_generator_template<system::host_system<UseHostFunc>,
                             rocrand_device::mrg31k3p_engine,
                             static_default_config_provider_t<ROCRAND_RNG_PSEUDO_MRG31K3P>>;

using mrg32k3a_generator
    = mrg_generator_template<system::device_system,
                             rocrand_device::mrg32k3a_engine,
                             default_config_provider<ROCRAND_RNG_PSEUDO_MRG32K3A>>;

template<bool UseHostFunc>
using mrg32k3a_generator_host
    = mrg_generator_template<system::host_system<UseHostFunc>,
                             rocrand_device::mrg32k3a_engine,
                             default_config_provider<ROCRAND_RNG_PSEUDO_MRG32K3A>>;

} // namespace rocrand_impl::host

#endif // ROCRAND_RNG_MRG_H_
