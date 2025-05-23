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

#ifndef ROCRAND_RNG_LFSR113_H_
#define ROCRAND_RNG_LFSR113_H_

// IWYU pragma: begin_keep
#include "config/lfsr113_config.hpp"
// IWYU pragma: end_keep

#include "common.hpp"
#include "config_types.hpp"
#include "distributions.hpp"
#include "generator_type.hpp"
#include "system.hpp"
#include "utils/cpp_utils.hpp"

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_lfsr113.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <variant>

namespace rocrand_impl::host
{

typedef ::rocrand_device::lfsr113_engine lfsr113_device_engine;

__host__ __device__ inline void init_lfsr113_engines(dim3 block_idx,
                                                     dim3 thread_idx,
                                                     dim3 /*grid_dim*/,
                                                     dim3                   block_dim,
                                                     lfsr113_device_engine* engines,
                                                     const unsigned int     start_engine_id,
                                                     const unsigned int     engines_size,
                                                     const uint4            seeds,
                                                     const unsigned int     offset)
{
    const unsigned int engine_id = block_idx.x * block_dim.x + thread_idx.x;
    if(engine_id < engines_size)
    {
        engines[engine_id] = lfsr113_device_engine(seeds,
                                                   engine_id,
                                                   offset + (engine_id < start_engine_id ? 1 : 0));
    }
}

template<class ConfigProvider, bool IsDynamic, class T, class Distribution>
__host__ __device__ __forceinline__ void generate_lfsr113(dim3 block_idx,
                                                          dim3 thread_idx,
                                                          dim3 grid_dim,
                                                          dim3 /*block_dim*/,
                                                          lfsr113_device_engine* engines,
                                                          const unsigned int     start_engine_id,
                                                          T*                     data,
                                                          const size_t           n,
                                                          Distribution           distribution)
{
    static_assert(is_single_tile_config<ConfigProvider, T>(IsDynamic),
                  "This kernel should only be used with single tile configs");
    constexpr unsigned int BlockSize    = get_block_size<ConfigProvider, T>(IsDynamic);
    constexpr unsigned int input_width  = Distribution::input_width;
    constexpr unsigned int output_width = Distribution::output_width;

    using vec_type = aligned_vec_type<T, output_width>;

    const unsigned int id          = block_idx.x * BlockSize + thread_idx.x;
    const unsigned int num_engines = grid_dim.x * BlockSize;

    const unsigned int    engine_id = (id + start_engine_id) & (num_engines - 1);
    lfsr113_device_engine engine    = engines[engine_id];

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

#if defined(__gfx90a__)
        // Workaround: The compiler hoists s_waitcnt vmcnt(..) out of the loops.
        // For some reason this optimization decreases performance of uniform distributions
        // on MI200. MI100 and MI300 are not affected.
        // Here we add s_waitcnt vmcnt(0)
        __builtin_amdgcn_s_waitcnt(/*vmcnt*/ 0 | (/*exp_cnt*/ 0x7 << 4) | (/*lgkmcnt*/ 0xf << 8));
#endif
        vec_data[index] = *reinterpret_cast<vec_type*>(output);
        index += num_engines;
    }

    if(output_width > 1 && index == vec_n)
    {
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

template<class System, class ConfigProvider>
class lfsr113_generator_template : public generator_impl_base
{
public:
    using system_type = System;
    using base_type   = generator_impl_base;
    using engine_type = lfsr113_device_engine;
    using poisson_distribution_manager_t
        = poisson_distribution_manager<DISCRETE_METHOD_ALIAS, system_type>;
    using poisson_distribution_t = typename poisson_distribution_manager_t::distribution_t;
    using poisson_approx_distribution_t =
        typename poisson_distribution_manager_t::approx_distribution_t;

    lfsr113_generator_template(uint4              seeds  = {ROCRAND_LFSR113_DEFAULT_SEED_X,
                                                            ROCRAND_LFSR113_DEFAULT_SEED_Y,
                                                            ROCRAND_LFSR113_DEFAULT_SEED_Z,
                                                            ROCRAND_LFSR113_DEFAULT_SEED_W},
                               unsigned long long offset = 0,
                               rocrand_ordering   order  = ROCRAND_ORDERING_PSEUDO_DEFAULT,
                               hipStream_t        stream = 0)
        : base_type(order, offset, stream), m_seed(seeds)
    {}

    lfsr113_generator_template(const lfsr113_generator_template&) = delete;

    lfsr113_generator_template(lfsr113_generator_template&& other)
        : base_type(other)
        , m_engines_initialized(other.m_engines_initialized)
        , m_engines(other.m_engines)
        , m_start_engine_id(other.m_start_engine_id)
        , m_engines_size(other.m_engines_size)
        , m_seed(other.m_seed)
        , m_poisson(std::move(other.m_poisson))
    {
        other.m_engines_initialized = false;
        other.m_engines             = nullptr;
    }

    lfsr113_generator_template& operator=(const lfsr113_generator_template&) = delete;

    lfsr113_generator_template& operator=(lfsr113_generator_template&& other)
    {
        *static_cast<base_type*>(this) = std::move(other);
        m_engines_initialized          = other.m_engines_initialized;
        m_engines                      = other.m_engines;
        m_start_engine_id              = other.m_start_engine_id;
        m_engines_size                 = other.m_engines_size;
        m_seed                         = other.m_seed;
        m_poisson                      = std::move(other.m_poisson);

        other.m_engines_initialized = false;
        other.m_engines             = nullptr;

        return *this;
    }

    ~lfsr113_generator_template()
    {
        if(m_engines != nullptr)
        {
            system_type::free(m_engines);
            m_engines = nullptr;
        }
    }

    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_PSEUDO_LFSR113;
    }

    void reset() override final
    {
        m_engines_initialized = false;
    }

    void set_seed(unsigned long long seed)
    {
        uint4 seeds = uint4{static_cast<unsigned int>(seed),
                            static_cast<unsigned int>(seed >> 16),
                            static_cast<unsigned int>(seed >> 32),
                            static_cast<unsigned int>(seed >> 48)};

        if(seeds.x < ROCRAND_LFSR113_DEFAULT_SEED_X)
            seeds.x += ROCRAND_LFSR113_DEFAULT_SEED_X;

        if(seeds.y < ROCRAND_LFSR113_DEFAULT_SEED_Y)
            seeds.y += ROCRAND_LFSR113_DEFAULT_SEED_Y;

        m_seed                = seeds;
        reset();
    }

    rocrand_status set_seed_uint4(uint4 seed) override final
    {
        if(seed.x < ROCRAND_LFSR113_DEFAULT_SEED_X)
            seed.x += ROCRAND_LFSR113_DEFAULT_SEED_X;

        if(seed.y < ROCRAND_LFSR113_DEFAULT_SEED_Y)
            seed.y += ROCRAND_LFSR113_DEFAULT_SEED_Y;

        if(seed.z < ROCRAND_LFSR113_DEFAULT_SEED_Z)
            seed.z += ROCRAND_LFSR113_DEFAULT_SEED_Z;

        if(seed.w < ROCRAND_LFSR113_DEFAULT_SEED_W)
            seed.w += ROCRAND_LFSR113_DEFAULT_SEED_W;

        m_seed                = seed;
        reset();
        return ROCRAND_STATUS_SUCCESS;
    }

    unsigned long long get_seed() const
    {
        return 0; // Not supported for this generator
    }

    uint4 get_seed_uint4() const
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

        constexpr unsigned int init_threads = 256;
        const unsigned int     init_blocks  = (m_engines_size + init_threads - 1) / init_threads;

        status = system_type::template launch<init_lfsr113_engines,
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

    template<class T, class Distribution = uniform_distribution<T>>
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
                    generate_lfsr113<ConfigProvider, is_dynamic, T, Distribution>,
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

        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

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
    bool         m_engines_initialized = false;
    engine_type* m_engines             = nullptr;
    unsigned int m_start_engine_id     = 0;
    unsigned int m_engines_size        = 0;
    uint4        m_seed;

    // For caching of Poisson for consecutive generations with the same lambda
    poisson_distribution_manager_t m_poisson;

    // m_seed from base_type
    // m_offset from base_type
};

using lfsr113_generator
    = lfsr113_generator_template<system::device_system,
                                 default_config_provider<ROCRAND_RNG_PSEUDO_LFSR113>>;

template<bool UseHostFunc>
using lfsr113_generator_host
    = lfsr113_generator_template<system::host_system<UseHostFunc>,
                                 default_config_provider<ROCRAND_RNG_PSEUDO_LFSR113>>;

} // namespace rocrand_impl::host

#endif // ROCRAND_RNG_LFSR113_H_
