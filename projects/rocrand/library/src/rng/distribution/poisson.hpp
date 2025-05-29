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

#ifndef ROCRAND_RNG_DISTRIBUTION_POISSON_H_
#define ROCRAND_RNG_DISTRIBUTION_POISSON_H_

#include "../common.hpp" // IWYU pragma: keep

#include "../system.hpp"
#include "discrete.hpp"

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_discrete_types.h>
#include <rocrand/rocrand_poisson.h>
#include <rocrand/rocrand_uniform.h>

#include <hip/hip_runtime_api.h>

#include <cassert>
#include <memory>
#include <mutex>
#include <variant>
#include <vector>

namespace rocrand_impl::host
{

// Precise calculation of Poisson distribution using table methods:
//  * the alias method (for PRNGs only as it does not preserve quasi-randomness);
//  * the binary search in the CDF (cumulative distribution function) table (suitable for both
//    QRNGs and PRNGs but not preferred for PRNGs because the alias method is faster);

template<discrete_method Method = DISCRETE_METHOD_ALIAS>
class poisson_distribution : private discrete_distribution_base<Method>
{
public:
    static constexpr inline unsigned int input_width  = 1;
    static constexpr inline unsigned int output_width = 1;

    using base_t = discrete_distribution_base<Method>;

    poisson_distribution(const rocrand_discrete_distribution_st& distribution)
        : base_t(distribution)
    {}

    template<class T>
    __forceinline__ __host__ __device__
    unsigned int
        operator()(T x) const
    {
        return base_t::operator()(x);
    }

    template<class T>
    __forceinline__ __host__ __device__
    void operator()(const T (&input)[1], unsigned int (&output)[1]) const
    {
        output[0] = (*this)(input[0]);
    }
};

// Approximation of Poisson distribution with normal distribution when lambda is large

class poisson_distribution_huge
{
public:
    static constexpr unsigned int input_width  = 1;
    static constexpr unsigned int output_width = 1;

    poisson_distribution_huge(const double lambda) : m_lambda(lambda), m_sqrt_lambda(sqrt(lambda))
    {}

    template<class T>
    __forceinline__ __host__ __device__
    unsigned int
        operator()(T x) const
    {
        const double normal_d = rocrand_device::detail::normal_distribution_double(x);
        return static_cast<unsigned int>(round(m_sqrt_lambda * normal_d + m_lambda));
    }

    template<class T>
    __forceinline__ __host__ __device__
    void operator()(const T (&input)[1], unsigned int (&output)[1]) const
    {
        output[0] = (*this)(input[0]);
    }

private:
    double m_lambda;
    double m_sqrt_lambda;
};

[[nodiscard]]
inline std::vector<double>
    calculate_poisson_probabilities(const double lambda, unsigned int& size, unsigned int& offset)
{
    const size_t        capacity = 2 * static_cast<size_t>(16.0 * (2.0 + std::sqrt(lambda)));
    std::vector<double> p(capacity);

    const double p_epsilon  = 1e-12;
    const double log_lambda = std::log(lambda);

    const int left = static_cast<int>(std::floor(lambda)) - capacity / 2;

    // Calculate probabilities starting from mean in both directions,
    // because only a small part of [0, lambda] has non-negligible values
    // (> p_epsilon).

    int lo = 0;
    for(int i = capacity / 2; i >= 0; i--)
    {
        const double x  = left + i;
        const double pp = std::exp(x * log_lambda - std::lgamma(x + 1.0) - lambda);
        if(pp < p_epsilon)
        {
            lo = i + 1;
            break;
        }
        p[i] = pp;
    }

    int hi = capacity - 1;
    for(int i = capacity / 2 + 1; i < static_cast<int>(capacity); i++)
    {
        const double x  = left + i;
        const double pp = std::exp(x * log_lambda - std::lgamma(x + 1.0) - lambda);
        if(pp < p_epsilon)
        {
            hi = i - 1;
            break;
        }
        p[i] = pp;
    }

    for(int i = lo; i <= hi; i++)
    {
        p[i - lo] = p[i];
    }

    size   = hi - lo + 1;
    offset = left + lo;

    return p;
}

inline void calculate_poisson_size(const double lambda, unsigned int& size, unsigned int& offset)
{
    (void)calculate_poisson_probabilities(lambda, size, offset);
}

// Handles caching of precomputed tables for the distribution and recomputes
// them only when lambda is changed (as these computations, device memory
// allocations and copying take time).
template<discrete_method Method = DISCRETE_METHOD_ALIAS, class System = system::device_system>
class poisson_distribution_manager
{
public:
    using factory_t             = discrete_distribution_factory<Method, !System::is_device()>;
    using distribution_t        = poisson_distribution<Method>;
    using approx_distribution_t = poisson_distribution_huge;

    poisson_distribution_manager() = default;

    poisson_distribution_manager(const poisson_distribution_manager&) = delete;

    poisson_distribution_manager(poisson_distribution_manager&& other)
        : m_initialized(std::exchange(other.m_initialized, false))
        , m_is_host_func_blocking(other.m_is_host_func_blocking)
        , m_stream(other.m_stream)
        , m_probability(std::exchange(other.m_probability, nullptr))
        , m_alias(std::exchange(other.m_alias, nullptr))
        , m_cdf(std::exchange(other.m_cdf, nullptr))
        , m_lambda(other.m_lambda)
        , m_distribution(std::exchange(other.m_distribution, {}))
    {}

    poisson_distribution_manager& operator=(const poisson_distribution_manager&) = delete;

    poisson_distribution_manager& operator=(poisson_distribution_manager&& other)
    {
        m_initialized           = other.m_initialized;
        m_is_host_func_blocking = other.m_is_host_func_blocking;
        m_stream                = other.m_stream;
        m_lambda                = other.m_lambda;
        std::swap(m_probability, other.m_probability);
        std::swap(m_alias, other.m_alias);
        std::swap(m_cdf, other.m_cdf);
        std::swap(m_distribution, other.m_distribution);

        return *this;
    }

    ~poisson_distribution_manager()
    {
        factory_t::deallocate(m_distribution);
        if constexpr((Method & DISCRETE_METHOD_ALIAS) != 0)
        {
            ROCRAND_HIP_FATAL_ASSERT(hipHostFree(m_probability));
            ROCRAND_HIP_FATAL_ASSERT(hipHostFree(m_alias));
        }
        if constexpr((Method & DISCRETE_METHOD_CDF) != 0)
        {
            ROCRAND_HIP_FATAL_ASSERT(hipHostFree(m_cdf));
        }
    }

    rocrand_status init()
    {
        if(m_initialized)
        {
            return ROCRAND_STATUS_SUCCESS;
        }

        unsigned int size;
        unsigned int offset;
        calculate_poisson_size(rocrand_device::detail::lambda_threshold_huge, size, offset);
        if constexpr((Method & DISCRETE_METHOD_ALIAS) != 0)
        {
            hipError_t error = hipHostMalloc(&m_probability, size * sizeof(*m_probability));
            if(error != hipSuccess)
            {
                return ROCRAND_STATUS_ALLOCATION_FAILED;
            }
            error = hipHostMalloc(&m_alias, size * sizeof(*m_alias));
            if(error != hipSuccess)
            {
                return ROCRAND_STATUS_ALLOCATION_FAILED;
            }
        }
        if constexpr((Method & DISCRETE_METHOD_CDF) != 0)
        {
            const hipError_t error = hipHostMalloc(&m_cdf, size * sizeof(*m_cdf));
            if(error != hipSuccess)
            {
                return ROCRAND_STATUS_ALLOCATION_FAILED;
            }
        }
        const rocrand_status status = factory_t::allocate(size, offset, m_distribution);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        m_initialized = true;
        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_status set_stream(const hipStream_t stream)
    {
        const rocrand_status status
            = System::is_host_func_blocking(stream, m_is_host_func_blocking);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }
        m_stream = stream;
        return ROCRAND_STATUS_SUCCESS;
    }

    std::variant<rocrand_status, distribution_t, approx_distribution_t>
        get_distribution(const double lambda)
    {
        if(!m_initialized)
        {
            const rocrand_status status = init();
            if(status != ROCRAND_STATUS_SUCCESS)
            {
                return status;
            }
        }

        if(lambda > rocrand_device::detail::lambda_threshold_huge)
        {
            return approx_distribution_t(lambda);
        }

        std::unique_lock lock(m_mutex, std::defer_lock_t{});
        if(!m_is_host_func_blocking)
        {
            lock.lock();
        }

        const bool changed = lambda != m_lambda;
        if(changed)
        {
            auto arg = std::make_unique<update_discrete_distribution_arg>(
                update_discrete_distribution_arg{lambda, this});
            const rocrand_status status
                = System::launch_host_func(m_stream, update_discrete_distribution, arg.release());
            if(status != ROCRAND_STATUS_SUCCESS)
            {
                return status;
            }
            if constexpr(System::is_device() && (Method & DISCRETE_METHOD_ALIAS))
            {
                hipError_t error
                    = hipMemcpyAsync(m_distribution.probability,
                                     m_probability,
                                     m_distribution.size * sizeof(*m_distribution.probability),
                                     hipMemcpyHostToDevice,
                                     m_stream);
                if(error != hipSuccess)
                {
                    return ROCRAND_STATUS_INTERNAL_ERROR;
                }
                error = hipMemcpyAsync(m_distribution.alias,
                                       m_alias,
                                       m_distribution.size * sizeof(*m_distribution.alias),
                                       hipMemcpyHostToDevice,
                                       m_stream);
                if(error != hipSuccess)
                {
                    return ROCRAND_STATUS_INTERNAL_ERROR;
                }
            }
            if constexpr(System::is_device() && (Method & DISCRETE_METHOD_CDF))
            {
                const hipError_t error
                    = hipMemcpyAsync(m_distribution.cdf,
                                     m_cdf,
                                     m_distribution.size * sizeof(*m_distribution.cdf),
                                     hipMemcpyHostToDevice,
                                     m_stream);
                if(error != hipSuccess)
                {
                    return ROCRAND_STATUS_INTERNAL_ERROR;
                }
            }
        }

        rocrand_discrete_distribution_st distribution_copy = m_distribution;
        calculate_poisson_size(lambda, distribution_copy.size, distribution_copy.offset);
        return distribution_t(distribution_copy);
    }

private:
    bool                             m_initialized           = false;
    bool                             m_is_host_func_blocking = true;
    hipStream_t                      m_stream                = 0;
    std::mutex                       m_mutex;
    double*                          m_probability  = nullptr;
    unsigned int*                    m_alias        = nullptr;
    double*                          m_cdf          = nullptr;
    double                           m_lambda       = 0;
    rocrand_discrete_distribution_st m_distribution = {};

    struct update_discrete_distribution_arg
    {
        double                        lambda;
        poisson_distribution_manager* manager;
    };

    static void update_discrete_distribution(void* user_data)
    {
        std::unique_ptr<update_discrete_distribution_arg> arg(
            reinterpret_cast<update_discrete_distribution_arg*>(user_data));
        std::unique_lock lock(arg->manager->m_mutex, std::defer_lock_t{});
        if(!arg->manager->m_is_host_func_blocking)
        {
            lock.lock();
        }
        unsigned int        size;
        unsigned int        offset;
        std::vector<double> poisson_probabilities
            = calculate_poisson_probabilities(arg->lambda, size, offset);
        assert(size <= arg->manager->m_distribution.size);
        factory_t::normalize(poisson_probabilities, size);
        if constexpr((Method & DISCRETE_METHOD_ALIAS) != 0)
        {
            factory_t::create_alias_table(poisson_probabilities,
                                          size,
                                          arg->manager->m_probability,
                                          arg->manager->m_alias);
        }
        if constexpr((Method & DISCRETE_METHOD_CDF) != 0)
        {
            factory_t::create_cdf(poisson_probabilities, size, arg->manager->m_cdf);
        }
        arg->manager->m_lambda = arg->lambda;
        if constexpr(!System::is_device())
        {
            if constexpr((Method & DISCRETE_METHOD_ALIAS) != 0)
            {
                std::copy_n(arg->manager->m_probability,
                            size,
                            arg->manager->m_distribution.probability);
                std::copy_n(arg->manager->m_alias, size, arg->manager->m_distribution.alias);
            }
            if constexpr((Method & DISCRETE_METHOD_CDF) != 0)
            {
                std::copy_n(arg->manager->m_cdf, size, arg->manager->m_distribution.cdf);
            }
        }
    }
};

// Mrg32k3a and Mrg31k3p

template<typename StateType,
         typename DistributionType = poisson_distribution<DISCRETE_METHOD_ALIAS>>
struct mrg_engine_poisson_distribution
{
    using distribution_type = DistributionType;

    static constexpr unsigned int input_width  = 1;
    static constexpr unsigned int output_width = 1;

    distribution_type dis;

    explicit mrg_engine_poisson_distribution(distribution_type dis) : dis(dis) {}

    __forceinline__ __host__ __device__
    void operator()(const unsigned int (&input)[1], unsigned int (&output)[1]) const
    {
        // Alias method requires x in [0, 1), uint must be in [0, UINT_MAX],
        // but MRG-based engine's "raw" output is in [1, MRG_M1],
        // so probabilities are slightly different than expected,
        // some values can not be generated at all.
        // Hence the "raw" value is remapped to [0, UINT_MAX]:
        unsigned int input2[1];
        input2[0] = rocrand_device::detail::mrg_uniform_distribution_uint<StateType>(input[0]);
        dis(input2, output);
    }
};

// Mrg32ka (compatibility API)

struct mrg_poisson_distribution : mrg_engine_poisson_distribution<rocrand_state_mrg32k3a>
{
    explicit mrg_poisson_distribution(poisson_distribution<DISCRETE_METHOD_ALIAS> dis)
        : mrg_engine_poisson_distribution(dis)
    {}
};

} // namespace rocrand_impl::host

#endif // ROCRAND_RNG_DISTRIBUTION_POISSON_H_
