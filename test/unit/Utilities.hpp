/**
 * Test suite utilites.
 */

#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <memory>
#include <random>

#include <rocRoller/Utilities/Logging.hpp>

template <typename T>
std::shared_ptr<T> make_shared_device(std::size_t n = 1, T init = 0)
{
    std::size_t size   = n * sizeof(T);
    T*          ptr    = nullptr;
    auto        result = hipMalloc(&ptr, size);
    if(result != hipSuccess)
    {
        throw std::runtime_error(hipGetErrorString(result));
    }

    result = hipMemset(ptr, init, size);
    if(result != hipSuccess)
    {
        throw std::runtime_error(hipGetErrorString(result));
    }

    return std::shared_ptr<T>(ptr, hipFree);
}

template <typename T>
std::shared_ptr<T> make_shared_device(const std::vector<T>& init)
{
    std::size_t size   = init.size() * sizeof(T);
    T*          ptr    = nullptr;
    auto        result = hipMalloc(&ptr, size);
    if(result != hipSuccess)
    {
        throw std::runtime_error(hipGetErrorString(result));
    }

    result = hipMemcpy(ptr, init.data(), size, hipMemcpyDefault);
    if(result != hipSuccess)
    {
        throw std::runtime_error(hipGetErrorString(result));
    }

    return std::shared_ptr<T>(ptr, hipFree);
}

MATCHER_P(HasHipSuccess, p, "")
{
    auto result = arg;
    if(result != hipSuccess)
    {
        std::string msg = hipGetErrorString(result);
        *result_listener << msg;
    }
    return result == hipSuccess;
}

template <typename T>
double norm(std::vector<T> a)
{
    double r = 0.0;
    for(int i = 0; i < a.size(); ++i)
    {
        r += a[i] * a[i];
    }
    return std::sqrt(r);
}

template <typename T>
double relativeNorm(std::vector<T> a, std::vector<T> b)
{
    double d = 0.0;
    double r = 0.0;
    for(size_t i = 0; i < a.size(); ++i)
    {
        d += double(a[i] - b[i]) * (a[i] - b[i]);
        r += b[i] * b[i];
    }
    return std::sqrt(d / r);
}

/*
 * Matrix multiply.
 */

namespace rocRoller
{
    void CPUMM(std::vector<float>&       D,
               const std::vector<float>& C,
               const std::vector<float>& A,
               const std::vector<float>& B,
               int                       M,
               int                       N,
               int                       K,
               float                     alpha,
               float                     beta,
               bool                      transposeB = true);
}

/*
 * Random vector generator.
 */

namespace rocRoller
{
    const std::string ENV_RANDOM_SEED = "ROCROLLER_RANDOM_SEED";

    /**
     * Random vector generator.
     *
     * A seed must be passed to the constructor.  If the environment
     * variable specified by `ENV_RANDOM_SEED` is present, it
     * supercedes the seed passed to the constructor.
     *
     * A seed may be set programmatically (at any time) by calling
     * seed().
     */
    class RandomGenerator
    {
    public:
        RandomGenerator() = delete;
        RandomGenerator(std::mt19937::result_type seed)
        {
            auto envSeed = getenv(ENV_RANDOM_SEED.c_str());
            if(envSeed)
            {
                seed = static_cast<std::mt19937::result_type>(atoi(envSeed));
            }

            rocRoller::Log::debug("Using random seed: {}; from env: {}", seed, bool(envSeed));
            m_gen.seed(seed);
        }

        /**
         * Set a new seed.
         */
        void seed(std::mt19937::result_type seed)
        {
            m_gen.seed(seed);
        }

        /**
         * Generate a random vector of length `nx`, with values
         * between `min` and `max`.
         */
        template <typename T>
        std::vector<T> vector(uint nx, T min, T max)
        {
            std::vector<T>                   x(nx);
            std::uniform_real_distribution<> udist(min, max);

            for(unsigned i = 0; i < nx; i++)
            {
                x[i] = static_cast<T>(udist(m_gen));
            }

            return x;
        }

    private:
        std::mt19937 m_gen;
    };
}
