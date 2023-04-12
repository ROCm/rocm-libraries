/**
 * Test suite utilites.
 */

#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <hip/amd_detail/amd_hip_fp16.h>
#include <hip/hip_runtime.h>
#include <memory>

#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Random.hpp>
#include <rocRoller/Utilities/Settings.hpp>

template <typename T>
std::shared_ptr<T> make_shared_device(std::size_t n = 1, T init = 0.0)
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

#pragma omp parallel for reduction(+ : r)
    for(int i = 0; i < a.size(); ++i)
    {
        r = r + a[i] * a[i];
    }
    return std::sqrt(r);
}

template <typename T>
double relativeNorm(std::vector<T> a, std::vector<T> b)
{
    double d = 0.0;
    double r = 0.0;

#pragma omp parallel for reduction(+ : d, r)
    for(size_t i = 0; i < a.size(); ++i)
    {
        d = d + double(a[i] - b[i]) * (a[i] - b[i]);
        r = r + b[i] * b[i];
    }

    return std::sqrt(d / r);
}

template <typename T>
double normInf(std::vector<T> a)
{
    double r = 0.0;

#pragma omp parallel for reduction(+ : r)
    for(int i = 0; i < a.size(); ++i)
    {
        r = r + fabs(double(a[i]));
    }
    return r;
}

template <typename T>
double relativeNormInf(std::vector<T> a, std::vector<T> b)
{
    double d = 0.0;
    double r = 0.0;

#pragma omp parallel for reduction(+ : d, r)
    for(size_t i = 0; i < a.size(); ++i)
    {
        d = d + fabs(double(a[i] - b[i]));
        r = r + fabs(double(b[i]));
    }

    return d / r;
}

int countSubstring(const std::string& str, const std::string& sub);

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
               bool                      transA = false,
               bool                      transB = true);

    void CPUMM(std::vector<__half>&       D,
               const std::vector<__half>& C,
               const std::vector<__half>& A,
               const std::vector<__half>& B,
               int                        M,
               int                        N,
               int                        K,
               float                      alpha,
               float                      beta,
               bool                       transA = false,
               bool                       transB = true);
}

/*
 * Random vector generator.
 */

namespace rocRoller
{
    template <typename T>
    void GenerateRandomInput(std::mt19937::result_type seed,
                             std::vector<T>&           A,
                             size_t                    sizeA,
                             std::vector<T>&           B,
                             size_t                    sizeB,
                             std::vector<T>&           C,
                             size_t                    sizeC)
    {
        auto rngA = RandomGenerator(seed + 1);
        auto rngB = RandomGenerator(seed + 2);
        auto rngC = RandomGenerator(seed + 3);

#pragma omp parallel sections
        {
#pragma omp section
            {
                A = rngA.vector<T>(sizeA, -1.f, 1.f);
            }

#pragma omp section
            {
                B = rngB.vector<T>(sizeB, -1.f, 1.f);
            }

#pragma omp section
            {
                C = rngC.vector<T>(sizeC, -1.f, 1.f);
            }
        }
    }

    template <typename T>
    void SetIdentityMatrix(std::vector<T>& mat, size_t cols, size_t rows)
    {
        for(size_t i = 0; i < cols; i++)
            for(size_t j = 0; j < rows; j++)
                mat[i + j * cols] = i == j ? static_cast<T>(1.0) : static_cast<T>(0.0);
    }
}
