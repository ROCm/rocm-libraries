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
std::shared_ptr<T> make_shared_device(std::size_t n = 1, T init = {})
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
std::shared_ptr<T> make_shared_device(const std::vector<T>& init, size_t padding = 0)
{
    std::size_t size   = init.size() * sizeof(T);
    T*          ptr    = nullptr;
    auto        result = hipMalloc(&ptr, size + padding);
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

    void CPUMM(std::vector<float>&       D,
               const std::vector<float>& C,
               const std::vector<FP8>&   A,
               const std::vector<FP8>&   B,
               int                       M,
               int                       N,
               int                       K,
               float                     alpha,
               float                     beta,
               bool                      transA = false,
               bool                      transB = true);

    void CPUMM(std::vector<float>&       D,
               const std::vector<float>& C,
               const std::vector<BF8>&   A,
               const std::vector<BF8>&   B,
               int                       M,
               int                       N,
               int                       K,
               float                     alpha,
               float                     beta,
               bool                      transA = false,
               bool                      transB = true);

    void CPUMM(std::vector<float>&        D,
               const std::vector<float>&  C,
               const std::vector<FP6x16>& A,
               const std::vector<FP6x16>& B,
               int                        M,
               int                        N,
               int                        K,
               float                      alpha,
               float                      beta,
               bool                       transA = false,
               bool                       transB = true);

    void CPUMM(std::vector<float>&        D,
               const std::vector<float>&  C,
               const std::vector<BF6x16>& A,
               const std::vector<BF6x16>& B,
               int                        M,
               int                        N,
               int                        K,
               float                      alpha,
               float                      beta,
               bool                       transA = false,
               bool                       transB = true);

    void CPUMM(std::vector<float>&       D,
               const std::vector<float>& C,
               const std::vector<FP4x8>& A,
               const std::vector<FP4x8>& B,
               int                       M,
               int                       N,
               int                       K,
               float                     alpha,
               float                     beta,
               bool                      transA = false,
               bool                      transB = true);

}

/*
 * Random vector generator.
 */

namespace rocRoller
{
    template <typename TA, typename TB, typename TC>
    void GenerateRandomInput(std::mt19937::result_type seed,
                             std::vector<TA>&          A,
                             size_t                    sizeA,
                             std::vector<TB>&          B,
                             size_t                    sizeB,
                             std::vector<TC>&          C,
                             size_t                    sizeC,
                             float                     min = -1.f,
                             float                     max = 1.f)
    {
        auto rngA = RandomGenerator(seed + 1);
        auto rngB = RandomGenerator(seed + 2);
        auto rngC = RandomGenerator(seed + 3);

        auto generateVector = [&](auto& vec, RandomGenerator& rng, size_t sz) {
            using elemT = typename std::remove_reference_t<decltype(vec)>::value_type;
            if constexpr(std::is_same_v<elemT, FP4x8>)
                vec = rng.vector<FP4>(sz, min, max);
            else if constexpr(std::is_same_v<elemT, FP6x16>)
                vec = rng.vector<FP6>(sz, min, max);
            else if constexpr(std::is_same_v<elemT, BF6x16>)
                vec = rng.vector<BF6>(sz, min, max);
            else
                vec = rng.vector<elemT>(sz, min, max);
        };

#pragma omp parallel sections
        {
#pragma omp section
            {
                generateVector(A, rngA, sizeA);
            }

#pragma omp section
            {
                generateVector(B, rngB, sizeB);
            }

#pragma omp section
            {
                C = rngC.vector<TC>(sizeC, min, max);
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

    template <>
    inline void SetIdentityMatrix(std::vector<FP4x8>& mat, size_t cols, size_t rows)
    {
        std::fill(mat.begin(), mat.end(), FP4x8()); // zero out the matrix

        // Notice `cols` and `rows` are NOT the actual dimensions of `mat`,
        // they are the dimensions before packed into FP4x8.
        size_t const row_bytes = 4 * cols / 8; // number of bytes in a row
        uint8_t      even      = 0b00100000;
        uint8_t      odd       = 0b00000010;

        //  Generate FP4 identity matrix with bit pattern like this:
        //    0010 0000 0000 0000
        //    0000 0010 0000 0000
        //    0000 0000 0010 0000
        //    0000 0000 0000 0010
        //    ...
        for(size_t i = 0; i < std::min(rows, cols); i += 2)
            std::memcpy(
                reinterpret_cast<uint8_t*>(mat.data()) + (row_bytes * i) + (4 * i / 8), &even, 1);
        for(size_t i = 1; i < std::min(rows, cols); i += 2)
            std::memcpy(
                reinterpret_cast<uint8_t*>(mat.data()) + (row_bytes * i) + (4 * i / 8), &odd, 1);
    }

    template <typename F6x16>
    requires(
        std::is_same_v<
            F6x16,
            FP6x16> || std::is_same_v<F6x16, BF6x16>) inline void SetIdentityMatrix(std::
                                                                                        vector<
                                                                                            F6x16>&
                                                                                            mat,
                                                                                    size_t  cols,
                                                                                    size_t  rows)
    {
        std::fill(mat.begin(), mat.end(), F6x16()); // zero out the matrix

        // Notice `cols` and `rows` are NOT the actual dimensions of `mat`,
        // they are the dimensions before packed into F6x16.
        size_t const row_bytes = 6 * cols / 8; // number of bytes in a row
        // clang-format off
        auto constexpr patterns = [] {
        if constexpr (std::is_same_v<F6x16, FP6x16>)
            return std::to_array<uint8_t> ({  //  Bit pattern of FP6 identity matrix:
                0b00100000,                   //    001000 000000 000000 000000 000000
                0b10000000,                   //    000000 001000 000000 000000 000000
                0b00000010,                   //    000000 000000 001000 000000 000000
                0b00001000,                   //    000000 000000 000000 001000 000000
            });                               //    000000 000000 000000 000000 001000 (repeat 1st row)
        else
            return std::to_array<uint8_t> ({  //  Bit pattern of BF6 identity matrix:
                0b00110000,                   //    001100 000000 000000 000000 000000
                0b11000000,                   //    000000 001100 000000 000000 000000
                0b00000011,                   //    000000 000000 001100 000000 000000
                0b00001100,                   //    000000 000000 000000 001100 000000
             });                              //    000000 000000 000000 000000 001100 (repeat 1st row)
        }();
        // clang-format on
        std::array constexpr shift = {0, 1, 0, 0};

        for(size_t i = 0; i < std::min(rows, cols); i++)
        {
            int byte_offset = (i * 6) / 8 + shift[i % 4];
            std::memcpy(reinterpret_cast<uint8_t*>(mat.data()) + (row_bytes * i) + byte_offset,
                        &patterns[i % 4],
                        1);
        }
    }
}
