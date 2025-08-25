/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

/**
 * Test suite utilites.
 */

#pragma once

#include <cmath>
#include <cstdlib>
#include <memory>
#include <sstream>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/DataTypes/DataTypes_Utils.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Random.hpp>
#include <rocRoller/Utilities/Settings.hpp>

template <typename Transform, typename... Args>
rocRoller::KernelGraph::KernelGraph transform(rocRoller::KernelGraph::KernelGraph& graph,
                                              Args... args)
{
    auto xform = std::make_shared<Transform>(std::forward<Args>(args)...);
    return graph.transform(xform);
}

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

template <typename T, typename U>
std::shared_ptr<T> make_shared_device(std::vector<U> const& init, size_t padding = 0)
{
    std::size_t size   = init.size() * sizeof(U);
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

auto make_shared_device(std::ranges::range auto const& init, size_t padding = 0)
{
    using T = std::remove_cvref_t<decltype(init.front())>;
    return make_shared_device<T, T>(init, padding);
}

template <typename T>
double normL2(std::vector<T> a)
{
    double r = 0.0;

#pragma omp parallel for reduction(+ : r)
    for(int i = 0; i < a.size(); ++i)
    {
        double t = double(a[i]);
        r        = r + t * t;
    }
    return std::sqrt(r);
}

template <typename T>
double relativeNormL2(std::vector<T> a, std::vector<T> b)
{
    double d = 0.0;
    double r = 0.0;

#pragma omp parallel for reduction(+ : d, r)
    for(size_t i = 0; i < a.size(); ++i)
    {
        double td = double(a[i] - b[i]);
        double t  = double(b[i]);
        d         = d + td * td;
        r         = r + t * t;
    }

    return std::sqrt(d / r);
}

template <typename T>
double normInf(std::vector<T> a)
{
    double r = 0.0;

#pragma omp parallel for reduction(max : r)
    for(int i = 0; i < a.size(); ++i)
    {
        double t = fabs(double(a[i]));
        if(t > r)
            r = t;
    }
    return r;
}

template <typename T>
double relativeNormInf(std::vector<T> a, std::vector<T> b)
{
    double d = 0.0;
    double r = 0.0;

#pragma omp parallel for reduction(max : d, r)
    for(size_t i = 0; i < a.size(); ++i)
    {
        double td = fabs(double(a[i] - b[i]));
        double t  = fabs(double(b[i]));

        if(td > d)
            d = td;
        if(t > r)
            r = t;
    }

    return d / r;
}

struct AcceptableError
{
    double      relativeL2Tolerance;
    std::string reasoning;
};

struct ComparisonResult
{
    bool ok;

    double relativeNormL2, relativeNormInf;
    double referenceNormL2, referenceNormInf;
    double xNormL2, xNormInf;

    AcceptableError acceptableError;

    std::string message() const
    {
        std::stringstream ss;
        ss << (ok ? "Comparison PASSED." : "Comparison FAILED.");
        ss << "  Relative norms:";
        ss << " L2 " << std::scientific << relativeNormL2;
        ss << " Inf " << std::scientific << relativeNormInf;
        ss << "  Norms (x/ref):";
        ss << " L2 " << std::scientific << xNormL2 << "/" << referenceNormL2;
        ss << " Inf " << std::scientific << xNormInf << "/" << referenceNormInf;
        ss << "  Tolerance: " << std::scientific << acceptableError.relativeL2Tolerance;
        ss << " (" << acceptableError.reasoning << ")";
        return ss.str();
    }
};

/**
 * Return expected machine epsilon for `T`.
 */
template <typename T>
double epsilon()
{
    using namespace rocRoller;

    double rv = 0.0;
    if constexpr(std::is_same_v<T, __half>)
        rv = std::pow(2.0, -10);
    else if constexpr(std::is_same_v<T, BFloat16>)
        rv = std::pow(2.0, -5);
    else if constexpr(std::is_same_v<T, FP8>)
        rv = std::pow(2.0, -3);
    else if constexpr(std::is_same_v<T, BF8>)
        rv = std::pow(2.0, -2);
    else
        rv = std::numeric_limits<T>::epsilon();

    AssertFatal(rv != 0.0, "Unknown data type.");

    return rv;
}

/**
 * Return acceptable error for GEMM problems.
 *
 * Currently scales epsilon with the square-root of `K`.
 *
 * This assumes that the routines that compute various norms used for
 * comparison do not accumulate a significant error themselves (if
 * they did, we would want to include `M` and `N` in the scaling).
 */
template <typename TA, typename TB, typename TD>
AcceptableError
    gemmAcceptableError(int M, int N, int K, rocRoller::GPUArchitectureTarget const& arch)
{
    double eps       = epsilon<TD>();
    double tolerance = 0.0;

    std::stringstream ss;
    ss << "Output epsilon: " << std::scientific << eps;

    if constexpr(std::is_same_v<TD, rocRoller::FP8> || std::is_same_v<TD, rocRoller::BF8>)
    {
        tolerance = eps;
        ss << " Error expected to be dominated by conversion.";
    }
    else
    {
        double scale = std::sqrt(K);
        double fudge = 5;

        ss << " K: " << K;
        ss << " Problem size scaling: " << scale;
        ss << " Fudge: " << fudge;

        if constexpr(std::is_same_v<TA, TB>)
        {
            if constexpr(std::is_same_v<TA, rocRoller::BF6>)
            {
                fudge *= 3;
                ss << " Increase fudge for BF6: " << fudge;
            }
            if constexpr(std::is_same_v<TA, rocRoller::BF8>)
            {
                fudge *= 5;
                ss << " Increase fudge for BF8: " << fudge;
            }
            if constexpr(std::is_same_v<TA, rocRoller::FP8>)
            {
                fudge *= 6.0 + (arch.gfx == rocRoller::GPUArchitectureGFX::GFX950 ? 0.5 : 0.0);

                ss << " Increase fudge for FP8: " << fudge;
            }
        }
        else
        {
            if constexpr(std::is_same_v<TA, rocRoller::BF8> || std::is_same_v<TB, rocRoller::BF8>)
            {
                fudge *= arch.gfx == rocRoller::GPUArchitectureGFX::GFX950 ? 7.58 : 6.0;
                ss << " Increase fudge for mixed BF8: " << fudge;
            }
            else if constexpr(std::is_same_v<TA,
                                             rocRoller::FP8> || std::is_same_v<TB, rocRoller::FP8>)
            {
                fudge *= 4.95;
                ss << " Increase fudge for mixed FP8: " << fudge;
            }
            else if constexpr(std::is_same_v<TA,
                                             rocRoller::BF6> || std::is_same_v<TB, rocRoller::BF6>)
            {
                fudge *= 3;
                ss << " Increase fudge for mixed BF6: " << fudge;
            }
        }
        tolerance = fudge * epsilon<TD>() * std::sqrt(K);
    }

    return {tolerance, ss.str()};
}

/**
 * @brief Compare `x` to a reference `r`.
 *
 * The boolean `ok` field of the return value is true if the relative
 * L2 norm between `x` and `r` is less than `scale` * `epsilon`.
 *
 * Various norms are computed and included in the return value.
 */
template <typename T>
ComparisonResult compare(std::vector<T> const&  x,
                         std::vector<T> const&  r,
                         AcceptableError const& acceptableError)
{
    ComparisonResult rv;

    rv.acceptableError  = acceptableError;
    rv.referenceNormL2  = normL2(r);
    rv.referenceNormInf = normInf(r);
    rv.xNormL2          = normL2(x);
    rv.xNormInf         = normInf(x);
    rv.relativeNormL2   = relativeNormL2(x, r);
    rv.relativeNormInf  = relativeNormInf(x, r);

    rv.ok = rv.relativeNormL2 < rv.acceptableError.relativeL2Tolerance;

    return rv;
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

    template <typename TA, typename TB, typename TC, typename TD>
    void CPUMM(std::vector<TD>&       D,
               const std::vector<TC>& C,
               const std::vector<TA>& A,
               const std::vector<TB>& B,
               int                    M,
               int                    N,
               int                    K,
               float                  alpha,
               float                  beta,
               bool                   transA = false,
               bool                   transB = true)
    {
        if constexpr(std::same_as<TD, float> && std::same_as<TC, float>)
        {
            CPUMM(D, C, unpackToFloat(A), unpackToFloat(B), M, N, K, alpha, beta, transA, transB);
        }
        else if constexpr((std::same_as<TD, __half> && std::same_as<TC, __half>)
                          || (std::same_as<TD, BFloat16> && std::same_as<TC, BFloat16>))
        {
            auto floatD = unpackToFloat(C);
            CPUMM(floatD,
                  floatD,
                  unpackToFloat(A),
                  unpackToFloat(B),
                  M,
                  N,
                  K,
                  alpha,
                  beta,
                  transA,
                  transB);
#pragma omp parallel for
            for(std::size_t i = 0; i != floatD.size(); ++i)
            {
                D[i] = TD(floatD[i]);
            }
        }
        else
        {
            Throw<FatalError>("CPUMM invalid types.");
        }
    }

    void ScaledCPUMM(std::vector<float>&         D,
                     const std::vector<float>&   C,
                     const std::vector<float>&   floatA,
                     const std::vector<float>&   floatB,
                     const std::vector<uint8_t>& AX,
                     const std::vector<uint8_t>& BX,
                     int                         M,
                     int                         N,
                     int                         K,
                     float                       alpha,
                     float                       beta,
                     bool                        transA,
                     bool                        transB,
                     const uint                  scaleBlockSize = 32,
                     const DataType              scaleTypeA     = DataType::E8M0,
                     const DataType              scaleTypeB     = DataType::E8M0);

    template <typename TA, typename TB, typename TC, typename TD>
    void ScaledCPUMM(std::vector<TD>&            D,
                     const std::vector<TC>&      C,
                     const std::vector<TA>&      A,
                     const std::vector<TB>&      B,
                     const std::vector<uint8_t>& AX,
                     const std::vector<uint8_t>& BX,
                     int                         M,
                     int                         N,
                     int                         K,
                     float                       alpha,
                     float                       beta,
                     bool                        transA,
                     bool                        transB,
                     const uint                  scaleBlockSize = 32,
                     const DataType              scaleTypeA     = DataType::E8M0,
                     const DataType              scaleTypeB     = DataType::E8M0)
    {
        if constexpr(std::same_as<TD, float> && std::same_as<TC, float>)
        {
            auto aConverted = unpackToFloat(A);
            auto bConverted = unpackToFloat(B);
            ScaledCPUMM(D,
                        C,
                        aConverted,
                        bConverted,
                        AX,
                        BX,
                        M,
                        N,
                        K,
                        alpha,
                        beta,
                        transA,
                        transB,
                        scaleBlockSize,
                        scaleTypeA,
                        scaleTypeB);
        }
        else if constexpr((std::same_as<TD, __half> && std::same_as<TC, __half>)
                          || (std::same_as<TD, BFloat16> && std::same_as<TC, BFloat16>))
        {
            auto aConverted = unpackToFloat(A);
            auto bConverted = unpackToFloat(B);
            auto floatD     = unpackToFloat(C);
            ScaledCPUMM(floatD,
                        floatD,
                        aConverted,
                        bConverted,
                        AX,
                        BX,
                        M,
                        N,
                        K,
                        alpha,
                        beta,
                        transA,
                        transB,
                        scaleBlockSize,
                        scaleTypeA,
                        scaleTypeB);
#pragma omp parallel for
            for(std::size_t i = 0; i != floatD.size(); ++i)
            {
                D[i] = TD(floatD[i]);
            }
        }
        else
        {
            auto dataTypeC = TypeInfo<TC>::Var.dataType;
            auto dataTypeD = TypeInfo<TD>::Var.dataType;

            Throw<FatalError>("ScaledCPUMM only supported for float, not ",
                              ShowValue(dataTypeC),
                              ShowValue(dataTypeD));
        }
    }
}

/*
 * Random vector generator.
 */

namespace rocRoller
{
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
    requires(CIsAnyOf<F6x16, FP6x16, BF6x16>) //
        inline void SetIdentityMatrix(std::vector<F6x16>& mat, size_t cols, size_t rows)
    {
        std::fill(mat.begin(), mat.end(), F6x16()); // zero out the matrix

        // Notice `cols` and `rows` are NOT the actual dimensions of `mat`,
        // they are the dimensions before packed into F6x16.
        size_t const row_bytes = 6 * cols / 8; // number of bytes in a row
        // clang-format off
        auto constexpr patterns = [] {
        if constexpr (std::is_same_v<F6x16, FP6x16>)
        {
            return std::to_array<uint8_t> ({  //  Bit pattern of FP6 identity matrix:
                0b00100000,                   //    001000 000000 000000 000000 000000
                0b10000000,                   //    000000 001000 000000 000000 000000
                0b00000010,                   //    000000 000000 001000 000000 000000
                0b00001000,                   //    000000 000000 000000 001000 000000
            });                               //    000000 000000 000000 000000 001000 (repeat 1st row)
        }
        else
        {
            return std::to_array<uint8_t> ({  //  Bit pattern of BF6 identity matrix:
                0b00110000,                   //    001100 000000 000000 000000 000000
                0b11000000,                   //    000000 001100 000000 000000 000000
                0b00000011,                   //    000000 000000 001100 000000 000000
                0b00001100,                   //    000000 000000 000000 001100 000000
             });                              //    000000 000000 000000 000000 001100 (repeat 1st row)
        }
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
