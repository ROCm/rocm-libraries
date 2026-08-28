// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Test suite utilites.
 */

#pragma once

#include <cmath>
#include <cstdlib>
#include <memory>
#include <regex>
#include <sstream>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/DataTypes/DataTypes_Utils.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/HostNumerics/HostReference.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>
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

/**
 * Return a new device array that contains the value stored in `arg`.
 */
std::shared_ptr<void> make_shared_device(rocRoller::CommandArgumentValue const& arg);

template <typename T>
roc::host_numerics::Tensor hostComparisonTensor(std::vector<T> const& values)
{
    constexpr size_t packing = rocRoller::TypeInfo<T>::Packing;
    if(values.size() > std::numeric_limits<size_t>::max() / packing)
        throw std::overflow_error("rocRoller comparison element count overflow.");
    const size_t logicalElements = values.size() * packing;
    const auto   type            = rocRoller::HostNumerics::hostScalarType(
        rocRoller::TypeInfo<T>::SegmentVariableType.dataType);
    return roc::host_numerics::Tensor::copyEncodedBackingStorage(
        type,
        roc::host_numerics::Layout::contiguousLastDimensionFastest(
            roc::host_numerics::Shape{logicalElements}),
        std::as_bytes(std::span<const T>(values)));
}

template <typename T>
double relativeNormL2(std::vector<T> const& observed, std::vector<T> const& expected)
{
    if(observed.size() != expected.size())
        throw std::invalid_argument("rocRoller comparison vectors have different sizes.");
    roc::host_numerics::ComparisonOptions options;
    options.pointwise                              = false;
    options.computePointwiseStatistics             = true;
    options.computeFrobenius                       = true;
    options.computeUlp                             = false;
    options.maxReportedMismatches                  = 0;
    options.zeroExpectedNormIsNaN                  = true;
    options.nonFiniteValuesInvalidateRelativeNorms = true;
    return roc::host_numerics::compare(
               hostComparisonTensor(observed), hostComparisonTensor(expected), options)
        .relativeFrobeniusError;
}

using AcceptableError  = rocRoller::HostNumerics::AcceptableGEMMError;
using ComparisonResult = rocRoller::HostNumerics::HostComparisonResult;

/**
 * Return expected machine epsilon for `T`.
 */
template <typename T>
double epsilon()
{
    return rocRoller::HostNumerics::hostReferenceEpsilon<T>();
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
AcceptableError gemmAcceptableError(int K, rocRoller::GPUArchitectureTarget const& arch)
{
    return rocRoller::HostNumerics::acceptableGEMMError<TA, TB, TD>(K, arch);
}

/**
 * @brief Compare `x` to a reference `r`.
 *
 * The `ok()` result is true if the relative
 * L2 norm between `x` and `r` is less than `scale` * `epsilon`.
 *
 * Various norms are computed and included in the return value.
 */
template <typename T>
ComparisonResult compare(std::vector<T> const&  x,
                         std::vector<T> const&  r,
                         AcceptableError const& acceptableError)
{
    return rocRoller::HostNumerics::compareHostReference(
        hostComparisonTensor(x), hostComparisonTensor(r), acceptableError);
}

int countSubstring(const std::string& str, const std::string& sub);

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

    inline std::string FixupInstructionStringsForVGPRIndexing(GPUArchitecture const& arch,
                                                              std::string            instr)
    {
        if(not arch.HasCapability(GPUCapability::HasVGPRIndexing))
            return instr;
        auto reservedRegionSize = Register::RegisterAllocatorDetail::ReservedRegionSize();

        std::string rv;
        // match v[#:#] or v#
        std::regex re{R"(v(?:\[(\d+):(\d+)\]|(\d+)))"};
        auto       begin   = ::std::sregex_iterator(instr.begin(), instr.end(), re);
        size_t     lastPos = 0;

        for(auto it = begin; it != std::sregex_iterator(); it++)
        {
            auto match = (*it);
            rv.append(instr, lastPos, it->position() - lastPos);
            if(match[1].matched or match[2].matched)
            {
                AssertFatal(match[1].matched and match[2].matched,
                            ShowValue(match[1].matched),
                            ShowValue(match[2].matched));
                // matched v[#:#]
                int start = std::stoi(match[1]) + reservedRegionSize;
                int end   = std::stoi(match[2]) + reservedRegionSize;
                rv += fmt::format("v[{}:{}]", start, end);
            }
            else
            {
                // matched v#
                rv += fmt::format("v{}", std::stoi(match[3]) + reservedRegionSize);
            }
            lastPos = it->position() + it->length();
        }
        rv.append(instr, lastPos, std::string::npos);

        return rv;
    }
}
