/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#ifndef HIPTENSOR_TEST_UTILS_HPP
#define HIPTENSOR_TEST_UTILS_HPP

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iterator>
#include <math.h>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <vector>

// hiptensor includes
#include <hiptensor/hiptensor.hpp>
#include <hiptensor/hiptensor_types.hpp>
#include <hiptensor/internal/hiptensor_utility.hpp>
#include <hiptensor/internal/types.hpp>

#include "device/common.hpp"
#include "hip_resource.hpp"

#define HIPTENSOR_FREE_DEVICE(ptr)     \
    if(ptr != nullptr)                 \
    {                                  \
        CHECK_HIP_ERROR(hipFree(ptr)); \
    }

#define HIPTENSOR_FREE_HOST(ptr)           \
    if(ptr != nullptr)                     \
    {                                      \
        CHECK_HIP_ERROR(hipHostFree(ptr)); \
    }

inline double getEpsilon(hiptensorComputeType_t id)
{
    auto toDouble = [](auto const& val) { return static_cast<double>(static_cast<float>(val)); };

    if(id == HIPTENSOR_COMPUTE_16F)
    {
        return toDouble(std::numeric_limits<_Float16>::epsilon());
    }
    else if(id == HIPTENSOR_COMPUTE_16BF)
    {
        return toDouble(std::numeric_limits<hip_bfloat16>::epsilon());
    }
    else if(id == HIPTENSOR_COMPUTE_32F)
    {
        return toDouble(std::numeric_limits<float>::epsilon());
    }
    else if(id == HIPTENSOR_COMPUTE_64F)
    {
        return toDouble(std::numeric_limits<double>::epsilon());
    }
    else if(id == HIPTENSOR_COMPUTE_8U)
    {
        return 0;
    }
    else if(id == HIPTENSOR_COMPUTE_8I)
    {
        return 0;
    }
    else if(id == HIPTENSOR_COMPUTE_32U)
    {
        return 0;
    }
    else if(id == HIPTENSOR_COMPUTE_32I)
    {
        return 0;
    }
    else if(id == HIPTENSOR_COMPUTE_C32F)
    {
        return toDouble(std::numeric_limits<float>::epsilon());
    }
    else if(id == HIPTENSOR_COMPUTE_C64F)
    {
        return toDouble(std::numeric_limits<double>::epsilon());
    }
    else
    {
#if !NDEBUG
        std::cout << "Unhandled hiptensorComputeType_t: " << id << std::endl;
#endif // !NDEBUG
        return 0;
    }
}

inline bool isF32Supported()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;

    CHECK_HIP_ERROR(hipGetDevice(&mHandle));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

    std::string deviceName(mProps.gcnArchName);

    return (deviceName.find("gfx908") != std::string::npos)
           || (deviceName.find("gfx90a") != std::string::npos)
           || (deviceName.find("gfx940") != std::string::npos)
           || (deviceName.find("gfx941") != std::string::npos)
           || (deviceName.find("gfx942") != std::string::npos);
}

inline bool isF64Supported()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;

    CHECK_HIP_ERROR(hipGetDevice(&mHandle));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

    std::string deviceName(mProps.gcnArchName);

    return (deviceName.find("gfx90a") != std::string::npos)
           || (deviceName.find("gfx940") != std::string::npos)
           || (deviceName.find("gfx941") != std::string::npos)
           || (deviceName.find("gfx942") != std::string::npos);
}

template <typename intT1,
          class = typename std::enable_if<std::is_integral<intT1>::value>::type,
          typename intT2,
          class = typename std::enable_if<std::is_integral<intT2>::value>::type>
static constexpr intT1 ceilDiv(const intT1 numerator, const intT2 divisor)
{
    return (numerator + divisor - 1) / divisor;
}

template <typename Container>
auto getProduct(const Container&               container,
                typename Container::value_type init = typename Container::value_type{1})
{
    return std::accumulate(std::begin(container),
                           std::end(container),
                           init,
                           std::multiplies<typename Container::value_type>{});
}

// fill kernel for 'elementSize' elements
template <typename DataType>
__host__ static inline void fillLaunchKernel(DataType* data, uint32_t elementSize, uint32_t seed)
{
    auto blockDim = dim3(1024, 1, 1);
    auto gridDim  = dim3(ceilDiv(elementSize, blockDim.x), 1, 1);
    hipLaunchKernelGGL((fillKernel<DataType>), gridDim, blockDim, 0, 0, data, elementSize, seed);
}

// fill kernel wrapper for 'elementSize' elements with a specific value
template <typename DataType>
__host__ static inline void
    fillValLaunchKernel(DataType* data, uint32_t elementSize, DataType value)
{
    auto blockDim = dim3(1024, 1, 1);
    auto gridDim  = dim3(ceilDiv(elementSize, blockDim.x), 1, 1);
    hipLaunchKernelGGL(
        (fillValKernel<DataType>), gridDim, blockDim, 0, 0, data, elementSize, value);
}

template <typename DDataType>
std::pair<bool, double> compareEqual(DDataType const*       deviceD,
                                     DDataType const*       hostD,
                                     std::size_t            elementsD,
                                     hiptensorComputeType_t computeType,
                                     double                 tolerance = 0.0)
{
    bool   retval             = true;
    double max_relative_error = 0.0;

    auto toDouble
        = [](DDataType const& val) { return static_cast<double>(static_cast<float>(val)); };

    bool       isInf = false;
    bool       isNaN = false;
    std::mutex writeMutex;

#pragma omp parallel for
    for(int i = 0; i < elementsD; ++i)
    {
        auto valDevice = deviceD[i];
        auto valHost   = hostD[i];

        auto numerator = fabs(toDouble(valDevice) - toDouble(valHost));
        auto divisor   = fabs(toDouble(valDevice)) + fabs(toDouble(valHost)) + 1.0;

        if(std::isinf(numerator) || std::isinf(divisor))
        {
#pragma omp atomic
            isInf |= true;
        }
        else
        {
            auto relative_error = numerator / divisor;
            if(std::isnan(relative_error))
            {
#pragma omp atomic
                isNaN |= true;
            }
            else if(relative_error > max_relative_error)
            {
                const std::lock_guard<std::mutex> guard(writeMutex);
                // Double check in case of stall
                if(relative_error > max_relative_error)
                {
                    max_relative_error = relative_error;
                }
            }
        }

        if(isInf || isNaN)
        {
            i = elementsD;
        }
    }

    if(tolerance == 0.0)
    {
        // use the same default tolerance value as CK
        if(computeType == HIPTENSOR_COMPUTE_16BF
           || std::is_same_v<DDataType, hiptensor::bfloat16_t>)
        {
            const double epsilon = std::pow(2, -7);
            tolerance            = epsilon * 2;
        }
        else if(computeType == HIPTENSOR_COMPUTE_16F
                || std::is_same_v<DDataType, hiptensor::float16_t>)
        {
            const double epsilon = std::pow(2, -10);
            tolerance            = epsilon * 2;
        }
        else
        {
            tolerance = 1e-5;
        }
    }

    if(isInf)
    {
        retval             = false;
        max_relative_error = std::numeric_limits<DDataType>::infinity();
    }
    else if(isNaN)
    {
        retval             = false;
        max_relative_error = std::numeric_limits<DDataType>::signaling_NaN();
    }
    else if(max_relative_error > tolerance)
    {
        retval = false;
    }

    return std::make_pair(retval, max_relative_error);
}

template <typename DDataType>
std::pair<bool, double> compareEqualLaunchKernel(DDataType*             deviceD,
                                                 DDataType*             hostD,
                                                 std::size_t            elementsD,
                                                 hiptensorComputeType_t computeType,
                                                 double                 tolerance = 0.0)
{
    static hiptensor::HipResource::DevicePtrT relativeErrorPtr;
    static std::size_t                        relativeErrorSize = 0;

    auto blockDim = dim3(1024, 1, 1);
    auto gridDim  = dim3(ceilDiv(elementsD, blockDim.x), 1, 1);

    double maxRelativeError;

    std::size_t requiredErrorSize = elementsD * sizeof(double);
    if(requiredErrorSize > relativeErrorSize)
    {
        hiptensor::HipResource::reallocDevice(relativeErrorPtr, requiredErrorSize);
        relativeErrorSize = requiredErrorSize;
    }

    hipEvent_t syncEvent;
    CHECK_HIP_ERROR(hipEventCreate(&syncEvent));

    // Calculate the relative error for each element of Tensor D
    hipLaunchKernelGGL((compareEqualKernel<DDataType>),
                       gridDim,
                       blockDim,
                       0,
                       0,
                       deviceD,
                       hostD,
                       static_cast<double*>(relativeErrorPtr.get()),
                       elementsD);
    CHECK_HIP_ERROR(hipEventRecord(syncEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(syncEvent));

    // Determine the maximum relative error
    blockDim             = dim3(512, 1, 1);
    uint32_t maxElements = 1024;
    uint32_t offset      = 1;

    for(uint32_t i = elementsD; i > 1; i = ceilDiv(i, maxElements))
    {
        gridDim       = dim3(ceilDiv(i, maxElements), 1, 1);
        auto elements = i > maxElements ? maxElements : i;

        hipLaunchKernelGGL((maxReduceKernel),
                           gridDim,
                           blockDim,
                           0,
                           0,
                           static_cast<double*>(relativeErrorPtr.get()),
                           elements,
                           offset,
                           elementsD);

        CHECK_HIP_ERROR(hipEventRecord(syncEvent));
        CHECK_HIP_ERROR(hipEventSynchronize(syncEvent));
        offset = offset * maxElements;
    }
    CHECK_HIP_ERROR(hipEventDestroy(syncEvent));

    CHECK_HIP_ERROR(hipMemcpy(&maxRelativeError,
                              static_cast<double*>(relativeErrorPtr.get()),
                              sizeof(double),
                              hipMemcpyDeviceToHost));

    bool retval = true;
    bool isNaN  = std::isnan(maxRelativeError);

    auto toDouble
        = [](DDataType const& val) { return static_cast<double>(static_cast<float>(val)); };

    if(tolerance == 0.0)
    {
        // use the same default tolerance value as CK
        if(computeType == HIPTENSOR_COMPUTE_16BF
           || std::is_same_v<DDataType, hiptensor::bfloat16_t>)
        {
            const double epsilon = std::pow(2, -7);
            tolerance            = epsilon * 2;
        }
        else if(computeType == HIPTENSOR_COMPUTE_16F
                || std::is_same_v<DDataType, hiptensor::float16_t>)
        {
            const double epsilon = std::pow(2, -10);
            tolerance            = epsilon * 2;
        }
        else
        {
            tolerance = 1e-5;
        }
    }

    if(isNaN)
    {
        retval           = false;
        maxRelativeError = std::numeric_limits<DDataType>::signaling_NaN();
    }
    else if(maxRelativeError > (tolerance))
    {
        retval = false;
    }

    return std::make_pair(retval, maxRelativeError);
}

namespace hiptensor
{
    template <typename T>
    struct has_const_iterator
    {
    private:
        typedef char yes;
        typedef struct
        {
            char array[2];
        } no;

        template <typename C>
        static yes test(typename C::const_iterator*);
        template <typename C>
        static no test(...);

    public:
        static const bool value = sizeof(test<T>(0)) == sizeof(yes);
        typedef T         type;
    };

    template <typename T>
    struct has_begin_end
    {
        template <typename C>
        static char (&f(
            typename std::enable_if<std::is_same<decltype(static_cast<typename C::const_iterator (
                                                              C::*)() const>(&C::begin)),
                                                 typename C::const_iterator (C::*)() const>::value,
                                    void>::type*))[1];

        template <typename C>
        static char (&f(...))[2];

        template <typename C>
        static char (&g(
            typename std::enable_if<std::is_same<decltype(static_cast<typename C::const_iterator (
                                                              C::*)() const>(&C::end)),
                                                 typename C::const_iterator (C::*)() const>::value,
                                    void>::type*))[1];

        template <typename C>
        static char (&g(...))[2];

        static bool const beg_value = sizeof(f<T>(0)) == 1;
        static bool const end_value = sizeof(g<T>(0)) == 1;
    };

    template <typename T>
    struct is_container
        : std::integral_constant<bool,
                                 has_const_iterator<T>::value && has_begin_end<T>::beg_value
                                     && has_begin_end<T>::end_value>
    {
    };

    // Write the content of arrays as value of a CSV field
    // do not use ',' since it is field separator in CSV
    // Example: arrays = [[1, 2], [3, 4]
    // Output:  '[[1 2] [3 4]]'

    // Main print function
    template <typename Container>
    std::ostream& printContainerInCsv(Container const& container, std::ostream& outStream)
    {
        outStream << "[";
        for(const auto& element : container)
        {
            if constexpr(is_container<typename Container::value_type>::value)
            {
                printContainerInCsv(element, outStream);
            }
            else
            {
                outStream << element << " ";
            }
        }
        outStream << "] ";
        return outStream;
    }
}

namespace std
{
    template <typename T>
    ostream& operator<<(ostream& os, const std::vector<T>& vec)
    {
        os << "[ ";
        for(auto i = 0; i < vec.size(); i++)
        {
            if(i < vec.size() - 1)
            {
                os << vec[i] << ", ";
            }
            else
            {
                os << vec[i];
            }
        }
        os << " ]";

        return os;
    }
}

#endif // HIPTENSOR_TEST_UTILS_HPP
