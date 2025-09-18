/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
 * ************************************************************************ */
#pragma once
#include "auto_testing_bad_arg.hpp"
#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "rocsparse.hpp"
#include "rocsparse_check.hpp"
#include "rocsparse_graph.hpp"
#include "rocsparse_matrix_factory.hpp"
#include "rocsparse_reproducibility.hpp"
#include "rocsparse_reproducibility_test_save.hpp"
#include "rocsparse_vector_utils.hpp"
#include "utility.hpp"

template <typename T>
inline T* rocsparse_fake_pointer()
{
    return static_cast<T*>((void*)0x4);
}

template <typename T>
inline T rocsparse_nan()
{
    return std::numeric_limits<T>::quiet_NaN();
}

template <>
inline rocsparse_float_complex rocsparse_nan<rocsparse_float_complex>()
{
    return rocsparse_float_complex(std::numeric_limits<float>::quiet_NaN(),
                                   std::numeric_limits<float>::quiet_NaN());
}

template <>
inline rocsparse_double_complex rocsparse_nan<rocsparse_double_complex>()
{
    return rocsparse_double_complex(std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::quiet_NaN());
}

template <typename T>
inline T rocsparse_inf()
{
    return std::numeric_limits<T>::infinity();
}

template <>
inline rocsparse_float_complex rocsparse_inf<rocsparse_float_complex>()
{
    return rocsparse_float_complex(std::numeric_limits<float>::infinity(),
                                   std::numeric_limits<float>::infinity());
}

template <>
inline rocsparse_double_complex rocsparse_inf<rocsparse_double_complex>()
{
    return rocsparse_double_complex(std::numeric_limits<double>::infinity(),
                                    std::numeric_limits<double>::infinity());
}

template <typename T>
floating_data_t<T> get_near_check_tol(const Arguments& arg)
{
    return arg.convert_to_int
               ? static_cast<floating_data_t<T>>(0)
               : static_cast<floating_data_t<T>>(arg.tolm) * default_tolerance<T>::value;
}

//
// Compute gflops
//

inline double get_gpu_gflops(double gpu_time_used, double gflop_count)
{
    return gflop_count / gpu_time_used * 1e6;
}

template <typename F, typename... Ts>
inline double get_gpu_gflops(double gpu_time_used, F count, Ts... ts)
{
    return get_gpu_gflops(gpu_time_used, count(ts...));
}

//
// Compute gbyte
//
inline double get_gpu_gbyte(double gpu_time_used, double gbyte_count)
{
    return gbyte_count / gpu_time_used * 1e6;
}

template <typename F, typename... Ts>
inline double get_gpu_gbyte(double gpu_time_used, F count, Ts... ts)
{
    return get_gpu_gbyte(gpu_time_used, count(ts...));
}

inline double get_gpu_time_msec(double gpu_time_used)
{
    return gpu_time_used / 1e3;
}

// Check hmm availability
inline bool is_hmm_enabled()
{
    int deviceID, hmm_enabled;
    if(hipGetDevice(&deviceID) != hipSuccess)
    {
        std::cout << "Failed to read device id..." << std::endl;
        return false;
    }
    if(hipDeviceGetAttribute(&hmm_enabled, hipDeviceAttributeManagedMemory, deviceID) != hipSuccess)
    {
        std::cout << "Failed to read device attribute..." << std::endl;
        return false;
    }

    return hmm_enabled;
}

namespace rocsparse_clients
{
    struct archnames
    {
        static constexpr const char* const gfx90a  = "gfx90a";
        static constexpr const char* const gfx942  = "gfx942";
        static constexpr const char* const gfx803  = "gfx803";
        static constexpr const char* const gfx900  = "gfx900";
        static constexpr const char* const gfx906  = "gfx906";
        static constexpr const char* const gfx908  = "gfx908";
        static constexpr const char* const gfx950  = "gfx950";
        static constexpr const char* const gfx1030 = "gfx1030";
        static constexpr const char* const gfx1100 = "gfx1100";
        static constexpr const char* const gfx1101 = "gfx1101";
        static constexpr const char* const gfx1102 = "gfx1102";
        static constexpr const char* const gfx1151 = "gfx1151";
        static constexpr const char* const gfx1200 = "gfx1200";
        static constexpr const char* const gfx1201 = "gfx1201";
    };

    inline bool archname_from(const char* archname, const char* name)
    {
        return (0 == strncmp(name, archname, strlen(name)));
    }

    template <typename... P>
    inline bool archname_from(const char* archname, const char* name, P... p)
    {
        if(0 == strncmp(name, archname, strlen(name)))
        {
            return true;
        }
        return rocsparse_clients::archname_from(archname, std::forward<P>(p)...);
    }

    template <typename... P>
    inline bool current_arch_from(P... p)
    {
        int             dev;
        hipDeviceProp_t prop;
        std::ignore = hipGetDevice(&dev);
        std::ignore = hipGetDeviceProperties(&prop, dev);
        return rocsparse_clients::archname_from(prop.gcnArchName, std::forward<P>(p)...);
    }
}

#define ROCSPARSE_DEBUG_VERBOSE_ON rocsparse_enable_debug_verbose()
#define ROCSPARSE_DEBUG_VERBOSE_OFF rocsparse_disable_debug_verbose()
