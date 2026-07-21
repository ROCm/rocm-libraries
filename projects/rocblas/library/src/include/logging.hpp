/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "handle.hpp"
#include "rocblas_ostream.hpp"
#include "tuple_helper.hpp"
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

/************************************************************************************
 * Profile kernel arguments
 ************************************************************************************/
template <typename TUP>
class rocblas_internal_argument_profile
{
    // Output stream
    mutable rocblas_internal_ostream os;

    // Mutex for multithreaded access to table
    mutable std::shared_timed_mutex mutex;

    // Table mapping argument tuples into counts
    // size_t is used for the map target type since atomic types are not movable, and
    // the map elements will only be moved when we hold an exclusive lock to the map.
    std::unordered_map<TUP,
                       size_t,
                       typename tuple_helper::hash_t<TUP>,
                       typename tuple_helper::equal_t<TUP>>
        map;

public:
    // A tuple of arguments is looked up in an unordered map.
    // A count of the number of calls with these arguments is kept.
    // arg is assumed to be an rvalue for efficiency
    void operator()(TUP&& arg)
    {
        { // Acquire a shared lock for reading map
            std::shared_lock<std::shared_timed_mutex> lock(mutex);

            // Look up the tuple in the map
            auto p = map.find(arg);

            // If tuple already exists, atomically increment count and return
            if(p != map.end())
            {
                __atomic_fetch_add(&p->second, 1, __ATOMIC_SEQ_CST);
                return;
            }
        } // Release shared lock

        { // Acquire an exclusive lock for modifying map
            std::lock_guard<std::shared_timed_mutex> lock(mutex);

            // If doesn't already exist, insert tuple by moving arg and initializing count to 0.
            // Increment the count after searching for tuple and returning old or new match.
            // We hold a lock to the map, so we don't have to increment the count atomically.
            map.emplace(std::move(arg), 0).first->second++;
        } // Release exclusive lock
    }

    // Constructor
    // We must duplicate the rocblas_internal_ostream to avoid dependence on static destruction order
    explicit rocblas_internal_argument_profile(rocblas_internal_ostream& os)
        : os(os.dup())
    {
    }

    // Dump the current profile
    void dump() const
    {
        // Acquire an exclusive lock to use map
        std::lock_guard<std::shared_timed_mutex> lock(mutex);

        // Clear the output buffer
        os.clear();

        // Print all of the tuples in the map
        for(const auto& p : map)
        {
            os << "- ";
            tuple_helper::print_tuple_pairs(
                os, std::tuple_cat(p.first, std::make_tuple("call_count", p.second)));
        }

        // Flush out the dump
        os.flush();
    }

    // Cleanup handler which dumps profile at destruction
    ~rocblas_internal_argument_profile()
    try
    {
        dump();
    }
    catch(...)
    {
        return;
    }
};

/*************************************************
 * Trace log scalar values pointed to by pointer *
 *************************************************/

template <typename T>
std::string rocblas_internal_log_trace_scalar_value(rocblas_handle handle, const T* value);

#define LOG_TRACE_SCALAR_VALUE(handle, value) rocblas_internal_log_trace_scalar_value(handle, value)

/*************************************************
 * Bench log scalar values pointed to by pointer *
 *************************************************/

template <typename T>
inline std::string rocblas_internal_log_bench_scalar_value(const char* name, const T* value)
{
    rocblas_internal_ostream ss;
    if constexpr(!rocblas_is_complex<T>)
    {
        ss << "--" << name << " " << (value ? *value : std::numeric_limits<T>::quiet_NaN());
    }
    else
    {
        ss << "--" << name << " "
           << (value ? std::real(*value)
                     : std::numeric_limits<typename T::value_type>::quiet_NaN());
        if(value && std::imag(*value))
            ss << " --" << name << "i " << std::imag(*value);
    }
    return ss.str();
}

template <>
inline std::string rocblas_internal_log_bench_scalar_value<rocblas_half>(const char*         name,
                                                                         const rocblas_half* value)
{
    rocblas_internal_ostream ss;
    ss << "--" << name << " " << (value ? float(*value) : std::numeric_limits<float>::quiet_NaN());
    return ss.str();
}

template <typename T>
std::string rocblas_internal_log_bench_scalar_value(rocblas_handle handle,
                                                    const char*    name,
                                                    const T*       value);

#define LOG_BENCH_SCALAR_VALUE(handle, name) \
    rocblas_internal_log_bench_scalar_value(handle, #name, name)

/*********************************************************************
 * Bench log precision for mixed precision scal_ex and nrm2_ex calls *
 *********************************************************************/
std::string rocblas_internal_log_bench_ex_precisions(rocblas_datatype a_type,
                                                     rocblas_datatype x_type,
                                                     rocblas_datatype ex_type);

/******************************************************************
 * Log alpha and beta with dynamic compute_type in *_ex functions *
 ******************************************************************/
rocblas_status rocblas_internal_log_trace_alpha_beta_ex(rocblas_datatype          compute_type,
                                                        const void*               alpha,
                                                        const void*               beta,
                                                        rocblas_internal_ostream& alphass,
                                                        rocblas_internal_ostream& betass);

rocblas_status rocblas_internal_log_bench_alpha_beta_ex(rocblas_datatype compute_type,
                                                        const void*      alpha,
                                                        const void*      beta,
                                                        std::string&     alphas,
                                                        std::string&     betas);

template <typename T>
double rocblas_internal_value_category(const T* beta, rocblas_datatype compute_type);

extern const char* c_rocblas_internal;

/******************************************************************
 * ROCBLAS LOGGER *
 ******************************************************************/

class rocblas_internal_logger
{
public:
    rocblas_internal_logger() = default;

    void log_endline(rocblas_internal_ostream& os);
    void log_cleanup();

    template <typename H, typename... Ts>
    void log_arguments(rocblas_internal_ostream& os, const char* sep, H&& head, Ts&&... xs)
    {
        os << std::forward<H>(head);
        // TODO: Replace with C++17 fold expression
        // ((os << sep << std::forward<Ts>(xs)), ...);
        (void)(int[]){(os << sep << std::forward<Ts>(xs), 0)...};

        log_endline(os);
    }

    // if trace logging is turned on with
    // (handle->layer_mode & rocblas_layer_mode_log_trace) != 0
    // log_function will call log_arguments to log arguments with a comma separator
    template <typename... Ts>
    void log_trace(rocblas_handle handle, Ts&&... xs)
    {
        log_arguments(*handle->log_trace_os, ",", std::forward<Ts>(xs)..., handle->atomics_mode);
    }

    // if bench logging is turned on with
    // (handle->layer_mode & rocblas_layer_mode_log_bench) != 0
    // log_bench will call log_arguments to log a string that
    // can be input to the executable rocblas-bench.
    template <typename... Ts>
    void log_bench(rocblas_handle handle, Ts&&... xs)
    {
        const char* atomics_str
            = handle->atomics_mode == rocblas_atomics_allowed ? "--atomics_allowed" : "";
        log_arguments(*handle->log_bench_os, " ", std::forward<Ts>(xs)..., atomics_str);
    }

    // if profile logging is turned on with
    // (handle->layer_mode & rocblas_layer_mode_log_profile) != 0
    // log_profile will call rocblas_internal_argument_profile to profile actual arguments,
    // keeping count of the number of times each set of arguments is used
    template <typename... Ts>
    void log_profile(rocblas_handle handle, const char* func, Ts&&... xs)
    {
        // Make a tuple with the arguments
        auto tup = std::make_tuple("rocblas_function",
                                   func,
                                   "atomics_mode",
                                   handle->atomics_mode,
                                   std::forward<Ts>(xs)...);

        // Set up profile
        static rocblas_internal_argument_profile<decltype(tup)> profile(*handle->log_profile_os);

        // Add at_quick_exit handler in case the program exits early
        static int aqe = at_quick_exit([] { profile.~rocblas_internal_argument_profile(); });

        // Profile the tuple
        profile(std::move(tup));
    }

    ~rocblas_internal_logger()
    {
        if(m_active)
        {
            log_cleanup(); // see log_endline
        }
    }

private:
    bool m_active{false};
};
