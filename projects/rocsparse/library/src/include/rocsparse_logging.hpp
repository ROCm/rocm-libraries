/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <fstream>
#include <string>

#include "rocsparse_handle.hpp"

#if defined(ROCSPARSE_BUILT_WITH_ROCTX)
#include "rocsparse_roctx.hpp"
#include <roctracer/roctx.h>
#endif

namespace rocsparse
{
    /**
 *  @brief Logging function
 *
 *  @details
 *  open_log_stream Open stream log_os for logging.
 *                  If the environment variable with name environment_variable_name
 *                  is not set, then stream log_os to std::cerr.
 *                  Else open a file at the full logfile path contained in
 *                  the environment variable.
 *                  If opening the file suceeds, stream to the file
 *                  else stream to std::cerr.
 *
 *  @param[in]
 *  environment_variable_name   std::string
 *                              Name of environment variable that contains
 *                              the full logfile path.
 *
 *  @parm[out]
 *  log_os      std::ostream**
 *              Output stream. Stream to std:err if environment_variable_name
 *              is not set, else set to stream to log_ofs
 *
 *  @parm[out]
 *  log_ofs     std::ofstream*
 *              Output file stream. If log_ofs->is_open()==true, then log_os
 *              will stream to log_ofs. Else it will stream to std::cerr.
 */

    inline void open_log_stream(std::ostream** log_os,
                                std::ofstream* log_ofs,
                                std::string    environment_variable_name)
    {
        *log_os = &std::cerr;

        char const* environment_variable_value = getenv(environment_variable_name.c_str());

        if(environment_variable_value != NULL)
        {
            // if environment variable is set, open file at logfile_pathname contained in the
            // environment variable
            std::string logfile_pathname = (std::string)environment_variable_value;
            log_ofs->open(logfile_pathname);

            // if log_ofs is open, then stream to log_ofs, else log_os is already
            // set equal to std::cerr
            if(log_ofs->is_open() == true)
            {
                *log_os = log_ofs;
            }
        }
    }

    /**
 * @brief Invoke functor for each argument in variadic parameter pack.
 * @details
 * The variatic template function each_args applies the functor f
 * to each argument in the expansion of the parameter pack xs...

 * Note that in ((void)f(xs),0) the C/C++ comma operator evaluates
 * the first expression (void)f(xs) and discards the output, then
 * it evaluates the second expression 0 and returns the output 0.

 * It thus calls (void)f(xs) on each parameter in xs... as a bye-product of
 * building the initializer_list 0,0,0,...0. The initializer_list is discarded.
 *
 * @param f functor to apply to each argument
 *
 * @parm xs variadic parameter pack with list of arguments
 */
    template <typename F, typename... Ts>
    void each_args(F f, Ts&&... xs)
    {
        (void)std::initializer_list<int>{((void)f(xs), 0)...};
    }

    /**
 * @brief Workaround for gcc warnings when each_args called with single argument
 *        and no parameter pack.
 */
    template <typename F>
    void each_args(F)
    {
    }

    /**
 * @brief Functor for logging arguments
 *
 * @details Functor to log single argument to ofs.
 * The overloaded () in log_arg is the function call operator.
 * The definition in log_arg says "objects of type log_arg can have
 * the function call operator () applied to them with operand x,
 * and it will output x to ofs and return void".
 */
    struct log_arg
    {
        log_arg(std::ostream& os, std::string& separator)
            : os_(os)
            , separator_(separator)
        {
        }

        /// Generic overload for () operator.
        template <typename T>
        void operator()(T& x) const
        {
            os_ << separator_ << x;
        }

        /// Overload () operator for rocsparse_float_complex.
        void operator()(const rocsparse_float_complex complex_value) const
        {
            os_ << separator_ << std::real(complex_value) << separator_ << std::imag(complex_value);
        }

        /// Overload () operator for rocsparse_double_complex.
        void operator()(const rocsparse_double_complex complex_value) const
        {
            os_ << separator_ << std::real(complex_value) << separator_ << std::imag(complex_value);
        }

    private:
        std::ostream& os_; ///< Output stream.
        std::string&  separator_; ///< Separator: output preceding argument.
    };

    /**
 * @brief Logging function
 *
 * @details
 * log_arguments   Log arguments to output file stream. Arguments
 *                 are preceded by new line, and separated by separator.
 *
 * @param[in]
 * ofs             std::ofstream
 *                 Open output stream file.
 *
 * @param[in]
 * separator       std::string
 *                 Separator to print between arguments.
 *
 * @param[in]
 * head            <typename H>
 *                 First argument to log. It is preceded by newline.
 *
 * @param[in]
 * xs              <typename... Ts>
 *                 Variadic parameter pack. Each argument in variadic
 *                 parameter pack is logged, and it is preceded by
 *                 separator.
 */
    template <typename H, typename... Ts>
    void log_arguments(std::ostream& os, std::string& separator, H head, Ts&&... xs)
    {
        os << " [Note: trace, debug, and bench logging is deprecated and will be removed in a "
              "future release] \n"
           << head;
        rocsparse::each_args(log_arg{os, separator}, std::forward<Ts>(xs)...);
    }

    /**
 * @brief Logging function
 *
 * @details
 * log_arguments   Log argument to output file stream. Argument
 *                 is preceded by new line.
 *
 * @param[in]
 * ofs             std::ofstream
 *                 open output stream file.
 *
 * @param[in]
 * separator       std::string
 *                 Not used.
 *
 * @param[in]
 * head            <typename H>
 *                 Argument to log. It is preceded by newline.
 */
    template <typename H>
    void log_argument(std::ostream& os, std::string& separator, H head)
    {
        os << " [Note: trace, debug, and bench logging is deprecated and will be removed in a "
              "future release] \n"
           << head;
    }

    /**
 * @brief Logging function
 *
 * @details
 * log_arguments   Log argument to output file stream. Argument
 *                 is preceded by new line.
 *
 * @param[in]
 * ofs             std::ofstream
 *                 open output stream file.
 *
 * @param[in]
 * head            <typename H>
 *                 Argument to log. It is preceded by newline.
 */
    template <typename H>
    void log_argument(std::ostream& os, H head)
    {
        os << " [Note: trace, debug, and bench logging is deprecated and will be removed in a "
              "future release] \n"
           << head;
    }

    // if trace logging is turned on with
    // (handle->layer_mode & rocsparse_layer_mode_log_trace) == true
    // then
    // log_function will call log_arguments to log function
    // arguments with a comma separator
    template <typename H, typename... Ts>
    void log_trace(rocsparse_handle handle, H head, Ts&&... xs)
    {
        if(nullptr != handle)
        {
            if(handle->layer_mode & rocsparse_layer_mode_log_trace)
            {
                std::string comma_separator = ",";

                std::ostream* os = handle->log_trace_os;
                rocsparse::log_arguments(*os, comma_separator, head, std::forward<Ts>(xs)...);
            }
        }
    }

    template <typename H, typename... Ts>
    void log_trace(H head, rocsparse_handle handle, Ts&&... xs)
    {
        rocsparse::log_trace(handle, head, xs...);
    }

    // if bench logging is turned on with
    // (handle->layer_mode & rocsparse_layer_mode_log_bench) == true
    // then
    // log_bench will call log_arguments to log a string that
    // can be input to the executable rocsparse-bench.
    template <typename H, typename... Ts>
    void log_bench(rocsparse_handle handle, H head, std::string precision, Ts&&... xs)
    {
        if(nullptr != handle)
        {
            if(handle->layer_mode & rocsparse_layer_mode_log_bench)
            {
                std::string space_separator = " ";

                std::ostream* os = handle->log_bench_os;
                rocsparse::log_arguments(
                    *os, space_separator, head, precision, std::forward<Ts>(xs)...);
            }
        }
    }

    // if debug logging is turned on with
    // (handle->layer_mode & rocsparse_layer_mode_log_debug) == true
    // then
    // log_debug will call log_arguments to log a error message
    // when a routine returns a status which is not rocsparse_status_success.
    static inline void log_debug(rocsparse_handle handle, std::string message)
    {
        if(nullptr != handle)
        {
            if(handle->layer_mode & rocsparse_layer_mode_log_debug)
            {
                std::string space_separator = " ";

                std::ostream* os = handle->log_debug_os;
                rocsparse::log_arguments(*os, space_separator, message);
            }
        }
    }

    // Trace log scalar values pointed to by pointer
    template <typename T>
    T log_trace_scalar_value(const T* value)
    {
        return value ? *value : std::numeric_limits<T>::quiet_NaN();
    }

    template <typename T>
    T log_trace_scalar_value(rocsparse_handle handle, const T* value)
    {
        if(nullptr != handle)
        {
            if(handle->layer_mode & rocsparse_layer_mode_log_trace)
            {
                T host;
                if(value && handle->pointer_mode == rocsparse_pointer_mode_device)
                {
                    hipStreamCaptureStatus capture_status;
                    RETURN_IF_HIP_ERROR(hipStreamIsCapturing(handle->stream, &capture_status));

                    if(capture_status == hipStreamCaptureStatusNone)
                    {
                        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                            &host, value, sizeof(host), hipMemcpyDeviceToHost, handle->stream));
                        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
                        value = &host;
                    }
                    else
                    {
                        value = nullptr;
                    }
                }
                return rocsparse::log_trace_scalar_value(value);
            }
        }
        return T{};
    }

#define LOG_TRACE_SCALAR_VALUE(handle, value) rocsparse::log_trace_scalar_value(handle, value)

    // Bench log scalar values pointed to by pointer
    template <typename T>
    T log_bench_scalar_value(const T* value)
    {
        return (value ? *value : std::numeric_limits<T>::quiet_NaN());
    }

    template <typename T>
    T log_bench_scalar_value(rocsparse_handle handle, const T* value)
    {
        if(handle->layer_mode & rocsparse_layer_mode_log_bench)
        {
            T host;
            if(value && handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                hipStreamCaptureStatus capture_status;
                RETURN_IF_HIP_ERROR(hipStreamIsCapturing(handle->stream, &capture_status));

                if(capture_status == hipStreamCaptureStatusNone)
                {
                    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                        &host, value, sizeof(host), hipMemcpyDeviceToHost, handle->stream));
                    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
                    value = &host;
                }
                else
                {
                    value = nullptr;
                }
            }
            return rocsparse::log_bench_scalar_value(value);
        }
        return T{};
    }

#define LOG_BENCH_SCALAR_VALUE(handle, name) log_bench_scalar_value(handle, name)

    // replaces X in string with s, d, c, z or h depending on typename T
    template <typename T>
    std::string replaceX(std::string input_string)
    {
        if(std::is_same<T, float>::value)
        {
            std::replace(input_string.begin(), input_string.end(), 'X', 's');
        }
        else if(std::is_same<T, double>::value)
        {
            std::replace(input_string.begin(), input_string.end(), 'X', 'd');
        }
        /*
    else if(std::is_same<T, rocsparse_float_complex>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'c');
    }
    else if(std::is_same<T, rocsparse_double_complex>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'z');
    }
    else if(std::is_same<T, rocsparse_half>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'h');
    }
    */
        return input_string;
    }

#if defined(ROCSPARSE_BUILT_WITH_ROCTX)
    class internal_roctx
    {
    public:
        internal_roctx(const char* name)
        {
            if(rocsparse_roctx_variables.get_roctx_enabled())
            {
                roctxRangePush(name);
            }
        }

        ~internal_roctx()
        {
            if(rocsparse_roctx_variables.get_roctx_enabled())
            {
                roctxRangePop();
            }
        }
    };
#define ROCSPARSE_ROCTX_TRACE rocsparse::internal_roctx roctx(__FUNCTION__);
#else
#define ROCSPARSE_ROCTX_TRACE
#endif

#define ROCSPARSE_ROUTINE_TRACE ROCSPARSE_ROCTX_TRACE
}
