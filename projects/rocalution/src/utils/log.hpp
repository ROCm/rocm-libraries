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

#ifndef ROCALUTION_UTILS_LOG_HPP_
#define ROCALUTION_UTILS_LOG_HPP_

#include "../base/backend_manager.hpp"
#include "def.hpp"

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>

namespace rocalution
{

    void _rocalution_open_log_file(void);
    void _rocalution_close_log_file(void);

    template <typename F, typename... Ts>
    void each_args(F f, Ts&... xs)
    {
        (void)std::initializer_list<int>{((void)f(xs), 0)...};
    }

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

    private:
        std::ostream& os_;
        std::string&  separator_;
    };

    template <typename P, typename F, typename... Ts>
    void log_arguments(std::ostream& os, std::string& separator, int rank, P ptr, F fct, Ts&... xs)
    {
        os << "\n[rank:" << rank << "]# ";
        os << "Obj addr: " << ptr << "; ";
        os << "fct: " << fct;
        each_args(log_arg{os, separator}, xs...);
    }

    template <typename P, typename F, typename... Ts>
    void log_debug(P ptr, F fct, Ts&... xs)
    {
        if(_get_backend_descriptor()->log_file != NULL)
        {
            // LCOV_EXCL_START
            std::string   comma_separator = ", ";
            std::ostream* os              = _get_backend_descriptor()->log_file;
            log_arguments(*os, comma_separator, _get_backend_descriptor()->rank, ptr, fct, xs...);
            // LCOV_EXCL_STOP
        }
    }

} // namespace rocalution

// Do not edit
#ifdef DEBUG_MODE

#undef VERBOSE_LEVEL
#define VERBOSE_LEVEL 10

#endif

// Do not edit
#define LOG_STREAM std::cout

// LOG ERROR
#define FATAL_ERROR(file, line)                                    \
    {                                                              \
        LOG_INFO("Fatal error - the program will be terminated "); \
        LOG_INFO("File: " << file << "; line: " << line);          \
        exit(1);                                                   \
    }

// LOG VERBOSE
#ifdef VERBOSE_LEVEL

#define LOG_VERBOSE_INFO(level, stream) \
    {                                   \
        if(level <= VERBOSE_LEVEL)      \
            LOG_INFO(stream);           \
    }

#else

#define LOG_VERBOSE_INFO(level, stream) ;

#endif

#ifdef LOG_MPI_RANK

#define LOG_INFO(stream)                                    \
    {                                                       \
        if(_get_backend_descriptor()->rank == LOG_MPI_RANK) \
            LOG_STREAM << stream << std::endl;              \
    }

#else // LOG_MPI_RANK

#define LOG_INFO(stream)                                                                         \
    {                                                                                            \
        LOG_STREAM << "[rank:" << _get_backend_descriptor()->rank << "]" << stream << std::endl; \
    }

#endif // LOG_MPI_RANK

#endif // ROCALUTION_UTILS_LOG_HPP_
