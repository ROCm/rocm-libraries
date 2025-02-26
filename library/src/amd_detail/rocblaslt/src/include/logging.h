/*! \file */
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc.
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

#pragma once
#ifndef LOGGING_H
#define LOGGING_H

#include "tuple_helper.hpp"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <sys/types.h>
#include <tuple>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <utility>

/************************************************************************************
 * Profile kernel arguments
 ************************************************************************************/
template <typename TUP>
class argument_profile
{
    // Output stream
    mutable std::ostream* os;

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
    explicit argument_profile(std::ostream* os)
        : os(os)
    {
    }

    // Dump the current profile
    void dump() const
    {
        // Acquire an exclusive lock to use map
        std::lock_guard<std::shared_timed_mutex> lock(mutex);

        // Clear the output buffer
        os->clear();

        // Print all of the tuples in the map
        for(const auto& p : map)
        {
            *os << "- ";
            tuple_helper::print_tuple_pairs(
                *os, std::tuple_cat(p.first, std::make_tuple("call_count", p.second)));
        }

        // Flush out the dump
        os->flush();
    }

    // Cleanup handler which dumps profile at destruction
    ~argument_profile()
    try
    {
        dump();
    }
    catch(...)
    {
        return;
    }
};

/**
 *  @brief Logging function
 *
 *  @details
 *  open_log_stream Open stream log_os for logging.
 *                  If the environment variable with name
 * environment_variable_name is not set, then stream log_os to std::cerr. Else
 * open a file at the full logfile path contained in the environment variable.
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
        // if environment variable is set, open file at logfile_pathname contained
        // in the environment variable
        std::string logfile_pathname = (std::string)environment_variable_value;

        log_ofs->exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try
        {
            size_t pos = logfile_pathname.find("%i");
            if(pos != std::string::npos)
                logfile_pathname.replace(pos, 2, std::to_string(getpid()));
            log_ofs->open(logfile_pathname);

            // if log_ofs is open, then stream to log_ofs, else log_os is already
            // set equal to std::cerr
            if(log_ofs->is_open() == true)
            {
                *log_os = log_ofs;
            }
        }
        catch(std::ofstream::failure& e)
        {
            std::cerr << "exception occured when writing to file: " << logfile_pathname << "\n"
                      << e.what() << std::endl;
        }
    }
}

class LoggerSingleton
{
public:
    std::ostream*           log_os         = nullptr;
    uint32_t                env_layer_mode = 0;
    static LoggerSingleton& getInstance()
    {
        static LoggerSingleton gInstance;
        return gInstance;
    }

    // copy contructor
    LoggerSingleton(const LoggerSingleton&) = delete;
    // assignment operator
    LoggerSingleton& operator=(const LoggerSingleton&) = delete;

private:
    // logging streams
    std::ofstream log_file_ofs;

    LoggerSingleton()
    {
        // Get Layer mode
        char* str_layer_mode;
        if((str_layer_mode = getenv("HIPBLASLT_LOG_LEVEL")) == NULL)
        {
            if((str_layer_mode = getenv("HIPBLASLT_LOG_MASK")) != NULL)
            {
                env_layer_mode = strtol(str_layer_mode, nullptr, 0);
            }
        }
        else
        {
            switch(atoi(str_layer_mode))
            {
            case rocblaslt_layer_level_log_api:
                env_layer_mode |= rocblaslt_layer_mode_log_api;
            case rocblaslt_layer_level_log_info:
                env_layer_mode |= rocblaslt_layer_mode_log_info;
            case rocblaslt_layer_level_log_hints:
                env_layer_mode |= rocblaslt_layer_mode_log_hints;
            case rocblaslt_layer_level_log_trace:
                env_layer_mode |= rocblaslt_layer_mode_log_trace;
            case rocblaslt_layer_level_log_error:
                env_layer_mode |= rocblaslt_layer_mode_log_error;
                break;
            default:
                env_layer_mode = rocblaslt_layer_mode_none;
                break;
            }
        }

        // Open log file
        if(env_layer_mode != rocblaslt_layer_mode_none)
        {
            open_log_stream(&log_os, &log_file_ofs, "HIPBLASLT_LOG_FILE");
        }
    }

    ~LoggerSingleton()
    {
        if(log_file_ofs.is_open())
        {
            log_file_ofs.close();
        }
    }
};
/**
 * @brief Invoke functor for each argument in variadic parameter pack.
 * @detail
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

template <typename T, typename... Ts>
inline void log_arg_data(std::ostream& os, std::string& separator, T& x, Ts&&... xs);

template <typename T, typename... Ts>
inline void log_arg_head(std::ostream& os, std::string& separator, T& x, Ts&&... xs)
{
    os << x;
    if constexpr(sizeof...(xs))
        log_arg_data(os, separator, xs...);
}

template <typename T, typename... Ts>
inline void log_arg_data(std::ostream& os, std::string& separator, T& x, Ts&&... xs)
{
    os << "=" << x << separator;
    if constexpr(sizeof...(xs))
        log_arg_head(os, separator, xs...);
}

template <typename H, typename... Ts>
void log_arguments(
    std::ostream& os, std::string& separator, std::string& prefix, H head, Ts&&... xs)
{
    os << prefix << " " << head;
    if constexpr(sizeof...(xs))
        log_arg_data(os, separator, xs...);
    os << "\n";
}

template <typename S, typename T, typename... Ts>
void log_arguments_bench(S& os, T& x, Ts&&... xs)
{
    if constexpr(std::is_same_v<T, const char*>)
    {
        if(strlen(x) && strcmp(x, "invalid"))
            os << x << " ";
    }
    else
    {
        os << x << " ";
    }
    if constexpr(sizeof...(xs))
        log_arguments_bench(os, xs...);
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
    os << "\n" << head;
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
    os << "\n" << head;
}

#endif // LOGGING_H
