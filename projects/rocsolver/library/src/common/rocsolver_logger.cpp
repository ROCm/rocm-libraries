/* **************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <string>

#include "rocblas_utility.hpp"
#include "rocsolver_logger.hpp"

#define STRINGIFY(s) STRINGIFY_HELPER(s)
#define STRINGIFY_HELPER(s) #s

ROCSOLVER_BEGIN_NAMESPACE

// initialize the static variable
rocsolver_logger* rocsolver_logger::_instance = nullptr;
std::mutex rocsolver_logger::_mutex;

static std::string rocblas_version()
{
    size_t size;
    rocblas_get_version_string_size(&size);
    std::string str(size - 1, '\0');
    rocblas_get_version_string(str.data(), size);
    return str;
}

static std::string rocsolver_version()
{
    size_t size;
    rocsolver_get_version_string_size(&size);
    std::string str(size - 1, '\0');
    rocsolver_get_version_string(str.data(), size);
    return str;
}

/***************************************************************************
 * Open logging streams
 ***************************************************************************/

std::ostream* rocsolver_logger::open_log_stream(const char* environment_variable)
{
    const char* logfile;
    if((logfile = std::getenv(environment_variable)) != nullptr
       || (logfile = std::getenv("ROCSOLVER_LOG_PATH")) != nullptr)
    {
        file_streams.emplace_front(logfile);
        std::ostream& os = file_streams.front();

        // print version info only once per file
        if(&os != trace_os && &os != bench_os && &os != profile_os)
        {
            fmt::print(os,
                       "ROCSOLVER LOG FILE\n"
                       "rocSOLVER Version: {}\nrocBLAS Version: {}\n",
                       rocsolver_version(), rocblas_version());
            os.flush();
        }
        return &os;
    }
    else
        return &std::cerr;
}

/***************************************************************************
 * Call stack manipulation
 ***************************************************************************/

rocsolver_log_entry& rocsolver_logger::push_log_entry(rocblas_handle handle, std::string&& name)
{
    std::vector<rocsolver_log_entry>& stack = call_stack[handle];
    stack.push_back(rocsolver_log_entry());

    rocsolver_log_entry& result = stack.back();
    result.name = std::move(name);
    result.level = stack.size() - 1;
    result.start_time = get_time_us_no_sync();

    for(int i = 1; i < stack.size() - 1; i++)
        result.callers.push_back(stack[i].name);

    return result;
}

rocsolver_log_entry& rocsolver_logger::peek_log_entry(rocblas_handle handle)
{
    std::vector<rocsolver_log_entry>& stack = call_stack[handle];
    rocsolver_log_entry& result = stack.back();
    return result;
}

rocsolver_log_entry rocsolver_logger::pop_log_entry(rocblas_handle handle)
{
    std::vector<rocsolver_log_entry>& stack = call_stack[handle];
    rocsolver_log_entry result = stack.back();
    stack.pop_back();

    if(stack.empty())
        call_stack.erase(handle);

    return result;
}

/***************************************************************************
 * Profile log printing
 ***************************************************************************/

void rocsolver_logger::append_profile(std::string& str,
                                      rocsolver_profile_map::iterator start,
                                      rocsolver_profile_map::iterator end)
{
    for(auto it = start; it != end; ++it)
    {
        rocsolver_profile_entry& entry = it->second;

        constexpr int shift_width = 4;
        int indent_level = entry.level - 1;
        int indent = shift_width * indent_level;

        str += fmt::format("{: <{}}{}: Calls: {}, Total Time: {:.3f} ms", "", indent, it->first,
                           entry.calls, entry.time * 1e-3);

        if(entry.internal_calls)
        {
            double internal_time = 0;
            for(const auto& nested : *entry.internal_calls)
                internal_time += nested.second.time;

            str += fmt::format(" (in nested functions: {:.3f} ms)\n", internal_time * 1e-3);

            if(entry.level < max_levels)
                append_profile(str, entry.internal_calls->begin(), entry.internal_calls->end());
        }
        else
            str += '\n';
    }
}

rocblas_status rocsolver_log_begin_impl()
{
    const std::lock_guard<std::mutex> lock(rocsolver_logger::_mutex);

    // if there is no logger, create one and:
    if(rocsolver_logger::_instance != nullptr)
        return rocblas_status_internal_error;

    auto logger = rocsolver_logger::_instance = new rocsolver_logger();

    // set layer_mode from environment variable ROCSOLVER_LAYER or to default
    if(const char* str_layer_mode = std::getenv("ROCSOLVER_LAYER"))
    {
        errno = 0;
        long value = strtol(str_layer_mode, 0, 0);
        if(errno || value < 0 || size_t(value) > size_t(UINT32_MAX))
            return rocblas_status_internal_error;
        else
            logger->layer_mode = static_cast<rocblas_layer_mode_flags>(value);
    }
    else
        logger->layer_mode = rocblas_layer_mode_none;

    // set max_levels from value of environment variable ROCSOLVER_LEVELS or to default
    if(const char* str_max_level = std::getenv("ROCSOLVER_LEVELS"))
    {
        errno = 0;
        long value = strtol(str_max_level, 0, 0);
        if(errno || value < 1 || size_t(value) > size_t(INT_MAX))
            return rocblas_status_internal_error;
        else
            logger->max_levels = static_cast<int>(value);
    }
    else
        logger->max_levels = 1;

    // create output streams (specified by env variables or default to stderr)
    logger->trace_os = logger->open_log_stream("ROCSOLVER_LOG_TRACE_PATH");
    logger->bench_os = logger->open_log_stream("ROCSOLVER_LOG_BENCH_PATH");
    logger->profile_os = logger->open_log_stream("ROCSOLVER_LOG_PROFILE_PATH");
    if(logger->trace_os->good() && logger->bench_os->good() && logger->profile_os->good())
        return rocblas_status_success;
    else
        return rocblas_status_internal_error;
}

rocblas_status rocsolver_log_end_impl()
{
    const std::lock_guard<std::mutex> lock(rocsolver_logger::_mutex);

    // if there is an active logger:
    if(rocsolver_logger::_instance == nullptr)
        return rocblas_status_internal_error;

    auto logger = rocsolver_logger::_instance;

    // if there are pending log_exit calls:
    if(!rocsolver_logger::_instance->call_stack.empty())
        return rocblas_status_internal_error;

    // print profile logging results
    if(logger->layer_mode & rocblas_layer_mode_log_profile && !logger->profile.empty())
    {
        std::string profile_str;
        logger->append_profile(profile_str, logger->profile.begin(), logger->profile.end());
        fmt::print(*logger->profile_os, "------- PROFILE -------\n{}\n", profile_str);
        logger->profile_os->flush();
    }

    // delete the logger
    delete rocsolver_logger::_instance;
    rocsolver_logger::_instance = nullptr;

    return rocblas_status_success;
}

rocblas_status rocsolver_log_set_layer_mode_impl(const rocblas_layer_mode_flags layer_mode)
{
    const std::lock_guard<std::mutex> lock(rocsolver_logger::_mutex);

    // if there is an active logger:
    if(rocsolver_logger::_instance == nullptr)
        return rocblas_status_internal_error;

    auto logger = rocsolver_logger::_instance;

    // change to user specified mode.
    // output streams remain the same defined at logger creation
    logger->layer_mode = layer_mode;

    return rocblas_status_success;
}

rocblas_status rocsolver_log_set_max_levels_impl(const rocblas_int max_levels)
{
    const std::lock_guard<std::mutex> lock(rocsolver_logger::_mutex);

    // if there is an active logger:
    if(rocsolver_logger::_instance == nullptr)
        return rocblas_status_internal_error;
    if(max_levels < 1)
        return rocblas_status_invalid_value;

    auto logger = rocsolver_logger::_instance;

    // change to user specified levels.
    // output streams remain the same defined at logger creation
    logger->max_levels = max_levels;

    return rocblas_status_success;
}

rocblas_status rocsolver_log_restore_defaults_impl()
{
    const std::lock_guard<std::mutex> lock(rocsolver_logger::_mutex);

    // if there is an active logger:
    if(rocsolver_logger::_instance == nullptr)
        return rocblas_status_internal_error;

    auto logger = rocsolver_logger::_instance;

    // reset to no logging
    logger->max_levels = 1;
    logger->layer_mode = rocblas_layer_mode_none;

    return rocblas_status_success;
}

rocblas_status rocsolver_log_write_profile_impl()
{
    const std::lock_guard<std::mutex> lock(rocsolver_logger::_mutex);

    // if there is an active logger:
    if(rocsolver_logger::_instance == nullptr)
        return rocblas_status_internal_error;

    auto logger = rocsolver_logger::_instance;

    // print profile logging results
    if(logger->layer_mode & rocblas_layer_mode_log_profile && !logger->profile.empty())
    {
        std::string profile_str;
        logger->append_profile(profile_str, logger->profile.begin(), logger->profile.end());
        fmt::print(*logger->profile_os, "------- PROFILE -------\n{}\n", profile_str);
        logger->profile_os->flush();
    }
    return rocblas_status_success;
}

rocblas_status rocsolver_log_flush_profile_impl()
{
    const std::lock_guard<std::mutex> lock(rocsolver_logger::_mutex);

    // if there is an active logger:
    if(rocsolver_logger::_instance == nullptr)
        return rocblas_status_internal_error;

    auto logger = rocsolver_logger::_instance;

    // print and clear profile logging results
    if(logger->layer_mode & rocblas_layer_mode_log_profile && !logger->profile.empty())
    {
        std::string profile_str;
        logger->append_profile(profile_str, logger->profile.begin(), logger->profile.end());
        fmt::print(*logger->profile_os, "------- PROFILE -------\n{}\n", profile_str);
        logger->profile_os->flush();

        logger->profile.clear();
    }
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

/***************************************************************************
 * Logging set-up and tear-down
 ***************************************************************************/

extern "C" {

rocblas_status rocsolver_log_begin()
try
{
    return rocsolver::rocsolver_log_begin_impl();
}
catch(...)
{
    return rocsolver::exception_to_rocblas_status();
}

rocblas_status rocsolver_log_end()
try
{
    return rocsolver::rocsolver_log_end_impl();
}
catch(...)
{
    return rocsolver::exception_to_rocblas_status();
}

rocblas_status rocsolver_log_set_layer_mode(const rocblas_layer_mode_flags layer_mode)
try
{
    return rocsolver::rocsolver_log_set_layer_mode_impl(layer_mode);
}
catch(...)
{
    return rocsolver::exception_to_rocblas_status();
}

rocblas_status rocsolver_log_set_max_levels(const rocblas_int max_levels)
try
{
    return rocsolver::rocsolver_log_set_max_levels_impl(max_levels);
}
catch(...)
{
    return rocsolver::exception_to_rocblas_status();
}

rocblas_status rocsolver_log_restore_defaults(void)
try
{
    return rocsolver::rocsolver_log_restore_defaults_impl();
}
catch(...)
{
    return rocsolver::exception_to_rocblas_status();
}

rocblas_status rocsolver_log_write_profile(void)
try
{
    return rocsolver::rocsolver_log_write_profile_impl();
}
catch(...)
{
    return rocsolver::exception_to_rocblas_status();
}

rocblas_status rocsolver_log_flush_profile(void)
try
{
    return rocsolver::rocsolver_log_flush_profile_impl();
}
catch(...)
{
    return rocsolver::exception_to_rocblas_status();
}
}
