/******************************************************************************
* Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#include "../../shared/device_properties.h"
#include "../../shared/environment.h"
#include "../../shared/rocfft_hip.h"
#include "logging.h"
#include "repo.h"
#include "rocfft/rocfft.h"
#include "rocfft_exception.h"
#include "rocfft_ostream.hpp"
#include "rtc_cache.h"
#include "solution_map.h"
#include "tuning_helper.h"
#include <fcntl.h>
#include <memory>

/*******************************************************************************
 * Static handle data
 ******************************************************************************/
int log_trace_fd    = -1;
int log_bench_fd    = -1;
int log_profile_fd  = -1;
int log_plan_fd     = -1;
int log_kernelio_fd = -1;
int log_rtc_fd      = -1;
int log_tuning_fd   = -1;
int log_graph_fd    = -1;

/**
 *  @brief Logging function
 *
 *  @details
 *  open_log_stream Open a file descriptor for logging.
 *                  If the environment variable with name
 * environment_variable_name
 *                  is not set, then leave the fd untouched.
 *                  Else open a file at the full logfile path contained in
 *                  the environment variable.
 *
 *  @param[in]
 *  environment_variable_name   const char*
 *                              Name of environment variable that contains
 *                              the full logfile path.
 *
 *  @parm[out]
 *  log_fd      int&
 *              Output file descriptor.
 */

static void open_log_stream(const char* environment_variable_name, int& log_fd)

{
    // if environment variable is set, open file at logfile_pathname contained in
    // the
    // environment variable
    auto logfile_pathname = rocfft_getenv(environment_variable_name);
    if(!logfile_pathname.empty())
    {
        log_fd = OPEN(logfile_pathname.c_str());
    }
}

// library setup function, called once in program at the start of library use
rocfft_status rocfft_setup()
try
{
    rocfft_ostream::setup();
    RTCCache::single = std::make_unique<RTCCache>();

    // set layer_mode from value of environment variable ROCFFT_LAYER
    auto str_layer_mode = rocfft_getenv("ROCFFT_LAYER");

    if(!str_layer_mode.empty())
    {
        rocfft_layer_mode layer_mode
            = static_cast<rocfft_layer_mode>(strtol(str_layer_mode.c_str(), 0, 0));
        LogSingleton::GetInstance().SetLayerMode(layer_mode);

        // open log_trace file
        if(layer_mode & rocfft_layer_mode_log_trace)
            open_log_stream("ROCFFT_LOG_TRACE_PATH", log_trace_fd);

        // open log_bench file
        if(layer_mode & rocfft_layer_mode_log_bench)
            open_log_stream("ROCFFT_LOG_BENCH_PATH", log_bench_fd);

        // open log_profile file
        if(layer_mode & rocfft_layer_mode_log_profile)
            open_log_stream("ROCFFT_LOG_PROFILE_PATH", log_profile_fd);

        // open log_plan file
        if(layer_mode & rocfft_layer_mode_log_plan)
            open_log_stream("ROCFFT_LOG_PLAN_PATH", log_plan_fd);

        // open log_kernelio file
        if(layer_mode & rocfft_layer_mode_log_kernelio)
            open_log_stream("ROCFFT_LOG_KERNELIO_PATH", log_kernelio_fd);

        // open log_rtc file
        if(layer_mode & rocfft_layer_mode_log_rtc)
            open_log_stream("ROCFFT_LOG_RTC_PATH", log_rtc_fd);

        // open log_tuning file
        if(layer_mode & rocfft_layer_mode_log_tuning)
            open_log_stream("ROCFFT_LOG_TUNING_PATH", log_tuning_fd);

        // open log_graph file
        if(layer_mode & rocfft_layer_mode_log_graph)
            open_log_stream("ROCFFT_LOG_GRAPH_PATH", log_graph_fd);
    }

    // setup solution map once in program at the start of library use
    int devcount = 0;
    if(hipGetDeviceCount(&devcount) != hipSuccess)
    {
        return rocfft_status_failure;
    }
    if(devcount == 0)
    {
        return rocfft_status_failure;
    }
    auto arch_name = get_arch_name(get_curr_device_prop());
    solution_map::get_solution_map().setup(arch_name);
    TuningBenchmarker::GetSingleton().Setup();

    log_trace(__func__);
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

// library cleanup function, called once in program after end of library use
rocfft_status rocfft_cleanup()
try
{
    // Logging is potentially unsafe if we're in the middle of static
    // deinitialization, as log structures might have already been
    // cleaned up.
    // log_trace(__func__);

    // close the RTC cache and clear the repo, so that subsequent
    // rocfft_setup() + plan creation will start from scratch
    Repo::Clear();
    RTCCache::single.reset();

    TuningBenchmarker::GetSingleton().Clean();

    LogSingleton::GetInstance().SetLayerMode(rocfft_layer_mode_none);
    // Close log files
    if(log_trace_fd != -1)
    {
        CLOSE(log_trace_fd);
        log_trace_fd = -1;
    }
    if(log_bench_fd != -1)
    {
        CLOSE(log_bench_fd);
        log_bench_fd = -1;
    }
    if(log_profile_fd != -1)
    {
        CLOSE(log_profile_fd);
        log_profile_fd = -1;
    }
    if(log_plan_fd != -1)
    {
        CLOSE(log_plan_fd);
        log_plan_fd = -1;
    }
    if(log_kernelio_fd != -1)
    {
        CLOSE(log_kernelio_fd);
        log_kernelio_fd = -1;
    }
    if(log_rtc_fd != -1)
    {
        CLOSE(log_rtc_fd);
        log_rtc_fd = -1;
    }
    if(log_tuning_fd != -1)
    {
        CLOSE(log_tuning_fd);
        log_tuning_fd = -1;
    }
    if(log_graph_fd != -1)
    {
        CLOSE(log_graph_fd);
        log_graph_fd = -1;
    }

    // stop all log worker threads
    rocfft_ostream::cleanup();

    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

#ifdef ROCFFT_BUILD_OFFLINE_TUNER
rocfft_status rocfft_get_offline_tuner_handle(void** offline_tuner)
try
{
    TuningBenchmarker::GetSingleton().SetBindingSolutionMap(&solution_map::get_solution_map());
    *offline_tuner = &(TuningBenchmarker::GetSingleton());
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}
#endif
