/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas_test.hpp"

#ifdef WIN32
// Must include windows.h before dependent headers.
// Specifically this must be before client_utility.hpp uses `#pragma GCC poison`
// to poison 'ctime' and 'abort', as those are used in standard library headers.
#include <windows.h>
#define strcasecmp(A, B) _stricmp(A, B)
#else
#include <pthread.h>
#include <unistd.h>
#endif

#include "client_utility.hpp"

#include <cstdlib>
#include <exception>
#include <regex>

/*********************************************
 * thread pool functions
 *********************************************/
void thread_pool::worker_thread()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    while(!m_done)
    {
        m_cond.wait(lock, [&] { return m_done || !m_work_queue.empty(); });
        if(m_done)
            break;
        auto task    = std::move(m_work_queue.front().first);
        auto promise = std::move(m_work_queue.front().second);
        m_work_queue.pop();
        lock.unlock();
        task();
        promise.set_value();
        lock.lock();
    }
}

thread_pool::thread_pool()
{
    auto thread_count = std::thread::hardware_concurrency();
    do
        m_threads.push_back(std::thread([&] { worker_thread(); }));
    while(thread_count-- > 1);
}

thread_pool::~thread_pool()
{
    m_done = true;
    m_cond.notify_all();
    for(auto& thread : m_threads)
        thread.join();
}

void thread_pool::submit(std::function<void()> func, std::promise<void> promise)
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_work_queue.push(std::make_pair(std::move(func), std::move(promise)));
    }
    m_cond.notify_one();
}

/*********************************************
 * stream pool functions
 *********************************************/
void stream_pool::reset(size_t numDevices, size_t numStreams)
{
    for(auto& streamvec : m_streams)
        for(auto& stream : streamvec)
            CHECK_HIP_ERROR(hipStreamDestroy(stream));

    m_streams.clear();

    for(size_t i = 0; i < (numDevices > 1 ? numDevices : 1); ++i)
    {
        if(numDevices)
            CHECK_HIP_ERROR(hipSetDevice(i));
        std::vector<hipStream_t> temp;
        for(size_t j = 0; j < numStreams; ++j)
        {
            hipStream_t stream;
            CHECK_HIP_ERROR(hipStreamCreate(&stream));
            temp.push_back(stream);
        }
        m_streams.push_back(std::move(temp));
    }
}

/*********************************************
 * thread and stream pool variables
 *********************************************/
thread_pool g_thread_pool;
stream_pool g_stream_pool;

void launch_test_on_streams(std::function<void()> test, size_t numStreams, size_t numDevices)
{
    size_t devices = numDevices > 1 ? numDevices : 1;
    size_t streams = numStreams > 1 ? numStreams : 1;
    for(size_t i = 0; i < devices; ++i)
    {
        hipSetDevice(i);
        for(size_t j = 0; j < streams; ++j)
        {
            if(numStreams)
                t_set_stream_callback.reset(
                    new std::function<void(rocblas_handle)>([=](rocblas_handle handle) {
                        rocblas_set_stream(handle, g_stream_pool[i][j]);
                    }));
            catch_signals_and_exceptions_as_failures(test, true);
        }
    }
}

void launch_test_on_threads(std::function<void()> test,
                            size_t                numThreads,
                            size_t                numStreams,
                            size_t                numDevices)
{
    auto promise = std::make_unique<std::promise<void>[]>(numThreads);
    auto future  = std::make_unique<std::future<void>[]>(numThreads);

    for(size_t i = 0; i < numThreads; ++i)
        future[i] = promise[i].get_future();

    for(size_t i = 0; i < numThreads; ++i)
        g_thread_pool.submit([=] { launch_test_on_streams(test, numStreams, numDevices); },
                             std::move(promise[i]));

    for(size_t i = 0; i < numThreads; ++i)
        future[i].get(); //Wait for tasks to complete
}

// Convert stream to normalized Google Test name
std::string RocBLAS_TestName_to_string(std::unordered_map<std::string, size_t>& table,
                                       const std::ostringstream&                str)
{
    std::string name{str.str()};

    // Remove trailing underscore
    if(!name.empty() && name.back() == '_')
        name.pop_back();

    // If name is empty, make it 1
    if(name.empty())
        name = "1";

    // Warn about unset letter parameters
    if(name.find('*') != name.npos)
        rocblas_cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                        "Warning: Character * found in name."
                        " This means a required letter parameter\n"
                        "(e.g., transA, diag, etc.) has not been set in the YAML file."
                        " Check the YAML file.\n"
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                     << std::endl;

    // Replace non-alphanumeric characters with letters
    std::replace(name.begin(), name.end(), '-', 'n'); // minus
    std::replace(name.begin(), name.end(), '.', 'p'); // decimal point

    // Complex (A,B) is replaced with ArBi
    name.erase(std::remove(name.begin(), name.end(), '('), name.end());
    std::replace(name.begin(), name.end(), ',', 'r');
    std::replace(name.begin(), name.end(), ')', 'i');

    // If parameters are repeated, append an incrementing suffix
    auto p = table.find(name);
    if(p != table.end())
        name += "_t" + std::to_string(++p->second);
    else
        table[name] = 1;

    return name;
}

static const char* const validCategories[]
    = {"quick", "pre_checkin", "nightly", "multi_gpu", "HMM", "known_bug", NULL};

static bool valid_category(const char* category)
{
    int i = 0;
    while(validCategories[i])
    {
        if(!strcmp(category, validCategories[i++]))
            return true;
    }
    return false;
}

bool rocblas_client_global_filters(const Arguments& args)
{
    static std::string gpu_arch = rocblas_internal_get_arch_name();

#ifdef WIN32
    static constexpr rocblas_client_os os = rocblas_client_os::WINDOWS;
#else
    static constexpr rocblas_client_os os = rocblas_client_os::LINUX;
#endif
    if(!(args.os_flags & os))
        return false;

    if(args.gpu_arch[0] && !gpu_arch_match(gpu_arch, args.gpu_arch))
        return false;

#ifndef BUILD_WITH_TENSILE
    if(args.initialization == rocblas_initialization::denorm2)
        return false; // source gemms don't support
#endif

    return true;
}

/********************************************************************************************
 * Function which matches Arguments with a category, accounting for arg.known_bug_platforms *
 ********************************************************************************************/
bool match_test_category(const Arguments& arg, const char* category)
{
    // category is currently unused as "_" for all categories
    if(*arg.known_bug_platforms)
    {
        // Regular expression for token delimiters
        static const std::regex regex{"[:, \\f\\n\\r\\t\\v]+", std::regex_constants::optimize};

        // The name of the current GPU platform
        static const std::string platform = rocblas_internal_get_arch_name();

        // Token iterator
        std::cregex_token_iterator iter{arg.known_bug_platforms,
                                        arg.known_bug_platforms + strlen(arg.known_bug_platforms),
                                        regex,
                                        -1};

        // Iterate across tokens in known_bug_platforms, looking for matches with platform
        for(; iter != std::cregex_token_iterator(); ++iter)
        {
            // If a platform matches, set category to "known_bug"
            if(!strcasecmp(iter->str().c_str(), platform.c_str()))
            {
                // We know that underlying arg object is non-const, so we can use const_cast
                strcpy(const_cast<char*>(arg.category), "known_bug");
                break;
            }
        }
    }

    // we are now bypassing the category key
    // Return whether arg.category matches the requested category
    // return !strcmp(arg.category, category);

    // valid_category can be used if we add unused category
    // return valid_category(arg.category);

    return true;
}
