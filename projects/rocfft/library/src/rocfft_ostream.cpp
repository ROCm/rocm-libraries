/******************************************************************************
* Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

// Predeclare rocfft_abort_once() for friend declaration in rocfft_ostream.hpp
static int rocfft_abort_once();

#include "rocfft_ostream.hpp"
#include <csignal>
#include <fcntl.h>
#include <iostream>
#include <type_traits>
#ifdef WIN32
#include <windows.h>
#endif

// static data
std::unique_ptr<rocfft_ostream::worker_map_t> rocfft_ostream::worker_map;
std::unique_ptr<std::recursive_mutex>         rocfft_ostream::worker_map_mutex;

/***********************************************************************
 * rocfft_ostream functions                                           *
 ***********************************************************************/

// Abort function which is called only once by rocfft_abort
static int rocfft_abort_once()
{
    // Make sure the alarm and abort actions are default
#ifndef WIN32
    signal(SIGALRM, SIG_DFL);
    signal(SIGABRT, SIG_DFL);

    // Unblock the alarm and abort signals
    sigset_t set[1];
    sigemptyset(set);
    sigaddset(set, SIGALRM);
    sigaddset(set, SIGABRT);
    sigprocmask(SIG_UNBLOCK, set, nullptr);

    // Timeout in case of deadlock
    alarm(5);
#endif

    // Clear the map, stopping all workers
    rocfft_ostream::cleanup();

    // Flush all
    fflush(NULL);

    // Abort
    std::abort();
    return 0;
}

// Abort function which safely flushes all IO
extern "C" void rocfft_abort()
{
    // If multiple threads call rocfft_abort(), the first one wins
    static int once = rocfft_abort_once();
    // suppress unused variable warning
    (void)once;
}

// Get worker for writing to a file descriptor
std::shared_ptr<rocfft_ostream::worker> rocfft_ostream::get_worker(int fd)
{
    // For a file descriptor indicating an error, return a nullptr
    if(fd == -1)
        return nullptr;

    // if logging is not initialized, there's no worker
    if(!worker_map_mutex || !worker_map)
        return nullptr;

    // C++ allows type punning of common initial sequences
    union
    {
        struct stat statbuf;
        file_id_t   file_id;
    };

    // Verify common initial sequence
    static_assert(std::is_standard_layout<file_id_t>{} && std::is_standard_layout<struct stat>{}
                      && offsetof(file_id_t, st_dev) == 0 && offsetof(struct stat, st_dev) == 0
                      && offsetof(file_id_t, st_ino) == offsetof(struct stat, st_ino)
                      && std::is_same<decltype(file_id_t::st_dev), decltype(stat::st_dev)>{}
                      && std::is_same<decltype(file_id_t::st_ino), decltype(stat::st_ino)>{},
                  "struct stat and file_id_t are not layout-compatible");

#ifndef WIN32
    // Get the device ID and inode, to detect common files
    if(fstat(fd, &statbuf))
    {
        perror("Error executing fstat()");
        return nullptr;
    }
#else
    HANDLE                     fh = (HANDLE)_get_osfhandle(fd);
    BY_HANDLE_FILE_INFORMATION bhfi;

    if(GetFileInformationByHandle(fh, &bhfi))
    {
        // Index info should be unique
        file_id.st_dev = bhfi.nFileIndexLow;
        file_id.st_ino = bhfi.nFileIndexHigh;
    }
    else
    {
        // assign what should be unique
        file_id.st_dev = fd;
        file_id.st_ino = 0;
    }
#endif

    // Lock the map from file_id -> std::shared_ptr<rocfft_ostream::worker>
    std::lock_guard<std::recursive_mutex> lock(*worker_map_mutex);

    // Insert a nullptr map element if file_id doesn't exist in map already
    // worker_ptr is a reference to the std::shared_ptr<rocfft_ostream::worker>
    auto& worker_ptr = worker_map->emplace(file_id, nullptr).first->second;

    // If a new entry was inserted, or an old entry is empty, create new worker
    if(!worker_ptr)
        worker_ptr = std::make_shared<worker>(fd);

    // Return the existing or new worker matching the file
    return worker_ptr;
}

// Construct rocfft_ostream from a file descriptor
rocfft_ostream::rocfft_ostream(int fd)
    : worker_ptr(get_worker(fd))
{
}

// Construct rocfft_ostream from a filename opened for writing with truncation
rocfft_ostream::rocfft_ostream(const char* filename)
{
    int fd     = OPEN(filename);
    worker_ptr = get_worker(fd);
    if(!worker_ptr)
    {
        std::cerr << "Cannot open " << filename << std::endl;
        rocfft_abort();
    }
    CLOSE(fd);
}

rocfft_ostream::~rocfft_ostream()
{
    flush(); // Flush any pending IO
}

// Flush the output
void rocfft_ostream::flush()
{
    // Flush only if this stream contains a worker (i.e., is not a string)
    if(worker_ptr)
    {
        // The contents of the string buffer
        auto str = os.str();

        // Empty string buffers kill the worker thread, so they are not flushed here
        if(str.size())
            worker_ptr->send(std::move(str));

        // Clear the string buffer
        clear();
    }
}

void rocfft_ostream::setup()
{
    if(worker_map_mutex && worker_map)
        return;
    worker_map_mutex = std::make_unique<std::recursive_mutex>();
    worker_map       = std::make_unique<worker_map_t>();
}

void rocfft_ostream::cleanup()
{
    if(worker_map_mutex && worker_map)
    {
        std::lock_guard<std::recursive_mutex> lock(*worker_map_mutex);
        worker_map.reset();
    }
    worker_map_mutex.reset();
}

/***********************************************************************
 * Formatted Output                                                    *
 ***********************************************************************/

// Floating-point output
rocfft_ostream& operator<<(rocfft_ostream& os, double x)
{
    char        s[32];
    const char* out;

    if(std::isnan(x))
        out = "nan";
    else if(std::isinf(x))
        out = (x < 0 ? "-inf" : "inf");
    else
    {
        out = s;
        snprintf(s, sizeof(s) - 2, "%.4g", x);

        // If no decimal point or exponent, append .0 to indicate floating point
        for(char* end = s; *end != '.' && *end != 'e' && *end != 'E'; ++end)
        {
            if(!*end)
            {
                end[0] = '.';
                end[1] = '0';
                end[2] = '\0';
                break;
            }
        }
    }
    os.os << out;
    return os;
}

rocfft_ostream& operator<<(rocfft_ostream& os, float f)
{
    return os << static_cast<double>(f);
}

// bool output
rocfft_ostream& operator<<(rocfft_ostream& os, bool b)
{
    os.os << (b ? 1 : 0);
    return os;
}

// Character output
rocfft_ostream& operator<<(rocfft_ostream& os, char c)
{
    os.os << c;
    return os;
}

// String output
rocfft_ostream& operator<<(rocfft_ostream& os, const char* s)
{
    os.os << s;
    return os;
}

rocfft_ostream& operator<<(rocfft_ostream& os, const std::string& s)
{
    os.os << s;
    return os;
}

// IO Manipulators
rocfft_ostream& operator<<(rocfft_ostream& os, std::ostream& (*pf)(std::ostream&))
{
    // Output the manipulator to the buffer
    os.os << pf;

    // If the manipulator is std::endl or std::flush, flush the output
    if(pf == static_cast<std::ostream& (*)(std::ostream&)>(std::endl)
       || pf == static_cast<std::ostream& (*)(std::ostream&)>(std::flush))
    {
        os.flush();
    }
    return os;
}

/***********************************************************************
 * rocfft_ostream::worker functions handle logging in a single thread *
 ***********************************************************************/

// Send a string to the worker thread for this stream's device/inode
// Empty strings tell the worker thread to exit
void rocfft_ostream::worker::send(std::string str)
{
    // task_t consists of string and promise
    // std::move transfers ownership of str and promise to task
    task_t worker_task(std::move(str));

    // The future indicating when the operation has completed
    auto future = worker_task.get_future();

    // Submit the task to the worker assigned to this device/inode
    // Hold mutex for as short as possible, to reduce contention
    {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(std::move(worker_task));
    }
    // no lock needed for notification
    cond.notify_one();

    // Wait for the task to be completed, to ensure flushed IO
#ifdef WIN32
    // NOTE: this is a hack to avoid hangs at shutdown
    //
    // cppcheck-suppress accessMoved
    if(worker_task.size())
#endif
        future.get();
}

// Worker thread which serializes data to be written to a device/inode
void rocfft_ostream::worker::thread_function()
{
    // Clear any errors in the FILE
    clearerr(file);

    // Lock the mutex in preparation for cond.wait
    std::unique_lock<std::mutex> lock(mutex);

    while(true)
    {
        // Wait for any data, ignoring spurious wakeups, locks lock on continue
        cond.wait(lock, [&] { return !queue.empty(); });

        // With the mutex locked, get and pop data from the front of queue
        task_t task = std::move(queue.front());
        queue.pop();

        // Temporarily unlock queue mutex, unblocking other threads
        lock.unlock();

        // An empty message indicates the closing of the stream
        if(!task.size())
        {
            // Tell future to wake
            task.set_value();
            break;
        }

        // Write the data
        fwrite(task.data(), 1, task.size(), file);

        // Detect any error and flush the C FILE stream
        if(ferror(file) || fflush(file))
        {
            perror("Error writing log file");

            // Tell future to wake up
            task.set_value();
            break;
        }

        // Promise that the data has been written
        task.set_value();

        // Re-lock the mutex in preparation for cond.wait
        lock.lock();
    }
}

// Constructor creates a worker thread from a file descriptor
rocfft_ostream::worker::worker(int fd)
{
    // The worker duplicates the file descriptor (RAII)
#ifdef WIN32
    fd = _dup(fd);
#else
    fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
#endif

    // If the dup fails or fdopen fails, print error and abort
    if(fd == -1 || !(file = FDOPEN(fd, "a")))
    {
        perror("fdopen() error");
        rocfft_abort();
    }

    // Create a worker thread, capturing *this
    thread = std::thread([=] { thread_function(); });

    // Detatch from the worker thread
    thread.detach();
}

rocfft_ostream::worker::~worker()
{
    // Tell worker thread to exit, by sending it an empty string
    send({});

    // drain the queue to ensure that outstanding std::promises
    // aren't broken when all of these tasks are torn down
    {
        std::unique_lock<std::mutex> lock(mutex);
        while(!queue.empty())
        {
            task_t task = std::move(queue.front());
            queue.pop();
            task.set_value();
        }
    }

    // Close the FILE
    if(file)
        fclose(file);
}

// output of rocfft-specific types
rocfft_ostream& operator<<(rocfft_ostream& os, rocfft_transform_type type)
{
    switch(type)
    {
    case rocfft_transform_type_complex_forward:
        os << "complex_forward";
        break;
    case rocfft_transform_type_complex_inverse:
        os << "complex_inverse";
        break;
    case rocfft_transform_type_real_forward:
        os << "real_forward";
        break;
    case rocfft_transform_type_real_inverse:
        os << "real_inverse";
        break;
    }
    return os;
}
rocfft_ostream& operator<<(rocfft_ostream& os, rocfft_precision precision)
{
    switch(precision)
    {
    case rocfft_precision_half:
        os << "half";
        break;
    case rocfft_precision_single:
        os << "single";
        break;
    case rocfft_precision_double:
        os << "double";
        break;
    }
    return os;
}
rocfft_ostream& operator<<(rocfft_ostream& os, rocfft_result_placement placement)
{
    switch(placement)
    {
    case rocfft_placement_inplace:
        os << "inplace";
        break;
    case rocfft_placement_notinplace:
        os << "notinplace";
        break;
    }
    return os;
}
rocfft_ostream& operator<<(rocfft_ostream& os, rocfft_array_type type)
{
    switch(type)
    {
    case rocfft_array_type_complex_interleaved:
        os << "complex_interleaved";
        break;
    case rocfft_array_type_complex_planar:
        os << "complex_planar";
        break;
    case rocfft_array_type_real:
        os << "real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        os << "hermitian_interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        os << "hermitian_planar";
        break;
    case rocfft_array_type_unset:
        os << "unset";
        break;
    }
    return os;
}
rocfft_ostream& operator<<(rocfft_ostream& os, std::pair<const size_t*, size_t> array)
{
    os << "[";
    if(array.first)
    {
        for(const size_t* s = array.first; s != array.first + array.second; ++s)
        {
            if(s != array.first)
                os << ",";
            os << *s;
        }
    }
    os << "]";
    return os;
}
