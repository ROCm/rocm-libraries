/*! \file */
/* ************************************************************************
* Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "hipsparse_bench.hpp"
#include "hipsparse_bench_cmdlines.hpp"

// Return version.
std::string hipsparse_get_version()
{
    int  hipsparse_ver;
    char hipsparse_rev[256];

    hipsparseStatus_t status;

    hipsparseHandle_t handle;
    status = hipsparseCreate(&handle);
    if(HIPSPARSE_STATUS_SUCCESS != status)
    {
        std::cerr << "The creation of the hipsparseHandle_t failed." << std::endl;
        throw(status);
    }

    status = hipsparseGetVersion(handle, &hipsparse_ver);
    if(HIPSPARSE_STATUS_SUCCESS != status)
    {
        std::cerr << "hipsparseGetVersion failed." << std::endl;
        throw(status);
    }

    status = hipsparseGetGitRevision(handle, hipsparse_rev);
    if(HIPSPARSE_STATUS_SUCCESS != status)
    {
        std::cerr << "hipsparseGetGitRevision failed." << std::endl;
        throw(status);
    }

    status = hipsparseDestroy(handle);

    if(HIPSPARSE_STATUS_SUCCESS != status)
    {
        std::cerr << "rocsparse_destroy_handle failed." << std::endl;
        throw(status);
    }

    std::ostringstream os;
    os << hipsparse_ver / 100000 << "." << hipsparse_ver / 100 % 1000 << "." << hipsparse_ver % 100
       << "-" << hipsparse_rev;
    return os.str();
}

void hipsparse_bench::parse(int& argc, char**& argv, hipsparse_arguments_config& config)
{
    config.set_description(this->desc);
    config.unit_check = 0;
    config.timing     = 1;

    int i = config.parse(argc, argv, this->desc);
    if(i == -1)
    {
        throw HIPSPARSE_STATUS_INTERNAL_ERROR;
    }
    else if(i == -2)
    {
        // Help.
        hipsparse_bench_cmdlines::help(std::cout);
        exit(0);
    }
}

hipsparse_bench::hipsparse_bench()
    : desc("hipsparse client command line options")
{
}

hipsparse_bench::hipsparse_bench(int& argc, char**& argv)
    : desc("hipsparse client command line options")
{
    this->parse(argc, argv, this->config);
    routine(this->config.function_name.c_str());

    // Device query
    int devs;
    if(hipGetDeviceCount(&devs) != hipSuccess)
    {
        std::cerr << "hipsparse_bench error: cannot get device count" << std::endl;
        exit(-1);
    }
    auto device_id = this->config.device_id;

    // Set device
    if(hipSetDevice(device_id) != hipSuccess || device_id >= devs)
    {
        std::cerr << "hipsparse_bench error: cannot set device ID " << device_id << std::endl;
        exit(-1);
    }
}

hipsparse_bench& hipsparse_bench::operator()(int& argc, char**& argv)
{
    this->parse(argc, argv, this->config);
    routine(this->config.function_name.c_str());
    return *this;
}

hipsparseStatus_t hipsparse_bench::run()
{
    return this->routine.dispatch(this->config.precision, this->config.indextype, this->config);
}

int hipsparse_bench::get_device_id() const
{
    return this->config.device_id;
}

// This is used for backward compatibility.
void hipsparse_bench::info_devices(std::ostream& out_) const
{
    int devs;
    if(hipGetDeviceCount(&devs) != hipSuccess)
    {
        std::cerr << "hipsparse_bench error: cannot get device count" << std::endl;
        exit(1);
    }

    std::cout << "Query device success: there are " << devs << " devices" << std::endl;
    for(int i = 0; i < devs; ++i)
    {
        hipDeviceProp_t prop;
        if(hipGetDeviceProperties(&prop, i) != hipSuccess)
        {
            std::cerr << "hipsparse_bench error: cannot get device properties" << std::endl;
            exit(1);
        }

        out_ << "Device ID " << i << ": " << prop.name << std::endl;

        gpu_config g(prop);
        g.print(out_);
    }

    // Print header.
    {
        int             device_id = this->get_device_id();
        hipDeviceProp_t prop;
        if(hipGetDeviceProperties(&prop, device_id) != hipSuccess)
        {
            std::cerr << "hipsparse_bench error: cannot get device properties" << std::endl;
            exit(1);
        }
        out_ << "Using device ID " << device_id << " (" << prop.name << ") for hipSPARSE"
             << std::endl
             << "-------------------------------------------------------------------------"
             << std::endl
             << "hipSPARSE version: " << hipsparse_get_version() << std::endl
             << std::endl;
    }
}
