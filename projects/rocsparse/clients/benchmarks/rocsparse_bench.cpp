/*! \file */
/* ************************************************************************
* Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <iostream>
#include <vector>

#include "rocsparse_bench.hpp"
#include "rocsparse_bench_cmdlines.hpp"
#include "test_check.hpp"
bool test_check::s_auto_testing_bad_arg;

// Return version.
std::string rocsparse_get_version()
{
    int  rocsparse_ver;
    char rocsparse_rev[64];

    rocsparse_handle handle;
    rocsparse_status status = rocsparse_create_handle(&handle);
    if(rocsparse_status_success != status)
    {
        std::cerr << "The creation of the rocsparse_handle failed." << std::endl;
        if(0 == rocsparse_state_debug())
        {
            std::cerr << "To get more information, please export the ROCSPARSE_DEBUG environment "
                         "variable:"
                      << std::endl;
            std::cerr << "export ROCSPARSE_DEBUG=1" << std::endl;
        }
        throw(status);
    }

    status = rocsparse_get_version(handle, &rocsparse_ver);
    if(rocsparse_status_success != status)
    {
        std::cerr << "rocsparse_get_version failed." << std::endl;
        throw(status);
    }

    status = rocsparse_get_git_rev(handle, rocsparse_rev);
    if(rocsparse_status_success != status)
    {
        std::cerr << "rocsparse_get_git_rev failed." << std::endl;
        throw(status);
    }

    status = rocsparse_destroy_handle(handle);
    if(rocsparse_status_success != status)
    {
        std::cerr << "rocsparse_destroy_handle failed." << std::endl;
        throw(status);
    }

    std::ostringstream os;
    os << rocsparse_ver / 100000 << "." << rocsparse_ver / 100 % 1000 << "." << rocsparse_ver % 100
       << "-" << rocsparse_rev;
    return os.str();
}

void rocsparse_bench::parse(int& argc, char**& argv, rocsparse_arguments_config& config)
{
    config.set_description(this->desc);
    config.unit_check          = 0;
    config.timing              = 1;
    config.alphai              = 0.0;
    config.betai               = 0.0;
    config.threshold           = 0.0;
    config.percentage          = 0.0;
    config.itilu0_alg          = rocsparse_itilu0_alg_default;
    config.sddmm_alg           = rocsparse_sddmm_alg_default;
    config.spmv_alg            = rocsparse_spmv_alg_default;
    config.spsv_alg            = rocsparse_spsv_alg_default;
    config.sptrsv_alg          = rocsparse_sptrsv_alg_default;
    config.spitsv_alg          = rocsparse_spitsv_alg_default;
    config.spsm_alg            = rocsparse_spsm_alg_default;
    config.sptrsm_alg          = rocsparse_sptrsm_alg_default;
    config.spmm_alg            = rocsparse_spmm_alg_default;
    config.spgemm_alg          = rocsparse_spgemm_alg_default;
    config.spgeam_alg          = rocsparse_spgeam_alg_default;
    config.sparse_to_dense_alg = rocsparse_sparse_to_dense_alg_default;
    config.dense_to_sparse_alg = rocsparse_dense_to_sparse_alg_default;
    config.precision           = 's';
    config.indextype           = 's';
    int i                      = config.parse(argc, argv, this->desc);
    if(i == -1)
    {
        throw rocsparse_status_internal_error;
    }
    else if(i == -2)
    {
        //
        // Help.
        //
        rocsparse_bench_cmdlines::help(std::cout);
        exit(0);
    }
}

rocsparse_bench::rocsparse_bench()
    : desc("rocsparse client command line options")
{
}

rocsparse_bench::rocsparse_bench(int& argc, char**& argv)
    : desc("rocsparse client command line options")
{
    this->parse(argc, argv, this->config);
    routine(this->config.function_name.c_str());

    // Device query
    int devs;
    if(hipGetDeviceCount(&devs) != hipSuccess)
    {
        std::cerr << "Error: cannot get device count" << std::endl;
        exit(-1);
    }
    auto device_id = this->config.device_id;
    // Set device
    if(hipSetDevice(device_id) != hipSuccess || device_id >= devs)
    {
        std::cerr << "Error: cannot set device ID " << device_id << std::endl;
        exit(-1);
    }
}

rocsparse_bench& rocsparse_bench::operator()(int& argc, char**& argv)
{
    this->parse(argc, argv, this->config);
    routine(this->config.function_name.c_str());
    return *this;
}

rocsparse_status rocsparse_bench::run()
{
    return this->routine.dispatch(this->config.precision, this->config.indextype, this->config);
}

rocsparse_int rocsparse_bench::get_device_id() const
{
    return this->config.device_id;
}

// This is used for backward compatibility.
void rocsparse_bench::info_devices(std::ostream& out_) const
{
    int devs;
    if(hipGetDeviceCount(&devs) != hipSuccess)
    {
        std::cerr << "Error: cannot get device count" << std::endl;
        exit(1);
    }

    std::cout << "Query device success: there are " << devs << " devices" << std::endl;
    for(int i = 0; i < devs; ++i)
    {
        hipDeviceProp_t prop;
        if(hipGetDeviceProperties(&prop, i) != hipSuccess)
        {
            std::cerr << "rocsparse-bench error: cannot get device properties" << std::endl;
            exit(1);
        }

        out_ << "Device ID " << i << ": " << prop.name << std::endl;

        gpu_config g(prop);
        g.print(out_);
    }

    //
    // Print header.
    //
    {
        rocsparse_int   device_id = this->get_device_id();
        hipDeviceProp_t prop;
        if(hipGetDeviceProperties(&prop, device_id) != hipSuccess)
        {
            std::cerr << "rocsparse-bench error: cannot get device properties" << std::endl;
            exit(1);
        }

        out_ << "Using device ID " << device_id << " (" << prop.name << ") for rocSPARSE"
             << std::endl
             << "-------------------------------------------------------------------------"
             << std::endl
             << "rocSPARSE version: " << rocsparse_get_version() << std::endl
             << std::endl;
    }
}
