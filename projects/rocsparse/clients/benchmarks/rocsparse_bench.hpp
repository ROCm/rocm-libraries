/*! \file */
/* ************************************************************************
* Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <iostream>
#include <vector>

#include "rocsparse_arguments_config.hpp"
#include "rocsparse_routine.hpp"

struct gpu_config
{
    char name[32];
    long memory_MB;
    long clockRate_MHz;
    long major;
    long minor;
    long maxGridSizeX;
    long sharedMemPerBlock_KB;
    long maxThreadsPerBlock;
    long warpSize;

    explicit gpu_config(const hipDeviceProp_t& prop)
    {
        strcpy(this->name, prop.name);
        this->memory_MB            = (prop.totalGlobalMem >> 20);
        this->clockRate_MHz        = prop.clockRate / 1000;
        this->major                = prop.major;
        this->minor                = prop.minor;
        this->maxGridSizeX         = prop.maxGridSize[0];
        this->sharedMemPerBlock_KB = (prop.sharedMemPerBlock >> 10);
        this->maxThreadsPerBlock   = prop.maxThreadsPerBlock;
        this->warpSize             = prop.warpSize;
    }

    void print(std::ostream& out_)
    {
        out_ << "-------------------------------------------------------------------------"
             << std::endl
             << "with " << this->memory_MB << "MB memory, clock rate " << this->clockRate_MHz
             << "MHz @ computing capability " << this->major << "." << this->minor << std::endl

             << "maxGridDimX " << this->maxGridSizeX << ", sharedMemPerBlock "
             << this->sharedMemPerBlock_KB << "KB, maxThreadsPerBlock " << this->maxThreadsPerBlock
             << std::endl

             << "wavefrontSize " << this->warpSize << std::endl
             << "-------------------------------------------------------------------------"
             << std::endl;
    }

    void print_json(std::ostream& out)
    {
        out << std::endl
            << "\"config gpu\": {" << std::endl

            << "  \"memory\"             : \"" << this->memory_MB << "\"," << std::endl

            << "  \"clockrate\"          : \"" << this->clockRate_MHz << "\"," << std::endl

            << "  \"capability\"         : \"" << this->major << "." << this->minor << "\","
            << std::endl

            << "  \"dimension\"          : \"" << this->maxGridSizeX << "\"," << std::endl

            << "  \"shared memory\"      : \"" << this->sharedMemPerBlock_KB << "\"," << std::endl

            << "  \"max thread per block\": \"" << this->maxThreadsPerBlock << "\"," << std::endl

            << "  \"wavefront size\"     : \"" << this->warpSize << "\"}," << std::endl;
    }
};

class rocsparse_bench
{
private:
    void parse(int& argc, char**& argv, rocsparse_arguments_config& config);

    options_description        desc;
    rocsparse_arguments_config config{};
    rocsparse_routine          routine{};

public:
    rocsparse_bench();
    rocsparse_bench(int& argc, char**& argv);
    rocsparse_bench& operator()(int& argc, char**& argv);
    rocsparse_status run();
    rocsparse_int    get_device_id() const;
    void             info_devices(std::ostream& out_) const;
};

std::string rocsparse_get_version();
