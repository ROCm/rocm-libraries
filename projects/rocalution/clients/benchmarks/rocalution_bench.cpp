/*! \file */
/* ************************************************************************
* Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocalution_bench.hpp"
#include "rocalution_bench_template.hpp"

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

std::string rocalution_get_version()
{
    int                   rocalution_ver = __ROCALUTION_VER;
    char                  rocalution_rev[64];
    static constexpr char v[] = TO_STR(__ROCALUTION_VER_TWEAK);
    memcpy(rocalution_rev, v, sizeof(v));
    std::ostringstream os;
    os << rocalution_ver / 100000 << "." << rocalution_ver / 100 % 1000 << "."
       << rocalution_ver % 100 << "-" << rocalution_rev;
    return os.str();
}

//
//
//
void rocalution_bench::parse(int& argc, char**& argv, rocalution_arguments_config& config)
{
    config.set_description(this->desc);
    config.precision = 's';
    config.indextype = 's';
    config.parse(argc, argv, this->desc);
}

//
//
//
rocalution_bench::rocalution_bench()
    : desc("rocalution client command line options")
{
}

//
// Execute the benchmark.
//
bool rocalution_bench::execute()
{
    //
    // Set up rocalution.
    //
    set_device_rocalution(device);
    init_rocalution();

    //
    // Run the benchmark.
    //
    bool success;
    success = rocalution_bench_template<double>(this->config);
    if(!success)
    {
        rocalution_bench_errmsg << "rocalution_bench_template failed." << std::endl;
    }

    //
    // Stop rocalution.
    //
    stop_rocalution();
    return success;
}

//
//
//
rocalution_bench::rocalution_bench(int& argc, char**& argv)
    : desc("rocalution client command line options")
{
    //
    // Parse the command line for configuration.
    //
    this->parse(argc, argv, this->config);
}

rocalution_bench& rocalution_bench::operator()(int& argc, char**& argv)
{
    this->parse(argc, argv, this->config);
    return *this;
}

//
// Run (execute).
//
bool rocalution_bench::run()
{
    return this->execute();
}

int rocalution_bench::get_device_id() const
{
    return this->config.device_id;
}

void rocalution_bench::info_devices(std::ostream& out_) const {}
