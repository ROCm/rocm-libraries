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
#pragma once

#include "rocalution_arguments_config.hpp"
#include "rocalution_enum_itsolver.hpp"

//
// @brief Class responsible of configuring the benchmark from parsing command line arguments
// and execute it.
//
class rocalution_bench
{
public:
    //
    // @brief Default contructor.
    //
    rocalution_bench();

    //
    // @brief Contructor with command line arguments.
    //
    rocalution_bench(int& argc, char**& argv);

    //
    // @brief Parenthesis operator equivalent to the contructor.
    //
    rocalution_bench& operator()(int& argc, char**& argv);

    //
    // @brief Run the benchmark.
    //
    bool run();

    //
    // @brief Execute the benchmark.
    //
    bool execute();

    //
    // @brief Get device id.
    //
    int get_device_id() const;

    //
    // @brief Get info devices.
    //
    void info_devices(std::ostream& out_) const;

private:
    //
    // @brief Convert the command line arguments to rocalution_arguments_config.
    //
    void parse(int& argc, char**& argv, rocalution_arguments_config& config);

    //
    // @brief Description of the command line options.
    //
    options_description desc;

    //
    // @brief Configuration of the benchmark.
    //
    rocalution_arguments_config config;
};

//
// @brief Get rocalution version.
//
std::string rocalution_get_version();
