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

#include "rocsparse_bench_cmdlines.hpp"

//
// @brief Get the output filename.
//
const char* rocsparse_bench_cmdlines::get_ofilename() const
{
    return this->m_cmd.get_ofilename();
}

//
// @brief Get the number of samples..
//
int rocsparse_bench_cmdlines::get_nsamples() const
{
    return this->m_cmd.get_nsamples();
};
int rocsparse_bench_cmdlines::get_option_index_x() const
{
    return this->m_cmd.get_option_index_x();
};

int rocsparse_bench_cmdlines::get_option_nargs(int i)
{
    return this->m_cmd.get_option_nargs(i);
}
const char* rocsparse_bench_cmdlines::get_option_arg(int i, int j)
{
    return this->m_cmd.get_option_arg(i, j);
}
const char* rocsparse_bench_cmdlines::get_option_name(int i)
{
    return this->m_cmd.get_option_name(i);
}
int rocsparse_bench_cmdlines::get_noptions_x() const
{
    return this->m_cmd.get_noptions_x();
};
int rocsparse_bench_cmdlines::get_noptions() const
{
    return this->m_cmd.get_noptions();
};
bool rocsparse_bench_cmdlines::is_stdout_disabled() const
{
    return this->m_cmd.is_stdout_disabled();
};
bool rocsparse_bench_cmdlines::no_rawdata() const
{
    return this->m_cmd.no_rawdata();
};

//
// @brief Get the number of runs per sample.
//
int rocsparse_bench_cmdlines::get_nruns() const
{
    return this->m_cmd.get_nruns();
};

//
// @brief Copy the command line arguments corresponding to a given sample.
//
void rocsparse_bench_cmdlines::get(int isample, int& argc, char** argv) const
{
    const auto& cmdsample = this->m_cmdset[isample];
    for(int j = 0; j < cmdsample.argc; ++j)
    {
        argv[j] = cmdsample.argv[j];
    }
    argc = cmdsample.argc;
}

void rocsparse_bench_cmdlines::get_argc(int isample, int& argc_) const
{
    argc_ = this->m_cmdset[isample].argc;
}

rocsparse_bench_cmdlines::~rocsparse_bench_cmdlines()
{
    if(this->m_cmdset != nullptr)
    {
        delete[] this->m_cmdset;
        this->m_cmdset = nullptr;
    }
}

//
// @brief Constructor.
//
rocsparse_bench_cmdlines::rocsparse_bench_cmdlines(int argc, char** argv)
    : m_cmd(argc, argv)
{
    //
    // Expand the command line .
    //
    this->m_cmdset = new val[this->m_cmd.get_nsamples()];
    this->m_cmd.expand(this->m_cmdset);
}

bool rocsparse_bench_cmdlines::applies(int argc, char** argv)
{
    for(int i = 1; i < argc; ++i)
    {
        if(!strcmp(argv[i], "--bench-x"))
        {
            return true;
        }
    }
    return false;
}

void rocsparse_bench_cmdlines::info() const
{
    int nsamples = this->m_cmd.get_nsamples();
    for(int isample = 0; isample < nsamples; ++isample)
    {
        const auto& cmdsample = this->m_cmdset[isample];
        const auto  argc      = cmdsample.argc;
        const auto  argv      = cmdsample.argv;
        std::cout << "sample[" << isample << "/" << nsamples << "], argc = " << argc << std::endl;

        for(int jarg = 0; jarg < argc; ++jarg)
        {
            std::cout << " " << argv[jarg];
        }
        std::cout << std::endl;
    }
}
