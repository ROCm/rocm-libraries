/*! \file */

/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse-auxiliary.h"

#include "control.h"
#include "envariables.h"
#include <map>
#include <mutex>

#include "debug.h"

static std::mutex s_mutex;

bool rocsparse::debug_variables_st::get_debug() const
{
    return this->debug;
}

bool rocsparse::debug_variables_st::get_debug_force_host_assert() const
{
    return this->debug_force_host_assert;
}

bool rocsparse::debug_variables_st::get_debug_verbose() const
{
    return this->debug_verbose;
}

bool rocsparse::debug_variables_st::get_debug_kernel_launch() const
{
    return this->debug_kernel_launch;
}

bool rocsparse::debug_variables_st::get_debug_arguments() const
{
    return this->debug_arguments;
}

bool rocsparse::debug_variables_st::get_debug_arguments_verbose() const
{
    return this->debug_arguments_verbose;
}

void rocsparse::debug_variables_st::set_debug(bool value)
{
    if(value != this->debug)
    {
        s_mutex.lock();
        this->debug = value;
        s_mutex.unlock();
    }
}

void rocsparse::debug_variables_st::set_debug_verbose(bool value)
{
    if(value != this->debug_verbose)
    {
        s_mutex.lock();
        this->debug_verbose = value;
        s_mutex.unlock();
    }
}

void rocsparse::debug_variables_st::set_debug_force_host_assert(bool value)
{
    if(value != this->debug_force_host_assert)
    {
        s_mutex.lock();
        this->debug_force_host_assert = value;
        s_mutex.unlock();
    }
}

void rocsparse::debug_variables_st::set_debug_arguments(bool value)
{
    if(value != this->debug_arguments)
    {
        s_mutex.lock();
        this->debug_arguments = value;
        s_mutex.unlock();
    }
}

void rocsparse::debug_variables_st::set_debug_kernel_launch(bool value)
{
    if(value != this->debug_kernel_launch)
    {
        s_mutex.lock();
        this->debug_kernel_launch = value;
        s_mutex.unlock();
    }
}

void rocsparse::debug_variables_st::set_debug_arguments_verbose(bool value)
{
    if(value != this->debug_arguments_verbose)
    {
        s_mutex.lock();
        this->debug_arguments_verbose = value;
        s_mutex.unlock();
    }
}

extern "C" {

int rocsparse_state_debug_arguments_verbose()
{
    return rocsparse_debug_variables.get_debug_arguments_verbose() ? 1 : 0;
}

int rocsparse_state_debug_verbose()
{
    return rocsparse_debug_variables.get_debug_verbose() ? 1 : 0;
}

int rocsparse_state_debug_force_host_assert()
{
    return rocsparse_debug_variables.get_debug_force_host_assert() ? 1 : 0;
}

int rocsparse_state_debug_kernel_launch()
{
    return rocsparse_debug_variables.get_debug_kernel_launch() ? 1 : 0;
}

int rocsparse_state_debug_arguments()
{
    return rocsparse_debug_variables.get_debug_arguments() ? 1 : 0;
}

int rocsparse_state_debug()
{
    return rocsparse_debug_variables.get_debug() ? 1 : 0;
}

void rocsparse_enable_debug_arguments_verbose()
{
    rocsparse_debug_variables.set_debug_arguments_verbose(true);
}

void rocsparse_disable_debug_arguments_verbose()
{
    rocsparse_debug_variables.set_debug_arguments_verbose(false);
}

void rocsparse_enable_debug_kernel_launch()
{
    rocsparse_debug_variables.set_debug_kernel_launch(true);
}

void rocsparse_disable_debug_kernel_launch()
{
    rocsparse_debug_variables.set_debug_kernel_launch(false);
}

void rocsparse_enable_debug_arguments()
{
    rocsparse_debug_variables.set_debug_arguments(true);
    rocsparse_enable_debug_arguments_verbose();
}

void rocsparse_disable_debug_arguments()
{
    rocsparse_debug_variables.set_debug_arguments(false);
    rocsparse_disable_debug_arguments_verbose();
}

void rocsparse_enable_debug_verbose()
{
    rocsparse_debug_variables.set_debug_verbose(true);
    rocsparse_enable_debug_arguments_verbose();
}

void rocsparse_disable_debug_verbose()
{
    rocsparse_debug_variables.set_debug_verbose(false);
    rocsparse_disable_debug_arguments_verbose();
}

void rocsparse_enable_debug_force_host_assert()
{
    rocsparse_debug_variables.set_debug_force_host_assert(true);
}

void rocsparse_disable_debug_force_host_assert()
{
    rocsparse_debug_variables.set_debug_force_host_assert(false);
}

void rocsparse_enable_debug()
{
    rocsparse_debug_variables.set_debug(true);
    rocsparse_enable_debug_arguments();
}

void rocsparse_disable_debug()
{
    rocsparse_debug_variables.set_debug(false);
    rocsparse_disable_debug_arguments();
}
}
