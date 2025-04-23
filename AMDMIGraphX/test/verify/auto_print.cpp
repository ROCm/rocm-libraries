/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
 */
#include "auto_print.hpp"
#include <map>
#include <exception>
#include <iostream>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

using handler_map = std::map<std::string, std::function<void()>>;

static handler_map create_handlers()
{
    handler_map m;
    for(const auto& name : migraphx::get_targets())
        m[name] = [] {};
    return m;
}

std::function<void()>& auto_print::get_handler(const std::string& name)
{
    // NOLINTNEXTLINE
    static handler_map handlers = create_handlers();
    return handlers.at(name);
}

void auto_print::set_terminate_handler(const std::string& name)
{
    // NOLINTNEXTLINE
    static std::string pname;
    pname = name;
    std::set_terminate(+[] {
        std::cout << "FAILED: " << pname << std::endl;
        try
        {
            std::rethrow_exception(std::current_exception());
        }
        catch(const std::exception& e)
        {
            std::cout << "    what(): " << e.what() << std::endl;
        }
        std::cout << std::endl;
        for(const auto& tname : migraphx::get_targets())
            get_handler(tname)();
    });
}

static bool in_exception()
{
#if __cplusplus >= 201703L
    return std::uncaught_exceptions() > 0;
#else
    return std::uncaught_exception();
#endif
}

auto_print::~auto_print()
{
    if(in_exception())
    {
        std::cout << std::endl;
        for(const auto& tname : migraphx::get_targets())
            get_handler(tname)();
    }
    get_handler(name) = [] {};
}
