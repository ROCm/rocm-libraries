/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_AUTO_REGISTER_VERIFY_PROGRAM_HPP
#define MIGRAPHX_GUARD_AUTO_REGISTER_VERIFY_PROGRAM_HPP

#include <functional>
#include <migraphx/auto_register.hpp>
#include <migraphx/program.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/ranges.hpp>

struct program_info
{
    std::string name;
    std::string section;
    std::size_t tolerance;
    std::function<migraphx::program()> get_program;
    migraphx::compile_options compile_options;
};

void register_program_info(const program_info& pi);
const std::vector<program_info>& get_programs();

struct register_verify_program_action
{
    template <class T>
    static void apply()
    {
        T x;
        program_info pi;
        const std::string& test_type_name             = migraphx::get_type_name<T>();
        const auto& split_name                        = migraphx::split_string(test_type_name, ':');
        std::vector<std::string> name_without_version = {};
        // test_type_name could contain internal namespace name with version_x_y_z i.e.
        // test_instancenorm<migraphx::version_1::shape::float_type> remove version and construct
        // test_name such as test_instancenorm<migraphx::shape::float_type>
        std::copy_if(
            split_name.begin(),
            split_name.end(),
            std::back_inserter(name_without_version),
            [&](const auto& i) { return not i.empty() and not migraphx::contains(i, "version"); });
        pi.name            = migraphx::trim(migraphx::join_strings(name_without_version, "::"));
        pi.section         = x.section();
        pi.tolerance       = x.get_tolerance();
        pi.get_program     = [x] { return x.create_program(); };
        pi.compile_options = x.get_compile_options();
        register_program_info(pi);
    }
};

template <class T>
using auto_register_verify_program = migraphx::auto_register<register_verify_program_action, T>;

template <class T>
struct verify_program : auto_register_verify_program<T>
{
    std::string section() const { return "general"; };
    migraphx::compile_options get_compile_options() const { return migraphx::compile_options{}; };
    std::size_t get_tolerance() const { return 80; };
};

#endif
