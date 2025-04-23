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
#ifndef MIGRAPHX_GUARD_TEST_RUN_VERIFY_HPP
#define MIGRAPHX_GUARD_TEST_RUN_VERIFY_HPP

#include <migraphx/program.hpp>
#include <functional>
#include <map>

struct program_info;
struct target_info
{
    using validation_function =
        std::function<void(const migraphx::program& p, const migraphx::parameter_map& m)>;
    bool parallel = true;
    validation_function validate;
    std::vector<std::string> disabled_tests;
};

struct run_verify
{
    std::pair<migraphx::program, std::vector<migraphx::argument>>
    run_ref(migraphx::program p,
            migraphx::parameter_map inputs,
            const migraphx::compile_options& c_opts) const;

    std::pair<migraphx::program, std::vector<migraphx::argument>>
    run_target(const migraphx::target& t,
               migraphx::program p,
               const migraphx::parameter_map& inputs,
               const migraphx::compile_options& c_opts) const;
    void validate(const migraphx::target& t,
                  const migraphx::program& p,
                  const migraphx::parameter_map& m) const;
    void verify(const program_info& pi) const;
    void run(int argc, const char* argv[]) const;

    target_info get_target_info(const std::string& name) const;
    void disable_parallel_for(const std::string& name);
    void add_validation_for(const std::string& name, target_info::validation_function v);
    void disable_test_for(const std::string& name, const std::vector<std::string>& tests);

    private:
    std::map<std::string, target_info> info{};
};

#endif
