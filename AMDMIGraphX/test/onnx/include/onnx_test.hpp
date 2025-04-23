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

#ifndef MIGRAPHX_GUARD_TEST_ONNX_ONNX_TEST_HPP
#define MIGRAPHX_GUARD_TEST_ONNX_ONNX_TEST_HPP

#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/rewrite_quantization.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>
#include <migraphx/tmp_dir.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/stringutils.hpp>
#include <onnx_files.hpp>
#include <test.hpp>

inline static std::string
read_weight_files(const std::unordered_map<std::string_view, std::string_view>& onnx_files)
{
    static migraphx::tmp_dir td{"weights"};
    for(const auto& i : onnx_files)
    {
        if(not migraphx::ends_with(std::string{i.first}, "weight"))
            continue;
        migraphx::fs::path full_path   = td.path / i.first;
        migraphx::fs::path parent_path = full_path.parent_path();
        migraphx::fs::create_directories(parent_path);
        migraphx::write_buffer(full_path, i.second.data(), i.second.size());
    }
    return td.path.string();
}

inline migraphx::program read_onnx(const std::string& name,
                                   migraphx::onnx_options options = migraphx::onnx_options{})
{
    static auto onnx_files{::onnx_files()};
    static std::string external_data_path = read_weight_files(onnx_files);
    options.external_data_path            = external_data_path;
    if(onnx_files.find(name) == onnx_files.end())
    {
        std::cerr << "ONNX model file: " << name << " not found, aborting the test." << std::endl;
        std::abort();
    }
    auto prog = migraphx::parse_onnx_buffer(std::string{onnx_files.at(name)}, options);
    return prog;
}

inline migraphx::program optimize_onnx(const std::string& name, bool run_passes = false)
{
    migraphx::onnx_options options;
    options.skip_unknown_operators = true;
    auto prog                      = read_onnx(name, options);
    auto* mm                       = prog.get_main_module();
    if(run_passes)
        migraphx::run_passes(*mm,
                             {migraphx::rewrite_quantization{}, migraphx::dead_code_elimination{}});

    // remove the last identity instruction
    auto last_ins = std::prev(mm->end());
    if(last_ins->name() == "@return")
    {
        mm->remove_instruction(last_ins);
    }

    return prog;
}

#endif
