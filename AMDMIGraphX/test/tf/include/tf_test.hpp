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

#ifndef MIGRAPHX_GUARD_TEST_TF_TF_TEST_HPP
#define MIGRAPHX_GUARD_TEST_TF_TF_TEST_HPP

#include <iostream>
#include <vector>
#include <unordered_map>
#include <pb_files.hpp>
#include <migraphx/common.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/tf.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/reduce_mean.hpp>
#include <migraphx/op/pooling.hpp>

#include <migraphx/serialize.hpp>

#include "test.hpp"

inline migraphx::program read_pb_file(const std::string& name, const migraphx::tf_options& options)
{
    static auto pb_files{::pb_files()};
    if(pb_files.find(name) == pb_files.end())
    {
        std::cerr << "Can not find TensorFlow Protobuf file by name: " << name
                  << " , aborting the program\n"
                  << std::endl;
        std::abort();
    }
    return migraphx::parse_tf_buffer(std::string{pb_files.at(name)}, options);
}

inline migraphx::program
parse_tf(const std::string& name,
         bool is_nhwc,
         const std::unordered_map<std::string, std::vector<std::size_t>>& dim_params = {},
         const std::vector<std::string>& output_node_names                           = {})
{

    return read_pb_file(name, migraphx::tf_options{is_nhwc, 1, dim_params, output_node_names});
}

inline migraphx::program optimize_tf(const std::string& name, bool is_nhwc)
{
    auto prog = read_pb_file(name, migraphx::tf_options{is_nhwc, 1});
    auto* mm  = prog.get_main_module();
    if(is_nhwc)
        migraphx::run_passes(*mm,
                             {migraphx::simplify_reshapes{},
                              migraphx::dead_code_elimination{},
                              migraphx::eliminate_identity{}});

    // remove the last return instruction

    if(mm->size() > 0)
    {
        auto last_ins = std::prev(mm->end());
        if(last_ins->name() == "@return")
        {
            mm->remove_instruction(last_ins);
        }
    }
    return prog;
}

#endif
