/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

// MIGraphX C++ API
#include <migraphx/migraphx.hpp>

char* getCmdOption(char**, char**, const std::string&);

bool cmdOptionExists(char**, char**, const std::string&);

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <input_file> "
                  << "[options]" << std::endl;
        std::cout << "options:" << std::endl;
        std::cout << "\t--parse onnx" << std::endl;
        std::cout << "\t--load  json/msgpack" << std::endl;
        std::cout << "\t--save  <output_file>" << std::endl;
        return 0;
    }

    char* load_arg         = getCmdOption(argv + 2, argv + argc, "--load");
    char* save_arg         = getCmdOption(argv + 2, argv + argc, "--save");
    const char* input_file = argv[1];

    migraphx::program p;

    if(cmdOptionExists(argv + 2, argv + argc, "--parse") or
       not cmdOptionExists(argv + 2, argv + argc, "--load"))
    {
        std::cout << "Parsing ONNX File" << std::endl;
        migraphx::onnx_options options;
        p = parse_onnx(input_file, options);
    }
    else if(load_arg != nullptr)
    {
        std::cout << "Loading Graph File" << std::endl;
        std::string format = load_arg;
        if(format == "json")
        {
            migraphx::file_options options;
            options.set_file_format("json");
            p = migraphx::load(input_file, options);
        }
        else if(format == "msgpack")
        {
            migraphx::file_options options;
            options.set_file_format("msgpack");
            p = migraphx::load(input_file, options);
        }
        else
            p = migraphx::load(input_file);
    }
    else
    {
        std::cout << "Error: Incorrect Usage" << std::endl;
        std::cout << "Usage: " << argv[0] << " <input_file> "
                  << "[options]" << std::endl;
        std::cout << "options:" << std::endl;
        std::cout << "\t--parse onnx" << std::endl;
        std::cout << "\t--load  json/msgpack" << std::endl;
        std::cout << "\t--save  <output_file>" << std::endl;
        return 0;
    }

    std::cout << "Input Graph: " << std::endl;
    p.print();
    std::cout << std::endl;

    if(cmdOptionExists(argv + 2, argv + argc, "--save"))
    {
        std::cout << "Saving program..." << std::endl;
        std::string output_file;
        output_file = save_arg == nullptr ? "out" : save_arg;
        output_file.append(".mxr");

        migraphx::file_options options;
        options.set_file_format("msgpack");
        migraphx::save(p, output_file.c_str(), options);
        std::cout << "Program has been saved as ./" << output_file << std::endl;
    }

    return 0;
}

char* getCmdOption(char** begin, char** end, const std::string& option)
{
    char** itr = std::find(begin, end, option);
    if(itr != end and ++itr != end)
    {
        return *itr;
    }

    return nullptr;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}
