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
#include <numeric>
#include <algorithm>
#include <chrono>
#include <random>
#include <migraphx/migraphx.hpp>

void read_nth_digit(const int, std::vector<float>&);

int main(int argc, char** argv)
{
    if(argc == 1)
    {
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                  << "options:" << std::endl
                  << "\t -c, --cpu      Compile for CPU" << std::endl
                  << "\t -g, --gpu      Compile for GPU" << std::endl
                  << "\t -f, --fp16     FP16 Quantization" << std::endl
                  << "\t -i, --int8     Int8 Quantization" << std::endl
                  << "\t       --cal    Int8 Calibration ON" << std::endl
                  << "\t -p, --print    Print Graph at Each Stage" << std::endl
                  << std::endl
                  << std::endl;
    }

    char** begin   = argv + 1;
    char** end     = argv + argc;
    const bool CPU = (std::find(begin, end, std::string("-c")) != end) or
                     std::find(begin, end, std::string("--cpu")) != end;
    const bool GPU = std::find(begin, end, std::string("-g")) != end or
                     std::find(begin, end, std::string("--gpu")) != end;
    const bool FP16 = std::find(begin, end, std::string("-f")) != end or
                      std::find(begin, end, std::string("--fp16")) != end;
    const bool INT8 = std::find(begin, end, std::string("-i")) != end or
                      std::find(begin, end, std::string("--int8")) != end;
    const bool CALIB = std::find(begin, end, std::string("--cal")) != end;
    const bool PRINT = std::find(begin, end, std::string("-p")) != end or
                       std::find(begin, end, std::string("--print")) != end;

    migraphx::program prog;
    migraphx::onnx_options onnx_opts;
    prog = parse_onnx("../mnist-8.onnx", onnx_opts);

    std::cout << "Parsing ONNX model..." << std::endl;
    if(PRINT)
        prog.print();
    std::cout << std::endl;

    std::string target_str;
    if(CPU)
        target_str = "cpu";
    else if(GPU)
        target_str = "gpu";
    else
        target_str = "ref";
    migraphx::target targ = migraphx::target(target_str.c_str());

    if(FP16)
    {
        migraphx::quantize_fp16(prog);

        std::cout << "Quantizing program for FP16..." << std::endl;
        if(PRINT)
            prog.print();
        std::cout << std::endl;
    }
    else if(INT8)
    {
        if(CALIB)
        {
            std::cout << "Calibration data: " << std::endl;
            std::vector<float> calib_dig;
            read_nth_digit(9, calib_dig);

            migraphx::quantize_int8_options quant_opts;
            migraphx::program_parameters quant_params;
            auto param_shapes = prog.get_parameter_shapes();
            for(auto&& name : param_shapes.names())
            {
                quant_params.add(name, migraphx::argument(param_shapes[name], calib_dig.data()));
            }

            quant_opts.add_calibration_data(quant_params);
            migraphx::quantize_int8(prog, targ, quant_opts);
        }
        else
        {
            migraphx::quantize_int8(prog, targ, migraphx::quantize_int8_options());
        }

        std::cout << "Quantizing program for INT8..." << std::endl;
        if(PRINT)
            prog.print();
        std::cout << std::endl;
    }

    if(GPU)
    {
        migraphx::compile_options comp_opts;
        comp_opts.set_offload_copy();
        prog.compile(targ, comp_opts);
    }
    else
    {
        prog.compile(targ);
    }

    std::cout << "Compiling program for " << target_str << "..." << std::endl;
    if(PRINT)
        prog.print();
    std::cout << std::endl;

    std::vector<float> digit;
    std::random_device rd;
    std::uniform_int_distribution<int> dist(0, 9);
    const int rand_digit = dist(rd);
    std::cout << "Model input: " << std::endl;
    read_nth_digit(rand_digit, digit);

    migraphx::program_parameters prog_params;
    auto param_shapes = prog.get_parameter_shapes();
    auto input        = param_shapes.names().front();
    prog_params.add(input, migraphx::argument(param_shapes[input], digit.data()));

    std::cout << "Model evaluating input..." << std::endl;
    auto start   = std::chrono::high_resolution_clock::now();
    auto outputs = prog.eval(prog_params);
    auto stop    = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Inference complete" << std::endl;
    std::cout << "Inference time: " << elapsed.count() * 1e-3 << "ms" << std::endl;

    auto shape   = outputs[0].get_shape();
    auto lengths = shape.lengths();
    auto num_results =
        std::accumulate(lengths.begin(), lengths.end(), 1, std::multiplies<size_t>());
    float* results   = reinterpret_cast<float*>(outputs[0].data());
    const float* max = std::max_element(results, results + num_results);
    int answer       = max - results;

    std::cout << std::endl
              << "Randomly chosen digit: " << rand_digit << std::endl
              << "Result from inference: " << answer << std::endl
              << std::endl
              << (answer == rand_digit ? "CORRECT" : "INCORRECT") << std::endl
              << std::endl;

    return 0;
}

void read_nth_digit(const int n, std::vector<float>& digit)
{
    const std::string SYMBOLS = "@0#%=+*-.  ";
    std::ifstream file("../digits.txt");
    const int DIGITS = 10;
    const int HEIGHT = 28;
    const int WIDTH  = 28;

    if(not file.is_open())
    {
        return;
    }

    for(int d = 0; d < DIGITS; ++d)
    {
        for(int i = 0; i < HEIGHT * WIDTH; ++i)
        {
            unsigned char temp = 0;
            file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
            if(d == n)
            {
                float data = temp / 255.0;
                digit.push_back(data);
                std::cout << SYMBOLS[static_cast<int>(data * 10) % 11];
                if((i + 1) % WIDTH == 0)
                    std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
}
