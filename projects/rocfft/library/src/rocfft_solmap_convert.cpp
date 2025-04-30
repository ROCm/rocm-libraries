// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../shared/CLI11.hpp"
#include "../../shared/environment.h"
#include "rocfft/rocfft.h"
#include "solution_map.h"

int main(int argc, char* argv[])
{
    // This helps with mixing output of both wide and narrow characters to the screen
    std::ios::sync_with_stdio(false);

    std::string input_filename  = "";
    std::string output_filename = "";

    // Declare the supported options.
    CLI::App app{"rocfft solution map converter command line options"};

    app.add_option("--input_file", input_filename, "filename of base-solution-map")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--output_file", output_filename, "filename of new-solution-map")
        ->required()
        ->check(CLI::ExistingFile);

    CLI11_PARSE(app, argc, argv);

    // don't use anything from solutions.cpp
    rocfft_setenv("ROCFFT_USE_EMPTY_SOL_MAP", "1");

    SolutionMapConverter converter;
    bool check_result = converter.VersionCheckAndConvert(input_filename, output_filename);

    if(!check_result)
    {
        std::cout << "Converting Solution Map Failed" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
