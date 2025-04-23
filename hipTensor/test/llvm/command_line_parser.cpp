/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <hiptensor/hiptensor-version.hpp>
#include <llvm/Support/CommandLine.h>

#include "command_line_parser.hpp"
#include "hiptensor_options.hpp"

// Get input/output file names
llvm::cl::OptionCategory   HiptensorCategory("hipTensor Options",
                                           "Options for hipTensor testing framework");
llvm::cl::opt<std::string> hiptensorInputFilename("y",
                                                  llvm::cl::desc("Specify input YAML filename"),
                                                  llvm::cl::value_desc("filename"),
                                                  llvm::cl::cat(HiptensorCategory));
llvm::cl::opt<std::string> hiptensorOutputFilename("o",
                                                   llvm::cl::desc("Specify output filename"),
                                                   llvm::cl::value_desc("filename"),
                                                   llvm::cl::cat(HiptensorCategory));
llvm::cl::opt<std::string>
    hiptensorValidationOption("v",
                              llvm::cl::desc("Specify whether to perform validation"),
                              llvm::cl::value_desc("ON/OFF"),
                              llvm::cl::cat(HiptensorCategory));
llvm::cl::opt<int32_t>
    hiptensorHotRuns("hot_runs",
                     llvm::cl::desc("Specify number of benchmark runs to include in the timing"),
                     llvm::cl::value_desc("integer number"),
                     llvm::cl::init(1),
                     llvm::cl::cat(HiptensorCategory));

llvm::cl::opt<int32_t> hiptensorColdRuns(
    "cold_runs",
    llvm::cl::desc(
        "Specify number of benchmark runs to exclude from timing, but to warm up frequency"),
    llvm::cl::value_desc("integer number"),
    llvm::cl::init(0),
    llvm::cl::cat(HiptensorCategory));

llvm::cl::opt<int32_t>
    hiptensorOmitMask("omit",
                      llvm::cl::desc("Output verbosity omission\n 0x1 - Skipped Result\n 0x2 - "
                                     "Failed Result\n 0x4 - Passed Result\n 0x8 - Cout Messages"),
                      llvm::cl::value_desc("Bitmask [3:0]"),
                      llvm::cl::cat(HiptensorCategory));

namespace hiptensor
{
    void parseOptions(int argc, char** argv)
    {
        using Options = hiptensor::HiptensorOptions;
        auto& options = Options::instance();

        // Setup LLVM command line parser
        llvm::cl::SetVersionPrinter([](llvm::raw_ostream& os) {
            os << "hipTensor version: " << std::to_string(hiptensorGetVersion()) << "\n";
        });

        llvm::cl::HideUnrelatedOptions(HiptensorCategory);
        llvm::cl::ParseCommandLineOptions(argc, argv);

        // set I/O files if present
        options->setInputYAMLFilename(hiptensorInputFilename);
        options->setOutputStreamFilename(hiptensorOutputFilename);

        options->setOmits(hiptensorOmitMask);

        options->setValidation(hiptensorValidationOption);

        options->setHotRuns(hiptensorHotRuns);
        options->setColdRuns(hiptensorColdRuns);

        // Load testing params from YAML file if present
        if(!options->inputFilename().empty())
        {
            options->setDefaultParams(false);
        }

        // Initialize output stream
        if(!options->outputFilename().empty())
        {
            options->setOstream(options->outputFilename());
        }
    }
}
