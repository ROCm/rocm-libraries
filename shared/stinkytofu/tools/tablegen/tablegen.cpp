/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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

#include <iostream> // fixme: don't use iostream.
#include <string>

#include "gfx/GpuArchManager.hpp"

namespace stinkytofu
{
    bool genAllArchDefinitions(GpuArchManager& manager, const std::string& outdir);
    bool genAllArchRocisaMappings(GpuArchManager& manager, const std::string& outdir);
    bool genPeepholePatterns(const std::string& patternFile, const std::string& outdir);
    bool genHighLevelIR(const std::string& outdir);
}

using namespace stinkytofu;

void usage()
{
    std::cout << "Usage: tablegen <outdir> <hardwareDir>\n"
              << "  outdir: The directory to output the generated files.\n"
              << "  hardwareDir: The directory to the hardware data.\n"
              << "\n"
              << "Example: tablegen out shared/stinkytofu/hardware\n";
}

int main(int argc, char** argv)
{
    constexpr int SUCCESS = 0;
    constexpr int FAILURE = 1;

    if(argc < 3)
    {
        usage();
        return FAILURE;
    }
    std::string outdir      = argv[1];
    std::string hardwareDir = argv[2];

    bool success = true;

    GpuArchManager manager;

    success &= GpuArchManager::initAllArchs(manager, hardwareDir);

    if(!success)
    {
        std::cerr << "Failed to initialize all architectures\n";
        return FAILURE;
    }

    success &= genAllArchDefinitions(manager, outdir);
    success &= genAllArchRocisaMappings(manager, outdir);

    // Generate peephole optimization patterns (assembly IR)
    std::string asmPatternFile
        = hardwareDir + "/../lib/Dialect/Transforms/PeepholePatterns.pattern";
    success &= genPeepholePatterns(asmPatternFile, outdir);

    // Generate high-level IR optimization patterns
    std::string hlirPatternFile
        = hardwareDir + "/../lib/Dialect/Transforms/HighLevelIR/HighLevelIRPatterns.pattern";
    success &= genPeepholePatterns(hlirPatternFile, outdir);

    // Generate high-level IR instruction classes and mappings
    success &= genHighLevelIR(outdir);

    return success ? SUCCESS : FAILURE;
}
