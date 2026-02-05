/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 * Standalone main for instruction generation (no gfxisa dependency).
 * ************************************************************************ */

#include <iostream>
#include <string>

namespace stinkytofu
{
    bool genInstructions(const std::string& arch,
                         const std::string& inputDir,
                         const std::string& outputDir);
}

int main(int argc, char** argv)
{
    if(argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " --arch=<arch> --input-dir=<dir> --output-dir=<dir>\n"
                  << "Example: " << argv[0]
                  << " --arch=gfx1250 --input-dir=hardware/defs --output-dir=hardware/generated\n";
        return 1;
    }

    std::string arch, inputDir, outputDir;
    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if(arg.find("--arch=") == 0)
            arch = arg.substr(7);
        else if(arg.find("--input-dir=") == 0)
            inputDir = arg.substr(12);
        else if(arg.find("--output-dir=") == 0)
            outputDir = arg.substr(13);
    }

    auto stripQuotes = [](std::string& s) {
        if(s.size() >= 2 && s.front() == '"' && s.back() == '"')
            s = s.substr(1, s.size() - 2);
        else if(s.size() >= 1 && (s.front() == '"' || s.back() == '"'))
        {
            if(s.front() == '"')
                s.erase(0, 1);
            if(!s.empty() && s.back() == '"')
                s.pop_back();
        }
    };
    stripQuotes(arch);
    stripQuotes(inputDir);
    stripQuotes(outputDir);

    if(arch.empty() || inputDir.empty() || outputDir.empty())
    {
        std::cerr << "Error: Missing --arch, --input-dir, or --output-dir\n";
        return 1;
    }

    bool ok = stinkytofu::genInstructions(arch, inputDir, outputDir);
    return ok ? 0 : 1;
}
