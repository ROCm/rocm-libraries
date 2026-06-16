/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 * Standalone main for instruction generation (no gfxisa dependency).
 * ************************************************************************ */

#include <iostream>
#include <string>

namespace stinkytofu {
bool genInstructions(const std::string& arch, const std::string& inputDir,
                     const std::string& outputDir);
// Generate for all archs and emit ISA .inc; no --arch needed.
bool genAllInstructions(const std::string& inputDir, const std::string& outputDir);
}  // namespace stinkytofu

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " --input-dir=<dir> --output-dir=<dir> [--arch=<arch>]\n"
                  << "  Without --arch: generate for all archs + ISA .inc (single-build mode).\n"
                  << "  With --arch: generate for one arch only (costs, init, operands).\n";
        return 1;
    }

    std::string arch, inputDir, outputDir;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.starts_with("--arch="))
            arch = arg.substr(7);
        else if (arg.starts_with("--input-dir="))
            inputDir = arg.substr(12);
        else if (arg.starts_with("--output-dir="))
            outputDir = arg.substr(13);
    }

    auto stripQuotes = [](std::string& s) {
        if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
            s = s.substr(1, s.size() - 2);
        else if (s.size() >= 1 && (s.front() == '"' || s.back() == '"')) {
            if (s.front() == '"') s.erase(0, 1);
            if (!s.empty() && s.back() == '"') s.pop_back();
        }
    };
    stripQuotes(arch);
    stripQuotes(inputDir);
    stripQuotes(outputDir);

    if (inputDir.empty() || outputDir.empty()) {
        std::cerr << "Error: Missing --input-dir or --output-dir\n";
        return 1;
    }

    bool ok = false;
    if (arch.empty())
        ok = stinkytofu::genAllInstructions(inputDir, outputDir);
    else
        ok = stinkytofu::genInstructions(arch, inputDir, outputDir);
    return ok ? 0 : 1;
}
