/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 * Standalone main for instruction generation (no gfxisa dependency).
 * ************************************************************************ */

#include <iostream>
#include <string>

namespace stinkytofu {
// Generate the requested arch(s)' full, buildable set (costs, init, operands, ISA, GfxArchDefines).
// arch may be a comma/semicolon separated list; genAllInstructions splits it.
bool genAllInstructions(const std::string& inputDir, const std::string& outputDir,
                        const std::string& arch);
}  // namespace stinkytofu

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr
            << "Usage: " << argv[0]
            << " --input-dir=<dir> --output-dir=<dir> [--arch=<Arch>[,<Arch>...]]\n"
            << "  Without --arch: generate the default build (Gfx1250 + Gfx1250v0).\n"
            << "  With --arch:    generate that stepping (or comma-separated list) instead.\n";
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

    // No --arch builds both gfx12.5 steppings so a default build ships one library holding v0 and
    // v1 (their identity, not the shared {12,5,0} triple, selects the cost table). --arch selects a
    // single stepping (or an explicit comma-separated subset). Both take the same buildable path.
    const std::string selected = arch.empty() ? "Gfx1250,Gfx1250v0" : arch;
    return stinkytofu::genAllInstructions(inputDir, outputDir, selected) ? 0 : 1;
}
