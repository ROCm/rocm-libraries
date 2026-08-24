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

/**
 * @file test_gen_instructions.cpp
 * @brief Test for instruction generation from .def files (proof-of-concept)
 */

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#include <process.h>
#define STINKYTOFU_GETPID _getpid
#else
#include <unistd.h>
#define STINKYTOFU_GETPID getpid
#endif

namespace stinkytofu {
extern bool genInstructions(const std::string& arch, const std::string& inputDir,
                            const std::string& outputDir);
extern bool genAllInstructions(const std::string& inputDir, const std::string& outputDir,
                               const std::string& requestedArch);
}  // namespace stinkytofu

using namespace stinkytofu;

// Helper: Check if file exists and contains expected string
static bool fileContains(const std::string& filepath, const std::string& expected) {
    std::ifstream ifs(filepath);
    if (!ifs) {
        std::cerr << "Error: Cannot open " << filepath << "\n";
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    return content.find(expected) != std::string::npos;
}

int main(int argc, char** argv) {
    std::cout << "Running instruction generation test...\n";

    // Get source directory (passed as argument or use current dir)
    std::string sourceDir = argc > 1 ? argv[1] : ".";

    // The architectures the build was configured for. A build may select more than one (the
    // gfx12.5 v0/v1 steppings share an ISA triple but are distinct GfxArchID identities), passed
    // here as separate arguments; joined into the single comma-separated name genAllInstructions
    // takes, and kept as a list so the full-generation check sees every declared arch.
    std::vector<std::string> builtArchs;
    for (int i = 2; i < argc; ++i) builtArchs.emplace_back(argv[i]);
    if (builtArchs.empty()) {
        std::cerr << "FAIL: expected at least one architecture; usage: " << argv[0]
                  << " <source-dir> <Arch>...\n";
        return 1;
    }
    std::string builtArchList;
    for (size_t i = 0; i < builtArchs.size(); ++i)
        builtArchList += (i != 0 ? "," : "") + builtArchs[i];

    // Output under build folder (WORKING_DIRECTORY is CMAKE_BINARY_DIR) with unique subdir
    // to avoid collisions when tests run in parallel and to keep paths portable.
    std::filesystem::path buildDir = std::filesystem::current_path();
    std::ostringstream uniqueName;
    uniqueName << "test_gen_instructions_" << STINKYTOFU_GETPID();
    std::filesystem::path outputPath = buildDir / uniqueName.str();
    std::string outputDir = outputPath.string();

    if (!std::filesystem::create_directories(outputPath)) {
        std::cerr << "FAIL: Could not create output directory " << outputDir << "\n";
        return 1;
    }

    // Test Gfx1250
    {
        std::string inputDir = sourceDir + "/hardware/src/gfx";
        std::string arch = "Gfx1250";

        std::cout << "\nTesting " << arch << "...\n";

        bool success = genInstructions(arch, inputDir, outputDir);
        if (!success) {
            std::cerr << "FAIL: genInstructions returned false for " << arch << "\n";
            return 1;
        }

        // Check generated files exist
        std::string costsFile = outputDir + "/" + arch + "_costs.inc";
        std::string operFile = outputDir + "/" + arch + "_operands.inc";
        std::string initFile = outputDir + "/" + arch + "_init.inc";
        std::string hwregFile = outputDir + "/" + arch + "_hwreg.inc";

        if (!std::filesystem::exists(costsFile)) {
            std::cerr << "FAIL: Cost file not generated: " << costsFile << "\n";
            return 1;
        }

        if (!std::filesystem::exists(operFile)) {
            std::cerr << "FAIL: Operand file not generated: " << operFile << "\n";
            return 1;
        }

        if (!std::filesystem::exists(initFile)) {
            std::cerr << "FAIL: Init file not generated: " << initFile << "\n";
            return 1;
        }

        if (!std::filesystem::exists(hwregFile)) {
            std::cerr << "FAIL: HwReg file not generated: " << hwregFile << "\n";
            return 1;
        }

        // Check init file contains v_add_f32 (all instructions should be in init file)
        if (!fileContains(initFile, "v_add_f32")) {
            std::cerr << "FAIL: Init file doesn't contain v_add_f32\n";
            return 1;
        }

        // Check cost file has header (cost file may not contain all instructions, only non-default
        // costs)
        if (!fileContains(costsFile, "InstructionCost")) {
            std::cerr << "FAIL: Cost file missing header\n";
            return 1;
        }

        // Check hwreg file has both per-arch sorted arrays and a known entry.
        if (!fileContains(hwregFile, "kHwregByName_" + arch)) {
            std::cerr << "FAIL: HwReg file missing kHwregByName_" << arch << " array\n";
            return 1;
        }
        if (!fileContains(hwregFile, "kHwregById_" + arch)) {
            std::cerr << "FAIL: HwReg file missing kHwregById_" << arch << " array\n";
            return 1;
        }
        if (!fileContains(hwregFile, "HW_REG_WAVE_SCHED_MODE")) {
            std::cerr << "FAIL: HwReg file missing HW_REG_WAVE_SCHED_MODE entry\n";
            return 1;
        }

        std::cout << "PASS: " << arch << " generation successful\n";
    }

    // The caller supplies the architecture list, so these pin that one full-generation pass emits
    // every file CMake declared as an output for every requested arch, and that the shared tables
    // (gfxIsa/GfxArchDefines/ArchHelper_includes) cover all of them.
    {
        std::string inputDir = sourceDir + "/hardware/src/gfx";

        std::cout << "\nTesting full generation for";
        for (const std::string& a : builtArchs) std::cout << " " << a;
        std::cout << "...\n";

        std::filesystem::path allOut = outputPath / "full_gen";
        if (!genAllInstructions(inputDir, allOut.string(), builtArchList)) {
            std::cerr << "FAIL: genAllInstructions returned false for the built arch list\n";
            return 1;
        }

        // Shared, arch-independent (or list-shaped) outputs: produced once regardless of count.
        std::vector<std::filesystem::path> expected = {
            allOut / "hardware/gfxIsa.inc", allOut / "hardware/generated/GfxArchDefines.cpp",
            allOut / "hardware/generated/GfxLogicalMaps.cpp",
            allOut / "hardware/arch_headers/ArchHelper_includes.inc"};
        // Per-arch outputs: one set for each requested arch.
        for (const std::string& builtArch : builtArchs) {
            expected.push_back(allOut / "hardware" / (builtArch + "Isa.inc"));
            expected.push_back(allOut / "hardware/arch_headers" / (builtArch + ".hpp"));
            for (const char* suffix :
                 {"_costs.inc", "_init.inc", "_operands.inc", "_hwreg.inc", "_block.inc"})
                expected.push_back(allOut / "hardware/generated" / (builtArch + suffix));
        }
        for (const std::filesystem::path& p : expected) {
            if (!std::filesystem::exists(p)) {
                std::cerr << "FAIL: declared output not generated: " << p.string() << "\n";
                return 1;
            }
        }

        // ArchHelper_includes.inc must name every generated arch, so it catches a generator that
        // dropped one of a multi-arch list (the v0/v1 collapse this whole change removes).
        std::filesystem::path includes = allOut / "hardware/arch_headers/ArchHelper_includes.inc";
        for (const std::string& builtArch : builtArchs) {
            if (!fileContains(includes.string(), "\"" + builtArch + ".hpp\"")) {
                std::cerr << "FAIL: no include emitted for requested arch " << builtArch << "\n";
                return 1;
            }
        }

        // An empty name would emit a unified opcode enum with no architecture in it and
        // silently satisfy every declared build output, so it must be rejected.
        if (genAllInstructions(inputDir, (outputPath / "empty_arch").string(), "")) {
            std::cerr << "FAIL: genAllInstructions accepted an empty architecture\n";
            return 1;
        }

        // A name with no .def directory must fail loudly; skipping it would produce a
        // library whose declared outputs are missing.
        if (genAllInstructions(inputDir, (outputPath / "bad_arch").string(), "GfxNotReal")) {
            std::cerr << "FAIL: genAllInstructions accepted an unknown architecture\n";
            return 1;
        }

        std::cout << "PASS: full generation honored for the built arch list\n";
    }

    // Clean up the temporary output directory
    std::error_code ec;
    std::filesystem::remove_all(outputPath, ec);
    if (ec)
        std::cerr << "Note: Could not remove test output dir " << outputDir << ": " << ec.message()
                  << "\n";

    std::cout << "\n===================\n";
    std::cout << "ALL TESTS PASSED! ?\n";
    std::cout << "===================\n";

    return 0;
}
