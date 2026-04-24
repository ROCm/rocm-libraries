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
#include "stinkytofu-opt.hpp"

#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"
#include "stinkytofu/support/DAGScheduleJsonWriter.hpp"
#include "stinkytofu/support/PassOrderSnapshotJson.hpp"

using namespace stinkytofu;

namespace {
/**
 * DeserializeStinkytofuIRPass is a pass that deserializes Stinkytofu IR from a file.
 */
class DeserializeStinkytofuIRPass : public StinkyInstPass {
   public:
    static char ID;

    DeserializeStinkytofuIRPass(const std::string& stinkytofuIRFile)
        : stinkytofuIRFile(stinkytofuIRFile) {}

    const char* getName() const override {
        return "DeserializeStinkytofuIRPass";
    }

    PassID getPassID() const override {
        return &DeserializeStinkytofuIRPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        GfxArchID arch =
            getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                         passCtx.getGemmTileConfig().arch[2]);

        std::string irText = readFile(stinkytofuIRFile);

        // Use the shared conversion logic from StinkyIRConverter (creates entry block)
        StinkyErrorCode result =
            StinkyIRConverter::populateFunctionFromString(irText, func, passCtx, arch);
        if (result != StinkyErrorCode::SUCCESS) {
            if (result == StinkyErrorCode::PASSCTX_EMPTY) {
                std::cerr << "No PassContext available. Call convertToIRList first." << "\n";
            } else {
                std::cerr << "Error: Failed to populate IRList from string. Error code: "
                          << static_cast<int>(result) << "\n";
            }
            return PreservedAnalyses::none();
        }
        return PreservedAnalyses::none();
    }

   private:
    const std::string stinkytofuIRFile;

    std::string readFile(const std::string& filename) {
        std::ifstream file(filename);
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
};

char DeserializeStinkytofuIRPass::ID = 0;

// Function to print available passes
void printAvailablePasses() {
    std::cout << "Available passes:\n";
    std::cout << "=================\n";
    for (const auto& passInfo : availablePasses) {
        std::cout << "  --" << passInfo.name << "\n";
    }
}

// Function to find and create a pass by name
std::unique_ptr<Pass> createPassByName(const std::string& passName) {
    for (const auto& passInfo : availablePasses) {
        if (passName == passInfo.name) {
            return passInfo.creator();
        }
    }
    return nullptr;
}

// Function to parse command-line arguments for passes
std::vector<std::string> parsePassNames(int argc, char** argv, int startIdx) {
    std::vector<std::string> passNames;
    for (int i = startIdx; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.substr(0, 2) == "--") {
            static constexpr char kSnapJson[] = "--pass-order-snapshot-json=";
            static constexpr char kSnapAfter[] = "--pass-order-snapshot-after-passes=";
            if (arg.rfind(kSnapJson, 0) == 0 || arg.rfind(kSnapAfter, 0) == 0 ||
                arg == "--print-output" || arg.rfind("--ds-read-order=", 0) == 0)
                continue;
            passNames.push_back(arg.substr(2));  // Remove "--" prefix
        }
    }
    return passNames;
}

std::string extractPassOrderSnapshotJsonPath(int argc, char** argv) {
    static constexpr char kPrefix[] = "--pass-order-snapshot-json=";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind(kPrefix, 0) == 0) return a.substr(std::strlen(kPrefix));
    }
    return {};
}

static void trimWhitespace(std::string& s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(0, 1);
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
}

static std::vector<std::string> splitCommaPassNames(const char* prefix, const std::string& a) {
    if (a.rfind(prefix, 0) != 0) return {};
    std::string rest = a.substr(std::strlen(prefix));
    std::vector<std::string> out;
    size_t start = 0;
    while (start < rest.size()) {
        size_t comma = rest.find(',', start);
        std::string token =
            rest.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        trimWhitespace(token);
        if (!token.empty()) {
            out.push_back(std::move(token));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return out;
}

/// Comma-separated `Pass::getName()` strings; if omitted, default is StinkyDAGSchedulerPass only.
std::vector<std::string> extractPassOrderSnapshotAfterPasses(int argc, char** argv) {
    static constexpr char kPrefix[] = "--pass-order-snapshot-after-passes=";
    for (int i = 1; i < argc; ++i) {
        std::vector<std::string> v = splitCommaPassNames(kPrefix, argv[i]);
        if (!v.empty()) return v;
    }
    return {};
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [options] <ir_file> [--pass1] [--pass2] ...\n\n";
        std::cerr << "Options:\n";
        std::cerr << "  --arch <arch>    Target architecture (gfx1250)\n";
        std::cerr << "  --pass-order-snapshot-json=<path>  Before/after instruction order JSON "
                     "(stinkytofu-analysis)\n";
        std::cerr << "  --pass-order-snapshot-after-passes=A,B  Pass::getName() allow-list "
                     "(optional; default: scheduler only)\n";
        std::cerr << "  --list-passes    List all available passes\n";
        std::cerr << "  --help           Show this help message\n\n";
        std::cerr << "Example:\n";
        std::cerr << "  " << argv[0] << " --arch gfx1250 input.txt --StinkyDAGSchedulerPass\n";
        return 1;
    }

    // Check for special options
    std::string firstArg = argv[1];
    if (firstArg == "--list-passes") {
        printAvailablePasses();
        return 0;
    }
    if (firstArg == "--help") {
        std::cerr << "stinkytofu-opt - StinkyTofu IR optimizer\n\n";
        std::cerr << "Usage: " << argv[0] << " [options] <ir_file> [--pass1] [--pass2] ...\n\n";
        std::cerr << "Options:\n";
        std::cerr << "  --arch <arch>    Target architecture (gfx1250)\n";
        std::cerr << "  --pass-order-snapshot-json=<path>  Before/after instruction order JSON "
                     "(stinkytofu-analysis)\n";
        std::cerr << "  --pass-order-snapshot-after-passes=A,B  Pass::getName() allow-list "
                     "(optional; default: scheduler only)\n";
        std::cerr << "  --list-passes    List all available passes\n";
        std::cerr << "  --help           Show this help message\n\n";
        printAvailablePasses();
        return 0;
    }

    // Parse architecture option
    std::array<int, 3> arch = {12, 5, 0};  // default gfx1250
    int irFileIdx = 1;
    int passStartIdx = 2;

    if (firstArg == "--arch") {
        if (argc < 4) {
            std::cerr << "Error: --arch requires an architecture argument\n";
            std::cerr << "Supported architectures: gfx1250\n";
            return 1;
        }

        std::string archStr = argv[2];
        if (archStr == "gfx1250") {
            arch = {12, 5, 0};
        } else {
            std::cerr << "Error: Unsupported architecture '" << archStr << "'\n";
            std::cerr << "Supported architectures: gfx1250\n";
            return 1;
        }

        irFileIdx = 3;
        passStartIdx = 4;
        std::cout << "Target architecture: " << archStr << "\n";
    }

    std::string filename = argv[irFileIdx];

    stinkytofu::PassFeatureConfig passFeatureConfig = getPassFeatureConfig();
    passFeatureConfig.passOrderSnapshot.jsonPath = extractPassOrderSnapshotJsonPath(argc, argv);
    passFeatureConfig.passOrderSnapshot.dumpAfterPasses =
        extractPassOrderSnapshotAfterPasses(argc, argv);

    // Parse --ds-read-order=ProgramOrder|Ascending|AscendingCache
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--ds-read-order=", 0) == 0) {
            std::string val = a.substr(16);
            if (val == "ProgramOrder")
                passFeatureConfig.dagFeatures.dsReadOrder =
                    stinkytofu::PassFeatureConfig::DsReadOrder::ProgramOrder;
            else if (val == "Ascending")
                passFeatureConfig.dagFeatures.dsReadOrder =
                    stinkytofu::PassFeatureConfig::DsReadOrder::Ascending;
            else if (val == "AscendingCache")
                passFeatureConfig.dagFeatures.dsReadOrder =
                    stinkytofu::PassFeatureConfig::DsReadOrder::AscendingCache;
        }
    }

    // Parse and validate user-specified passes from command line
    std::vector<std::string> requestedPasses = parsePassNames(argc, argv, passStartIdx);

    if (!requestedPasses.empty()) {
        std::cout << "\n=== Adding Passes ===\n";
        for (const auto& passName : requestedPasses) {
            if (createPassByName(passName))
                std::cout << "Adding pass: " << passName << "\n";
            else {
                std::cerr << "Warning: Unknown pass '" << passName << "' - skipping\n";
                std::cerr << "Use --list-passes to see available passes\n";
            }
        }
        std::cout << "\n";
    } else {
        std::cout << "\n=== No optimization passes specified ===\n";
        std::cout << "Only deserialization will be performed.\n";
        std::cout << "Use --list-passes to see available passes.\n\n";
    }

    // Check for --print-output flag
    bool printOutput = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--print-output") printOutput = true;
    }

    // Read and parse all functions from the input file
    std::ifstream inputFile(filename);
    std::stringstream fileBuffer;
    fileBuffer << inputFile.rdbuf();
    std::string fileContent = fileBuffer.str();

    GfxArchID archID = getGfxArchID(arch[0], arch[1], arch[2]);

    auto parsed = stinkytofu::parseAllSourceStringsWithDiagnostics(fileContent);
    if (parsed.hasErrors()) {
        std::cerr << "Error: Failed to parse input file\n";
        for (const auto& diag : parsed.diagnostics) std::cerr << "  " << diag.getMessage() << "\n";
        return 1;
    }

    // Fall back to single-function parsing for flat format (no st.func)
    if (parsed.functions.empty()) {
        auto singleResult = stinkytofu::parseSourceStringWithDiagnostics(fileContent);
        if (singleResult.parsedFunction)
            parsed.functions.push_back(std::move(singleResult.parsedFunction));
    }

    // Process each function independently
    for (auto& parsedFunc : parsed.functions) {
        stinkytofu::PassManager passManager;
        stinkytofu::registerAllAnalyses(passManager.getAnalysisManager());

        passManager.addInstrumentation(createDebugPrintInstrumentation());
        if (!passFeatureConfig.passOrderSnapshot.jsonPath.empty()) {
            auto collector = std::make_shared<stinkytofu::DAGScheduleJsonCollector>(
                passFeatureConfig.passOrderSnapshot.jsonPath, parsedFunc->funcName);
            passManager.addInstrumentation(
                std::make_shared<stinkytofu::PassOrderSnapshotInstrumentation>(
                    std::move(collector)));
        }
        passManager.setPassFeatureConfig(passFeatureConfig);
        setKernelConfig(passManager, arch);

        // Add user-specified passes
        for (const auto& passName : requestedPasses) {
            auto pass = createPassByName(passName);
            if (pass) passManager.addPass(std::move(pass));
        }

        stinkytofu::Function func(parsedFunc->funcName);
        func.setGemmTileConfig(passManager.getPassContext().getGemmTileConfig());

        auto result =
            stinkytofu::StinkyIRConverter::populateFunctionFromParsed(*parsedFunc, func, archID);
        if (result != stinkytofu::StinkyErrorCode::SUCCESS) {
            std::cerr << "Error: Failed to populate function '" << parsedFunc->funcName << "'\n";
            continue;
        }

        passManager.run(func);

        if (printOutput) func.dump(std::cout);
    }

    return 0;
}
