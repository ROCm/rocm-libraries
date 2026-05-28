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

#include "stinkytofu/Version.h"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/ToolchainCaps.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkySignature.hpp"
#include "stinkytofu/pipeline/Backend.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"
#include "stinkytofu/serialization/asm/RawAsmParser.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
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
                std::cerr << "No PassContext available. Call convertToIRList first.\n";
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

int extractOptLevel(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-O0") return 0;
        if (a == "-O1") return 1;
        if (a == "-O2") return 2;
        if (a == "-O3") return 3;
    }
    return -1;
}

// Function to parse command-line arguments for passes
std::vector<std::string> parsePassNames(int argc, char** argv, int startIdx) {
    std::vector<std::string> passNames;
    for (int i = startIdx; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-O0" || arg == "-O1" || arg == "-O2" || arg == "-O3") continue;
        if (arg.substr(0, 2) == "--") {
            static constexpr char kSnapJson[] = "--pass-order-snapshot-json=";
            static constexpr char kSnapAfter[] = "--pass-order-snapshot-after-passes=";
            if (arg.starts_with(kSnapJson) || arg.starts_with(kSnapAfter) ||
                arg == "--print-output" || arg == "--emit-asm" || arg == "--remarks" ||
                arg == "--preserve-symbolic-regs" || arg == "--preserve-comments" ||
                arg.starts_with("--ds-read-order=") || arg == "--from-label" || arg == "--to-label")
                continue;
            // Two-arg flags: skip both the flag and its value so the value
            // doesn't get mistaken for a pass name and the flag doesn't get
            // mistaken for a missing pass (e.g. `--debug-pass FooPass` was
            // emitting `Warning: Unknown pass 'debug-pass'` because the
            // `--` prefix was stripped and `debug-pass` wasn't a known pass).
            if (arg == "-o" || arg == "--debug-pass") {
                ++i;  // skip the value argument
                continue;
            }
            passNames.push_back(arg.substr(2));  // Remove "--" prefix
        }
    }
    return passNames;
}

std::string extractPassOrderSnapshotJsonPath(int argc, char** argv) {
    static constexpr char kPrefix[] = "--pass-order-snapshot-json=";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.starts_with(kPrefix)) return a.substr(std::strlen(kPrefix));
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
    if (!a.starts_with(prefix)) return {};
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
    BackendRegistry::registerAllBackends();

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [options] <ir_file> [--pass1] [--pass2] ...\n\n";
        std::cerr << "Options:\n";
        std::cerr << "  --arch <arch>    Target architecture. Supported:";
        for (const auto& key : BackendRegistry::getRegisteredArchKeys()) std::cerr << " " << key;
        std::cerr << "\n";
        std::cerr << "  --pass-order-snapshot-json=<path>  Before/after instruction order JSON "
                     "(stinkytofu-analysis)\n";
        std::cerr << "  --pass-order-snapshot-after-passes=A,B  Pass::getName() allow-list "
                     "(optional; default: scheduler only)\n";
        std::cerr << "  -O<N>            Run the registered pipeline at opt level N (0-3)\n";
        std::cerr << "  --remarks        Enable optimization remarks on stderr\n";
        std::cerr << "  --list-passes    List all available passes\n";
        std::cerr << "  --version        Show version information\n";
        std::cerr << "  --help           Show this help message\n\n";
        std::cerr << "Input formats:\n";
        std::cerr << "  <file>.stir      StinkyTofu IR text format (default)\n";
        std::cerr << "  <file>.s         Raw GPU assembly (auto-detected by extension)\n\n";
        std::cerr << "Output flags:\n";
        std::cerr << "  --print-output   Emit optimized IR (StinkyTofu text format)\n";
        std::cerr << "  --emit-asm       Emit optimized GPU assembly (always on for .s input)\n";
        std::cerr << "  -o <file>        Write output to file instead of stdout\n";
        std::cerr
            << "  --from-label <label>  Start of region to optimize (label name, inclusive)\n";
        std::cerr << "  --to-label <label>    End of region to optimize (label name, inclusive)\n";
        std::cerr << "                        Required for .s input when passes are specified.\n";
        std::cerr
            << "                        Labels can be any identifier added to the asm file.\n";
        std::cerr << "  --preserve-symbolic-regs  Preserve and re-emit symbolic register names\n";
        std::cerr << "                            (e.g. v[vgprSerialPersist-768]) instead of the\n";
        std::cerr << "                            resolved numeric form (v255). Best used with\n";
        std::cerr << "                            no passes, since passes operate on numeric\n";
        std::cerr << "                            indices and can leave symbolic names stale.\n";
        std::cerr << "  --preserve-comments       Preserve and re-emit trailing source comments\n";
        std::cerr << "                            (// ... or ;) attached to each instruction,\n";
        std::cerr << "                            label, or .set directive. Best used with no\n";
        std::cerr << "                            passes; passes that rewrite or insert\n";
        std::cerr << "                            instructions may leave comments stale.\n\n";
        std::cerr << "Example:\n";
        std::cerr << "  " << argv[0] << " --arch <arch> input.stir --StinkyDAGSchedulerPass\n";
        std::cerr << "  " << argv[0]
                  << " --arch <arch> input.s   --StinkyDAGSchedulerPass --emit-asm\n";
        return 1;
    }

    // Check for special options
    std::string firstArg = argv[1];
    if (firstArg == "--list-passes") {
        printAvailablePasses();
        return 0;
    }
    if (firstArg == "--version") {
        std::cout << "stinkytofu-opt " << STINKYTOFU_VERSION_MAJOR << "."
                  << STINKYTOFU_VERSION_MINOR << "." << STINKYTOFU_VERSION_PATCH;
        constexpr char tweak[] = STINKYTOFU_VERSION_TWEAK;
        if (tweak[0] != '\0') std::cout << "-" << tweak;
        std::cout << "\n";
        return 0;
    }
    if (firstArg == "--help") {
        std::cerr << "stinkytofu-opt - StinkyTofu IR optimizer\n\n";
        std::cerr << "Usage: " << argv[0] << " [options] <ir_file> [--pass1] [--pass2] ...\n\n";
        std::cerr << "Options:\n";
        std::cerr << "  --arch <arch>    Target architecture. Supported:";
        for (const auto& key : BackendRegistry::getRegisteredArchKeys()) std::cerr << " " << key;
        std::cerr << "\n";
        std::cerr << "  --pass-order-snapshot-json=<path>  Before/after instruction order JSON "
                     "(stinkytofu-analysis)\n";
        std::cerr << "  --pass-order-snapshot-after-passes=A,B  Pass::getName() allow-list "
                     "(optional; default: scheduler only)\n";
        std::cerr << "  -O<N>            Run the registered pipeline at opt level N (0-3)\n";
        std::cerr << "  --remarks        Enable optimization remarks on stderr\n";
        std::cerr << "  --list-passes    List all available passes\n";
        std::cerr << "  --version        Show version information\n";
        std::cerr << "  --help           Show this help message\n\n";
        std::cerr << "Input formats:\n";
        std::cerr << "  <file>.stir      StinkyTofu IR text format (default)\n";
        std::cerr << "  <file>.s         Raw GPU assembly (auto-detected by extension)\n\n";
        std::cerr << "Output flags:\n";
        std::cerr << "  --print-output   Emit optimized IR (StinkyTofu text format)\n";
        std::cerr << "  --emit-asm       Emit optimized GPU assembly (always on for .s input)\n";
        std::cerr << "  -o <file>        Write output to file instead of stdout\n";
        std::cerr
            << "  --from-label <label>  Start of region to optimize (label name, inclusive)\n";
        std::cerr << "  --to-label <label>    End of region to optimize (label name, inclusive)\n";
        std::cerr << "                        Required for .s input when passes are specified.\n";
        std::cerr
            << "                        Labels can be any identifier added to the asm file.\n";
        std::cerr << "  --preserve-symbolic-regs  Preserve and re-emit symbolic register names\n";
        std::cerr << "                            (e.g. v[vgprSerialPersist-768]) instead of the\n";
        std::cerr << "                            resolved numeric form (v255). Best used with\n";
        std::cerr << "                            no passes, since passes operate on numeric\n";
        std::cerr << "                            indices and can leave symbolic names stale.\n";
        std::cerr << "  --preserve-comments       Preserve and re-emit trailing source comments\n";
        std::cerr << "                            (// ... or ;) attached to each instruction,\n";
        std::cerr << "                            label, or .set directive. Best used with no\n";
        std::cerr << "                            passes; passes that rewrite or insert\n";
        std::cerr << "                            instructions may leave comments stale.\n\n";
        printAvailablePasses();
        return 0;
    }

    // Parse architecture option
    std::array<int, 3> arch = {12, 5, 0};  // default gfx1250
    int irFileIdx = 1;
    int passStartIdx = 2;

    if (firstArg == "--arch") {
        if (argc < 4) {
            std::cerr << "Error: --arch requires an architecture argument. Supported:";
            for (const auto& key : BackendRegistry::getRegisteredArchKeys())
                std::cerr << " " << key;
            std::cerr << "\n";
            return 1;
        }

        std::string archStr = argv[2];
        if (!BackendRegistry::parseArchKey(archStr, arch)) {
            std::cerr << "Error: Invalid architecture format '" << archStr
                      << "'. Expected gfx<major><minor><stepping> (e.g. gfx1250)\n";
            return 1;
        }

        if (!BackendRegistry::getArchPipeline(arch)) {
            std::cerr << "Error: Unsupported architecture '" << archStr << "'\n";
            std::cerr << "Supported architectures:";
            for (const auto& key : BackendRegistry::getRegisteredArchKeys())
                std::cerr << " " << key;
            std::cerr << "\n";
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
        if (a.starts_with("--ds-read-order=")) {
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
    int optLevel = extractOptLevel(argc, argv);

    if (optLevel >= 0 && !requestedPasses.empty()) {
        std::cerr << "Error: -O<N> (pipeline mode) and individual --<Pass> flags are mutually "
                     "exclusive\n";
        return 1;
    }

    if (!requestedPasses.empty()) {
        std::cerr << "\n=== Adding Passes ===\n";
        for (const auto& passName : requestedPasses) {
            if (createPassByName(passName))
                std::cerr << "Adding pass: " << passName << "\n";
            else {
                std::cerr << "Warning: Unknown pass '" << passName << "' - skipping\n";
                std::cerr << "Use --list-passes to see available passes\n";
            }
        }
        std::cerr << "\n";
    } else {
        std::cerr << "\n=== No optimization passes specified ===\n";
        std::cerr << "Only deserialization will be performed.\n";
        std::cerr << "Use --list-passes to see available passes.\n\n";
    }

    // Check for --print-output, --emit-asm, -o, --from-label, --to-label,
    // --preserve-symbolic-regs, --preserve-comments flags
    bool printOutput = false;
    bool emitAsm = false;
    bool enableRemarks = false;
    bool preserveSymbolicRegs = false;
    bool preserveComments = false;
    std::string outputFile;
    std::string fromLabel;
    std::string toLabel;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--print-output") printOutput = true;
        if (std::string(argv[i]) == "--emit-asm") emitAsm = true;
        if (std::string(argv[i]) == "--remarks") enableRemarks = true;
        if (std::string(argv[i]) == "--preserve-symbolic-regs") preserveSymbolicRegs = true;
        if (std::string(argv[i]) == "--preserve-comments") preserveComments = true;
        if (std::string(argv[i]) == "--debug-pass" && i + 1 < argc) {
            stinkytofu::PassManagerDebugConfig::addDebugOnly(argv[++i]);
        }
        if (std::string(argv[i]) == "-o" && i + 1 < argc) outputFile = argv[++i];
        if (std::string(argv[i]) == "--from-label" && i + 1 < argc) fromLabel = argv[++i];
        if (std::string(argv[i]) == "--to-label" && i + 1 < argc) toLabel = argv[++i];
    }

    // Auto-detect assembly input by file extension (.s)
    bool isAsmInput = filename.size() >= 2 && filename.substr(filename.size() - 2) == ".s";
    if (isAsmInput) emitAsm = true;  // default output for asm input is asm

    // For .s input with passes, --from-label and --to-label are required so passes
    // run only on the intended region (loop body) rather than the entire kernel.
    if (isAsmInput && !requestedPasses.empty() && (fromLabel.empty() || toLabel.empty())) {
        std::cerr << "Error: --from-label and --to-label are required when running passes on "
                     ".s input.\n";
        std::cerr << "  Add labels to your assembly to mark the region to optimize, e.g.:\n";
        std::cerr << "    my_region_start:\n";
        std::cerr << "    <instructions>\n";
        std::cerr << "    my_region_end:\n";
        std::cerr << "  Then run: --from-label my_region_start --to-label my_region_end\n";
        return 1;
    }

    // Read the input file
    std::ifstream inputFile(filename);
    if (!inputFile) {
        std::cerr << "Error: Cannot open input file '" << filename << "'\n";
        return 1;
    }
    std::stringstream fileBuffer;
    fileBuffer << inputFile.rdbuf();
    std::string fileContent = fileBuffer.str();

    GfxArchID archID = getGfxArchID(arch[0], arch[1], arch[2]);

    stinkytofu::MultiParseResult parsed;
    std::shared_ptr<stinkytofu::SignatureBase> asmSignature;  // non-null for .s input with header

    if (isAsmInput) {
        // Parse raw GPU assembly via RawAsmParser → ParsedFunction
        stinkytofu::RawAsmParserOptions parserOpts;
        parserOpts.preserveSymbolicNames = preserveSymbolicRegs;
        parserOpts.preserveComments = preserveComments;
        auto asmResult = stinkytofu::parseRawAsmString(fileContent, archID, parserOpts);
        if (asmResult.hasErrors()) {
            std::cerr << "Error: Failed to parse assembly input file\n";
            for (const auto& diag : asmResult.diagnostics)
                std::cerr << "  " << diag.getMessage() << "\n";
            return 1;
        }
        for (const auto& diag : asmResult.diagnostics)
            std::cerr << "Warning: " << diag.getMessage() << "\n";
        asmSignature = std::move(asmResult.signature);
        if (asmResult.parsedFunction)
            parsed.functions.push_back(std::move(asmResult.parsedFunction));
    } else {
        // Parse StinkyTofu IR text format
        parsed = stinkytofu::parseAllSourceStringsWithDiagnostics(fileContent);
        if (parsed.hasErrors()) {
            std::cerr << "Error: Failed to parse input file\n";
            for (const auto& diag : parsed.diagnostics)
                std::cerr << "  " << diag.getMessage() << "\n";
            return 1;
        }
        // Fall back to single-function parsing for flat format (no st.func)
        if (parsed.functions.empty()) {
            auto singleResult = stinkytofu::parseSourceStringWithDiagnostics(fileContent);
            if (singleResult.parsedFunction)
                parsed.functions.push_back(std::move(singleResult.parsedFunction));
        }
    }

    // If --from-label / --to-label are given, split the flat ParsedFunction into three
    // sections: pre (verbatim), sched (passes applied), post (verbatim).
    // pre and post are wrapped in their own single-block ParsedFunctions so the normal
    // emit path handles them without any passes.
    stinkytofu::MultiParseResult preResult, postResult;
    if (!fromLabel.empty() || !toLabel.empty()) {
        if (fromLabel.empty() || toLabel.empty()) {
            std::cerr << "Error: --from-label and --to-label must both be specified together\n";
            return 1;
        }
        if (parsed.functions.size() != 1) {
            std::cerr << "Error: --from-label/--to-label requires exactly one function\n";
            return 1;
        }
        // Take ownership of the original instruction list and name before splitting.
        // parsed.functions[0] will be replaced below, which would destroy the block
        // and invalidate any reference into it.
        std::string origFuncName = parsed.functions[0]->funcName;
        auto originalInsts = std::move(parsed.functions[0]->blocks[0]->instructions);

        // Find from-label and to-label positions
        int fromIdx = -1, toIdx = -1;
        for (int idx = 0; idx < (int)originalInsts.size(); ++idx) {
            if (originalInsts[idx]->isLabel && originalInsts[idx]->opcodeStr == fromLabel)
                fromIdx = idx;
            if (originalInsts[idx]->isLabel && originalInsts[idx]->opcodeStr == toLabel)
                toIdx = idx;
        }
        if (fromIdx < 0) {
            std::cerr << "Error: --from-label '" << fromLabel << "' not found in assembly\n";
            return 1;
        }
        if (toIdx < 0) {
            std::cerr << "Error: --to-label '" << toLabel << "' not found in assembly\n";
            return 1;
        }
        if (toIdx < fromIdx) {
            std::cerr << "Error: --to-label appears before --from-label in the assembly\n";
            return 1;
        }

        // Build pre: [0, fromIdx)
        auto preFunc = std::make_unique<stinkytofu::ParsedFunction>();
        preFunc->funcName = origFuncName + "_pre";
        auto preBlock = std::make_unique<stinkytofu::ParsedBlock>();
        preBlock->blockId = "entry";
        for (int idx = 0; idx < fromIdx; ++idx)
            preBlock->instructions.push_back(std::move(originalInsts[idx]));
        preFunc->blocks.push_back(std::move(preBlock));
        preResult.functions.push_back(std::move(preFunc));

        // Build sched: [fromIdx, toIdx]  (inclusive of both labels)
        auto schedFunc = std::make_unique<stinkytofu::ParsedFunction>();
        schedFunc->funcName = origFuncName;
        auto schedBlock = std::make_unique<stinkytofu::ParsedBlock>();
        schedBlock->blockId = "entry";
        for (int idx = fromIdx; idx <= toIdx; ++idx)
            schedBlock->instructions.push_back(std::move(originalInsts[idx]));
        schedFunc->blocks.push_back(std::move(schedBlock));
        parsed.functions[0] = std::move(schedFunc);

        // Build post: (toIdx, end)
        auto postFunc = std::make_unique<stinkytofu::ParsedFunction>();
        postFunc->funcName = origFuncName + "_post";
        auto postBlock = std::make_unique<stinkytofu::ParsedBlock>();
        postBlock->blockId = "entry";
        for (int idx = toIdx + 1; idx < (int)originalInsts.size(); ++idx)
            postBlock->instructions.push_back(std::move(originalInsts[idx]));
        postFunc->blocks.push_back(std::move(postBlock));
        postResult.functions.push_back(std::move(postFunc));
    }

    // Open output stream (-o <file> or stdout)
    std::ofstream outputFileStream;
    if (!outputFile.empty()) {
        outputFileStream.open(outputFile);
        if (!outputFileStream) {
            std::cerr << "Error: Cannot open output file '" << outputFile << "'\n";
            return 1;
        }
    }
    std::ostream& out = outputFile.empty() ? std::cout : outputFileStream;

    // Helper: emit a Function to the output stream.
    auto emitFunction = [&](stinkytofu::Function& func) {
        if (emitAsm) {
            stinkytofu::AsmEmitterOptions opts;
            opts.emitComments = preserveComments;
            opts.indent = 0;
            opts.useSymbolicNames = preserveSymbolicRegs;
            stinkytofu::StinkyAsmEmitter emitter(opts);
            emitter.emit(out, func);
        } else if (printOutput) {
            func.dump(out);
        }
    };

    // Helper: emit a ParsedFunction verbatim (no passes) to the output stream.
    auto emitVerbatim = [&](stinkytofu::MultiParseResult& mr) {
        for (auto& pf : mr.functions) {
            stinkytofu::Function func(pf->funcName);
            stinkytofu::StinkyIRConverter::populateFunctionFromParsed(*pf, func, archID);
            emitFunction(func);
        }
    };

    // Emit header: use the parsed SignatureBase when available, otherwise verbatim pre-region.
    if (asmSignature && emitAsm) {
        out << asmSignature->toString();
    } else {
        emitVerbatim(preResult);
    }

    // Process each function independently
    for (auto& parsedFunc : parsed.functions) {
        if (optLevel >= 0) {
            // Pipeline mode: create a StinkyAsmModule and run the registered pipeline
            stinkytofu::StinkyAsmModule::ModuleOptions moduleOpts{};
            moduleOpts.OptLevel = optLevel;
            moduleOpts.EnableRemarks = enableRemarks;
            stinkytofu::StinkyAsmModule module(parsedFunc->funcName, arch, moduleOpts);

            stinkytofu::Function& func = module.getFunction();
            auto result = stinkytofu::StinkyIRConverter::populateFunctionFromParsed(*parsedFunc,
                                                                                    func, archID);
            if (result != stinkytofu::StinkyErrorCode::SUCCESS) {
                std::cerr << "Error: Failed to populate function '" << parsedFunc->funcName
                          << "'\n";
                continue;
            }

            stinkytofu::Backend backend(module);
            backend.runOptimization();

            emitFunction(func);
        } else {
            // Individual pass mode
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
            passManager.setAsmCapsConfig(stinkytofu::ToolchainCaps::probe(archID));
            if (enableRemarks) passManager.getPassContext().setRemarksEnabled(true);

            for (const auto& passName : requestedPasses) {
                auto pass = createPassByName(passName);
                if (pass) passManager.addPass(std::move(pass));
            }

            stinkytofu::Function func(parsedFunc->funcName);
            func.setGemmTileConfig(passManager.getPassContext().getGemmTileConfig());

            auto result = stinkytofu::StinkyIRConverter::populateFunctionFromParsed(*parsedFunc,
                                                                                    func, archID);
            if (result != stinkytofu::StinkyErrorCode::SUCCESS) {
                std::cerr << "Error: Failed to populate function '" << parsedFunc->funcName
                          << "'\n";
                continue;
            }

            passManager.run(func);

            emitFunction(func);
        }
    }

    emitVerbatim(postResult);

    return 0;
}
