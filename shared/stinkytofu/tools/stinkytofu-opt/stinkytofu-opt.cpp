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

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "stinkytofu/Version.h"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkySignature.hpp"
#include "stinkytofu/pipeline/Backend.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/pipeline/PassCatalog.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"
#include "stinkytofu/serialization/asm/RawAsmParser.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
#include "stinkytofu/support/PassArguments.hpp"
#include "stinkytofu/support/StandardInstrumentations.hpp"
#include "stinkytofu/support/TimePassesInstrumentation.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRulesRegistry.hpp"
#include "stinkytofu/transforms/asm/ra/AllocatorRegistry.hpp"

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
        GfxArchID arch = getGfxArchID(passCtx.getGemmTileConfig().arch);

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
    for (const auto& passInfo : passCatalog()) {
        std::cout << "  --" << passInfo.name << "\n";
    }
}

// A user-specified pass: bare name + optional comma-separated argument list
// parsed out of `--PassName=arg1,arg2`.
struct RequestedPass {
    std::string name;
    std::vector<std::string> args;
};

// Flags of the removed cap / throttle ds_load pacing and hide-budget prescan. Still accepted,
// no longer read.
static bool isRemovedDsPacingFlag(const std::string& a) {
    for (const char* flag : {"--ds-read-throttle-latency", "--ds-read-throttle-transition-factor",
                             "--ds-read-throttle-transition-entries", "--ds-read-per-cap",
                             "--ds-issue-cap-span-cycles", "--ds-issue-cap-mode", "--ds-slot-first",
                             "--enable-wmma-hide-budget-prescan", "--barrier-half-slack"}) {
        const std::string f = flag;
        if (a == f || a.starts_with(f + "=")) return true;
    }
    return false;
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

bool isKernelConfigArg(const std::string& arg) {
    for (const GemmTileField& field : kGemmTileFields) {
        const std::string flag = std::string("--") + field.name;
        if (arg == flag || arg.starts_with(flag + "=")) return true;
    }
    return false;
}

// Function to parse command-line arguments for passes.
//
// Accepts both `--PassName` and `--PassName=arg1,arg2,...` forms.
// In the second form, the comma-separated tokens after `=` are passed
// to the pass's creator function.
std::vector<RequestedPass> parsePassNames(int argc, char** argv, int startIdx) {
    std::vector<RequestedPass> passes;
    for (int i = startIdx; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-O0" || arg == "-O1" || arg == "-O2" || arg == "-O3") continue;
        if (arg.substr(0, 2) == "--") {
            if (arg == "--print-output" || arg == "--emit-asm" || arg == "--remarks" ||
                arg == "--verify-each" || arg == "--dump-passes" || arg == "--time-passes" ||
                arg == "--preserve-symbolic-regs" || arg == "--preserve-comments" ||
                arg.starts_with("--ds-read-order=") || arg.starts_with("--ds-read-queue-depth=") ||
                arg.starts_with("--ds-read-drain-latency=") ||
                arg.starts_with("--ds-read-throttle-latency=") ||
                arg.starts_with("--ds-read-throttle-transition-factor=") ||
                arg.starts_with("--ds-read-throttle-transition-entries=") ||
                arg.starts_with("--ds-read-per-cap=") ||
                arg.starts_with("--ds-issue-cap-span-cycles=") ||
                arg.starts_with("--ds-issue-cap-mode=") ||
                arg.starts_with("--tensor-load-wmma-space=") ||
                arg.starts_with("--tensor-load-ds-load-gap-cycles=") ||
                arg.starts_with("--wmma-queue-depth=") ||
                arg.starts_with("--wmma-queue-cover-cycles=") ||
                arg.starts_with("--wmma-refill-mode=") || arg.starts_with("--wmma-batch-size=") ||
                arg == "--ds-spread" || arg == "--no-ds-spread" ||
                arg.starts_with("--barrier-half-slack=") ||
                arg.starts_with("--wmma-barrier-cover-cycles=") ||
                arg.starts_with("--ds-catch-up-pace=") || arg.starts_with("--ds-batch-wmmas=") ||
                arg.starts_with("--ds-read-cycles-per-load=") ||
                arg.starts_with("--wmma-refill-to-cycles=") ||
                arg.starts_with("--global-read-queue-depth=") ||
                arg.starts_with("--global-read-drain-latency=") ||
                arg.starts_with("--merge-barrier-threshold=") ||
                arg == "--enable-wmma-hide-budget-prescan" ||
                arg == "--enable-esm2-track-valu-vsrc" ||
                arg.starts_with("--wait-alu-hold-strict-count=") ||
                arg == "--wait-alu-scoreboard" || arg.starts_with("--prefetch-lead-wmmas=") ||
                arg.starts_with("--prefetch-lead-min-stage-wmmas=") ||
                arg.starts_with("--war-gate-wmmas=") || arg == "--ds-slot-first" ||
                arg.starts_with("--vgpr-msb-mode=") || arg == "--from-label" ||
                arg == "--to-label" || isKernelConfigArg(arg))
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
            std::string body = arg.substr(2);  // strip "--"
            RequestedPass rp;
            size_t eq = body.find('=');
            rp.name = body.substr(0, eq);
            if (eq != std::string::npos) rp.args = splitPassArgs(body.substr(eq + 1));
            passes.push_back(std::move(rp));
        }
    }
    return passes;
}

static bool parseUint32Value(const std::string& value, uint32_t& out) {
    if (value.empty()) return false;
    uint64_t result = 0;
    for (unsigned char c : value) {
        if (!std::isdigit(c)) return false;
        result = result * 10 + static_cast<uint64_t>(c - '0');
        if (result > std::numeric_limits<uint32_t>::max()) return false;
    }
    out = static_cast<uint32_t>(result);
    return true;
}

static bool parseGemmTileField(const char* flag, const std::string& value, uint32_t& field,
                               std::string& error) {
    if (parseUint32Value(value, field)) return true;
    error = std::string("Invalid value for ") + flag;
    return false;
}

static bool parseGemmTileConfigOptions(int argc, char** argv, stinkytofu::GemmTileConfig& config,
                                       std::string& error) {
    config = {};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        for (const GemmTileField& field : kGemmTileFields) {
            const std::string flag = std::string("--") + field.name;
            uint32_t& value = config.*field.member;
            if (a.starts_with(flag + "=")) {
                if (!parseGemmTileField(flag.c_str(), a.substr(flag.size() + 1), value, error))
                    return false;
                break;
            }
            if (a == flag) {
                if (i + 1 < argc && !parseGemmTileField(flag.c_str(), argv[++i], value, error))
                    return false;
                break;
            }
        }
    }
    return true;
}

static void printKernelConfigHelp(std::ostream& os) {
    os << "  Kernel configuration (GemmTileConfig fields):\n";
    for (const GemmTileField& field : kGemmTileFields) {
        std::string usage = std::string("--") + field.name + " <n>";
        usage.resize(20, ' ');
        os << "  " << usage << "(or --" << field.name << "=<n>)\n";
    }
}
}  // namespace

int main(int argc, char** argv) {
    BackendRegistry::registerAllBackends();
    AllocatorRegistry::registerAllAllocators();
    AllocationRulesRegistry::registerAll();

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [options] <ir_file> [--pass1] [--pass2] ...\n\n";
        std::cerr << "Options:\n";
        std::cerr << "  --arch <arch>    Target architecture. Supported:";
        for (const auto& key : BackendRegistry::getRegisteredArchKeys()) std::cerr << " " << key;
        std::cerr << "\n";
        std::cerr << "  -O<N>            Run the registered pipeline at opt level N (0-3)\n";
        std::cerr << "  --remarks        Enable optimization remarks on stderr\n";
        std::cerr << "  --verify-each    Verify StinkyTofu ASM IR after every pass\n";
        std::cerr << "  --dump-passes    Dump IR before/after each pass into before.txt and\n";
        std::cerr << "                   after.txt in the working directory (implied by\n";
        std::cerr << "                   --debug-pass)\n";
        std::cerr << "  --time-passes    Report per-pass wall time on stderr\n";
        std::cerr << "  --list-passes    List all available passes\n";
        std::cerr << "  --version        Show version information\n";
        std::cerr << "  --help           Show this help message\n\n";
        printKernelConfigHelp(std::cerr);
        std::cerr << "\nInput formats:\n";
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
        std::cerr << "  -O<N>            Run the registered pipeline at opt level N (0-3)\n";
        std::cerr << "  --remarks        Enable optimization remarks on stderr\n";
        std::cerr << "  --verify-each    Verify StinkyTofu ASM IR after every pass\n";
        std::cerr << "  --dump-passes    Dump IR before/after each pass into before.txt and\n";
        std::cerr << "                   after.txt in the working directory (implied by\n";
        std::cerr << "                   --debug-pass)\n";
        std::cerr << "  --time-passes    Report per-pass wall time on stderr\n";
        std::cerr << "  --list-passes    List all available passes\n";
        std::cerr << "  --version        Show version information\n";
        std::cerr << "  --help           Show this help message\n\n";
        printKernelConfigHelp(std::cerr);
        std::cerr << "\nInput formats:\n";
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
    // Concrete stepping identity when the user named one (e.g. "gfx1250v0"). Steppings that share
    // an ISA triple (gfx1250 v1 vs gfx1250v0 on {12,5,0}) are indistinguishable by triple, so the
    // triple-based getGfxArchID() first-matches v1. When the user names a stepping, resolve the
    // GfxArchID by identity instead. Empty => no name given, keep the legacy triple behavior.
    std::string resolvedArchName;
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
        // Resolve by concrete identity name first so a stepping variant sharing a triple
        // (e.g. gfx1250v0) is selectable at runtime. getArchInfo(name) returns nullptr on a name
        // not registered as an identity in this build (no assert), so a plain triple spelling
        // (e.g. gfx942 in a gfx1250 build) falls through to the triple parser, and a stepping name
        // absent from this build fails cleanly at the isArchSupported() check below.
        if (const auto* namedInfo = ArchHelper::getInstance().getArchInfo(archStr)) {
            arch = {static_cast<int>(namedInfo->major), static_cast<int>(namedInfo->minor),
                    static_cast<int>(namedInfo->stepping)};
            resolvedArchName = archStr;
        } else if (!BackendRegistry::parseArchKey(archStr, arch)) {
            std::cerr << "Error: Invalid architecture format '" << archStr
                      << "'. Expected gfx<major><minor><stepping> (e.g. gfx1250)\n";
            return 1;
        }

        if (!BackendRegistry::isArchSupported(arch)) {
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

    stinkytofu::GemmTileConfig gemmTileConfig;
    std::string gemmTileConfigError;
    if (!parseGemmTileConfigOptions(argc, argv, gemmTileConfig, gemmTileConfigError)) {
        std::cerr << "Error: " << gemmTileConfigError << "\n";
        return 1;
    }

    stinkytofu::PassFeatureConfig passFeatureConfig = stinkytofu::standalonePassFeatureConfig();

    // Parse --vgpr-msb-mode=none|msb8|msb16 (override of ToolchainCaps::probe).
    // Useful when running on a host whose comgr doesn't know the target ISA,
    // so the FileCheck/regression tests can pin the mode explicitly.
    std::optional<stinkytofu::VgprMsbMode> vgprMsbOverride;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.starts_with("--vgpr-msb-mode=")) {
            std::string val = a.substr(std::string("--vgpr-msb-mode=").size());
            if (val == "none")
                vgprMsbOverride = stinkytofu::VgprMsbMode::None;
            else if (val == "msb8")
                vgprMsbOverride = stinkytofu::VgprMsbMode::Msb8;
            else if (val == "msb16")
                vgprMsbOverride = stinkytofu::VgprMsbMode::Msb16;
            else {
                std::cerr << "Error: --vgpr-msb-mode expects none|msb8|msb16, got '" << val
                          << "'\n";
                return 1;
            }
        }
    }

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
        } else if (a.starts_with("--ds-read-queue-depth=")) {
            passFeatureConfig.dagFeatures.dsReadQueueDepth = std::stoi(a.substr(22));
        } else if (a.starts_with("--ds-read-drain-latency=")) {
            passFeatureConfig.dagFeatures.dsReadDrainLatency = std::stoi(a.substr(24));
        } else if (isRemovedDsPacingFlag(a)) {
            std::cerr << "warning: " << a.substr(0, a.find('=')) << " was removed with the legacy "
                      << "ds_load pacing and is ignored\n";
        } else if (a.starts_with("--tensor-load-wmma-space=")) {
            passFeatureConfig.dagFeatures.tensorLoadWmmaSpace = std::stoi(a.substr(25));
        } else if (a.starts_with("--tensor-load-ds-load-gap-cycles=")) {
            passFeatureConfig.dagFeatures.tensorLoadDsLoadGapCycles =
                std::stoi(a.substr(std::string("--tensor-load-ds-load-gap-cycles=").size()));
        } else if (a.starts_with("--wmma-queue-depth=")) {
            passFeatureConfig.dagFeatures.wmmaQueueDepth =
                std::stoi(a.substr(std::string("--wmma-queue-depth=").size()));
        } else if (a.starts_with("--wmma-queue-cover-cycles=")) {
            passFeatureConfig.dagFeatures.wmmaQueueCoverCycles =
                std::stoi(a.substr(std::string("--wmma-queue-cover-cycles=").size()));
        } else if (a.starts_with("--wmma-barrier-cover-cycles=")) {
            passFeatureConfig.dagFeatures.wmmaBarrierCoverCycles =
                std::stoi(a.substr(std::string("--wmma-barrier-cover-cycles=").size()));
        } else if (a == "--ds-spread") {
            passFeatureConfig.dagFeatures.dsSpread = true;
        } else if (a == "--no-ds-spread") {
            passFeatureConfig.dagFeatures.dsSpread = false;
        } else if (a.starts_with("--ds-batch-wmmas=")) {
            passFeatureConfig.dagFeatures.dsBatchWmmas =
                std::stoi(a.substr(std::string("--ds-batch-wmmas=").size()));
        } else if (a.starts_with("--ds-catch-up-pace=")) {
            passFeatureConfig.dagFeatures.dsCatchUpPace =
                std::stoi(a.substr(std::string("--ds-catch-up-pace=").size()));
        } else if (a.starts_with("--ds-read-cycles-per-load=")) {
            passFeatureConfig.dagFeatures.dsReadCyclesPerLoad =
                std::stoi(a.substr(std::string("--ds-read-cycles-per-load=").size()));
        } else if (a.starts_with("--wmma-refill-mode=")) {
            const std::string v = a.substr(std::string("--wmma-refill-mode=").size());
            passFeatureConfig.dagFeatures.wmmaRefillMode =
                (v == "dynamic" || v == "1")
                    ? stinkytofu::PassFeatureConfig::WmmaRefillMode::Dynamic
                    : stinkytofu::PassFeatureConfig::WmmaRefillMode::Fixed;
        } else if (a.starts_with("--wmma-batch-size=")) {
            passFeatureConfig.dagFeatures.wmmaBatchSize =
                std::max(1, std::stoi(a.substr(std::string("--wmma-batch-size=").size())));
        } else if (a.starts_with("--wmma-refill-to-cycles=")) {
            passFeatureConfig.dagFeatures.wmmaRefillToCycles =
                std::stoi(a.substr(std::string("--wmma-refill-to-cycles=").size()));
        } else if (a.starts_with("--global-read-queue-depth=")) {
            passFeatureConfig.dagFeatures.globalReadQueueDepth = std::stoi(a.substr(26));
        } else if (a.starts_with("--global-read-drain-latency=")) {
            passFeatureConfig.dagFeatures.globalReadDrainLatency = std::stoi(a.substr(28));
        } else if (a == "--enable-esm2-track-valu-vsrc") {
            passFeatureConfig.dagFeatures.enableESM2TrackValuVsrc = true;
        } else if (a.starts_with("--wait-alu-hold-strict-count=")) {
            passFeatureConfig.dagFeatures.waitAluHoldStrictCount = std::stoi(a.substr(29));
        } else if (a == "--wait-alu-scoreboard") {
            passFeatureConfig.dagFeatures.waitAluScoreboard = true;
        } else if (a.starts_with("--prefetch-lead-wmmas=")) {
            passFeatureConfig.dagFeatures.prefetchLeadWmmas = std::stoi(a.substr(22));
        } else if (a.starts_with("--prefetch-lead-min-stage-wmmas=")) {
            passFeatureConfig.dagFeatures.prefetchLeadMinStageWmmas = std::stoi(a.substr(32));
        } else if (a.starts_with("--war-gate-wmmas=")) {
            passFeatureConfig.dagFeatures.warGateWmmas = std::stoi(a.substr(17));
        } else if (a.starts_with("--merge-barrier-threshold=")) {
            passFeatureConfig.dagFeatures.mergeBarrierThreshold = std::stoi(a.substr(26));
        }
    }

    // Parse and validate user-specified passes from command line
    std::vector<RequestedPass> requestedPasses = parsePassNames(argc, argv, passStartIdx);
    int optLevel = extractOptLevel(argc, argv);

    if (optLevel >= 0 && !requestedPasses.empty()) {
        std::cerr << "Error: -O<N> (pipeline mode) and individual --<Pass> flags are mutually "
                     "exclusive\n";
        return 1;
    }

    if (!requestedPasses.empty()) {
        std::cerr << "\n=== Adding Passes ===\n";
        for (const auto& rp : requestedPasses) {
            // Name check only. Building the pass here would run the creator's
            // side effects (a dump directory, an error on bad arguments) and
            // then throw that instance away.
            if (findCatalogPass(rp.name) == nullptr) {
                std::cerr << "Warning: Unknown pass '" << rp.name << "' - skipping\n";
                std::cerr << "Use --list-passes to see available passes\n";
                continue;
            }
            std::cerr << "Adding pass: " << rp.name;
            if (!rp.args.empty()) {
                std::cerr << " (args:";
                for (const auto& a : rp.args) std::cerr << " " << a;
                std::cerr << ")";
            }
            std::cerr << "\n";
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
    bool verifyEach = false;
    // Off by default: createDebugPrintInstrumentation() opens before.txt/after.txt
    // eagerly, so installing it unconditionally littered the working directory of every
    // run. Mirrors the needsFileOutput gate the pipeline path already has.
    bool dumpPasses = false;
    bool timePasses = false;
    bool preserveSymbolicRegs = false;
    bool preserveComments = false;
    std::string outputFile;
    std::string fromLabel;
    std::string toLabel;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--print-output") printOutput = true;
        if (std::string(argv[i]) == "--emit-asm") emitAsm = true;
        if (std::string(argv[i]) == "--remarks") enableRemarks = true;
        if (std::string(argv[i]) == "--verify-each") verifyEach = true;
        if (std::string(argv[i]) == "--dump-passes") dumpPasses = true;
        if (std::string(argv[i]) == "--time-passes") timePasses = true;
        if (std::string(argv[i]) == "--preserve-symbolic-regs") preserveSymbolicRegs = true;
        if (std::string(argv[i]) == "--preserve-comments") preserveComments = true;
        if (std::string(argv[i]) == "--debug-pass" && i + 1 < argc) {
            // Naming a pass to debug is only meaningful with the dumps on.
            dumpPasses = true;
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

    // Prefer the named stepping's identity when the user gave one; the triple alone first-matches
    // v1 for steppings that share it. resolvedArchName is only set after getArchInfo(name)
    // confirmed the name is registered, so getGfxArchID(name) here never hits its assert path.
    GfxArchID archID =
        resolvedArchName.empty() ? getGfxArchID(arch) : getGfxArchID(resolvedArchName);

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
                std::cerr << "  " << diag.format() << "\n";
            return 1;
        }
        for (const auto& diag : asmResult.diagnostics) std::cerr << diag.format() << "\n";
        asmSignature = std::move(asmResult.signature);
        if (asmResult.parsedFunction)
            parsed.functions.push_back(std::move(asmResult.parsedFunction));
    } else {
        // Parse StinkyTofu IR text format
        parsed = stinkytofu::parseAllSourceStringsWithDiagnostics(fileContent);
        // Fall back to single-function parsing for flat format (no st.func)
        if (!parsed.hasErrors() && parsed.functions.empty()) {
            auto singleResult = stinkytofu::parseSourceStringWithDiagnostics(fileContent);
            parsed.diagnostics.insert(parsed.diagnostics.end(), singleResult.diagnostics.begin(),
                                      singleResult.diagnostics.end());
            if (singleResult.parsedFunction)
                parsed.functions.push_back(std::move(singleResult.parsedFunction));
        }
        if (parsed.hasErrors()) {
            std::cerr << "Error: Failed to parse input file\n";
            for (const auto& diag : parsed.diagnostics) std::cerr << "  " << diag.format() << "\n";
            return 1;
        }
        for (const auto& diag : parsed.diagnostics) std::cerr << diag.format() << "\n";
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
        std::string regionError;
        auto cut =
            stinkytofu::cutParsedRegion(*parsed.functions[0], fromLabel, toLabel, regionError);
        if (!cut) {
            std::cerr << "Error: --from-label/--to-label: " << regionError << "\n";
            return 1;
        }
        preResult.functions.push_back(std::move(cut->before));
        parsed.functions[0] = std::move(cut->region);
        postResult.functions.push_back(std::move(cut->after));
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

    // Set when an instruction or block edge could not be built; the run then exits 1.
    bool loadFailed = false;
    auto populate = [&](stinkytofu::ParsedFunction& pf, stinkytofu::Function& func) {
        std::vector<stinkytofu::Diagnostic> diagnostics;
        const auto result = stinkytofu::StinkyIRConverter::populateFunctionFromParsed(
            pf, func, archID, diagnostics);
        for (const auto& diag : diagnostics) std::cerr << diag.format() << "\n";
        if (result == stinkytofu::StinkyErrorCode::SUCCESS) return true;
        std::cerr << "Error: Failed to populate function '" << pf.funcName << "'\n";
        loadFailed = true;
        return false;
    };

    // Helper: emit a ParsedFunction verbatim (no passes) to the output stream.
    auto emitVerbatim = [&](stinkytofu::MultiParseResult& mr) {
        for (auto& pf : mr.functions) {
            stinkytofu::Function func(pf->funcName);
            if (populate(*pf, func)) emitFunction(func);
        }
    };

    // Emit header: use the parsed SignatureBase when available, otherwise verbatim pre-region.
    if (asmSignature && emitAsm) {
        out << asmSignature->toString();
    } else {
        emitVerbatim(preResult);
    }

    // Set to true if any analysis/verification pass reports a failure; surfaced as a
    // non-zero exit code after all functions and output have been processed.
    bool analysisFailed = false;
    // A creator that rejects its arguments is told once. Later functions skip
    // the name instead of constructing the pass again.
    std::vector<std::string> rejectedPassNames;

    // Process each function independently
    for (auto& parsedFunc : parsed.functions) {
        if (optLevel >= 0) {
            // Pipeline mode: create a StinkyAsmModule and run the registered pipeline
            stinkytofu::StinkyAsmModule::ModuleOptions moduleOpts{};
            moduleOpts.OptLevel = optLevel;
            moduleOpts.EnableRemarks = enableRemarks;
            moduleOpts.VerifyEach = verifyEach;
            moduleOpts.TimePasses = timePasses;
            // The --Tile*/--NumGR*/--NumWaves flags are parsed for both modes but
            // used to be applied only in individual-pass mode, so pipeline mode
            // silently ran every kernel with a zeroed tile config. Backend's entry
            // gate now rejects that, which is the right answer for a real kernel
            // but would make these flags unusable here. Forward them instead.
            // GemmTileConfig::NumWaves is WaveGroup0 * WaveGroup1 downstream, so a
            // flat --NumWaves goes in as WaveGroup0 with WaveGroup1 = 1.
            moduleOpts.TileA0 = static_cast<int>(gemmTileConfig.TileA0);
            moduleOpts.TileB0 = static_cast<int>(gemmTileConfig.TileB0);
            moduleOpts.TileM0 = static_cast<int>(gemmTileConfig.TileM0);
            moduleOpts.NumGRA = gemmTileConfig.NumGRA;
            moduleOpts.NumGRB = gemmTileConfig.NumGRB;
            moduleOpts.NumGRM = gemmTileConfig.NumGRM;
            moduleOpts.WaveGroup0 = static_cast<int>(gemmTileConfig.NumWaves);
            moduleOpts.WaveGroup1 = 1;
            // Backend reads these two from the module options, and there is no
            // rocisa here to supply them.
            stinkytofu::AsmCapsConfig standaloneCaps;
            stinkytofu::BackendRegistry::applyStandaloneCaps(arch, standaloneCaps);
            moduleOpts.RequiresXCntForVolatileVMEM = standaloneCaps.requiresXCntForVolatileVMEM;
            moduleOpts.EnableXnackReplay = standaloneCaps.enableXnackReplay;
            stinkytofu::StinkyAsmModule module(parsedFunc->funcName, arch, moduleOpts);

            stinkytofu::Function& func = module.getFunction();
            if (!populate(*parsedFunc, func)) continue;
            if (asmSignature) {
                func.setMetaData(stinkytofu::kSigTotalVgprsMetaKey,
                                 static_cast<uint64_t>(asmSignature->kernelDescriptor.totalVgprs));
            }

            stinkytofu::Backend backend(module);
            backend.runOptimization();

            emitFunction(func);
        } else {
            // Individual pass mode
            stinkytofu::PassManager passManager = stinkytofu::makeStandalonePassManager(arch);

            // Pipeline mode gets its session from Backend (ModuleOptions.TimePasses);
            // here there is no module, so the driver owns it.
            stinkytofu::TimePassesSession timing(timePasses, parsedFunc->funcName, std::cerr);

            if (dumpPasses) passManager.addInstrumentation(createDebugPrintInstrumentation());
            if (verifyEach) {
                passManager.addInstrumentation(
                    std::make_shared<stinkytofu::VerifyInstrumentation>());
            }
            if (auto timer = stinkytofu::getActiveTimePasses()) {
                passManager.addInstrumentation(std::move(timer));
            }
            passManager.setPassFeatureConfig(passFeatureConfig);
            gemmTileConfig.arch = arch;
            passManager.setGemmTileConfig(gemmTileConfig);
            if (vgprMsbOverride) {
                stinkytofu::AsmCapsConfig caps = passManager.getPassContext().getAsmCapsConfig();
                caps.vgprMsbMode = *vgprMsbOverride;
                passManager.setAsmCapsConfig(caps);
            }
            if (enableRemarks) passManager.getPassContext().setRemarksEnabled(true);

            for (const auto& rp : requestedPasses) {
                if (std::find(rejectedPassNames.begin(), rejectedPassNames.end(), rp.name) !=
                    rejectedPassNames.end())
                    continue;
                // Unknown names were reported before any function ran.
                if (findCatalogPass(rp.name) == nullptr) continue;
                auto pass = createCatalogPass(rp.name, rp.args);
                if (!pass) {
                    rejectedPassNames.push_back(rp.name);
                    std::cerr << "Warning: pass '" << rp.name
                              << "' rejected its arguments - skipping\n";
                    continue;
                }
                passManager.addPass(std::move(pass));
            }

            stinkytofu::Function func(parsedFunc->funcName);
            func.setGemmTileConfig(gemmTileConfig);

            if (!populate(*parsedFunc, func)) continue;
            if (asmSignature) {
                func.setMetaData(stinkytofu::kSigTotalVgprsMetaKey,
                                 static_cast<uint64_t>(asmSignature->kernelDescriptor.totalVgprs));
            }

            passManager.run(func);
            if (passManager.getPassContext().getAnalysisFailed()) analysisFailed = true;

            emitFunction(func);
        }
    }

    emitVerbatim(postResult);

    return (analysisFailed || loadFailed) ? 1 : 0;
}
