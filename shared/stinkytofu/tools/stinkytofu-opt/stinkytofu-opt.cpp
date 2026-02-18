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
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace stinkytofu;

namespace
{
    /**
     * DeserializeStinkytofuIRPass is a pass that deserializes Stinkytofu IR from a file.
     */
    class DeserializeStinkytofuIRPass : public StinkyInstPass
    {
    public:
        static char ID;

        DeserializeStinkytofuIRPass(const std::string& stinkytofuIRFile)
            : stinkytofuIRFile(stinkytofuIRFile)
        {
        }

        const char* getName() const override
        {
            return "DeserializeStinkytofuIRPass";
        }

        PassID getPassID() const override
        {
            return &DeserializeStinkytofuIRPass::ID;
        }

        void run(Function& func, PassContext& passCtx) override
        {
            GfxArchID arch = getGfxArchID(passCtx.getGemmTileConfig().arch[0],
                                         passCtx.getGemmTileConfig().arch[1],
                                         passCtx.getGemmTileConfig().arch[2]);

            std::string irText = readFile(stinkytofuIRFile);

            // Use the shared conversion logic from StinkyIRConverter (creates entry block)
            StinkyErrorCode result
                = StinkyIRConverter::populateFunctionFromString(irText, func, passCtx, arch);
            if(result != StinkyErrorCode::SUCCESS)
            {
                if(result == StinkyErrorCode::PASSCTX_EMPTY)
                {
                    std::cerr << "No PassContext available. Call convertToIRList first."
                              << "\n";
                }
                else
                {
                    std::cerr << "Error: Failed to populate IRList from string. Error code: "
                              << static_cast<int>(result) << "\n";
                }
            }
        }

    private:
        const std::string stinkytofuIRFile;

        std::string readFile(const std::string& filename)
        {
            std::ifstream     file(filename);
            std::stringstream buffer;
            buffer << file.rdbuf();
            return buffer.str();
        }
    };

    char DeserializeStinkytofuIRPass::ID = 0;

    // Function to print available passes
    void printAvailablePasses()
    {
        std::cout << "Available passes:\n";
        std::cout << "=================\n";
        for(const auto& passInfo : availablePasses)
        {
            std::cout << "  --" << passInfo.name << "\n";
        }
    }

    // Function to find and create a pass by name
    std::unique_ptr<Pass> createPassByName(const std::string& passName)
    {
        for(const auto& passInfo : availablePasses)
        {
            if(passName == passInfo.name)
            {
                return passInfo.creator();
            }
        }
        return nullptr;
    }

    // Function to parse command-line arguments for passes
    std::vector<std::string> parsePassNames(int argc, char** argv, int startIdx)
    {
        std::vector<std::string> passNames;
        for(int i = startIdx; i < argc; ++i)
        {
            std::string arg = argv[i];
            if(arg.substr(0, 2) == "--")
            {
                passNames.push_back(arg.substr(2)); // Remove "--" prefix
            }
        }
        return passNames;
    }
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " [options] <ir_file> [--pass1] [--pass2] ...\n\n";
        std::cerr << "Options:\n";
        std::cerr << "  --arch <arch>    Target architecture (gfx942, gfx950, gfx1250)\n";
        std::cerr << "  --list-passes    List all available passes\n";
        std::cerr << "  --help           Show this help message\n\n";
        std::cerr << "Example:\n";
        std::cerr << "  " << argv[0] << " --arch gfx942 input.txt --StinkyDAGSchedulerPass\n";
        return 1;
    }

    // Check for special options
    std::string firstArg = argv[1];
    if(firstArg == "--list-passes")
    {
        printAvailablePasses();
        return 0;
    }
    if(firstArg == "--help")
    {
        std::cerr << "stinkytofu-opt - StinkyTofu IR optimizer\n\n";
        std::cerr << "Usage: " << argv[0] << " [options] <ir_file> [--pass1] [--pass2] ...\n\n";
        std::cerr << "Options:\n";
        std::cerr << "  --arch <arch>    Target architecture (gfx942, gfx950, gfx1250)\n";
        std::cerr << "  --list-passes    List all available passes\n";
        std::cerr << "  --help           Show this help message\n\n";
        printAvailablePasses();
        return 0;
    }

    // Parse architecture option
    std::array<int, 3> arch         = {9, 4, 2}; // default gfx942
    int                irFileIdx    = 1;
    int                passStartIdx = 2;

    if(firstArg == "--arch")
    {
        if(argc < 4)
        {
            std::cerr << "Error: --arch requires an architecture argument\n";
            std::cerr << "Supported architectures: gfx942, gfx950, gfx1250\n";
            return 1;
        }

        std::string archStr = argv[2];
        if(archStr == "gfx942")
        {
            arch = {9, 4, 2};
        }
        else if(archStr == "gfx950")
        {
            arch = {9, 5, 0};
        }
        else if(archStr == "gfx1250")
        {
            arch = {12, 5, 0};
        }
        else
        {
            std::cerr << "Error: Unsupported architecture '" << archStr << "'\n";
            std::cerr << "Supported architectures: gfx942, gfx950, gfx1250\n";
            return 1;
        }

        irFileIdx    = 3;
        passStartIdx = 4;
        std::cout << "Target architecture: " << archStr << "\n";
    }

    std::string filename = argv[irFileIdx];

    auto                          debugConfig       = getPassManagerDebugConfig();
    stinkytofu::PassFeatureConfig passFeatureConfig = getPassFeatureConfig();

    stinkytofu::PassManager passManager;

    passManager.setDebugConfig(std::move(debugConfig));
    passManager.setPassFeatureConfig(passFeatureConfig);
    setKernelConfig(passManager, arch);

    // Add deserialization pass first to load the IR with the specified architecture
    passManager.addPass(std::make_unique<DeserializeStinkytofuIRPass>(filename));

    // Parse and add user-specified passes from command line
    std::vector<std::string> requestedPasses = parsePassNames(argc, argv, passStartIdx);

    if(!requestedPasses.empty())
    {
        std::cout << "\n=== Adding Passes ===\n";
        for(const auto& passName : requestedPasses)
        {
            auto pass = createPassByName(passName);
            if(pass)
            {
                std::cout << "Adding pass: " << passName << "\n";
                passManager.addPass(std::move(pass));
            }
            else
            {
                std::cerr << "Warning: Unknown pass '" << passName << "' - skipping\n";
                std::cerr << "Use --list-passes to see available passes\n";
            }
        }
        std::cout << "\n";
    }
    else
    {
        std::cout << "\n=== No optimization passes specified ===\n";
        std::cout << "Only deserialization will be performed.\n";
        std::cout << "Use --list-passes to see available passes.\n\n";
    }

    stinkytofu::Function func("kernel");
    passManager.run(func);

    return 0;
}
