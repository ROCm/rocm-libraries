/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
#pragma once
#include "code.hpp"
#include <memory>

// Forward declaration for StinkyTofu types
namespace stinkytofu
{
    class StinkyAsmModule;
}

namespace rocisa
{
    // External structures and functions
    struct rocIsaPassOption
    {
        std::string hardwareConfigPath = "."; // Path to the hardware config files

        bool stinkyOpt       = false;
        bool insertDelayAlu  = false;
        bool removeDupFunc   = true;
        bool removeDupAssign = true;
        bool getCycles       = true;
        int  numWaves        = 0; // is used when getCycles is true

        // This part is for kernel config to stinkytofu
        uint32_t TileA0        = 16;
        uint32_t TileB0        = 16;
        uint32_t TileM0        = 16;
        uint32_t NumGRA        = 16;
        uint32_t NumGRB        = 16;
        uint32_t NumGRM        = 16;
        uint32_t WavefrontSize = 32;

        bool doOpt() const
        {
            return removeDupAssign;
        }
    };

    struct rocIsaPassResult
    {
        int cycles = -1;
    };

    std::string getActFuncModuleName(int gwvw, int sgpr, int tmpVgpr, int tmpSgpr);
    std::string getActFuncBranchModuleName();

    // rocIsaPass with optional StinkyAsmModule parameter (for gfx1250)
    rocIsaPassResult rocIsaPass(std::shared_ptr<KernelBody>& kernel,
                                const rocIsaPassOption&      option);

    // Internal use only
    struct Graph
    {
        std::vector<std::vector<std::shared_ptr<Item>>> vgpr;
        std::vector<std::vector<std::shared_ptr<Item>>> sgpr;
        std::vector<std::vector<std::shared_ptr<Item>>> mgpr;

        std::vector<std::vector<std::shared_ptr<Item>>>& getGprRef(const std::string& gpr)
        {
            if(gpr == "v")
                return vgpr;
            else if(gpr == "s")
                return sgpr;
            else if(gpr == "m")
                return mgpr;
            else
                throw std::runtime_error("Invalid gpr type");
        }
    };

    // No opt item container class
    class NoOptItem : public Item
    {
    public:
        NoOptItem(std::shared_ptr<Item> item)
            : Item("")
            , _item(item)
        {
        }
        std::shared_ptr<Item> getItem() const
        {
            return _item;
        }

    private:
        std::shared_ptr<Item> _item;
    };

    void insertDelayAlu(std::shared_ptr<Module> module);
    void removeDuplicatedFunction(std::shared_ptr<Module> module);
    void compositeToInstruction(std::shared_ptr<Module>& module);
    void convertTextVariablesToRegisters(std::shared_ptr<Module> module);
    Graph buildGraph(
        std::shared_ptr<Module>               module,
        int                                   vgprMax,
        int                                   sgprMax); // Return value is std::move()d
    void removeDuplicateAssignment(Graph& graph);

    int getCycles(std::shared_ptr<Module> module, int numWaves);
} // namespace rocisa
