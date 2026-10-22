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
#include "container.hpp"
#include "code.hpp"
#include "instruction/branch.hpp"
#include "instruction/common.hpp"
#include "instruction/instruction.hpp"

namespace rocisa
{
    std::shared_ptr<Item> replaceHolder(std::shared_ptr<Item> inst, int dst)
    {
        if(auto mod = std::dynamic_pointer_cast<Module>(inst))
        {
            for(auto& item : mod->items())
            {
                // We don't need the output
                static_cast<void>(replaceHolder(item, dst));
            }
        }
        else if(auto _inst = std::dynamic_pointer_cast<Instruction>(inst))
        {
            for(auto& param : _inst->getParams())
            {
                if(auto container = std::get_if<std::shared_ptr<Container>>(&param))
                {
                    if(auto holder = std::dynamic_pointer_cast<HolderContainer>(*container))
                    {
                        holder->setRegNum(dst);
                        param = std::make_shared<RegisterContainer>(holder->getCopiedRC());
                    }
                }
            }
        }
        else if(auto waitcnt = std::dynamic_pointer_cast<SWaitCnt>(inst))
        {
            throw std::runtime_error("SWaitCnt is not supported yet");
        }
        return inst;
    }

    // Helper function to create GPR containers
    std::shared_ptr<RegisterContainer>
        createGPR(const std::string& gprType, const Holder& holder, float regNum)
    {
        if(holder.idx == -1)
        {
            return std::make_shared<HolderContainer>(gprType, *holder.name, regNum);
        }
        return std::make_shared<HolderContainer>(gprType, holder.idx, regNum);
    }

    std::shared_ptr<RegisterContainer> createGPR(const std::string& gprType, int idx, float regNum)
    {
        return std::make_shared<RegisterContainer>(gprType, std::nullopt, idx, regNum);
    }

    std::shared_ptr<RegisterContainer> createGPR(const std::string& gprType,
                                                 const std::string& name,
                                                 float              regNum  = 1.f,
                                                 bool               isMacro = false,
                                                 bool               isAbs   = false,
                                                 bool               isOff   = false)
    {
        RegName regname = generateRegName(name);
        return std::make_shared<RegisterContainer>(
            gprType, regname, isAbs, isMacro, isOff, -1, regNum);
    }

    // Overloaded functions to create specific GPR containers with default regNum = 1.f
    std::shared_ptr<RegisterContainer> vgpr(const Holder& holder, float regNum)
    {
        return createGPR("v", holder, regNum);
    }

    std::shared_ptr<RegisterContainer> vgpr(int idx, float regNum)
    {
        return createGPR("v", idx, regNum);
    }

    std::shared_ptr<RegisterContainer>
        vgpr(const std::string& name, float regNum, bool isMacro, bool isAbs, bool isOff)
    {
        return createGPR("v", name, regNum, isMacro, isAbs, isOff);
    }

    std::shared_ptr<RegisterContainer> sgpr(const Holder& holder, float regNum)
    {
        return createGPR("s", holder, regNum);
    }

    std::shared_ptr<RegisterContainer> sgpr(int idx, float regNum)
    {
        return createGPR("s", idx, regNum);
    }

    std::shared_ptr<RegisterContainer> sgpr(const std::string& name, float regNum, bool isMacro, bool isOff)
    {
        return createGPR("s", name, regNum, isMacro, false, isOff);
    }

    std::shared_ptr<RegisterContainer> accvgpr(const Holder& holder, float regNum)
    {
        return createGPR("acc", holder, regNum);
    }

    std::shared_ptr<RegisterContainer> accvgpr(int idx, float regNum)
    {
        return createGPR("acc", idx, regNum);
    }

    std::shared_ptr<RegisterContainer> accvgpr(const std::string& name, float regNum)
    {
        return createGPR("acc", name, regNum);
    }

    std::shared_ptr<RegisterContainer> mgpr(const Holder& holder, float regNum)
    {
        return createGPR("m", holder, regNum);
    }

    std::shared_ptr<RegisterContainer> mgpr(int idx, float regNum)
    {
        return createGPR("m", idx, regNum);
    }

    std::shared_ptr<RegisterContainer> mgpr(const std::string& name, float regNum)
    {
        return createGPR("m", name, regNum);
    }

} // namespace rocisa
