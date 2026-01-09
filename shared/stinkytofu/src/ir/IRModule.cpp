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

#include "ir/IRModule.hpp"
#include "ir/StinkyInstructions.hpp"
#include <iostream>

namespace stinkytofu
{
    struct IRModule::Impl
    {
        std::string                                 name;
        std::vector<std::shared_ptr<IRInstruction>> instructions;

        Impl(const std::string& name)
            : name(name)
        {
        }

        ~Impl()
        {
            // shared_ptr will automatically handle cleanup
        }
    };

    IRModule::IRModule(const std::string& name)
        : pImpl(std::make_unique<Impl>(name))
    {
    }

    IRModule::~IRModule() = default;

    IRModule::IRModule(IRModule&&) noexcept            = default;
    IRModule& IRModule::operator=(IRModule&&) noexcept = default;

    std::string IRModule::getName() const
    {
        return pImpl->name;
    }

    std::shared_ptr<IRInstruction> IRModule::add(std::shared_ptr<IRInstruction> inst)
    {
        if(inst)
        {
            pImpl->instructions.push_back(inst);
        }
        return inst;
    }

    const std::vector<std::shared_ptr<IRInstruction>>& IRModule::getInstructions() const
    {
        return pImpl->instructions;
    }

    size_t IRModule::size() const
    {
        return pImpl->instructions.size();
    }

    void IRModule::dump(std::ostream& out) const
    {
        out << "IRModule: " << pImpl->name << "\n";
        out << "Instructions: " << pImpl->instructions.size() << "\n";
        for(size_t i = 0; i < pImpl->instructions.size(); ++i)
        {
            out << "  [" << i << "] ";
            pImpl->instructions[i]->dump(out);
            out << "\n";
        }
    }

} // namespace stinkytofu
