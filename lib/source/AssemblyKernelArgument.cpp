/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

/**
 */

#include <string>

#include <rocRoller/AssemblyKernelArgument.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    bool AssemblyKernelArgument::operator==(AssemblyKernelArgument const& rhs) const
    {
        return name == rhs.name //
               && variableType == rhs.variableType //
               && dataDirection == rhs.dataDirection //
               && equivalent(expression, rhs.expression) //
               && offset == rhs.offset //
               && size == rhs.size;
    }

    std::string AssemblyKernelArgument::toString() const
    {
        auto rv = concatenate("KernelArg{", name, ", ", variableType);

        if(dataDirection != DataDirection::ReadOnly)
            rv += concatenate(", ", dataDirection);

        rv += concatenate(", ", expression);

        if(offset != -1)
            rv += concatenate(", o:", offset);

        if(size != -1)
            rv += concatenate(", s:", size);

        return rv + "}";
    }

    std::ostream& operator<<(std::ostream& stream, AssemblyKernelArgument const& arg)
    {
        return stream << arg.toString();
    }
}
