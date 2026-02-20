/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/TensorOps.hpp>
#include <Tensile/Utils.hpp>

namespace Tensile
{
    std::map<std::string, TensorOp::Type> TensorOp::typeNames;

    TensorOp::TensorOp() = default;
    TensorOp::TensorOp(Type type)
        : type(type)
    {
    }

    std::string ToString(TensorOp::Type t)
    {
        switch(t)
        {
        case TensorOp::Type::None:
            return "None";
        case TensorOp::Type::ComplexConjugate:
            return "ComplexConjugate";

        case TensorOp::Type::Count:;
        }

        return "Invalid";
    }

    std::string Suffix(TensorOp::Type t)
    {
        switch(t)
        {
        case TensorOp::Type::None:
            return "";
        case TensorOp::Type::ComplexConjugate:
            return "C";

        case TensorOp::Type::Count:;
        }

        return "Invalid";
    }

    TensorOp::Type TensorOp::GetType(std::string const& name)
    {
        static int call_once = (InitTypeNames(), 0);

        auto iter = typeNames.find(name);
        if(iter == typeNames.end())
            throw std::runtime_error(concatenate("Invalid TensorOp type: ", name));

        // Use variable to quiet the compiler.
        if(call_once)
            return iter->second;
        return iter->second;
    }

    void TensorOp::InitTypeNames()
    {
        for(int idx = 0; idx < static_cast<int>(Type::Count); idx++)
        {
            Type type                 = static_cast<Type>(idx);
            typeNames[ToString(type)] = type;
            typeNames[Suffix(type)]   = type;
        }
    }

    std::string TensorOp::name() const
    {
        return ToString(type);
    }

    std::string TensorOp::suffix() const
    {
        return Suffix(type);
    }

    std::ostream& operator<<(std::ostream& stream, TensorOp const& t)
    {
        return stream << t.type;
    }

    std::istream& operator>>(std::istream& stream, TensorOp& t)
    {
        return stream >> t.type;
    }

    std::ostream& operator<<(std::ostream& stream, TensorOp::Type const& t)
    {
        return stream << ToString(t);
    }

    std::istream& operator>>(std::istream& stream, TensorOp::Type& t)
    {
        std::string typeName;
        stream >> typeName;

        t = TensorOp::GetType(typeName);

        return stream;
    }
} // namespace Tensile
