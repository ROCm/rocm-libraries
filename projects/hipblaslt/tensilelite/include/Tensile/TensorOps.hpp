/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <Tensile/Tensile.hpp>
#include <Tensile/TensorDescriptor_fwd.hpp>
#include <Tensile/TensorOps_fwd.hpp>

namespace TensileLite
{
    /**
 * \addtogroup Problem
 * @{
 */

    using TensorOps = std::vector<TensorOp>;

    /**
 * Represents a tensor operation that can be performed inline to a kernel.
 * For now can represent a complex conjugate but this could be where we
 * represent inline activation layers later.
 */
    class TENSILE_API TensorOp
    {
    public:
        enum class Type : int
        {
            None,
            ComplexConjugate,
            Count
        };

        Type type = Type::None;

        TensorOp();
        TensorOp(Type type);

        static TensorOp ComplexConjugate()
        {
            return TensorOp(Type::ComplexConjugate);
        }

        std::string name() const;
        std::string suffix() const;

        // static TensorOp ReLU();
        // static TensorOp LeakyReLU(float alpha);
        // ...

        bool operator==(TensorOp const& rhs) const
        {
            return this->type == rhs.type;
        }
        bool operator<(TensorOp const& rhs) const
        {
            return this->type < rhs.type;
        }

        bool operator>(TensorOp const& rhs) const
        {
            return rhs < *this;
        }
        bool operator!=(TensorOp const& rhs) const
        {
            return !(*this == rhs);
        }
        bool operator<=(TensorOp const& rhs) const
        {
            return !(*this > rhs);
        }
        bool operator>=(TensorOp const& rhs) const
        {
            return !(*this < rhs);
        }

        static Type GetType(std::string const& name);

    private:
        static void                        InitTypeNames();
        static std::map<std::string, Type> typeNames;
    };

    std::string ToString(TensorOp::Type t);
    std::string Suffix(TensorOp::Type t);

    std::ostream& operator<<(std::ostream& stream, TensorOp const& t);
    std::istream& operator>>(std::istream& stream, TensorOp& t);

    std::ostream& operator<<(std::ostream& stream, TensorOp::Type const& t);
    std::istream& operator>>(std::istream& stream, TensorOp::Type& t);

    /**
 * @}
 */
} // namespace TensileLite
