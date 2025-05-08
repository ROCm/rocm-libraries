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

#pragma once

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Operations/CommandArgument.hpp>
#include <rocRoller/Operations/CommandArguments_fwd.hpp>
#include <rocRoller/Operations/Operation.hpp>
#include <rocRoller/Operations/TensorScalar_fwd.hpp>

namespace rocRoller
{
    namespace Operations
    {
        class Literal : public BaseOperation
        {
        public:
            Literal() = delete;
            Literal(CommandArgumentValue value);

            std::string          toString() const;
            CommandArgumentValue value() const;

            bool operator==(Literal const& rhs) const;

        private:
            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;

            CommandArgumentValue m_value;
        };

        std::ostream& operator<<(std::ostream& stream, Literal const& val);

        class Scalar : public BaseOperation
        {
        public:
            Scalar() = delete;
            Scalar(VariableType variableType);

            std::string toString() const;
            std::string toString(const unsigned char*) const;

            void allocateArguments();

            CommandArgumentPtr data() const;
            VariableType       variableType() const;

            bool operator==(Scalar const& rhs) const;

        private:
            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;

            std::string getArgumentString(const unsigned char*) const;

            CommandArgumentPtr m_pointer;
            VariableType       m_variableType;
        };

        std::ostream& operator<<(std::ostream& stream, Scalar const& val);

        class Tensor : public BaseOperation
        {
        public:
            Tensor() = delete;
            Tensor(int numDims, VariableType variableType);
            Tensor(int                        numDims,
                   VariableType               variableType,
                   std::vector<size_t> const& literalStrides);

            std::string toString() const;
            std::string toString(const unsigned char*) const;

            void allocateArguments();

            std::vector<size_t> const& literalStrides() const;

            std::vector<CommandArgumentPtr> const& strides() const;
            std::vector<CommandArgumentPtr> const& sizes() const;

            CommandArgumentPtr limit() const;
            CommandArgumentPtr data() const;

            VariableType variableType() const;
            DataType     dataType() const;

            bool operator==(Tensor const& rhs) const;

        private:
            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;

            std::string getArgumentString(const unsigned char*) const;

            VariableType m_variableType;
            int          m_numDims = -1;

            CommandArgumentPtr m_pointer;
            CommandArgumentPtr m_extent;

            std::vector<CommandArgumentPtr> m_sizes;
            std::vector<CommandArgumentPtr> m_strides;

            std::vector<size_t> m_literalStrides;
        };

        std::ostream& operator<<(std::ostream& stream, Tensor const& val);
    }
}

#include <rocRoller/Operations/TensorScalar_impl.hpp>
