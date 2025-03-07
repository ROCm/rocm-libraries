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

#include <rocRoller/Operations/Operation.hpp>

#include <rocRoller/Serialization/Base_fwd.hpp>

namespace rocRoller
{
    namespace Operations
    {
        class T_Load_Linear : public BaseOperation
        {
        public:
            T_Load_Linear() = delete;
            explicit T_Load_Linear(OperationTag tensor);

            OperationTag getSrcTag() const;
            std::string  toString() const;

            bool operator==(T_Load_Linear const& rhs) const;

        private:
            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;

            OperationTag m_tensorTag;
            OperationTag m_srcTag;
        };

        std::ostream& operator<<(std::ostream& stream, T_Load_Linear const& val);

        class T_Load_Scalar : public BaseOperation
        {
        public:
            T_Load_Scalar() = delete;
            explicit T_Load_Scalar(OperationTag scalar);

            OperationTag getSrcTag() const;
            std::string  toString() const;

            bool operator==(T_Load_Scalar const& rhs) const;

        private:
            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;

            OperationTag m_scalarTag;
            OperationTag m_srcTag;
        };

        std::ostream& operator<<(std::ostream& stream, T_Load_Scalar const& val);

        class T_Load_Tiled : public BaseOperation
        {
        public:
            T_Load_Tiled() = delete;
            explicit T_Load_Tiled(OperationTag tensor);

            OperationTag getSrcTag() const;
            std::string  toString() const;

            bool operator==(T_Load_Tiled const& rhs) const;

        private:
            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;

            OperationTag m_tensorTag;
            OperationTag m_srcTag;
        };

        std::ostream& operator<<(std::ostream& stream, T_Load_Tiled const& val);

        class T_Store_Linear : public BaseOperation
        {
            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;

        public:
            T_Store_Linear() = delete;
            T_Store_Linear(OperationTag source, OperationTag tensor);

            OperationTag getSrcTag() const;
            OperationTag getDstTag() const;
            std::string  toString() const;

            bool operator==(T_Store_Linear const& rhs) const;

        private:
            OperationTag m_srcTag;
            OperationTag m_dstTag;
        };

        std::ostream& operator<<(std::ostream& stream, T_Store_Linear const& val);

        class T_Store_Tiled : public BaseOperation
        {
        public:
            T_Store_Tiled() = delete;
            T_Store_Tiled(OperationTag source, OperationTag tensor);

            OperationTag getSrcTag() const;
            OperationTag getDstTag() const;
            std::string  toString() const;

            bool operator==(T_Store_Tiled const& rhs) const;

        private:
            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;

            OperationTag m_srcTag;
            OperationTag m_dstTag;
        };

        std::ostream& operator<<(std::ostream& stream, T_Store_Tiled const& val);
    }
}
