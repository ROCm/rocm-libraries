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

#include <rocRoller/Operations/TensorScalar.hpp>

namespace rocRoller
{
    namespace Operations
    {
        inline CommandArgumentValue Literal::value() const
        {
            return m_value;
        }

        inline bool Literal::operator==(Literal const& rhs) const
        {
            return m_tag == rhs.m_tag && m_value == rhs.m_value;
        }

        inline CommandArgumentPtr Scalar::data() const
        {
            return m_pointer;
        }

        inline VariableType Scalar::variableType() const
        {
            return m_variableType;
        }

        inline bool Scalar::operator==(Scalar const& rhs) const
        {
            if(m_pointer && !rhs.m_pointer)
                return false;
            if(!m_pointer && rhs.m_pointer)
                return false;
            if(!m_pointer && !rhs.m_pointer)
                return m_tag == rhs.m_tag && (m_variableType == rhs.m_variableType);
            return m_tag == rhs.m_tag && (*m_pointer) == (*rhs.m_pointer)
                   && (m_variableType == rhs.m_variableType);
        }

        inline std::vector<size_t> const& Tensor::literalStrides() const
        {
            return m_literalStrides;
        }

        inline std::vector<CommandArgumentPtr> const& Tensor::strides() const
        {
            return m_strides;
        }

        inline std::vector<CommandArgumentPtr> const& Tensor::sizes() const
        {
            return m_sizes;
        }

        inline CommandArgumentPtr Tensor::limit() const
        {
            return m_extent;
        }

        inline VariableType Tensor::variableType() const
        {
            return m_variableType;
        }

        inline DataType Tensor::dataType() const
        {
            return m_variableType.dataType;
        }

        inline CommandArgumentPtr Tensor::data() const
        {
            return m_pointer;
        }

        inline bool Tensor::operator==(Tensor const& rhs) const
        {
            if(m_pointer && !rhs.m_pointer)
                return false;
            if(!m_pointer && rhs.m_pointer)
                return false;
            if(m_extent && !rhs.m_extent)
                return false;
            if(!m_extent && rhs.m_extent)
                return false;
            if(m_sizes.size() != rhs.m_sizes.size())
                return false;
            if(m_strides.size() != rhs.m_strides.size())
                return false;

            bool equal = true;
            equal &= m_numDims == rhs.m_numDims;
            equal &= m_variableType == rhs.m_variableType;
            if(m_pointer)
                equal &= (*m_pointer) == (*rhs.m_pointer);
            if(m_extent)
                equal &= (*m_extent) == *(rhs.m_extent);
            for(auto i = 0; i < m_sizes.size(); ++i)
                equal &= (*m_sizes[i]) == (*rhs.m_sizes[i]);
            for(auto i = 0; i < m_strides.size(); ++i)
                equal &= (*m_strides[i]) == (*rhs.m_strides[i]);
            equal &= m_literalStrides == rhs.m_literalStrides;

            return equal;
        }
    }
}
