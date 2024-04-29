#pragma once

#include "TensorScalar.hpp"

namespace rocRoller
{
    namespace Operations
    {

        inline CommandArgumentPtr Scalar::data() const
        {
            return m_pointer;
        }

        inline VariableType Scalar::variableType() const
        {
            return m_variableType;
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
    }
}
