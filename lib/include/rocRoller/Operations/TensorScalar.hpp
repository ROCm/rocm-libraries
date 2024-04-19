#pragma once

#include "../DataTypes/DataTypes.hpp"
#include "CommandArgument.hpp"
#include "Operation.hpp"

namespace rocRoller
{
    namespace Operations
    {
        class Scalar : public BaseOperation
        {
        public:
            Scalar() = delete;
            Scalar(VariableType variableType);

            std::string toString() const;
            std::string toString(const unsigned char*) const;

            void allocateArguments();

            CommandArgumentPtr data() const
            {
                return m_pointer;
            }
            VariableType variableType() const
            {
                return m_variableType;
            }

        private:
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

            std::vector<size_t> literalStrides() const
            {
                return m_literalStrides;
            }
            std::vector<CommandArgumentPtr> strides() const
            {
                return m_strides;
            }
            std::vector<CommandArgumentPtr> sizes() const
            {
                return m_sizes;
            }
            CommandArgumentPtr limit() const
            {
                return m_extent;
            }
            VariableType variableType() const
            {
                return m_variableType;
            }
            DataType dataType() const
            {
                return m_variableType.dataType;
            }
            CommandArgumentPtr data() const
            {
                return m_pointer;
            }

        private:
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
