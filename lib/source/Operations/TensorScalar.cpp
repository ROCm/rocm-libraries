#include "rocRoller/Operations/TensorScalar.hpp"
#include "rocRoller/Operations/Command.hpp"
#include "rocRoller/Operations/Operation.hpp"

namespace rocRoller
{
    namespace Operations
    {
        std::string Tensor::getArgumentString(const unsigned char* runtime_args) const
        {
            std::ostringstream msg;

            if(m_variableType.dataType != DataType::None)
                msg << "." << m_variableType.dataType;
            msg << ".d" << m_numDims << " " << m_tag << ", "
                << "(base=";
            if(m_pointer)
            {
                if(runtime_args)
                    msg << *(reinterpret_cast<const size_t*>(runtime_args + m_pointer->offset()));
                else
                    msg << "&" << m_pointer->offset();
            }
            msg << ", lim=";
            if(m_extent)
            {
                if(runtime_args)
                    msg << *(reinterpret_cast<const size_t*>(runtime_args + m_extent->offset()));
                else
                    msg << "&" << m_extent->offset();
            }
            msg << ", sizes={";
            for(auto i : m_sizes)
            {
                if(runtime_args)
                    msg << *(reinterpret_cast<const size_t*>(runtime_args + i->offset())) << " ";
                else
                    msg << "&" << i->offset() << " ";
            }
            msg << "}, strides={";
            for(auto i : m_strides)
            {
                if(runtime_args)
                    msg << *(reinterpret_cast<const size_t*>(runtime_args + i->offset())) << " ";
                else
                    msg << "&" << i->offset() << " ";
            }
            msg << "})";

            return msg.str();
        }

        Tensor::Tensor(int numDims, VariableType variableType)
            : BaseOperation()
            , m_variableType(variableType)
            , m_numDims(numDims)
        {
        }

        Tensor::Tensor(int                        numDims,
                       VariableType               variableType,
                       std::vector<size_t> const& literalStrides)
            : Tensor(numDims, variableType)
        {
            AssertFatal(literalStrides.size() <= numDims,
                        "Cannot specify more literal strides than dimensions.");

            m_literalStrides = literalStrides;
        }

        void Tensor::allocateArguments()
        {
            if(auto ptr = m_command.lock())
            {
                std::string base = concatenate("Tensor_", m_tag);

                // update DataDirection
                m_pointer
                    = ptr->allocateArgument({m_variableType.dataType, PointerType::PointerGlobal},
                                            DataDirection::ReadWrite,
                                            base + "_pointer");

                m_extent = ptr->allocateArgument(
                    DataType::Int64, DataDirection::ReadOnly, base + "_extent");
                m_sizes = ptr->allocateArgumentVector(
                    DataType::Int64, m_numDims, DataDirection::ReadOnly, base + "_size");
                m_strides = ptr->allocateArgumentVector(
                    DataType::Int64, m_numDims, DataDirection::ReadOnly, base + "_stride");
            }
        }

        std::string Tensor::toString() const
        {
            return "Tensor" + getArgumentString(nullptr);
        }

        std::string Tensor::toString(const unsigned char* runtime_args) const
        {
            return "Tensor" + getArgumentString(runtime_args);
        }

        std::ostream& operator<<(std::ostream& stream, Tensor const& val)
        {
            return stream << val.toString();
        }

        std::string Scalar::getArgumentString(const unsigned char* runtime_args) const
        {
            std::ostringstream msg;

            if(m_variableType.dataType != DataType::None)
                msg << "." << m_variableType.dataType;
            msg << "." << m_tag;

            return msg.str();
        }

        Scalar::Scalar(VariableType variableType)
            : BaseOperation()
            , m_variableType(variableType)
        {
        }

        void Scalar::allocateArguments()
        {
            if(auto ptr = m_command.lock())
            {
                m_pointer = ptr->allocateArgument(m_variableType, DataDirection::ReadOnly);
            }
        }

        std::string Scalar::toString() const
        {
            return "Scalar" + getArgumentString(nullptr);
        }

        std::string Scalar::toString(const unsigned char* runtime_args) const
        {
            return "Scalar" + getArgumentString(runtime_args);
        }

        std::ostream& operator<<(std::ostream& stream, Scalar const& val)
        {
            return stream << val.toString();
        }

        Literal::Literal(CommandArgumentValue value)
            : BaseOperation()
            , m_value(value)
        {
        }

        std::string Literal::toString() const
        {
            return "Literal";
        }

        std::ostream& operator<<(std::ostream& stream, Literal const& val)
        {
            return stream << val.toString();
        }
    }
}
