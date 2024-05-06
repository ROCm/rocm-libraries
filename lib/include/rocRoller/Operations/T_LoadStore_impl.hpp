#pragma once

#include "T_LoadStore.hpp"

namespace rocRoller
{
    namespace Operations
    {
        inline T_Load_Linear::T_Load_Linear(OperationTag tensor)
            : BaseOperation()
            , m_tensorTag(tensor)
        {
        }

        inline OperationTag T_Load_Linear::getTensorTag() const
        {
            return m_tensorTag;
        }

        inline std::string T_Load_Linear::toString() const
        {
            std::ostringstream msg;

            msg << "T_LOAD_LINEAR " << m_tag << " Tensor " << m_tensorTag;

            return msg.str();
        }

        inline std::ostream& operator<<(std::ostream& stream, T_Load_Linear const& val)
        {
            return stream << val.toString();
        }

        inline T_Load_Scalar::T_Load_Scalar(OperationTag scalar)
            : BaseOperation()
            , m_scalarTag(scalar)
        {
        }

        inline OperationTag T_Load_Scalar::getScalarTag() const
        {
            return m_scalarTag;
        }

        inline std::string T_Load_Scalar::toString() const
        {
            std::ostringstream msg;

            msg << "T_LOAD_SCALAR " << m_tag << "Scalar " << m_scalarTag;

            return msg.str();
        }

        inline std::ostream& operator<<(std::ostream& stream, T_Load_Scalar const& val)
        {
            return stream << val.toString();
        }

        inline T_Load_Tiled::T_Load_Tiled(OperationTag tensor)
            : BaseOperation()
            , m_tensorTag(tensor)
        {
        }

        inline OperationTag T_Load_Tiled::getTensorTag() const
        {
            return m_tensorTag;
        }

        inline std::string T_Load_Tiled::toString() const
        {
            std::ostringstream msg;

            msg << "T_LOAD_TILED " << m_tag << " Tensor " << m_tensorTag;

            return msg.str();
        }

        inline std::ostream& operator<<(std::ostream& stream, T_Load_Tiled const& val)
        {
            return stream << val.toString();
        }

        inline T_Store_Linear::T_Store_Linear(OperationTag source, OperationTag tensor)
            : BaseOperation()
            , m_srcTag(source)
            , m_tensorTag(tensor)
        {
        }

        inline OperationTag T_Store_Linear::getSrcTag() const
        {
            return m_srcTag;
        }

        inline OperationTag T_Store_Linear::getTensorTag() const
        {
            return m_tensorTag;
        }

        inline std::string T_Store_Linear::toString() const
        {
            std::ostringstream msg;

            msg << "T_STORE_LINEAR " << m_tag << " Source " << m_srcTag << " Tensor "
                << m_tensorTag;

            return msg.str();
        }

        inline std::ostream& operator<<(std::ostream& stream, T_Store_Linear const& val)
        {
            return stream << val.toString();
        }

        inline T_Store_Tiled::T_Store_Tiled(OperationTag source, OperationTag tensor)
            : BaseOperation()
            , m_srcTag(source)
            , m_tensorTag(tensor)
        {
        }

        inline OperationTag T_Store_Tiled::getSrcTag() const
        {
            return m_srcTag;
        }

        inline OperationTag T_Store_Tiled::getTensorTag() const
        {
            return m_tensorTag;
        }

        inline std::string T_Store_Tiled::toString() const
        {
            std::ostringstream msg;

            msg << "T_STORE_TILED " << m_tag << " Source " << m_srcTag << " Tensor " << m_tensorTag;

            return msg.str();
        }

        inline std::ostream& operator<<(std::ostream& stream, T_Store_Tiled const& val)
        {
            return stream << val.toString();
        }
    }
}
