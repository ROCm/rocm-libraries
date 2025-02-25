#include <rocRoller/Operations/T_LoadStore.hpp>

#include <sstream>

namespace rocRoller
{
    namespace Operations
    {
        T_Load_Linear::T_Load_Linear(OperationTag src)
            : BaseOperation()
            , m_srcTag(src)
        {
        }

        OperationTag T_Load_Linear::getSrcTag() const
        {
            return m_srcTag;
        }

        std::string T_Load_Linear::toString() const
        {
            std::ostringstream msg;

            msg << "T_LOAD_LINEAR " << m_tag << " Source " << m_srcTag;

            return msg.str();
        }

        std::ostream& operator<<(std::ostream& stream, T_Load_Linear const& val)
        {
            return stream << val.toString();
        }

        bool T_Load_Linear::operator==(T_Load_Linear const& rhs) const
        {
            return m_tag == rhs.m_tag && m_tensorTag == rhs.m_tensorTag;
        }

        T_Load_Scalar::T_Load_Scalar(OperationTag src)
            : BaseOperation()
            , m_srcTag(src)
        {
        }

        OperationTag T_Load_Scalar::getSrcTag() const
        {
            return m_srcTag;
        }

        std::string T_Load_Scalar::toString() const
        {
            std::ostringstream msg;

            msg << "T_LOAD_SCALAR " << m_tag << "Source " << m_srcTag;

            return msg.str();
        }

        std::ostream& operator<<(std::ostream& stream, T_Load_Scalar const& val)
        {
            return stream << val.toString();
        }

        bool T_Load_Scalar::operator==(T_Load_Scalar const& rhs) const
        {
            return m_tag == rhs.m_tag && m_scalarTag == rhs.m_scalarTag;
        }

        T_Load_Tiled::T_Load_Tiled(OperationTag src)
            : BaseOperation()
            , m_srcTag(src)
        {
        }

        OperationTag T_Load_Tiled::getSrcTag() const
        {
            return m_srcTag;
        }

        std::string T_Load_Tiled::toString() const
        {
            std::ostringstream msg;

            msg << "T_LOAD_TILED " << m_tag << " Source " << m_srcTag;

            return msg.str();
        }

        std::ostream& operator<<(std::ostream& stream, T_Load_Tiled const& val)
        {
            return stream << val.toString();
        }

        bool T_Load_Tiled::operator==(T_Load_Tiled const& rhs) const
        {
            return m_tag == rhs.m_tag && m_tensorTag == rhs.m_tensorTag;
        }

        T_Store_Linear::T_Store_Linear(OperationTag source, OperationTag dest)
            : BaseOperation()
            , m_srcTag(source)
            , m_dstTag(dest)
        {
        }

        OperationTag T_Store_Linear::getSrcTag() const
        {
            return m_srcTag;
        }

        OperationTag T_Store_Linear::getDstTag() const
        {
            return m_dstTag;
        }

        std::string T_Store_Linear::toString() const
        {
            std::ostringstream msg;

            msg << "T_STORE_LINEAR " << m_tag << " Source " << m_srcTag << " Dest " << m_dstTag;

            return msg.str();
        }

        std::ostream& operator<<(std::ostream& stream, T_Store_Linear const& val)
        {
            return stream << val.toString();
        }

        bool T_Store_Linear::operator==(T_Store_Linear const& rhs) const
        {
            return m_tag == rhs.m_tag && m_dstTag == rhs.m_dstTag && m_srcTag == rhs.m_srcTag;
        }

        T_Store_Tiled::T_Store_Tiled(OperationTag source, OperationTag dest)
            : BaseOperation()
            , m_srcTag(source)
            , m_dstTag(dest)
        {
        }

        OperationTag T_Store_Tiled::getSrcTag() const
        {
            return m_srcTag;
        }

        OperationTag T_Store_Tiled::getDstTag() const
        {
            return m_dstTag;
        }

        std::string T_Store_Tiled::toString() const
        {
            std::ostringstream msg;

            msg << "T_STORE_TILED " << m_tag << " Source " << m_srcTag << " Dest " << m_dstTag;

            return msg.str();
        }

        std::ostream& operator<<(std::ostream& stream, T_Store_Tiled const& val)
        {
            return stream << val.toString();
        }

        bool T_Store_Tiled::operator==(T_Store_Tiled const& rhs) const
        {
            return m_tag == rhs.m_tag && m_dstTag == rhs.m_dstTag && m_srcTag == rhs.m_srcTag;
        }
    }
}
