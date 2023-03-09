#include <rocRoller/CodeGen/WaitCount.hpp>

#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    void WaitCount::toStream(std::ostream& os, LogLevel level) const
    {
        auto commentIter = level > LogLevel::Terse ? m_comments.begin() : m_comments.end();

        if(m_vmcnt >= 0 || m_lgkmcnt >= 0 || m_expcnt >= 0)
        {
            os << "s_waitcnt";

            if(m_vmcnt >= 0)
            {
                os << " vmcnt(" << m_vmcnt << ")";
            }

            if(m_lgkmcnt >= 0)
            {
                os << " lgkmcnt(" << m_lgkmcnt << ")";
            }

            if(m_expcnt >= 0)
            {
                os << " expcnt(" << m_expcnt << ")";
            }

            if(commentIter != m_comments.end())
            {
                os << " // " << *commentIter;
                commentIter++;
            }

            os << "\n";
        }

        if(m_vscnt >= 0)
        {
            os << "s_waitcnt_vscnt " << m_vscnt;

            if(commentIter != m_comments.end())
            {
                os << " // " << *commentIter;
                commentIter++;
            }

            os << "\n";
        }

        for(; commentIter != m_comments.end(); commentIter++)
        {
            os << "// " << *commentIter << "\n";
        }
    }

    std::string WaitCount::toString(LogLevel level) const
    {
        std::ostringstream oss;
        toStream(oss, level);
        return oss.str();
    }

    std::ostream& operator<<(std::ostream& stream, WaitCount const& wait)
    {
        auto logLevel = Settings::getInstance()->get(Settings::LogLvl);

        wait.toStream(stream, logLevel);
        return stream;
    }
}
