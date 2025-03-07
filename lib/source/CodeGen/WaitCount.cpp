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

#include "rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp"
#include "rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp"
#include <rocRoller/CodeGen/WaitCount.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    WaitCount::WaitCount(GPUArchitecture const& arch, std::string const& message)
    {
        m_target = arch.target();
        if(message.length() > 0)
        {
            m_comments = {message};
        }
    }

    WaitCount::WaitCount(GPUArchitecture const& arch,
                         int                    loadcnt,
                         int                    storecnt,
                         int                    vscnt,
                         int                    dscnt,
                         int                    kmcnt,
                         int                    expcnt)
        : m_loadcnt(loadcnt)
        , m_storecnt(storecnt)
        , m_vscnt(vscnt)
        , m_dscnt(dscnt)
        , m_kmcnt(kmcnt)
        , m_expcnt(expcnt)
        , m_target(arch.target())
    {
    }

    WaitCount::WaitCount(GPUArchitecture const& arch, GPUWaitQueue queue, int count)
        : m_loadcnt(-1)
        , m_storecnt(-1)
        , m_vscnt(-1)
        , m_dscnt(-1)
        , m_kmcnt(-1)
        , m_expcnt(-1)
        , m_target(arch.target())
    {
        switch(queue)
        {
        case GPUWaitQueue::LoadQueue:
            m_loadcnt = count;
            break;
        case GPUWaitQueue::StoreQueue:
            m_storecnt = count;
            break;
        case GPUWaitQueue::DSQueue:
            m_dscnt = count;
            break;
        case GPUWaitQueue::KMQueue:
            m_kmcnt = count;
            break;
        case GPUWaitQueue::EXPQueue:
            m_expcnt = count;
            break;
        case GPUWaitQueue::VSQueue:
            m_vscnt = count;
            break;
        }
    }

    WaitCount WaitCount::LoadCnt(GPUArchitecture const& arch, int value, std::string const& message)
    {
        WaitCount rv{arch, message};
        rv.m_loadcnt = value;

        return rv;
    }

    WaitCount
        WaitCount::StoreCnt(GPUArchitecture const& arch, int value, std::string const& message)
    {
        WaitCount rv{arch, message};
        rv.m_storecnt = value;

        return rv;
    }

    WaitCount WaitCount::VSCnt(GPUArchitecture const& arch, int value, std::string const& message)
    {
        WaitCount rv{arch, message};
        rv.m_vscnt = value;

        return rv;
    }

    WaitCount WaitCount::DSCnt(GPUArchitecture const& arch, int value, std::string const& message)
    {
        WaitCount rv{arch, message};
        rv.m_dscnt = value;

        return rv;
    }

    WaitCount WaitCount::KMCnt(GPUArchitecture const& arch, int value, std::string const& message)
    {
        WaitCount rv{arch, message};
        rv.m_kmcnt = value;

        return rv;
    }

    WaitCount WaitCount::EXPCnt(GPUArchitecture const& arch, int value, std::string const& message)
    {
        WaitCount rv{arch, message};
        rv.m_expcnt = value;

        return rv;
    }

    WaitCount WaitCount::Zero(GPUArchitecture const& architecture, std::string const& message)
    {
        WaitCount rv{architecture, message};

        rv.m_loadcnt  = 0;
        rv.m_storecnt = 0;
        if(architecture.HasCapability(GPUCapability::SeparateVscnt))
        {
            rv.m_vscnt = 0;
        }
        rv.m_dscnt  = 0;
        rv.m_kmcnt  = 0;
        rv.m_expcnt = 0;

        return rv;
    }

    int WaitCount::CombineValues(int lhs, int rhs)
    {
        if(lhs < 0)
        {
            return rhs;
        }

        if(rhs < 0)
        {
            return lhs;
        }

        return std::min(lhs, rhs);
    }

    void WaitCount::combine(WaitCount const& other)
    {
        m_loadcnt  = CombineValues(m_loadcnt, other.m_loadcnt);
        m_storecnt = CombineValues(m_storecnt, other.m_storecnt);
        m_vscnt    = CombineValues(m_vscnt, other.m_vscnt);
        m_dscnt    = CombineValues(m_dscnt, other.m_dscnt);
        m_kmcnt    = CombineValues(m_kmcnt, other.m_kmcnt);
        m_expcnt   = CombineValues(m_expcnt, other.m_expcnt);

        m_target = other.m_target;

        m_comments.insert(m_comments.end(), other.m_comments.begin(), other.m_comments.end());
    }

    int WaitCount::loadcnt() const
    {
        return m_loadcnt;
    }
    int WaitCount::storecnt() const
    {
        return m_storecnt;
    }
    int WaitCount::vscnt() const
    {
        return m_vscnt;
    }
    int WaitCount::dscnt() const
    {
        return m_dscnt;
    }
    int WaitCount::kmcnt() const
    {
        return m_kmcnt;
    }
    int WaitCount::expcnt() const
    {
        return m_expcnt;
    }

    int WaitCount::getCount(GPUWaitQueue queue) const
    {
        switch(queue)
        {
        case GPUWaitQueue::LoadQueue:
            return m_loadcnt;
        case GPUWaitQueue::StoreQueue:
            return m_storecnt;
        case GPUWaitQueue::DSQueue:
            return m_dscnt;
        case GPUWaitQueue::KMQueue:
            return m_kmcnt;
        case GPUWaitQueue::EXPQueue:
            return m_expcnt;
        case GPUWaitQueue::VSQueue:
            return m_vscnt;
        default:
            return -1;
        }
    }

    void WaitCount::setLoadcnt(int value)
    {
        m_loadcnt = value;
    }
    void WaitCount::setStorecnt(int value)
    {
        m_storecnt = value;
    }
    void WaitCount::setVscnt(int value)
    {
        m_vscnt = value;
    }
    void WaitCount::setDScnt(int value)
    {
        m_dscnt = value;
    }
    void WaitCount::setKMcnt(int value)
    {
        m_kmcnt = value;
    }
    void WaitCount::setExpcnt(int value)
    {
        m_expcnt = value;
    }

    void WaitCount::combineLoadcnt(int value)
    {
        m_loadcnt = CombineValues(m_loadcnt, value);
    }
    void WaitCount::combineStorecnt(int value)
    {
        m_storecnt = CombineValues(m_storecnt, value);
    }
    void WaitCount::combineVscnt(int value)
    {
        m_vscnt = CombineValues(m_vscnt, value);
    }
    void WaitCount::combineDScnt(int value)
    {
        m_dscnt = CombineValues(m_dscnt, value);
    }
    void WaitCount::combineKMcnt(int value)
    {
        m_kmcnt = CombineValues(m_kmcnt, value);
    }
    void WaitCount::combineExpcnt(int value)
    {
        m_expcnt = CombineValues(m_expcnt, value);
    }

    std::vector<std::string> const& WaitCount::comments() const
    {
        return m_comments;
    }

    void WaitCount::addComment(std::string const& comment)
    {
        m_comments.emplace_back(comment);
    }

    void WaitCount::addComment(std::string&& comment)
    {
        m_comments.emplace_back(std::move(comment));
    }

    WaitCount WaitCount::getAsSaturatedWaitCount(GPUArchitecture const& arch) const
    {
        int loadcnt  = m_loadcnt;
        int storecnt = m_storecnt;
        int vscnt    = m_vscnt;
        int dscnt    = m_dscnt;
        int kmcnt    = m_kmcnt;
        int expcnt   = m_expcnt;

        if(arch.HasCapability(GPUCapability::MaxVmcnt))
        {
            loadcnt  = std::min(loadcnt, arch.GetCapability(GPUCapability::MaxVmcnt));
            storecnt = std::min(storecnt, arch.GetCapability(GPUCapability::MaxVmcnt));
        }

        if(arch.HasCapability(GPUCapability::MaxLgkmcnt))
        {
            kmcnt = std::min(kmcnt, arch.GetCapability(GPUCapability::MaxLgkmcnt));
            dscnt = std::min(dscnt, arch.GetCapability(GPUCapability::MaxLgkmcnt));
        }

        if(arch.HasCapability(GPUCapability::MaxExpcnt))
        {
            expcnt = std::min(expcnt, arch.GetCapability(GPUCapability::MaxExpcnt));
        }

        return WaitCount(arch, loadcnt, storecnt, vscnt, dscnt, kmcnt, expcnt);
    }

    void WaitCount::toStream(std::ostream& os, LogLevel level) const
    {
        auto commentIter = level > LogLevel::Terse ? m_comments.begin() : m_comments.end();

        if(m_loadcnt >= 0 || m_storecnt >= 0 || m_kmcnt >= 0 || m_dscnt >= 0 || m_expcnt >= 0)
        {
            if(!m_target.isRDNA4GPU())
            {
                os << "s_waitcnt";

                if(m_loadcnt >= 0 || m_storecnt >= 0)
                {
                    auto vmcnt = WaitCount::CombineValues(m_loadcnt, m_storecnt);
                    os << " vmcnt(" << vmcnt << ")";
                }

                if(m_kmcnt >= 0 || m_dscnt >= 0)
                {

                    auto lgkmcnt = WaitCount::CombineValues(m_kmcnt, m_dscnt);
                    os << " lgkmcnt(" << lgkmcnt << ")";
                }

                if(m_expcnt >= 0)
                {
                    os << " expcnt(" << m_expcnt << ")";
                }
            }
            else
            {
                if(m_loadcnt >= 0)
                {
                    os << "s_wait_loadcnt " << m_loadcnt << std::endl;
                }

                if(m_storecnt >= 0)
                {
                    os << "s_wait_storecnt " << m_storecnt << std::endl;
                }

                if(m_kmcnt >= 0)
                {
                    os << "s_wait_kmcnt " << m_kmcnt << std::endl;
                }

                if(m_dscnt >= 0)
                {
                    os << "s_wait_dscnt " << m_dscnt << std::endl;
                }

                if(m_expcnt >= 0)
                {
                    os << "s_wait_expcnt " << m_expcnt << std::endl;
                }
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
            AssertFatal(!m_target.isRDNA4GPU(), "VSCnt is not a valid counter on RDNA4");

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
