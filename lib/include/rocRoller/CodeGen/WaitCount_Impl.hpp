/**
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include <rocRoller/CodeGen/WaitCount.hpp>

namespace rocRoller
{
    inline WaitCount::WaitCount() = default;

    inline WaitCount::WaitCount(std::string const& message)
        : m_comments({message})
    {
    }

    inline WaitCount::WaitCount(int vmcnt, int vscnt, int lgkmcnt, int expcnt)
        : m_vmcnt(vmcnt)
        , m_vscnt(vscnt)
        , m_lgkmcnt(lgkmcnt)
        , m_expcnt(expcnt)
    {
    }

    inline WaitCount::WaitCount(GPUWaitQueue queue, int count)
        : m_vmcnt(-1)
        , m_vscnt(-1)
        , m_lgkmcnt(-1)
        , m_expcnt(-1)
    {
        switch(queue)
        {
        case GPUWaitQueue::VMQueue:
            m_vmcnt = count;
            break;
        case GPUWaitQueue::LGKMQueue:
            m_lgkmcnt = count;
            break;
        case GPUWaitQueue::EXPQueue:
            m_expcnt = count;
            break;
        case GPUWaitQueue::VSQueue:
            m_vscnt = count;
            break;
        }
    }

    inline WaitCount::~WaitCount() = default;

    inline WaitCount WaitCount::VMCnt(int value)
    {
        WaitCount rv;
        rv.m_vmcnt = value;

        return rv;
    }

    inline WaitCount WaitCount::VSCnt(int value)
    {
        WaitCount rv;
        rv.m_vscnt = value;

        return rv;
    }

    inline WaitCount WaitCount::LGKMCnt(int value)
    {
        WaitCount rv;
        rv.m_lgkmcnt = value;

        return rv;
    }

    inline WaitCount WaitCount::EXPCnt(int value)
    {
        WaitCount rv;
        rv.m_expcnt = value;

        return rv;
    }

    inline WaitCount WaitCount::VMCnt(int value, std::string const& message)
    {
        WaitCount rv(message);
        rv.m_vmcnt = value;

        return rv;
    }

    inline WaitCount WaitCount::VSCnt(int value, std::string const& message)
    {
        WaitCount rv(message);
        rv.m_vscnt = value;

        return rv;
    }

    inline WaitCount WaitCount::LGKMCnt(int value, std::string const& message)
    {
        WaitCount rv(message);
        rv.m_lgkmcnt = value;

        return rv;
    }

    inline WaitCount WaitCount::EXPCnt(int value, std::string const& message)
    {
        WaitCount rv(message);
        rv.m_expcnt = value;

        return rv;
    }

    inline WaitCount WaitCount::Zero(GPUArchitecture const& architecture)
    {
        return Zero("", architecture);
    }

    inline WaitCount WaitCount::Zero(std::string const&     message,
                                     GPUArchitecture const& architecture)
    {
        WaitCount rv(message);

        rv.m_vmcnt = 0;
        if(architecture.HasCapability(GPUCapability::SeparateVscnt))
        {
            rv.m_vscnt = 0;
        }
        rv.m_lgkmcnt = 0;
        rv.m_expcnt  = 0;

        return rv;
    }

    inline int WaitCount::CombineValues(int lhs, int rhs)
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

    inline void WaitCount::combine(WaitCount const& other)
    {
        m_vmcnt   = CombineValues(m_vmcnt, other.m_vmcnt);
        m_vscnt   = CombineValues(m_vscnt, other.m_vscnt);
        m_lgkmcnt = CombineValues(m_lgkmcnt, other.m_lgkmcnt);
        m_expcnt  = CombineValues(m_expcnt, other.m_expcnt);

        m_comments.insert(m_comments.end(), other.m_comments.begin(), other.m_comments.end());
    }

    inline int WaitCount::vmcnt() const
    {
        return m_vmcnt;
    }
    inline int WaitCount::vscnt() const
    {
        return m_vscnt;
    }
    inline int WaitCount::lgkmcnt() const
    {
        return m_lgkmcnt;
    }
    inline int WaitCount::expcnt() const
    {
        return m_expcnt;
    }

    inline int WaitCount::getCount(GPUWaitQueue queue) const
    {
        switch(queue)
        {
        case GPUWaitQueue::VMQueue:
            return m_vmcnt;
        case GPUWaitQueue::LGKMQueue:
            return m_lgkmcnt;
        case GPUWaitQueue::EXPQueue:
            return m_expcnt;
        case GPUWaitQueue::VSQueue:
            return m_vscnt;
        default:
            return -1;
        }
    }

    inline void WaitCount::setVmcnt(int value)
    {
        m_vmcnt = value;
    }
    inline void WaitCount::setVscnt(int value)
    {
        m_vscnt = value;
    }
    inline void WaitCount::setLgkmcnt(int value)
    {
        m_lgkmcnt = value;
    }
    inline void WaitCount::setExpcnt(int value)
    {
        m_expcnt = value;
    }

    inline void WaitCount::combineVmcnt(int value)
    {
        m_vmcnt = CombineValues(m_vmcnt, value);
    }
    inline void WaitCount::combineVscnt(int value)
    {
        m_vscnt = CombineValues(m_vscnt, value);
    }
    inline void WaitCount::combineLgkmcnt(int value)
    {
        m_lgkmcnt = CombineValues(m_lgkmcnt, value);
    }
    inline void WaitCount::combineExpcnt(int value)
    {
        m_expcnt = CombineValues(m_expcnt, value);
    }

    inline std::vector<std::string> const& WaitCount::comments() const
    {
        return m_comments;
    }

    inline void WaitCount::addComment(std::string const& comment)
    {
        m_comments.emplace_back(comment);
    }

    inline void WaitCount::addComment(std::string&& comment)
    {
        m_comments.emplace_back(std::move(comment));
    }

    inline WaitCount WaitCount::getAsSaturatedWaitCount(GPUArchitecture const& architecture) const
    {
        int vmcnt   = m_vmcnt;
        int vscnt   = m_vscnt;
        int lgkmcnt = m_lgkmcnt;
        int expcnt  = m_expcnt;

        if(architecture.HasCapability(GPUCapability::MaxVmcnt))
        {
            vmcnt = std::min(vmcnt, architecture.GetCapability(GPUCapability::MaxVmcnt));
        }

        if(architecture.HasCapability(GPUCapability::MaxLgkmcnt))
        {
            lgkmcnt = std::min(lgkmcnt, architecture.GetCapability(GPUCapability::MaxLgkmcnt));
        }

        if(architecture.HasCapability(GPUCapability::MaxExpcnt))
        {
            expcnt = std::min(expcnt, architecture.GetCapability(GPUCapability::MaxExpcnt));
        }

        WaitCount wc = WaitCount(vmcnt, vscnt, lgkmcnt, expcnt);
        return wc;
    }
}
