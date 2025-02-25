/**
 * @file WaitCount.hpp
 * @brief Keeps track of required `waitcnt` instructions.
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include <string>
#include <vector>

#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Utilities/Settings_fwd.hpp>

namespace rocRoller
{

    /**
     * Represents whether we need to insert a `waitcnt` before a given instruction.
     *
     * Can represent waiting for any counter on any generation of device, including
     * architectures with separate `vscnt` counters.
     *
     * Internally represents -1 as not having to wait for a particular counter.
     */
    class WaitCount
    {
    public:
        WaitCount();
        explicit WaitCount(std::string const& message);
        WaitCount(int vmcnt, int vscnt, int lgkmcnt, int expcnt);
        WaitCount(GPUWaitQueue, int count);

        ~WaitCount();

        bool operator==(WaitCount a) const
        {
            return a.m_vmcnt == m_vmcnt && a.m_vscnt == m_vscnt && a.m_lgkmcnt == m_lgkmcnt
                   && a.m_expcnt == m_expcnt;
        }
        bool operator!=(WaitCount a) const
        {
            return a.m_vmcnt != m_vmcnt || a.m_vscnt != m_vscnt || a.m_lgkmcnt != m_lgkmcnt
                   || a.m_expcnt != m_expcnt;
        }

        static WaitCount VMCnt(int value);
        static WaitCount VSCnt(int value);
        static WaitCount LGKMCnt(int value);
        static WaitCount EXPCnt(int value);

        static WaitCount VMCnt(int value, std::string const& message);
        static WaitCount VSCnt(int value, std::string const& message);
        static WaitCount LGKMCnt(int value, std::string const& message);
        static WaitCount EXPCnt(int value, std::string const& message);

        static WaitCount Zero(GPUArchitecture const&);
        static WaitCount Zero(std::string const& message, GPUArchitecture const&);

        std::string toString(LogLevel level) const;
        void        toStream(std::ostream& os, LogLevel level) const;

        static inline int CombineValues(int lhs, int rhs);

        void combine(WaitCount const& other);

        int vmcnt() const;
        int vscnt() const;
        int lgkmcnt() const;
        int expcnt() const;

        int getCount(GPUWaitQueue) const;

        void setVmcnt(int value);
        void setVscnt(int value);
        void setLgkmcnt(int value);
        void setExpcnt(int value);

        void combineVmcnt(int value);
        void combineVscnt(int value);
        void combineLgkmcnt(int value);
        void combineExpcnt(int value);

        std::vector<std::string> const& comments() const;

        void addComment(std::string const& comment);
        void addComment(std::string&& comment);

        WaitCount getAsSaturatedWaitCount(GPUArchitecture const& architecture) const;

    private:
        /**
         * -1 means don't care.
         *
         * On machines without separate vscnt, the vscnt field should not be used.
         *
         */
        int m_vmcnt   = -1;
        int m_vscnt   = -1;
        int m_lgkmcnt = -1;
        int m_expcnt  = -1;

        std::vector<std::string> m_comments;
    };

    std::ostream& operator<<(std::ostream& stream, WaitCount const& wait);
}

#include <rocRoller/CodeGen/WaitCount_Impl.hpp>
