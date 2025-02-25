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
        WaitCount() = default;
        explicit WaitCount(GPUArchitecture const& arch, std::string const& message = "");
        WaitCount(GPUArchitecture const& arch,
                  int                    loadcnt,
                  int                    storecnt,
                  int                    vscnt,
                  int                    dscnt,
                  int                    kmcnt,
                  int                    expcnt);
        WaitCount(GPUArchitecture const& arch, GPUWaitQueue, int count);

        ~WaitCount() = default;

        bool operator==(WaitCount a) const
        {
            return a.m_loadcnt == m_loadcnt && a.m_storecnt == m_storecnt && a.m_vscnt == m_vscnt
                   && a.m_dscnt == m_dscnt && a.m_kmcnt == m_kmcnt && a.m_expcnt == m_expcnt;
        }
        bool operator!=(WaitCount a) const
        {
            return a.m_loadcnt != m_loadcnt || a.m_storecnt != m_storecnt || a.m_vscnt != m_vscnt
                   || a.m_dscnt != m_dscnt || a.m_kmcnt != m_kmcnt || a.m_expcnt != m_expcnt;
        }

        static WaitCount
            LoadCnt(GPUArchitecture const& arch, int value, std::string const& message = "");
        static WaitCount
            StoreCnt(GPUArchitecture const& arch, int value, std::string const& message = "");
        static WaitCount
            VSCnt(GPUArchitecture const& arch, int value, std::string const& message = "");
        static WaitCount
            DSCnt(GPUArchitecture const& arch, int value, std::string const& message = "");
        static WaitCount
            KMCnt(GPUArchitecture const& arch, int value, std::string const& message = "");
        static WaitCount
            EXPCnt(GPUArchitecture const& arch, int value, std::string const& message = "");

        static WaitCount Zero(GPUArchitecture const& arch, std::string const& message = " ");

        std::string toString(LogLevel level) const;
        void        toStream(std::ostream& os, LogLevel level) const;

        static int CombineValues(int lhs, int rhs);

        void combine(WaitCount const& other);

        int loadcnt() const;
        int storecnt() const;
        int vscnt() const;
        int dscnt() const;
        int kmcnt() const;
        int expcnt() const;

        int getCount(GPUWaitQueue) const;

        void setLoadcnt(int value);
        void setStorecnt(int value);
        void setVscnt(int value);
        void setDScnt(int value);
        void setKMcnt(int value);
        void setExpcnt(int value);

        void combineLoadcnt(int value);
        void combineStorecnt(int value);
        void combineVscnt(int value);
        void combineDScnt(int value);
        void combineKMcnt(int value);
        void combineExpcnt(int value);

        std::vector<std::string> const& comments() const;

        void addComment(std::string const& comment);
        void addComment(std::string&& comment);

        WaitCount getAsSaturatedWaitCount(GPUArchitecture const& arch) const;

    private:
        /**
         * -1 means don't care.
         *
         * On machines without separate vscnt, the vscnt field should not be used.
         *
         */
        int m_loadcnt  = -1;
        int m_storecnt = -1;
        int m_vscnt    = -1;
        int m_dscnt    = -1;
        int m_kmcnt    = -1;
        int m_expcnt   = -1;

        std::vector<std::string> m_comments;

        GPUArchitectureTarget m_target;
    };

    std::ostream& operator<<(std::ostream& stream, WaitCount const& wait);
}
