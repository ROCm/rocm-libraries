// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Utilities/EnumBitset.hpp>
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
                  int                    expcnt,
                  int                    tensorcnt);

        /// Issues a waitcnt with the given count for the given queue.
        WaitCount(GPUArchitecture const& arch, GPUWaitQueue queueForCount, int count);

        /// Instructs the WaitcntObserver to sync the given queues.
        WaitCount(GPUArchitecture const&       arch,
                  EnumBitset<GPUWaitQueueType> queuesToSync,
                  std::string const&           message = "");

        ~WaitCount() = default;

        bool operator==(WaitCount const& a) const
        {
            return a.m_loadcnt == m_loadcnt && a.m_storecnt == m_storecnt && a.m_vscnt == m_vscnt
                   && a.m_dscnt == m_dscnt && a.m_kmcnt == m_kmcnt && a.m_expcnt == m_expcnt
                   && a.m_tensorcnt == m_tensorcnt;
        }
        bool operator!=(WaitCount const& a) const
        {
            return !(*this == a);
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
        static WaitCount
            TensorCnt(GPUArchitecture const& arch, int value, std::string const& message = "");

        static WaitCount Zero(GPUArchitecture const& arch, std::string const& message = " ");

        static WaitCount Max(GPUArchitecture const& arch, std::string const& message = " ");

        /**
         * This means to empty the specified queue, i.e. include a waitcount of 0 if that queue
         * is not empty.
         */
        static WaitCount SyncQueue(GPUArchitecture const& arch,
                                   GPUWaitQueueType       queue,
                                   std::string const&     message = "");
        /**
         * This means to empty the specified queues, i.e. include a waitcount of 0 if any of the
         * specified queues are not empty.
         */
        static WaitCount SyncQueues(GPUArchitecture const&       arch,
                                    EnumBitset<GPUWaitQueueType> queues,
                                    std::string const&           message = "");

        std::string toString(LogLevel level) const;
        void        toStream(std::ostream& os, LogLevel level) const;

        static int CombineValues(int lhs, int rhs);

        WaitCount& combine(WaitCount const& other);

        int loadcnt() const;
        int storecnt() const;
        int vscnt() const;
        int dscnt() const;
        int kmcnt() const;
        int expcnt() const;
        int tensorcnt() const;

        /**
         * vmcnt is the combination of loadcnt and storecnt for non-split counters
         */
        int vmcnt() const;

        int getCount(GPUWaitQueue) const;

        void setLoadcnt(int value);
        void setStorecnt(int value);
        void setVscnt(int value);
        void setDScnt(int value);
        void setKMcnt(int value);
        void setExpcnt(int value);
        void setTensorcnt(int value);

        WaitCount& combineLoadcnt(int value);
        WaitCount& combineStorecnt(int value);
        WaitCount& combineVscnt(int value);
        WaitCount& combineDScnt(int value);
        WaitCount& combineKMcnt(int value);
        WaitCount& combineExpcnt(int value);
        WaitCount& combineTensorcnt(int value);

        std::vector<std::string> const& comments() const;

        void addComment(std::string const& comment);
        void addComment(std::string&& comment);

        WaitCount getAsSaturatedWaitCount(GPUArchitecture const& arch) const;

        EnumBitset<GPUWaitQueueType> const& queuesToSync() const;

    private:
        /**
         * -1 means don't care.
         *
         * On machines without separate vscnt, the vscnt field should not be used.
         *
         */
        int m_loadcnt   = -1;
        int m_storecnt  = -1;
        int m_vscnt     = -1;
        int m_dscnt     = -1;
        int m_kmcnt     = -1;
        int m_expcnt    = -1;
        int m_tensorcnt = -1;

        std::vector<std::string> m_comments;

        bool m_isSplitCounter = false;
        bool m_hasVSCnt       = false;
        bool m_hasEXPCnt      = false;
        bool m_hasTensorCnt   = false;

        EnumBitset<GPUWaitQueueType> m_queuesToSync;
    };

    std::ostream& operator<<(std::ostream& stream, WaitCount const& wait);
}
