/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2025 AMD ROCm(TM) Software
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

        bool m_isSplitCounter = false;
        bool m_hasVSCnt       = false;
    };

    std::ostream& operator<<(std::ostream& stream, WaitCount const& wait);
}
