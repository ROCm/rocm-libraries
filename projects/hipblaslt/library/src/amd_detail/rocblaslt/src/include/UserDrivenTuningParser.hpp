/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#pragma once

#include "UserDrivenTuningTypes.hpp"
#include "auxiliary.hpp"
#include "tensile_host.hpp"
#include <Tensile/DataTypes.hpp>
#include <mutex>
#include <shared_mutex>

#include <map>
#include <string>
#include <vector>

class OverrideSingleton
{
public:
    std::string file_path;
    bool        env_mode = false;

    static OverrideSingleton& getInstance()
    {
        static OverrideSingleton gInstance;
        return gInstance;
    }

    // Whether file_path's recorded build version matches the running build.
    // Computed lazily and cached on first call via std::call_once (see
    // m_versionCheckOnce) - hipblasLtMatmulAlgoGetHeuristic() can be called
    // concurrently from multiple threads for the same or different problems,
    // and a hand-rolled "if(!checked) { compute; checked = true; }" guard
    // would let two threads race on m_versionCurrent/m_versionChecked with no
    // synchronization between the write and other threads' reads - undefined
    // behavior even though every thread would compute the same value.
    // call_once guarantees exactly one thread runs the computation and that
    // its result is visible to every caller that returns from call_once
    // afterward. Entries that carry a solution name are validated directly
    // against the live solution library and never consult this; it exists
    // only so legacy entries (parsed before the solution_name column existed,
    // so they cannot be validated that way) keep the old fail-safe behavior
    // of falling back to default selection instead of trusting a
    // possibly-stale index - see
    // problem_override_from_file[_cpp]() in rocblaslt_auxiliary.cpp.
    bool isBuildVersionCurrent();

    // copy contructor
    OverrideSingleton(const OverrideSingleton&) = delete;
    // assignment operator
    OverrideSingleton& operator=(const OverrideSingleton&) = delete;

private:
    OverrideSingleton()
    {
        char* Env = getenv("HIPBLASLT_TUNING_OVERRIDE_FILE");
        if(Env)
        {
            file_path = Env;
            env_mode  = true;
        }
    }

    ~OverrideSingleton() {}

    std::once_flag m_versionCheckOnce;
    bool           m_versionCurrent = true;
};

namespace TensileLite
{

    enum class HeaderFields
    {
        transA = 0,
        transB,
        batch_count,
        m,
        n,
        k,
        a_type,
        b_type,
        c_type,
        compute_type,
        solution_index,
        solution_name,
        count
    };

    class ProblemOverride
    {
    public:
        ProblemOverride();
        ProblemOverride(bool             transA,
                        bool             transB,
                        rocisa::DataType inputTypeA,
                        rocisa::DataType inputTypeB,
                        rocisa::DataType computeType,
                        rocisa::DataType outputType,
                        size_t           m,
                        size_t           n,
                        size_t           k,
                        size_t           batchSize);
        ProblemOverride(const ProblemOverride& problem);

        inline bool transA() const
        {
            return m_transA;
        }
        inline bool transB() const
        {
            return m_transB;
        }
        inline rocisa::DataType inputTypeA() const
        {
            return m_inputTypeA;
        }
        inline rocisa::DataType inputTypeB() const
        {
            return m_inputTypeB;
        }
        inline rocisa::DataType computeType() const
        {
            return m_computeType;
        }
        inline rocisa::DataType outputType() const
        {
            return m_outputType;
        }
        inline size_t m() const
        {
            return m_m;
        }
        inline size_t n() const
        {
            return m_n;
        }
        inline size_t k() const
        {
            return m_k;
        }
        inline size_t batchSize() const
        {
            return m_batchSize;
        }

    private:
        bool             m_transA;
        bool             m_transB;
        rocisa::DataType m_inputTypeA;
        rocisa::DataType m_inputTypeB;
        rocisa::DataType m_computeType;
        rocisa::DataType m_outputType;
        size_t           m_m;
        size_t           m_n;
        size_t           m_k;
        size_t           m_batchSize;
    };

    std::pair<ProblemOverride, TunedSolution>
        problemFromEntries(const std::vector<std::string>& entries);

    void getContractionProblemsFromFile(const std::string& path);

    template <>
    struct Comparison<ProblemOverride>
    {
        enum
        {
            implemented = true
        };

        static int compare(ProblemOverride const& lhs, ProblemOverride const& rhs)
        {
            return LexicographicCompare(lhs.transA(),
                                        rhs.transA(),
                                        lhs.transB(),
                                        rhs.transB(),
                                        lhs.inputTypeA(),
                                        rhs.inputTypeA(),
                                        lhs.inputTypeB(),
                                        rhs.inputTypeB(),
                                        lhs.computeType(),
                                        rhs.computeType(),
                                        lhs.outputType(),
                                        rhs.outputType(),
                                        lhs.m(),
                                        rhs.m(),
                                        lhs.n(),
                                        rhs.n(),
                                        lhs.k(),
                                        rhs.k(),
                                        lhs.batchSize(),
                                        rhs.batchSize());
        }
    };

    class OverrideMap
    {
        using Map = std::multimap<ProblemOverride, TunedSolution>;

    public:
        // A zero-copy view over one problem's entries. The shared lock stays
        // alive for the view's lifetime, so iterators never escape their
        // synchronization scope and concurrent readers remain lock-shared.
        class CandidateRange
        {
        public:
            using const_iterator = Map::const_iterator;

            CandidateRange(std::shared_timed_mutex& mutex,
                           const Map&               overrideMap,
                           const ProblemOverride&   probKey)
                : m_lock(mutex)
            {
                auto range = overrideMap.equal_range(probKey);
                m_begin    = range.first;
                m_end      = range.second;
            }

            CandidateRange(const CandidateRange&)            = delete;
            CandidateRange& operator=(const CandidateRange&) = delete;
            CandidateRange(CandidateRange&&)                 = default;
            CandidateRange& operator=(CandidateRange&&)      = default;

            const_iterator begin() const
            {
                return m_begin;
            }

            const_iterator end() const
            {
                return m_end;
            }

            bool empty() const
            {
                return m_begin == m_end;
            }

        private:
            std::shared_lock<std::shared_timed_mutex> m_lock;
            const_iterator                            m_begin;
            const_iterator                            m_end;
        };

        static OverrideMap& getMap()
        {
            static OverrideMap gInstance;
            return gInstance;
        }

        OverrideMap() {}
        ~OverrideMap() {}
        // copy contructor
        OverrideMap(const OverrideMap&) = delete;
        // assignment operator
        OverrideMap& operator=(const OverrideMap&) = delete;

        int size()
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);
            auto                                      size = m_override.size();
            return size;
        }

        CandidateRange find(const ProblemOverride& prob_key)
        {
            return CandidateRange(m_mutex, m_override, prob_key);
        }

        // Add one entry unless this problem already carries the same index.
        // Duplicate filtering and insertion happen under one exclusive lock.
        bool add(const std::pair<ProblemOverride, TunedSolution>& problemSolution)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
            auto range = m_override.equal_range(problemSolution.first);
            for(auto it = range.first; it != range.second; ++it)
            {
                if(it->second.index == problemSolution.second.index)
                    return false;
            }
            m_override.insert(problemSolution);
            return true;
        }

        std::mutex& getLock()
        {
            return m_guard;
        }

    private:
        Map                     m_override;
        std::mutex              m_guard;
        std::shared_timed_mutex m_mutex;
    };

    // Parse path into the supplied map. This overload keeps parser tests and
    // other isolated consumers independent of the process-global map.
    void loadContractionProblemsFromFile(const std::string& path, OverrideMap& overrideMap);
} // namespace Tensile

namespace std
{
    template <>
    struct hash<TensileLite::ProblemOverride>
    {
        inline size_t operator()(TensileLite::ProblemOverride const& po) const
        {
            return TensileLite::hash_combine(po.transA(),
                                             po.transB(),
                                             po.inputTypeA(),
                                             po.inputTypeB(),
                                             po.computeType(),
                                             po.outputType(),
                                             po.m(),
                                             po.n(),
                                             po.k(),
                                             po.batchSize());
        }
    };
} // namespace std
