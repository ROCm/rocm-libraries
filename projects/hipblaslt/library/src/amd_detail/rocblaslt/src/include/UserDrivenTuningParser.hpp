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
    // afterward. Entries that carry a stable solution name are
    // validated/healed directly against the live solution library and never
    // consult this; it exists only so legacy entries (parsed before the
    // solution_name column existed, so they cannot be validated that way)
    // keep the old fail-safe behavior of falling back to default selection
    // instead of trusting a possibly-stale index - see
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

    // A tuning-cache entry: the solution index found during offline tuning,
    // plus (when available) the stable Tensile solution name it referred to
    // at tuning time. `index` is only a fast-path hint - it can go stale
    // after a rebuild reorders the solution library. `name` is what lets a
    // stale index be healed instead of silently used or the whole override
    // file being discarded. `name` is empty for legacy (pre solution_name
    // column) override files, which keeps them working exactly as before.
    struct TunedSolution
    {
        int         index = -1;
        std::string name;
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
    public:
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

        // Returns a snapshot (copy) of every TunedSolution stored for prob_key.
        // Deliberately NOT a pair of live iterators: callers used to hold onto
        // iterators returned from equal_range() after this function's lock was
        // released, then read `iterator->second` unsynchronized - which raced
        // with update()/erase() running concurrently on another thread (e.g.
        // one thread healing an entry while another thread reads it mid-
        // assignment). A snapshot has no such lifetime/synchronization
        // dependency on the map: once returned, it is exclusively owned by the
        // caller.
        std::vector<TunedSolution> find(const ProblemOverride& prob_key)
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);
            auto                                      range = m_override.equal_range(prob_key);
            std::vector<TunedSolution>                result;
            result.reserve(std::distance(range.first, range.second));
            for(auto it = range.first; it != range.second; ++it)
                result.push_back(it->second);
            return result;
        }

        void add(const std::pair<ProblemOverride, TunedSolution>& problemSolution)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
            m_override.insert(problemSolution);
        }

        // Heals a cache entry in place after its solution has been relocated
        // to a new index by a rebuild, so subsequent find() calls in this
        // process return the healed index directly. Re-locates the specific
        // entry to update under its own lock (matched by prob_key and the
        // pre-heal index recorded in `original`) rather than trusting a
        // previously-returned iterator/reference, so this can never race with
        // a concurrent find() the way updating via a stale iterator would.
        void update(const ProblemOverride& prob_key,
                    const TunedSolution&   original,
                    const TunedSolution&   healed)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
            auto                                      range = m_override.equal_range(prob_key);
            for(auto it = range.first; it != range.second; ++it)
            {
                if(it->second.index == original.index && it->second.name == original.name)
                {
                    it->second = healed;
                    break;
                }
            }
        }

        std::mutex& getLock()
        {
            return m_guard;
        }

    private:
        std::multimap<ProblemOverride, TunedSolution> m_override;
        std::mutex                                     m_guard;
        std::shared_timed_mutex                        m_mutex;
    };
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
