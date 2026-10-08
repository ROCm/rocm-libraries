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
#include <shared_mutex>

#include <map>
#include <optional>
#include <set>
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

    // copy contructor
    OverrideSingleton(const OverrideSingleton&) = delete;
    // assignment operator
    OverrideSingleton& operator=(const OverrideSingleton&) = delete;

    /**
     * Re-read HIPBLASLT_TUNING_OVERRIDE_FILE after the singleton exists.
     *
     * Tests only: they set and clear the variable within one process, and the
     * singleton otherwise reads it once, at its first use.
     */
    void reloadForTest()
    {
        file_path.clear();
        env_mode = false;
        load();
    }

private:
    OverrideSingleton()
    {
        load();
    }

    void load()
    {
        char* Env = getenv("HIPBLASLT_TUNING_OVERRIDE_FILE");
        if(Env)
        {
            file_path = Env;
            env_mode  = true;
        }
    }

    ~OverrideSingleton() {}
};

namespace TensileLite
{
    /**
     * What a tuning file row resolves to.
     *
     * A solution index is only a position in one build's kernel library, so on
     * its own it cannot tell whether it still names the kernel that was tuned.
     * The name recorded beside it is what lets replay check that: kernel_name
     * in files hipblaslt-bench writes, solution_name in some older files, and
     * neither in the oldest, which are trusted only when they were written by
     * the running build.
     */
    struct TunedEntry
    {
        int32_t                    solutionIndex = -1;
        std::optional<std::string> kernelName;
        std::optional<std::string> solutionName;

        bool sameIdentity(const TunedEntry& other) const
        {
            return solutionIndex == other.solutionIndex && kernelName == other.kernelName
                   && solutionName == other.solutionName;
        }
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

        /**
         * Copy out every entry recorded for a key, in file order. Copies, so
         * no caller walks the multimap outside the lock.
         */
        std::vector<TunedEntry> find(const ProblemOverride& prob_key)
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            std::vector<TunedEntry> found;
            auto                    range = m_override.equal_range(prob_key);
            for(auto it = range.first; it != range.second; ++it)
                found.push_back(it->second);
            return found;
        }

        /**
         * Insert unless this key already records the same entry, so a file that
         * repeats a row does not stack duplicates. The identity is the index and
         * the recorded names together: two rows can share an index while naming
         * different kernels, and only one of them can still be valid.
         */
        void addIfAbsent(const ProblemOverride& key, const TunedEntry& entry)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

            auto range = m_override.equal_range(key);
            for(auto it = range.first; it != range.second; ++it)
                if(it->second.sameIdentity(entry))
                    return;
            m_override.emplace(key, entry);
        }

        /**
         * Whether a path has already been read. Tracked per path rather than
         * inferred from the map, so a file that yields no usable rows is still
         * read only once.
         */
        bool isLoaded(const std::string& path)
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);
            return m_loaded.count(path) != 0;
        }

        void markLoaded(const std::string& path)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
            m_loaded.insert(path);
        }

        void resetForTest()
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
            m_override.clear();
            m_loaded.clear();
        }

        std::mutex& getLock()
        {
            return m_guard;
        }

    private:
        std::multimap<ProblemOverride, TunedEntry> m_override;
        std::set<std::string>                      m_loaded;
        std::mutex                                 m_guard;
        std::shared_timed_mutex                    m_mutex;
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
