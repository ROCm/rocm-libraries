/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    /**
 * \ingroup SolutionLibrary
 *
 * Holds the unparsed solution bytes of an indexed (`format_version: 2`)
 * library file plus a table describing where each solution lives, and
 * deserializes individual solutions on demand.
 *
 * A typical GEMM touches a handful of the thousands of solutions in a file,
 * so parsing them all at load time is almost entirely wasted work. Leaf nodes
 * of the library tree hold a shared_ptr to this cache and an index, and only
 * materialize when a query actually selects them.
 *
 * The blob is owned rather than referenced. Pointing into the msgpack zone
 * would be zero-copy but ties the cache's validity to loader internals that
 * differ across supported msgpack versions, and would pin the whole parsed
 * object graph rather than just these bytes.
 *
 * Deserialization is injected as a callable rather than called directly. This
 * header is compiled into both the msgpack and LLVM-YAML backends, which are
 * mutually exclusive at link time (`src/CMakeLists.txt`), so referring to
 * msgpack here would break the YAML build even though it never reads an
 * indexed file. It also lets tests drive the cache with a stub.
 */
    template <typename MySolution>
    class SolutionBlobCache
    {
    public:
        /// Byte offset and length of one solution's msgpack payload.
        using Slice = std::pair<size_t, size_t>;

        /// Parses one solution from `size` bytes at `data`. Returns nullptr on
        /// failure; the cache remembers the failure rather than retrying.
        using Deserializer
            = std::function<std::shared_ptr<MySolution>(const uint8_t* data, size_t size)>;

        SolutionBlobCache(std::vector<uint8_t>                blob,
                          std::unordered_map<int, Slice>      slices,
                          Deserializer                        deserialize)
            : m_blob(std::move(blob))
            , m_slices(std::move(slices))
            , m_deserialize(std::move(deserialize))
        {
        }

        /// True when `index` has a slice in this cache. Used at load time so a
        /// tree reference to a missing solution still fails the whole library,
        /// matching the legacy eager path, and at query time to pick which
        /// shard's cache owns an index.
        bool contains(int index) const
        {
            return m_slices.find(index) != m_slices.end();
        }

        size_t size() const
        {
            return m_slices.size();
        }

        /// Number of solutions parsed so far. Exposed for tests asserting that
        /// a query materialized only what it needed.
        size_t materializedCount() const
        {
            std::shared_lock<std::shared_mutex> lock(m_guard);
            return m_materialized.size();
        }

        /// Code object file name stamped onto every solution as it is
        /// materialized. Shard loading sets this, because the solutions it used
        /// to stamp in bulk no longer exist at merge time.
        void setCodeObjectFilename(std::string filename)
        {
            std::unique_lock<std::shared_mutex> lock(m_guard);
            m_codeObjectFilename = std::move(filename);
            // Anything already materialized predates the stamp.
            for(auto const& entry : m_materialized)
                if(entry.second)
                    entry.second->codeObjectFilename = m_codeObjectFilename;
        }

        /// Returns the solution for `index`, parsing it on first use.
        /// Returns nullptr for an unknown index or a payload that fails to
        /// parse. Safe to call concurrently for the same or different indices.
        ///
        /// The guarantee is **one retained object per index, not one parse**.
        /// Threads racing on the same index can each deserialize it; the first
        /// result to be published wins and every caller receives that one
        /// object, while the losers' copies are dropped. That is deliberate:
        /// parsing under the writer lock would serialize threads working on
        /// unrelated indices, which is the common case, to save duplicate work
        /// in the rare one. Callers must not rely on the deserializer running
        /// exactly once per index.
        std::shared_ptr<MySolution> get(int index) const
        {
            {
                std::shared_lock<std::shared_mutex> lock(m_guard);
                auto                                iter = m_materialized.find(index);
                if(iter != m_materialized.end())
                    return iter->second;
            }

            auto slice = m_slices.find(index);
            if(slice == m_slices.end())
                return nullptr;

            // Parse outside the lock: this is the expensive part, and holding
            // the writer lock across it would serialize unrelated indices.
            std::shared_ptr<MySolution> parsed;
            if(m_deserialize)
                parsed = m_deserialize(m_blob.data() + slice->second.first,
                                       slice->second.second);

            // A payload whose own index disagrees with the table means the file
            // is inconsistent; treat it as a parse failure rather than handing
            // back a solution under the wrong key.
            if(parsed && parsed->index != index)
                parsed.reset();

            std::unique_lock<std::shared_mutex> lock(m_guard);
            // Another thread may have finished the same index first; keep
            // whichever landed so callers never see two objects for one index.
            auto inserted = m_materialized.emplace(index, parsed);
            if(inserted.second && parsed && !m_codeObjectFilename.empty())
                parsed->codeObjectFilename = m_codeObjectFilename;
            return inserted.first->second;
        }

        /// Materializes every solution. Enumeration paths need the whole set,
        /// and doing it in one pass avoids repeated lock round-trips.
        void materializeAll() const
        {
            for(auto const& entry : m_slices)
                get(entry.first);
        }

        /// Every index this cache can supply.
        std::vector<int> indices() const
        {
            std::vector<int> rv;
            rv.reserve(m_slices.size());
            for(auto const& entry : m_slices)
                rv.push_back(entry.first);
            return rv;
        }

    private:
        std::vector<uint8_t>           m_blob;
        std::unordered_map<int, Slice> m_slices;
        Deserializer                   m_deserialize;

        mutable std::shared_mutex                            m_guard;
        mutable std::map<int, std::shared_ptr<MySolution>>   m_materialized;
        std::string                                          m_codeObjectFilename;
    };

} // namespace TensileLite
