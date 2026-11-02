// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <vector>

/*******************************************************************************
 * On-disk local kernel library, named by HIPBLASLT_LOCAL_KERNEL_LIBRARY.
 *
 *   <root>/objects/<hash>.dat[.zlib]   payload shards, content addressed
 *   <root>/objects/<hash>.co           payload code objects, content addressed
 *   <root>/<scope>/journal-<build>     append-only, one record per line:
 *
 *     V <format> <build>
 *     R <datHash> <coHash> <dat|dat.zlib> <sum>
 *     E <datHash> <coHash> <payloadIndex> <m> <n> <batch> <k> <sum>
 *
 * <scope> names the architecture and compute unit count of the device the
 * records were made on, e.g. gfx942_CU304: a payload's hardware predicate
 * usually names both, and a mapping is tuned for that device. <build> is the
 * hipBLASLt build stamp that user-driven tuning files are checked against. A
 * saved code object takes the kernel arguments its build's launch code
 * passes, so each build keeps its own journal. The V line starts every
 * journal, and a journal whose V line is not this build's is not read.
 *
 * <sum> is a checksum of the rest of the line, so a line cut short by a crash
 * is rejected even where what remains would parse. Records of unknown kinds
 * are skipped, as are lines that do not parse.
 *
 * A payload is identified by both hashes, since one shard can be paired with
 * different code objects. E records name a solution by its index within the
 * payload, because assigned indices differ from process to process.
 *
 * Whoever can write the library chooses which code objects are loaded, so
 * every directory in it and every journal must be owned by the effective user
 * and not be group- or world-writable.
 *
 * Writers hold an exclusive flock on the journal; readers hold a shared one.
 * Objects are written to a temporary name and renamed, so a reader never sees
 * a partial object, and are durable before a record refers to them.
 ******************************************************************************/

namespace rocblaslt
{
    namespace local_kernel_store
    {
        struct PayloadId
        {
            std::string datHash;
            std::string coHash;

            bool operator<(PayloadId const& other) const
            {
                return std::tie(datHash, coHash) < std::tie(other.datHash, other.coHash);
            }
            bool operator==(PayloadId const& other) const
            {
                return datHash == other.datHash && coHash == other.coHash;
            }
        };

        struct PayloadRecord
        {
            PayloadId id;
            bool      compressed = false;
        };

        struct ExactRecord
        {
            PayloadId id;
            int       payloadIndex = -1;
            uint64_t  m = 0, n = 0, batch = 0, k = 0;
        };

        struct JournalEntry
        {
            enum class Kind
            {
                Payload,
                Exact
            };

            Kind          kind = Kind::Payload;
            PayloadRecord payload;
            ExactRecord   exact;
        };

        /**
         * The shard file the loader reads for `datPath`: a ".dat.zlib" next to
         * a bare ".dat" path is preferred, as LoadLibraryFile does.
         */
        std::string resolveShardFile(std::string const& datPath);

        /** Hashes a payload's shard file (from resolveShardFile) and code object. */
        bool identifyPayload(std::string const& shardFile,
                             std::string const& coPath,
                             PayloadRecord*     out,
                             std::string*       error);

        class Store
        {
        public:
            /**
             * The library named by HIPBLASLT_LOCAL_KERNEL_LIBRARY, resolved
             * once per process. Null when the variable is unset, when the
             * process is privileged, or when the directory is refused; the
             * last two print a warning.
             */
            static Store const* configured();

            std::string const& root() const
            {
                return m_root;
            }

            std::string shardPath(PayloadRecord const& record) const;
            std::string codeObjectPath(PayloadId const& id) const;

            /**
             * Copies a payload into objects/, hashing the bytes as they are
             * written, and sets `record->id` to those hashes, which name the
             * copies. A file that changes after it was identified is therefore
             * saved under the name of what was copied.
             */
            bool storeObjects(std::string const& shardFile,
                              std::string const& coPath,
                              PayloadRecord*     record,
                              std::string*       error) const;

            /**
             * Whether a payload's saved objects are private to this user, as
             * the directories must be, and still hash to their names.
             */
            bool verifyObjects(PayloadRecord const& record, std::string* error) const;

            /** Appends an R record unless the journal already has one for this payload. */
            bool appendPayload(std::string const&   scope,
                               PayloadRecord const& record,
                               std::string*         error) const;

            /**
             * Appends an E record unless the journal's last mapping at this
             * shape is this one, which applying again would not change.
             */
            bool appendExact(std::string const& scope,
                             ExactRecord const& record,
                             std::string*       error) const;

            /**
             * Reads the records that start at or after byte `offset` of the
             * journal for `scope`, and sets `end` to the offset after the last
             * complete line. A missing journal reads as empty.
             */
            bool read(std::string const&         scope,
                      uint64_t                   offset,
                      std::vector<JournalEntry>* entries,
                      uint64_t*                  end,
                      std::string*               error) const;

        private:
            /** What appends have read of one journal, so each reads only what is new. */
            struct JournalState
            {
                uint64_t            device = 0, inode = 0, offset = 0;
                std::set<PayloadId> payloads;
                // The last mapping at each [m, n, batch, k].
                std::map<std::array<uint64_t, 4>, ExactRecord> lastExact;
            };

            explicit Store(std::string root)
                : m_root(std::move(root))
            {
            }

            static Store* open(std::string const& root, std::string* error);

            bool        checkLayout(std::string* error) const;
            std::string journalPath(std::string const& scope, std::string* error) const;

            /**
             * Locks the journal for `scope` exclusively, creating it with its
             * V line if needed, reads what was appended since `state` was last
             * brought up to date, and appends the line `makeLine` returns for
             * that state, if it is not empty.
             */
            bool appendUnlessKnown(std::string const&                                     scope,
                                   std::function<std::string(JournalState const&)> const& makeLine,
                                   std::string* error) const;

            std::string m_root;

            mutable std::mutex                          m_mutex;
            mutable std::map<std::string, JournalState> m_journals;
        };
    } // namespace local_kernel_store
} // namespace rocblaslt
