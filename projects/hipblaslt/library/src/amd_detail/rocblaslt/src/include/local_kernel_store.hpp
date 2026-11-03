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
 *     V <format> <build> <journalId>
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
 * <journalId> is random, so a journal that is deleted and created again is
 * told apart from the one a reader had read part of.
 *
 * <sum> is a checksum of the rest of the line, so a line cut short by a crash
 * is rejected even where what remains would parse. An append that fails is
 * truncated away, so the next one cannot complete it. Records of unknown
 * kinds are skipped, as are lines that do not parse.
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
 * Neither waits more than a few seconds for it, so a process stopped while
 * holding it makes the others fail rather than hang. Objects are written to a
 * temporary name and renamed, so a reader never sees a partial object, and
 * are durable before a record refers to them. Temporary files a crashed
 * writer left behind are removed after an hour.
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

        /** How far a reader has read one journal. */
        struct JournalCursor
        {
            // The journal's V line, which tells this file from one that replaced it.
            std::string journal;
            uint64_t    offset = 0;
        };

        /**
         * A payload's objects on their way into the library; see
         * Store::stageObjects. Temporary copies not yet published are removed
         * with it.
         */
        class StagedObjects
        {
        public:
            StagedObjects()                                = default;
            StagedObjects(StagedObjects const&)            = delete;
            StagedObjects& operator=(StagedObjects const&) = delete;
            ~StagedObjects();

            /** The file to read the shard from until it is published. */
            std::string const& shardFile() const
            {
                return m_shard.temp.empty() ? m_shard.final : m_shard.temp;
            }

        private:
            friend class Store;

            struct Object
            {
                std::string temp, final;
                // `final` existed but did not match its name.
                bool replace = false;
                // publishObjects created `final`.
                bool created = false;
            };

            Object m_shard, m_co;
        };

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
             * Prepares a payload identified by identifyPayload for saving. An
             * object already saved under a file's hash, and still matching
             * it, is used as it is. Otherwise the file is copied to a
             * temporary name in objects/, hashed as it is written, and
             * `record->id` takes that hash, so a file that changed after it
             * was identified is named by what was copied. Nothing is visible
             * under an object's name until publishObjects.
             */
            bool stageObjects(std::string const& shardFile,
                              std::string const& coPath,
                              PayloadRecord*     record,
                              StagedObjects*     staged,
                              std::string*       error) const;

            /** Moves staged copies to their names, which they then keep. */
            bool publishObjects(StagedObjects* staged, std::string* error) const;

            /**
             * Removes the objects publishObjects created, for a payload found
             * unusable after they were published. Objects that were already
             * there may belong to other payloads and are kept.
             */
            void unpublishObjects(StagedObjects* staged) const;

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
             * Reads the records of the journal for `scope` after `cursor`, and
             * moves the cursor past the last complete line. A journal that is
             * not the one the cursor was in, because it was replaced, or that
             * is now shorter, is read from the start. A missing journal reads
             * as empty. The cursor is unchanged on failure.
             */
            bool read(std::string const&         scope,
                      JournalCursor*             cursor,
                      std::vector<JournalEntry>* entries,
                      std::string*               error) const;

        private:
            /** What appends have read of one journal, so each reads only what is new. */
            struct JournalState
            {
                JournalCursor       cursor;
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
