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
 * <scope> names the kind of device the records were made on: its kernel
 * target, compute unit count and PCI chip ID, e.g. gfx942_CU304 or
 * gfx950_CU256_ID75a3. A payload's hardware predicate usually names these,
 * and a mapping is tuned for that kind of device. <build> is the
 * hipBLASLt version followed by the build stamp that user-driven tuning files
 * are checked against, e.g. 1.5.0-3c1c5d13b19; a build without git history
 * has only the version. A saved code object takes the kernel arguments its build's launch code
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
            // Where the record's line starts in the journal.
            uint64_t offset = 0;
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

            /**
             * Moves staged copies to their names, which they then keep, even
             * if the payload turns out unusable: another payload's
             * registration may already be using an object, and an object no
             * record names costs only space.
             */
            bool publishObjects(StagedObjects* staged, std::string* error) const;

            /**
             * Whether a payload's saved objects are private to this user, as
             * the directories must be, and still hash to their names.
             */
            bool verifyObjects(PayloadRecord const& record, std::string* error) const;

            /**
             * A cheap description of a payload's saved objects as they are
             * now, which changes whenever anything verifyObjects checks may
             * have: when either is written, replaced or removed, or when the
             * owner or mode of one of them or of the library's directories
             * changes.
             */
            std::string objectState(PayloadRecord const& record) const;

            /** Appends an R record unless the journal already has one for this payload. */
            bool appendPayload(std::string const&   scope,
                               PayloadRecord const& record,
                               std::string*         error) const;

            /**
             * Appends an E record unless the journal's last mapping at this
             * shape is this one, which applying again would not change. The
             * R record of `payload`, which `record` names, is appended first,
             * in the same write, if the journal does not have it.
             *
             * Sets `from` to where a read finds this mapping, appended or
             * not, in the journal it is in: a cursor at or before its line,
             * and after every record before it.
             */
            bool appendExact(std::string const&   scope,
                             PayloadRecord const& payload,
                             ExactRecord const&   record,
                             JournalCursor*       from,
                             std::string*         error) const;

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
                // The last mapping at each [m, n, batch, k], with where it is.
                std::map<std::array<uint64_t, 4>, JournalEntry> lastExact;
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
             * that state, if it is not empty. A non-null `from` is set, before
             * `makeLine` is called, to the journal and the offset its line
             * would be read from.
             */
            bool appendUnlessKnown(std::string const&                                     scope,
                                   std::function<std::string(JournalState const&)> const& makeLine,
                                   JournalCursor*                                         from,
                                   std::string* error) const;

            std::string m_root;

            // An append can wait for another process's lock on one scope's
            // journal, so each scope has its own mutex; m_mutex guards only
            // the map.
            struct ScopeJournal
            {
                std::mutex   mutex;
                JournalState state;
            };
            mutable std::mutex                          m_mutex;
            mutable std::map<std::string, ScopeJournal> m_journals;
        };
    } // namespace local_kernel_store
} // namespace rocblaslt
