// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

/*******************************************************************************
 * On-disk user kernel library, named by HIPBLASLT_USER_KERNEL_LIBRARY.
 *
 *   <root>/objects/<hash>.dat[.zlib]   payload shards, content addressed
 *   <root>/objects/<hash>.co           payload code objects, content addressed
 *   <root>/<arch>/journal              append-only, one record per line:
 *
 *     R <datHash> <coHash> <dat|dat.zlib>
 *     E <datHash> <coHash> <payloadIndex> <m> <n> <batch> <k>
 *
 * A payload is identified by both hashes, since one shard can be paired with
 * different code objects. E records name a solution by its index within the
 * payload, because assigned indices differ from process to process. Records of
 * unknown kinds are skipped, as are lines that do not parse.
 *
 * Writers hold an exclusive flock on the journal; readers hold a shared one.
 * Objects are written to a temporary name and renamed, so a reader never sees
 * a partial object, and are durable before a record refers to them.
 ******************************************************************************/

namespace rocblaslt
{
    namespace user_kernel_store
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
             * The library named by HIPBLASLT_USER_KERNEL_LIBRARY, resolved
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

            /** Copies a payload into objects/ unless it is already there. */
            bool storeObjects(std::string const&   shardFile,
                              std::string const&   coPath,
                              PayloadRecord const& record,
                              std::string*         error) const;

            /** Appends an R record unless the journal already has one for this payload. */
            bool appendPayload(std::string const&   arch,
                               PayloadRecord const& record,
                               std::string*         error) const;

            bool appendExact(std::string const& arch,
                             ExactRecord const& record,
                             std::string*       error) const;

            /**
             * Reads the records that start at or after byte `offset` of the
             * journal for `arch`, and sets `end` to the offset after the last
             * complete line. A missing journal reads as empty.
             */
            bool read(std::string const&         arch,
                      uint64_t                   offset,
                      std::vector<JournalEntry>* entries,
                      uint64_t*                  end,
                      std::string*               error) const;

        private:
            explicit Store(std::string root)
                : m_root(std::move(root))
            {
            }

            static Store* open(std::string const& root, std::string* error);

            std::string journalPath(std::string const& arch, std::string* error) const;

            std::string m_root;
        };
    } // namespace user_kernel_store
} // namespace rocblaslt
