// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

/*******************************************************************************
 * Persistent store for the user kernel library.
 *
 *   <root>/objects/<hash>.dat     payload shards, content addressed
 *   <root>/objects/<hash>.co      payload code objects, content addressed
 *   <root>/registry.log           append-only journal, one record per line:
 *
 *     R <datHash> <coHash>
 *     E <datHash> <shardIndex> <m> <n> <batch> <k>
 *
 * E records use the index within the payload, because assigned indices change
 * from run to run.
 *
 * Single writer only: concurrent registration from several processes into one
 * root can interleave journal records.
 ******************************************************************************/

namespace rocblaslt
{
    namespace user_kernel
    {
        struct ExactRecord
        {
            std::string datHash;
            int         shardIndex = -1;
            uint64_t    m = 0, n = 0, batch = 0, k = 0;
        };

        struct PayloadRecord
        {
            std::string datHash;
            std::string coHash;
        };

        struct Journal
        {
            std::vector<PayloadRecord> payloads;
            std::vector<ExactRecord>   exacts;
        };

        /**
         * Opens the store at `root`, creating it if absent. Refuses a root or
         * objects directory that is group- or world-writable, since whoever can
         * write there controls which code objects are loaded.
         */
        bool open(std::string const& root, std::string* error);

        /** Whether open() has succeeded. */
        bool isOpen();

        std::string root();

        /**
         * Copies a payload into the store and appends its journal record.
         * Returns the content hashes and the paths of the stored copies.
         */
        bool storePayload(std::string const& datPath,
                          std::string const& coPath,
                          std::string*       datHash,
                          std::string*       coHash,
                          std::string*       storedDat,
                          std::string*       storedCo,
                          std::string*       error);

        /** Appends one exact-match record. */
        bool recordExactMatch(ExactRecord const& record, std::string* error);

        /** Reads the journal. A missing journal reads as empty. */
        bool readJournal(Journal* journal, std::string* error);

        /** Path of a stored object, whether or not it exists. */
        std::string objectPath(std::string const& hash, std::string const& extension);
    } // namespace user_kernel
} // namespace rocblaslt
