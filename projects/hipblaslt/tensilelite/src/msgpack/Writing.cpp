// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/msgpack/Writing.hpp>

#include <Tensile/CachingLibrary.hpp>
#include <Tensile/SolutionBlobCache.hpp>
#include <Tensile/msgpack/MessagePack.hpp>

#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace TensileLite
{
    namespace
    {
        std::vector<uint8_t> finish(Serialization::MessagePackOutput const& out)
        {
            if(!out.error.empty())
            {
                std::ostringstream msg;
                msg << "Error writing msgpack data:" << std::endl;
                for(auto const& err : out.error)
                    msg << err << std::endl;

                throw std::runtime_error(msg.str());
            }

            auto data = reinterpret_cast<const uint8_t*>(out.buffer.data());
            return std::vector<uint8_t>(data, data + out.buffer.size());
        }

        MasterContractionLibrary& writableMaster(
            SolutionLibrary<ContractionProblemGemm, ContractionSolution> const& library)
        {
            auto master = dynamic_cast<MasterContractionLibrary const*>(&library);
            if(master == nullptr)
                throw std::runtime_error(concatenate(
                    "Only a MasterSolutionLibrary can be written as a library, not ",
                    library.type()));

            // Traits take a mutable reference. This writer does not modify the
            // library. The caller must not use it from another thread; see
            // MessagePackWriteLibraryData.
            return const_cast<MasterContractionLibrary&>(*master);
        }

        // format_version written by MessagePackWriteIndexedLibraryData. Taken
        // from the reader's own constant so a bump cannot be half-applied.
        constexpr int IndexedFormatVersion
            = Serialization::MappingTraits<MasterContractionLibrary,
                                           Serialization::MessagePackOutput>::
                  IndexedFormatVersion;

        /// What to write for one solution index: either an object to encode,
        /// or the bytes a blob cache already holds for it.
        struct IndexedPayload
        {
            /// Null when this index is carried by the slice below.
            std::shared_ptr<ContractionSolution> solution;
            const uint8_t*                       data = nullptr;
            size_t                               size = 0;
        };

        /// Every index the library can produce, in index order, paired with the
        /// bytes to write. Objects in `lib.solutions` win over the blob, which
        /// matches the eager path.
        ///
        /// A solution that is still only bytes stays bytes: the blob already
        /// holds exactly what encoding it would produce, so parsing it would
        /// fill the cache with every solution in the file -- the state the lazy
        /// format exists to avoid -- and re-encoding could only drift from the
        /// original. `cache` is held for the call so the slices stay alive.
        std::map<int, IndexedPayload> payloadsForIndex(
            MasterContractionLibrary&                               lib,
            std::shared_ptr<SolutionBlobCache<ContractionSolution>> cache)
        {
            std::map<int, IndexedPayload> payloads;
            {
                std::lock_guard<std::mutex> lock(lib.solutionsGuard);
                for(auto const& entry : lib.solutions)
                    payloads[entry.first] = IndexedPayload{entry.second, nullptr, 0};
            }

            if(cache)
            {
                for(int index : cache->indices())
                {
                    if(payloads.count(index))
                        continue;

                    auto slice = cache->rawSlice(index);
                    if(slice.first == nullptr || slice.second == 0)
                        throw std::runtime_error(concatenate(
                            "Solution ", index, " has no bytes in solutions_blob."));
                    payloads[index] = IndexedPayload{nullptr, slice.first, slice.second};
                }
            }

            return payloads;
        }

        std::vector<uint8_t> libraryTreeBytes(MasterContractionLibrary& lib)
        {
            using Inner = SolutionLibrary<ContractionProblemGemm, ContractionSolution>;
            std::shared_ptr<Inner> inner;
            auto cache = std::dynamic_pointer_cast<
                CachingLibrary<ContractionProblemGemm, ContractionSolution>>(lib.library);
            if(cache)
                inner = cache->library();
            else
                inner = lib.library;

            // Snapshot under the lock. The tree walk below reads that copy.
            // blobCache is still the live cache: nothing else may fill it
            // while this write runs.
            decltype(lib.solutions) solutions;
            {
                std::lock_guard<std::mutex> lock(lib.solutionsGuard);
                solutions = lib.solutions;
            }

            LibraryIOContext<ContractionSolution> context{std::string(""), {}, &solutions};
            context.blobCache = lib.blobCache;

            Serialization::MessagePackOutput out(&context);
            out.output(inner);
            return finish(out);
        }
    } // anonymous namespace

    // The traits take mutable references in both directions; writing reads
    // through them without modifying anything.
    std::vector<uint8_t> MessagePackWriteLibraryData(
        SolutionLibrary<ContractionProblemGemm, ContractionSolution> const& library)
    {
        auto& lib = writableMaster(library);

        // Snapshot under the lock, as libraryTreeBytes does. Handing the
        // context a pointer to the live map would read it unlocked while the
        // trait that this call drives copies the same map under solutionsGuard.
        decltype(lib.solutions) solutions;
        {
            std::lock_guard<std::mutex> lock(lib.solutionsGuard);
            solutions = lib.solutions;
        }

        LibraryIOContext<ContractionSolution> context{std::string(""), {}, &solutions};

        Serialization::MessagePackOutput out(&context);
        out.output(lib);
        return finish(out);
    }

    std::vector<uint8_t> MessagePackWriteIndexedLibraryData(
        SolutionLibrary<ContractionProblemGemm, ContractionSolution> const& library)
    {
        auto& lib = writableMaster(library);

        auto cache = lib.blobCache;

        std::vector<int64_t> table;
        std::vector<uint8_t> blob;
        for(auto const& entry : payloadsForIndex(lib, cache))
        {
            std::vector<uint8_t> encoded;
            if(entry.second.solution)
                encoded = MessagePackWriteSolutionData(*entry.second.solution);

            auto   data = entry.second.solution ? encoded.data() : entry.second.data;
            size_t size = entry.second.solution ? encoded.size() : entry.second.size;

            if(blob.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max())
               || size > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
                throw std::runtime_error("solutions_blob exceeds indexed format offsets");

            table.push_back(entry.first);
            table.push_back(static_cast<int64_t>(blob.size()));
            table.push_back(static_cast<int64_t>(size));
            blob.insert(blob.end(), data, data + size);
        }

        if(blob.size() > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("solutions_blob exceeds msgpack bin size");

        auto libraryBytes = libraryTreeBytes(lib);

        const bool hasVersion = !lib.version.empty();

        msgpack::sbuffer                  buffer;
        msgpack::packer<msgpack::sbuffer> packer(buffer);
        packer.pack_map(hasVersion ? 5u : 4u);

        packer.pack(std::string("format_version"));
        packer.pack(IndexedFormatVersion);

        if(hasVersion)
        {
            packer.pack(std::string("version"));
            packer.pack(lib.version);
        }

        packer.pack(std::string("solutions_index"));
        packer.pack(table);

        packer.pack(std::string("solutions_blob"));
        packer.pack_bin(static_cast<uint32_t>(blob.size()));
        packer.pack_bin_body(reinterpret_cast<const char*>(blob.data()),
                             static_cast<uint32_t>(blob.size()));

        packer.pack(std::string("library"));
        buffer.write(reinterpret_cast<const char*>(libraryBytes.data()), libraryBytes.size());

        auto data = reinterpret_cast<const uint8_t*>(buffer.data());
        return std::vector<uint8_t>(data, data + buffer.size());
    }

    std::vector<uint8_t> MessagePackWriteSolutionData(ContractionSolution const& solution)
    {
        Serialization::MessagePackOutput out;
        out.output(const_cast<ContractionSolution&>(solution));
        return finish(out);
    }

    std::vector<uint8_t> MessagePackWriteSolutionListData(
        std::vector<std::shared_ptr<ContractionSolution>> const& solutions)
    {
        Serialization::MessagePackOutput out;
        out.output(const_cast<std::vector<std::shared_ptr<ContractionSolution>>&>(solutions));
        return finish(out);
    }

    std::vector<uint8_t> MessagePackWriteLibraryMapping(std::map<int, std::string> const& mapping)
    {
        msgpack::sbuffer buffer;
        msgpack::packer<msgpack::sbuffer> packer(buffer);

        packer.pack_map(static_cast<uint32_t>(mapping.size()));
        for(auto const& pair : mapping)
        {
            packer.pack(std::to_string(pair.first));
            packer.pack(pair.second);
        }

        auto data = reinterpret_cast<const uint8_t*>(buffer.data());
        return std::vector<uint8_t>(data, data + buffer.size());
    }
}
