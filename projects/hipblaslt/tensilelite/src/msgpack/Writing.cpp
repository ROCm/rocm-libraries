// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/msgpack/Writing.hpp>

#include <Tensile/CachingLibrary.hpp>
#include <Tensile/msgpack/MessagePack.hpp>

#include <limits>
#include <map>
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

        std::map<int, std::shared_ptr<ContractionSolution>>
            solutionsForIndex(MasterContractionLibrary& lib)
        {
            std::map<int, std::shared_ptr<ContractionSolution>> written;
            {
                std::lock_guard<std::mutex> lock(lib.solutionsGuard);
                written.insert(lib.solutions.begin(), lib.solutions.end());
            }

            if(lib.blobCache)
            {
                for(int index : lib.blobCache->indices())
                {
                    if(written.count(index))
                        continue;

                    auto solution = lib.blobCache->get(index);
                    if(solution == nullptr)
                        throw std::runtime_error(concatenate(
                            "Solution ", index, " in solutions_blob does not parse."));
                    written.emplace(index, std::move(solution));
                }
            }

            return written;
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

        std::vector<int64_t> table;
        std::vector<uint8_t> blob;
        for(auto const& entry : solutionsForIndex(lib))
        {
            auto bytes = MessagePackWriteSolutionData(*entry.second);
            if(blob.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max())
               || bytes.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
                throw std::runtime_error("solutions_blob exceeds indexed format offsets");

            table.push_back(entry.first);
            table.push_back(static_cast<int64_t>(blob.size()));
            table.push_back(static_cast<int64_t>(bytes.size()));
            blob.insert(blob.end(), bytes.begin(), bytes.end());
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
