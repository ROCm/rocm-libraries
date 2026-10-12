// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/msgpack/Writing.hpp>

#include <Tensile/msgpack/MessagePack.hpp>

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
    } // anonymous namespace

    // The traits take mutable references in both directions; writing reads
    // through them without modifying anything.
    std::vector<uint8_t> MessagePackWriteLibraryData(
        SolutionLibrary<ContractionProblemGemm, ContractionSolution> const& library)
    {
        auto master = dynamic_cast<MasterContractionLibrary const*>(&library);
        if(master == nullptr)
            throw std::runtime_error(concatenate(
                "Only a MasterSolutionLibrary can be written as a library, not ", library.type()));

        auto& lib = const_cast<MasterContractionLibrary&>(*master);

        LibraryIOContext<ContractionSolution> context{std::string(""), {}, &lib.solutions};

        Serialization::MessagePackOutput out(&context);
        out.output(lib);
        return finish(out);
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
