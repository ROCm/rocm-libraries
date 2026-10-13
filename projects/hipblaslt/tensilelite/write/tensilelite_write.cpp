// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>

#include <Tensile/msgpack/SolutionState.hpp>
#include <Tensile/msgpack/Writing.hpp>

#include <msgpack.hpp>

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace
{
    std::vector<uint8_t> asVector(nb::bytes const& data)
    {
        auto const* bytes = reinterpret_cast<uint8_t const*>(data.c_str());
        return std::vector<uint8_t>(bytes, bytes + data.size());
    }

    nb::bytes asBytes(std::vector<uint8_t> const& data)
    {
        return nb::bytes(reinterpret_cast<char const*>(data.data()), data.size());
    }
} // namespace

NB_MODULE(tensilelite_write, m)
{
    m.doc() = "Write TensileLite MessagePack libraries, and build a solution record from a "
              "solution state, with the C++ serialization traits.";

    m.def(
        "write_library",
        [](nb::bytes data) {
            return asBytes(TensileLite::MessagePackWriteLibraryFromBytes(asVector(data)));
        },
        nb::arg("msgpack_bytes"),
        "Return the eager MasterSolutionLibrary document for msgpack_bytes.");

    m.def(
        "write_library_indexed",
        [](nb::bytes data) {
            return asBytes(TensileLite::MessagePackWriteIndexedLibraryFromBytes(asVector(data)));
        },
        nb::arg("msgpack_bytes"),
        "Return the format_version 2 indexed document for msgpack_bytes.");

    m.def(
        "write_mapping",
        [](nb::bytes data) {
            auto bytes = asVector(data);
            auto handle
                = msgpack::unpack(reinterpret_cast<char const*>(bytes.data()), bytes.size());
            auto const& object = handle.get();
            if(object.type != msgpack::type::MAP)
                throw std::runtime_error("library mapping must be a msgpack map");
            std::map<int, std::string> mapping;
            for(uint32_t i = 0; i < object.via.map.size; ++i)
            {
                auto const& key = object.via.map.ptr[i].key;
                auto const& value = object.via.map.ptr[i].val;
                int index = 0;
                if(key.type == msgpack::type::POSITIVE_INTEGER)
                    index = static_cast<int>(key.via.u64);
                else if(key.type == msgpack::type::NEGATIVE_INTEGER)
                    index = static_cast<int>(key.via.i64);
                else if(key.type == msgpack::type::STR)
                    index = std::stoi(std::string(key.via.str.ptr, key.via.str.size));
                else
                    throw std::runtime_error("library mapping key must be an index");
                if(value.type != msgpack::type::STR)
                    throw std::runtime_error("library mapping value must be a file prefix");
                mapping.emplace(index, std::string(value.via.str.ptr, value.via.str.size));
            }
            return asBytes(TensileLite::MessagePackWriteLibraryMapping(mapping));
        },
        nb::arg("msgpack_bytes"),
        "Return the lazy-library index-to-file-prefix table.");

    m.def(
        "solution_from_state",
        [](nb::bytes data) {
            auto bytes = asVector(data);
            auto handle
                = msgpack::unpack(reinterpret_cast<char const*>(bytes.data()), bytes.size());
            auto const& state = handle.get();
            if(state.type != msgpack::type::MAP)
                throw std::runtime_error("solution state must be a msgpack map");
            auto solution = TensileLite::solutionFromState(state);
            return asBytes(TensileLite::MessagePackWriteSolutionData(*solution));
        },
        nb::arg("msgpack_bytes"),
        "Return the MessagePack solution record for one solution-state map.");
}
