// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>

#include <Tensile/msgpack/Writing.hpp>

#include <cstdint>
#include <stdexcept>
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
    m.doc() = "Write TensileLite MessagePack libraries with the C++ serialization traits.";

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
}
