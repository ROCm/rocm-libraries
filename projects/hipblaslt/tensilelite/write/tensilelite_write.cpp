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

    void packValue(msgpack::packer<msgpack::sbuffer>& packer, nb::handle handle)
    {
        if(!handle.is_valid() || handle.is_none())
        {
            packer.pack_nil();
            return;
        }
        if(PyBool_Check(handle.ptr()))
        {
            packer.pack(handle.ptr() == Py_True);
            return;
        }
        if(PyLong_Check(handle.ptr()))
        {
            int       overflow = 0;
            long long value    = PyLong_AsLongLongAndOverflow(handle.ptr(), &overflow);
            if(overflow > 0)
            {
                unsigned long long wide = PyLong_AsUnsignedLongLong(handle.ptr());
                if(PyErr_Occurred())
                {
                    PyErr_Clear();
                    throw std::runtime_error("solution state integer is out of range");
                }
                packer.pack(wide);
            }
            else if(value == -1 && PyErr_Occurred())
            {
                PyErr_Clear();
                throw std::runtime_error("solution state integer is out of range");
            }
            else
            {
                packer.pack(value);
            }
            return;
        }
        if(PyFloat_Check(handle.ptr()))
        {
            packer.pack(PyFloat_AsDouble(handle.ptr()));
            return;
        }
        if(PyUnicode_Check(handle.ptr()))
        {
            Py_ssize_t  size = 0;
            char const* text = PyUnicode_AsUTF8AndSize(handle.ptr(), &size);
            if(!text)
                throw std::runtime_error("solution state string is not UTF-8");
            packer.pack_str(static_cast<uint32_t>(size));
            packer.pack_str_body(text, static_cast<uint32_t>(size));
            return;
        }
        if(PyBytes_Check(handle.ptr()))
        {
            char*      bytes = nullptr;
            Py_ssize_t size  = 0;
            if(PyBytes_AsStringAndSize(handle.ptr(), &bytes, &size) != 0)
                throw std::runtime_error("solution state bytes cannot be read");
            packer.pack_bin(static_cast<uint32_t>(size));
            packer.pack_bin_body(bytes, static_cast<uint32_t>(size));
            return;
        }
        if(PyList_Check(handle.ptr()) || PyTuple_Check(handle.ptr()))
        {
            Py_ssize_t count = PySequence_Size(handle.ptr());
            packer.pack_array(static_cast<uint32_t>(count));
            for(Py_ssize_t i = 0; i < count; ++i)
            {
                nb::object item = nb::steal(PySequence_GetItem(handle.ptr(), i));
                packValue(packer, item);
            }
            return;
        }
        if(PyDict_Check(handle.ptr()))
        {
            packer.pack_map(static_cast<uint32_t>(PyDict_Size(handle.ptr())));
            PyObject *key = nullptr, *value = nullptr;
            Py_ssize_t pos = 0;
            while(PyDict_Next(handle.ptr(), &pos, &key, &value))
            {
                packValue(packer, nb::handle(key));
                packValue(packer, nb::handle(value));
            }
            return;
        }
        if(PyObject_HasAttrString(handle.ptr(), "state"))
        {
            nb::object state = nb::steal(PyObject_GetAttrString(handle.ptr(), "state"));
            if(PyDict_Check(state.ptr()))
            {
                packValue(packer, state);
                return;
            }
            if(PyCallable_Check(state.ptr()))
            {
                nb::object called = nb::steal(PyObject_CallNoArgs(state.ptr()));
                if(!called.is_valid())
                {
                    PyErr_Clear();
                    throw std::runtime_error("solution state() failed");
                }
                packValue(packer, called);
                return;
            }
        }
        throw std::runtime_error("cannot encode a solution state value");
    }

    std::vector<uint8_t> packObject(nb::handle handle)
    {
        msgpack::sbuffer                      buffer;
        msgpack::packer<msgpack::sbuffer>     packer(buffer);
        packValue(packer, handle);
        auto const* data = reinterpret_cast<uint8_t const*>(buffer.data());
        return std::vector<uint8_t>(data, data + buffer.size());
    }

    std::vector<uint8_t> asDocument(nb::object const& data)
    {
        if(PyBytes_Check(data.ptr()))
            return asVector(nb::cast<nb::bytes>(data));
        return packObject(data);
    }

    nb::object toPython(msgpack::object const& object)
    {
        switch(object.type)
        {
        case msgpack::type::NIL:
            return nb::none();
        case msgpack::type::BOOLEAN:
            return nb::bool_(object.via.boolean);
        case msgpack::type::POSITIVE_INTEGER:
            return nb::int_(object.via.u64);
        case msgpack::type::NEGATIVE_INTEGER:
            return nb::int_(object.via.i64);
        case msgpack::type::FLOAT32:
        case msgpack::type::FLOAT64:
            return nb::float_(object.via.f64);
        case msgpack::type::STR:
            return nb::str(object.via.str.ptr, object.via.str.size);
        case msgpack::type::BIN:
            return nb::bytes(object.via.bin.ptr, object.via.bin.size);
        case msgpack::type::ARRAY:
        {
            nb::list values;
            for(uint32_t i = 0; i < object.via.array.size; ++i)
                values.append(toPython(object.via.array.ptr[i]));
            return values;
        }
        case msgpack::type::MAP:
        {
            nb::dict values;
            for(uint32_t i = 0; i < object.via.map.size; ++i)
            {
                values[toPython(object.via.map.ptr[i].key)]
                    = toPython(object.via.map.ptr[i].val);
            }
            return values;
        }
        default:
            throw std::runtime_error("cannot return this MessagePack value to Python");
        }
    }
} // namespace

NB_MODULE(tensilelite_write, m)
{
    m.doc() = "Write TensileLite MessagePack libraries, and build a solution record from a "
              "solution state, with the C++ serialization traits.";

    m.def(
        "write_library",
        [](nb::object data) {
            return asBytes(TensileLite::MessagePackWriteLibraryFromBytes(asDocument(data)));
        },
        nb::arg("library"),
        "Return the eager MasterSolutionLibrary document for a library state.");

    m.def(
        "write_library_indexed",
        [](nb::object data) {
            return asBytes(
                TensileLite::MessagePackWriteIndexedLibraryFromBytes(asDocument(data)));
        },
        nb::arg("library"),
        "Return the format_version 2 indexed document for a library state.");

    m.def(
        "write_mapping",
        [](nb::object data) {
            auto bytes = asDocument(data);
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
        nb::arg("mapping"),
        "Return the lazy-library index-to-file-prefix table.");

    m.def(
        "solution_from_state",
        [](nb::object data) {
            auto bytes = asDocument(data);
            auto handle
                = msgpack::unpack(reinterpret_cast<char const*>(bytes.data()), bytes.size());
            auto const& state = handle.get();
            if(state.type != msgpack::type::MAP)
                throw std::runtime_error("solution state must be a map");
            auto solution = TensileLite::solutionFromState(state);
            auto record   = TensileLite::MessagePackWriteSolutionData(*solution);
            auto written  = msgpack::unpack(reinterpret_cast<char const*>(record.data()),
                                           record.size());
            return toPython(written.get());
        },
        nb::arg("state"),
        "Return the solution record for one solution state.");

    m.def(
        "encode",
        [](nb::object data) { return asBytes(packObject(data)); },
        nb::arg("value"),
        "Return the MessagePack encoding of a Python value.");

    m.def(
        "decode",
        [](nb::bytes data) {
            auto bytes  = asVector(data);
            auto handle = msgpack::unpack(reinterpret_cast<char const*>(bytes.data()), bytes.size());
            return toPython(handle.get());
        },
        nb::arg("data"),
        "Return the Python value stored in a MessagePack document.");
}
