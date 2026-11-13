// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "cudnn_compat.hpp"
#include "dlpack_adapter.hpp"

#include <nanobind/stl/string.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace nb = nanobind;
using hipdnn_frontend::DataType;
using hipdnn_frontend::ErrorCode;

namespace
{

class GraphNotSupportedException : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// Names shared by torch (minus its "torch." prefix) and numpy dtypes.
std::optional<DataType> dataTypeFromDtypeName(std::string_view name)
{
    struct Entry
    {
        std::string_view name;
        DataType type;
    };
    static constexpr std::array<Entry, 14> ENTRIES{{
        {"float16", DataType::HALF},
        {"bfloat16", DataType::BFLOAT16},
        {"float32", DataType::FLOAT},
        {"float64", DataType::DOUBLE},
        {"int8", DataType::INT8},
        {"int32", DataType::INT32},
        {"int64", DataType::INT64},
        {"uint8", DataType::UINT8},
        {"bool", DataType::BOOLEAN},
        {"float8_e4m3fn", DataType::FP8_E4M3},
        {"float8_e5m2", DataType::FP8_E5M2},
        {"float8_e8m0fnu", DataType::FP8_E8M0},
        {"float8_e4m3fnuz", DataType::FP8_E4M3_FNUZ},
        {"float8_e5m2fnuz", DataType::FP8_E5M2_FNUZ},
    }};
    for(const auto& entry : ENTRIES)
    {
        if(entry.name == name)
        {
            return entry.type;
        }
    }
    return std::nullopt;
}

bool startsWith(std::string_view text, std::string_view prefix)
{
    return text.substr(0, prefix.size()) == prefix;
}

} // namespace

namespace hipdnn_python
{

void throwUnsupportedParameter(const char* parameter, const char* accepted)
{
    const std::string message = std::string("hipDNN does not support the cuDNN parameter '")
                                + parameter + "'; it only accepts " + accepted + ".";
    PyErr_SetString(PyExc_NotImplementedError, message.c_str());
    throw nb::python_error();
}

void requireNone(const nb::handle& value, const char* parameter)
{
    if(!value.is_none())
    {
        throwUnsupportedParameter(parameter, "None");
    }
}

void throwIfBad(const hipdnn_frontend::Error& error)
{
    if(!error.is_bad())
    {
        return;
    }

    // Same grouping as cuDNN's throw_if (python/pycudnn.cpp in cudnn-frontend).
    switch(error.get_code())
    {
    case ErrorCode::ATTRIBUTE_NOT_SET:
    case ErrorCode::SHAPE_DEDUCTION_FAILED:
    case ErrorCode::INVALID_TENSOR_NAME:
    case ErrorCode::INVALID_VARIANT_PACK:
        throw nb::value_error(error.get_message().c_str());
    case ErrorCode::GRAPH_NOT_SUPPORTED:
    case ErrorCode::GRAPH_EXECUTION_PLAN_CREATION_FAILED:
    case ErrorCode::HEURISTIC_QUERY_FAILED:
    case ErrorCode::UNSUPPORTED_GRAPH_FORMAT:
        throw GraphNotSupportedException(error.get_message());
    default:
        throw std::runtime_error(error.get_message());
    }
}

hipdnnHandle_t rawHandleOrNull(const nb::handle& handle)
{
    if(handle.is_none())
    {
        return nullptr;
    }
    auto handlePtr = handle.attr("get")();
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<hipdnnHandle_t>(nb::cast<uintptr_t>(handlePtr));
}

void* workspacePointer(const nb::handle& workspace)
{
    return workspace.is_none() ? nullptr : toDataPointer(workspace, "workspace");
}

std::vector<uint8_t> bytesToVector(const nb::bytes& data)
{
    const auto* ptr = reinterpret_cast<const uint8_t*>(data.c_str());
    return {ptr, ptr + data.size()};
}

std::optional<DataType> dataTypeFromObject(const nb::handle& value)
{
    if(nb::isinstance<DataType>(value))
    {
        return nb::cast<DataType>(value);
    }

    // Identify torch and numpy dtypes by module, so neither library is imported for an
    // object that did not come from it.
    std::string name;
    const bool isTypeObject = PyType_Check(value.ptr()) != 0;
    const nb::handle owner = isTypeObject ? value : value.type();
    if(!nb::hasattr(owner, "__module__"))
    {
        return std::nullopt;
    }
    const auto module = nb::cast<std::string>(owner.attr("__module__"));
    if(module == "torch" && !isTypeObject)
    {
        // str(torch.float16) == "torch.float16"; torch.half prints the same.
        name = nb::cast<std::string>(nb::str(value));
        if(!startsWith(name, "torch."))
        {
            return std::nullopt;
        }
        name = name.substr(6);
    }
    else if(startsWith(module, "numpy") && isTypeObject)
    {
        // A numpy scalar type such as numpy.float32; numpy.dtype itself is not a dtype.
        if(!nb::hasattr(value, "dtype"))
        {
            return std::nullopt;
        }
        // The dtype's name is canonical, so aliases such as numpy.intc map like
        // numpy.dtype(numpy.intc). Abstract types such as numpy.floating have no dtype
        // and keep their class name, which the table rejects.
        try
        {
            name = nb::cast<std::string>(
                nb::module_::import_("numpy").attr("dtype")(value).attr("name"));
        }
        catch(const nb::python_error&)
        {
            name = nb::cast<std::string>(value.attr("__name__"));
        }
    }
    else if(startsWith(module, "numpy") && nb::hasattr(value, "kind") && nb::hasattr(value, "name"))
    {
        name = nb::cast<std::string>(value.attr("name"));
    }
    else
    {
        return std::nullopt;
    }

    if(auto type = dataTypeFromDtypeName(name))
    {
        return type;
    }
    throw nb::value_error(("hipDNN has no DataType for dtype '" + name + "'").c_str());
}

void compatBindings(nb::module_& m)
{
    // Constructing the exception registers it with the module and with nanobind's
    // C++-to-Python translation; the object itself is not needed afterwards.
    [[maybe_unused]] const nb::exception<GraphNotSupportedException> registered(
        m, "GraphNotSupportedError", PyExc_RuntimeError);
}

} // namespace hipdnn_python
