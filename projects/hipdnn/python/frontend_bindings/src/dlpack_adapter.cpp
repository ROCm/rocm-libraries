// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "dlpack_adapter.hpp"
#include "device_buffer.hpp"

#include <hipdnn_frontend/Types.hpp>
#include <nanobind/ndarray.h>

#include <optional>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;
using hipdnn_frontend::DataType;
using hipdnn_frontend::graph::TensorAttributes;

namespace hipdnn_python
{
namespace
{

// Imports a __dlpack__ producer without conversion, so the data is never copied.
// Read-only capsules are accepted: as in cuDNN, the writable flag is not checked.
nb::ndarray<nb::ro> importDlpack(nb::handle obj, const std::string& what)
{
    nb::ndarray<nb::ro> array;
    if(!nb::try_cast(obj, array, /*convert=*/false))
    {
        throw nb::value_error(
            (what + ": __dlpack__ did not return a valid DLPack capsule").c_str());
    }
    return array;
}

// Python int or any other __index__ integer (for example np.int64), which the
// earlier nanobind map conversion also accepted. bool is rejected.
std::optional<nb::int_> asIndexInteger(nb::handle value)
{
    if(nb::isinstance<nb::bool_>(value) || PyIndex_Check(value.ptr()) == 0)
    {
        return std::nullopt;
    }
    PyObject* index = PyNumber_Index(value.ptr());
    if(index == nullptr)
    {
        throw nb::python_error();
    }
    return nb::steal<nb::int_>(index);
}

std::string dtypeText(const nb::dlpack::dtype& dt)
{
    return "(code=" + std::to_string(static_cast<int>(dt.code))
           + ", bits=" + std::to_string(static_cast<int>(dt.bits)) + ")";
}

std::optional<DataType> toDataType(const nb::dlpack::dtype& dt)
{
    using Code = nb::dlpack::dtype_code;
    const auto code = static_cast<Code>(dt.code);
    const int bits = dt.bits;
    switch(code)
    {
    case Code::Float:
        if(bits == 16)
        {
            return DataType::HALF;
        }
        if(bits == 32)
        {
            return DataType::FLOAT;
        }
        if(bits == 64)
        {
            return DataType::DOUBLE;
        }
        break;
    case Code::Bfloat:
        if(bits == 16)
        {
            return DataType::BFLOAT16;
        }
        break;
    case Code::Int:
        if(bits == 8)
        {
            return DataType::INT8;
        }
        if(bits == 32)
        {
            return DataType::INT32;
        }
        if(bits == 64)
        {
            return DataType::INT64;
        }
        break;
    case Code::UInt:
        if(bits == 8)
        {
            return DataType::UINT8;
        }
        break;
    case Code::Bool:
        if(bits == 8)
        {
            return DataType::BOOLEAN;
        }
        break;
    case Code::Float8_E4M3FN:
        if(bits == 8)
        {
            return DataType::FP8_E4M3;
        }
        break;
    case Code::Float8_E4M3FNUZ:
        if(bits == 8)
        {
            return DataType::FP8_E4M3_FNUZ;
        }
        break;
    case Code::Float8_E5M2:
        if(bits == 8)
        {
            return DataType::FP8_E5M2;
        }
        break;
    case Code::Float8_E5M2FNUZ:
        if(bits == 8)
        {
            return DataType::FP8_E5M2_FNUZ;
        }
        break;
    case Code::Float8_E8M0FNU:
        if(bits == 8)
        {
            return DataType::FP8_E8M0;
        }
        break;
    case Code::Float6_E2M3FN:
        if(bits == 6)
        {
            return DataType::FP6_E2M3;
        }
        break;
    case Code::Float6_E3M2FN:
        if(bits == 6)
        {
            return DataType::FP6_E3M2;
        }
        break;
    case Code::Float4_E2M1FN:
        if(bits == 4)
        {
            return DataType::FP4_E2M1;
        }
        break;
    default:
        break;
    }
    return std::nullopt;
}

// Same device set cuDNN accepts (kDLCPU, kDLCUDAHost, kDLCUDA), mapped to ROCm.
bool isSupportedDevice(int deviceType)
{
    return deviceType == nb::device::cpu::value || deviceType == nb::device::rocm::value
           || deviceType == nb::device::rocm_host::value;
}

std::string unsupportedDeviceText(int deviceType)
{
    return "unsupported DLPack device_type=" + std::to_string(deviceType)
           + "; expected cpu (1), rocm (10), or rocm_host (11)";
}

} // namespace

void* toDataPointer(nb::handle value, const std::string& what)
{
    // bool is a subclass of int in Python, so reject it before the int check.
    if(nb::isinstance<nb::bool_>(value))
    {
        throw nb::type_error((what + ": bool is not a data pointer").c_str());
    }
    if(nb::isinstance<nb::int_>(value))
    {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        return reinterpret_cast<void*>(nb::cast<uintptr_t>(value));
    }
    if(nb::isinstance<DeviceBuffer>(value))
    {
        return nb::cast<DeviceBuffer&>(value).ptr();
    }
    // cuDNN order: data_ptr() before the DLPack fallback. For torch it is the
    // same address as DLPack (storage offset included) without a capsule round trip.
    if(nb::hasattr(value, "data_ptr"))
    {
        const nb::object dataPtr = value.attr("data_ptr");
        if(nb::isinstance<nb::callable>(dataPtr))
        {
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            return reinterpret_cast<void*>(nb::cast<uintptr_t>(dataPtr()));
        }
    }
    if(nb::hasattr(value, "__dlpack__"))
    {
        const nb::ndarray<nb::ro> array = importDlpack(value, what);
        if(!isSupportedDevice(array.device_type()))
        {
            throw nb::value_error(
                (what + ": " + unsupportedDeviceText(array.device_type())).c_str());
        }
        return const_cast<void*>(array.data());
    }
    // Checked last: a single-element integer tensor also implements __index__,
    // and its value is not a pointer.
    if(const auto index = asIndexInteger(value))
    {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        return reinterpret_cast<void*>(nb::cast<uintptr_t>(*index));
    }
    throw nb::type_error(
        (what
         + ": expected an integer, DeviceBuffer, an object with data_ptr(), or an object "
           "implementing "
           "__dlpack__, got "
         + nb::type_name(value.type()).c_str())
            .c_str());
}

std::unordered_map<int64_t, void*> toVariantPack(const nb::dict& variantPack)
{
    std::unordered_map<int64_t, void*> result;
    result.reserve(variantPack.size());
    for(const auto& [key, value] : variantPack)
    {
        int64_t uid = 0;
        if(nb::isinstance<TensorAttributes>(key))
        {
            const auto& tensor = nb::cast<const TensorAttributes&>(key);
            if(!tensor.has_uid())
            {
                throw nb::value_error(
                    ("variant_pack key tensor '" + tensor.get_name() + "' has no uid").c_str());
            }
            uid = tensor.get_uid();
        }
        else if(const auto index = asIndexInteger(key))
        {
            uid = nb::cast<int64_t>(*index);
        }
        else
        {
            throw nb::type_error("variant_pack keys must be Tensor objects or int tensor UIDs");
        }
        result[uid] = toDataPointer(value, "variant_pack[" + std::to_string(uid) + "]");
    }
    return result;
}

std::shared_ptr<TensorAttributes> tensorAttributesFromDlpack(nb::handle obj,
                                                             const std::string& name)
{
    if(!nb::hasattr(obj, "__dlpack__"))
    {
        throw nb::type_error(
            "tensor_like() expects a hipdnn Tensor or an object implementing __dlpack__");
    }
    const nb::ndarray<nb::ro> array = importDlpack(obj, "tensor_like()");

    const int deviceType = array.device_type();
    if(!isSupportedDevice(deviceType))
    {
        throw nb::value_error(("tensor_like(): " + unsupportedDeviceText(deviceType)).c_str());
    }
    const auto dt = array.dtype();
    if(dt.lanes != 1)
    {
        throw nb::value_error("tensor_like(): vector DLPack dtypes (lanes != 1) are unsupported");
    }
    const auto dataType = toDataType(dt);
    if(!dataType)
    {
        throw nb::value_error(("tensor_like(): unsupported DLPack dtype " + dtypeText(dt)).c_str());
    }

    const size_t ndim = array.ndim();
    std::vector<int64_t> dims(ndim);
    std::vector<int64_t> strides(ndim);
    for(size_t i = 0; i < ndim; ++i)
    {
        dims[i] = static_cast<int64_t>(array.shape(i));
    }
    if(array.stride_ptr() == nullptr)
    {
        // DLPack: null strides mean compact row-major.
        int64_t stride = 1;
        for(size_t i = ndim; i > 0; --i)
        {
            strides[i - 1] = stride;
            stride *= dims[i - 1];
        }
    }
    else
    {
        for(size_t i = 0; i < ndim; ++i)
        {
            strides[i] = array.stride(i);
        }
    }

    // As in cuDNN, host memory marks a runtime pass-by-value tensor: execute()
    // takes its host pointer in the variant pack.
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim(dims)
        .set_stride(strides)
        .set_data_type(*dataType)
        .set_is_pass_by_value(deviceType == nb::device::cpu::value)
        .set_name(name);
    return tensor;
}

} // namespace hipdnn_python
