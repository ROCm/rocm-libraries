// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "dlpack_adapter.hpp"
#include "device_buffer.hpp"

#include <hip/hip_runtime.h>
#include <hipdnn_frontend/Types.hpp>
#include <nanobind/ndarray.h>

#include <cstring>
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
nb::ndarray<> importDlpack(nb::handle obj, const std::string& what)
{
    nb::ndarray<> array;
    if(!nb::try_cast(obj, array, /*convert=*/false))
    {
        throw nb::value_error(
            (what + ": __dlpack__ did not return a valid DLPack capsule").c_str());
    }
    return array;
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

template <typename T>
T readScalar(const void* src)
{
    T value{};
    std::memcpy(&value, src, sizeof(T));
    return value;
}

void setHostScalar(TensorAttributes& tensor,
                   DataType dataType,
                   const void* src,
                   const nb::dlpack::dtype& dt)
{
    switch(dataType)
    {
    case DataType::FLOAT:
        tensor.set_value(readScalar<float>(src));
        return;
    case DataType::DOUBLE:
        tensor.set_value(readScalar<double>(src));
        return;
    case DataType::HALF:
        tensor.set_value(readScalar<hipdnn_frontend::half>(src));
        return;
    case DataType::BFLOAT16:
        tensor.set_value(readScalar<hipdnn_frontend::bfloat16>(src));
        return;
    case DataType::UINT8:
        tensor.set_value(readScalar<uint8_t>(src));
        return;
    case DataType::INT32:
        tensor.set_value(readScalar<int32_t>(src));
        return;
    case DataType::INT64:
        tensor.set_value(readScalar<int64_t>(src));
        return;
    case DataType::BOOLEAN:
        tensor.set_value(readScalar<uint8_t>(src) != 0);
        return;
    default:
        throw nb::value_error(("tensor_like(): host scalar dtype " + dtypeText(dt)
                               + " is not supported for pass-by-value")
                                  .c_str());
    }
}

} // namespace

void* toDevicePointer(nb::handle value, const std::string& what)
{
    // bool is a subclass of int in Python, so reject it before the int check.
    if(nb::isinstance<nb::bool_>(value))
    {
        throw nb::type_error((what + ": bool is not a device pointer").c_str());
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
    if(nb::hasattr(value, "__dlpack__"))
    {
        const nb::ndarray<> array = importDlpack(value, what);
        if(array.device_type() != nb::device::rocm::value)
        {
            throw nb::value_error(
                (what
                 + ": DLPack tensor must be ROCm device memory (device_type=10), got device_type="
                 + std::to_string(array.device_type()))
                    .c_str());
        }
        int currentDevice = 0;
        const auto status = hipGetDevice(&currentDevice);
        if(status != hipSuccess)
        {
            throw std::runtime_error("hipGetDevice failed: "
                                     + std::string(hipGetErrorString(status)));
        }
        if(array.device_id() != currentDevice)
        {
            throw nb::value_error(
                (what + ": DLPack tensor is on device " + std::to_string(array.device_id())
                 + " but the current HIP device is " + std::to_string(currentDevice))
                    .c_str());
        }
        return array.data();
    }
    throw nb::type_error(
        (what + ": expected int, DeviceBuffer, or an object implementing __dlpack__, got "
         + nb::type_name(value.type()).c_str())
            .c_str());
}

std::unordered_map<int64_t, void*> toVariantPack(const nb::dict& variantPack)
{
    std::unordered_map<int64_t, void*> result;
    result.reserve(variantPack.size());
    for(const auto& [key, value] : variantPack)
    {
        if(!nb::isinstance<nb::int_>(key) || nb::isinstance<nb::bool_>(key))
        {
            throw nb::type_error("variant_pack keys must be int tensor UIDs");
        }
        const auto uid = nb::cast<int64_t>(key);
        result[uid] = toDevicePointer(value, "variant_pack[" + std::to_string(uid) + "]");
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
    const nb::ndarray<> array = importDlpack(obj, "tensor_like()");

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

    auto tensor = std::make_shared<TensorAttributes>();
    const int deviceType = array.device_type();
    if(deviceType == nb::device::cpu::value && array.size() == 1)
    {
        setHostScalar(*tensor, *dataType, array.data(), dt);
    }
    else if(deviceType == nb::device::cpu::value || deviceType == nb::device::rocm::value)
    {
        const size_t ndim = array.ndim();
        std::vector<int64_t> dims(ndim);
        std::vector<int64_t> strides(ndim);
        for(size_t i = 0; i < ndim; ++i)
        {
            dims[i] = static_cast<int64_t>(array.shape(i));
        }
        if(ndim == 0)
        {
            dims = strides = {1};
        }
        else if(array.stride_ptr() == nullptr)
        {
            strides[ndim - 1] = 1;
            for(size_t i = ndim - 1; i > 0; --i)
            {
                strides[i - 1] = strides[i] * dims[i];
            }
        }
        else
        {
            for(size_t i = 0; i < ndim; ++i)
            {
                strides[i] = array.stride(i);
            }
        }
        tensor->set_dim(dims).set_stride(strides).set_data_type(*dataType);
    }
    else
    {
        throw nb::value_error(("tensor_like(): unsupported DLPack device_type="
                               + std::to_string(deviceType) + "; expected cpu (1) or rocm (10)")
                                  .c_str());
    }
    tensor->set_name(name);
    return tensor;
}

} // namespace hipdnn_python
