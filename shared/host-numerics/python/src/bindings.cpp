// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "bindings.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <roc/host_numerics/validation.hpp>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::host_numerics;

namespace {
std::vector<size_t> dimensions(const Shape& shape) {
    return {shape.dimensions().begin(), shape.dimensions().end()};
}

std::vector<ptrdiff_t> strides(const Layout& layout) {
    return {layout.strides().begin(), layout.strides().end()};
}

template <typename Function>
void forEachIndex(const Shape& shape, Function&& function) {
    const size_t count = shape.elementCount();
    std::vector<size_t> indices(shape.rank(), 0);
    for (size_t linear = 0; linear < count; ++linear) {
        function(std::span<const size_t>(indices));
        for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            if (++indices[index] < shape[index]) break;
            indices[index] = 0;
        }
    }
}

template <typename Value, typename Tag>
Value loadTensorValue(std::span<const std::byte> storage, ptrdiff_t logicalOffset) {
    return detail::decodeScalarKnown<Tag::type, Value>(storage, logicalOffset);
}

template <typename Value>
void appendTensorValues(nb::list& result, Tensor tensor) {
    using LoadFunction = Value (*)(std::span<const std::byte>, ptrdiff_t);
    const LoadFunction load = visitScalarType(
        tensor.type(), []<typename Tag>() -> LoadFunction { return &loadTensorValue<Value, Tag>; });
    const auto storage = tensor.rawEncodedBackingStorage();
    const auto& layout = tensor.layout();
    forEachIndex(tensor.shape(), [&](std::span<const size_t> indices) {
        result.append(load(storage, layout.elementOffset(indices)));
    });
}

nb::list tensorValues(Tensor tensor) {
    nb::list result;
    const ScalarCategory category = scalarTypeInfo(tensor.type()).category;
    switch (category) {
        case ScalarCategory::Boolean:
            appendTensorValues<bool>(result, tensor);
            break;
        case ScalarCategory::SignedInteger:
            appendTensorValues<int64_t>(result, tensor);
            break;
        case ScalarCategory::UnsignedInteger:
            appendTensorValues<uint64_t>(result, tensor);
            break;
        case ScalarCategory::Complex:
            appendTensorValues<std::complex<double>>(result, tensor);
            break;
        case ScalarCategory::FloatingPoint:
        case ScalarCategory::Scale:
            appendTensorValues<double>(result, tensor);
            break;
    }
    return result;
}

nb::object tensorItem(const Tensor& tensor) {
    const Scalar value = tensor.item();
    switch (scalarTypeInfo(value.type()).category) {
        case ScalarCategory::Boolean:
            return nb::cast(value.as<bool>());
        case ScalarCategory::SignedInteger:
            return nb::cast(value.as<int64_t>());
        case ScalarCategory::UnsignedInteger:
            return nb::cast(value.as<uint64_t>());
        case ScalarCategory::Complex:
            return nb::cast(value.as<std::complex<double>>());
        case ScalarCategory::FloatingPoint:
        case ScalarCategory::Scale:
            return nb::cast(value.as<double>());
    }
    throw std::invalid_argument("Tensor item has an invalid scalar type.");
}

struct NumpyStorageType {
    ScalarType type;
    size_t itemSize;
};

class ReadOnlyPythonBuffer {
   public:
    explicit ReadOnlyPythonBuffer(nb::handle owner) {
        if (PyObject_GetBuffer(owner.ptr(), &m_view, PyBUF_RECORDS_RO) != 0)
            throw nb::python_error();
    }

    ReadOnlyPythonBuffer(const ReadOnlyPythonBuffer&) = delete;
    ReadOnlyPythonBuffer& operator=(const ReadOnlyPythonBuffer&) = delete;

    ~ReadOnlyPythonBuffer() {
        PyBuffer_Release(&m_view);
    }

    const Py_buffer& view() const {
        return m_view;
    }

   private:
    Py_buffer m_view{};
};

NumpyStorageType numpyStorageType(nb::handle array) {
    const nb::object dtype = array.attr("dtype");
    if (!nb::cast<bool>(dtype.attr("isnative")))
        throw nb::type_error("Tensor.from_numpy requires a native-endian NumPy dtype.");

    const std::string kind = nb::cast<std::string>(dtype.attr("kind"));
    const size_t itemSize = nb::cast<size_t>(dtype.attr("itemsize"));
    if (kind.size() == 1) {
        switch (kind[0]) {
            case 'b':
                if (itemSize == 1) return {ScalarType::Boolean, itemSize};
                break;
            case 'u':
                if (itemSize == 1) return {ScalarType::UInt8, itemSize};
                if (itemSize == 2) return {ScalarType::UInt16, itemSize};
                if (itemSize == 4) return {ScalarType::UInt32, itemSize};
                if (itemSize == 8) return {ScalarType::UInt64, itemSize};
                break;
            case 'i':
                if (itemSize == 1) return {ScalarType::Int8, itemSize};
                if (itemSize == 2) return {ScalarType::Int16, itemSize};
                if (itemSize == 4) return {ScalarType::Int32, itemSize};
                if (itemSize == 8) return {ScalarType::Int64, itemSize};
                break;
            case 'f':
                if (itemSize == 2) return {ScalarType::Float16, itemSize};
                if (itemSize == 4) return {ScalarType::Float32, itemSize};
                if (itemSize == 8) return {ScalarType::Float64, itemSize};
                break;
            case 'c':
                if (itemSize == 8) return {ScalarType::ComplexFloat32, itemSize};
                if (itemSize == 16) return {ScalarType::ComplexFloat64, itemSize};
                break;
        }
    }

    throw nb::type_error(
        "Tensor.from_numpy supports only exact native NumPy bool, integer, "
        "float16/32/64, and complex64/128 storage dtypes.");
}

ptrdiff_t checkedElementDelta(ptrdiff_t stride, size_t elementCount) {
    if (stride == 0 || elementCount == 0) return 0;
    if (!std::in_range<ptrdiff_t>(elementCount))
        throw std::overflow_error("NumPy Tensor stride extent exceeds ptrdiff_t.");

    const ptrdiff_t signedElementCount = static_cast<ptrdiff_t>(elementCount);
    if (stride > 0) {
        if (signedElementCount > std::numeric_limits<ptrdiff_t>::max() / stride)
            throw std::overflow_error("NumPy Tensor stride extent overflow.");
    } else if (stride == -1) {
        return -signedElementCount;
    } else if (signedElementCount > std::numeric_limits<ptrdiff_t>::min() / stride) {
        throw std::overflow_error("NumPy Tensor stride extent overflow.");
    }
    return stride * signedElementCount;
}

ptrdiff_t checkedElementOffsetAdd(ptrdiff_t left, ptrdiff_t right) {
    if ((right > 0 && left > std::numeric_limits<ptrdiff_t>::max() - right) ||
        (right < 0 && left < std::numeric_limits<ptrdiff_t>::min() - right))
        throw std::overflow_error("NumPy Tensor element offset overflow.");
    return left + right;
}

size_t checkedStorageBytes(size_t elementCount, size_t itemSize) {
    if (itemSize != 0 && elementCount > std::numeric_limits<size_t>::max() / itemSize)
        throw std::overflow_error("NumPy Tensor storage byte count overflow.");
    return elementCount * itemSize;
}

Tensor tensorFromNumpy(nb::object array, std::optional<ScalarType> requestedType) {
    const nb::object numpyArrayType = nb::module_::import_("numpy").attr("ndarray");
    if (!nb::isinstance(array, numpyArrayType))
        throw nb::type_error("Tensor.from_numpy requires a NumPy ndarray.");

    const NumpyStorageType storageType = numpyStorageType(array);
    if (requestedType && *requestedType != storageType.type)
        throw std::invalid_argument(
            "Tensor.from_numpy scalar_type must exactly match the NumPy storage dtype.");
    const ScalarType type = requestedType.value_or(storageType.type);

    const ReadOnlyPythonBuffer buffer(array);
    const Py_buffer& view = buffer.view();
    if (view.ndim < 0 || (view.ndim > 0 && (!view.shape || !view.strides)))
        throw std::invalid_argument("NumPy Tensor buffer geometry is invalid.");
    if (view.itemsize <= 0 || static_cast<size_t>(view.itemsize) != storageType.itemSize)
        throw std::invalid_argument("NumPy Tensor buffer item size does not match its dtype.");
    if (!std::in_range<ptrdiff_t>(storageType.itemSize))
        throw std::overflow_error("NumPy Tensor item size exceeds ptrdiff_t.");
    const ptrdiff_t signedItemSize = static_cast<ptrdiff_t>(storageType.itemSize);

    std::vector<size_t> tensorDimensions;
    std::vector<ptrdiff_t> tensorStrides;
    tensorDimensions.reserve(static_cast<size_t>(view.ndim));
    tensorStrides.reserve(static_cast<size_t>(view.ndim));
    bool empty = false;
    for (Py_ssize_t dimension = 0; dimension < view.ndim; ++dimension) {
        if (view.shape[dimension] < 0)
            throw std::invalid_argument("NumPy Tensor extent is negative.");
        if (!std::in_range<ptrdiff_t>(view.strides[dimension]))
            throw std::overflow_error("NumPy Tensor byte stride exceeds ptrdiff_t.");
        const size_t extent = static_cast<size_t>(view.shape[dimension]);
        const ptrdiff_t byteStride = static_cast<ptrdiff_t>(view.strides[dimension]);
        if (byteStride % signedItemSize != 0)
            throw std::invalid_argument(
                "NumPy Tensor byte strides must be exact multiples of item size.");
        tensorDimensions.push_back(extent);
        tensorStrides.push_back(byteStride / signedItemSize);
        empty = empty || extent == 0;
    }

    Shape shape(std::move(tensorDimensions));
    if (empty) {
        return Tensor::copyEncodedBackingStorage(
            type, Layout(std::move(shape), std::move(tensorStrides)),
            std::span<const std::byte>(reinterpret_cast<const std::byte*>(view.buf), 0));
    }
    if (!view.buf) throw std::invalid_argument("Nonempty NumPy Tensor has a null data pointer.");

    ptrdiff_t lowerOffset = 0;
    ptrdiff_t upperOffset = 0;
    for (size_t dimension = 0; dimension < tensorStrides.size(); ++dimension) {
        const ptrdiff_t delta = checkedElementDelta(tensorStrides[dimension], shape[dimension] - 1);
        if (delta < 0)
            lowerOffset = checkedElementOffsetAdd(lowerOffset, delta);
        else
            upperOffset = checkedElementOffsetAdd(upperOffset, delta);
    }

    if (lowerOffset == std::numeric_limits<ptrdiff_t>::min())
        throw std::overflow_error("NumPy Tensor base offset overflow.");
    const ptrdiff_t normalizedOffset = -lowerOffset;
    const ptrdiff_t normalizedUpper = checkedElementOffsetAdd(upperOffset, normalizedOffset);
    if (!std::in_range<size_t>(normalizedOffset) || !std::in_range<size_t>(normalizedUpper))
        throw std::overflow_error("NumPy Tensor addressed range exceeds size_t.");

    const size_t prefixBytes =
        checkedStorageBytes(static_cast<size_t>(normalizedOffset), storageType.itemSize);
    const size_t normalizedUpperSize = static_cast<size_t>(normalizedUpper);
    if (normalizedUpperSize == std::numeric_limits<size_t>::max())
        throw std::overflow_error("NumPy Tensor addressed range overflow.");
    const size_t storageBytes = checkedStorageBytes(normalizedUpperSize + 1, storageType.itemSize);

    const uintptr_t logicalAddress = reinterpret_cast<uintptr_t>(view.buf);
    if (prefixBytes > logicalAddress)
        throw std::overflow_error("NumPy Tensor base address underflow.");
    const uintptr_t storageAddress = logicalAddress - prefixBytes;
    if (storageBytes > std::numeric_limits<uintptr_t>::max() - storageAddress)
        throw std::overflow_error("NumPy Tensor storage address overflow.");

    return Tensor::copyEncodedBackingStorage(
        type, Layout(std::move(shape), std::move(tensorStrides), normalizedOffset),
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(storageAddress),
                                   storageBytes));
}

Tensor tensorFromStorage(ScalarType type, std::vector<size_t> dimensions, nb::bytes rawStorage,
                         std::optional<std::vector<ptrdiff_t>> tensorStrides, ptrdiff_t offset) {
    std::vector<std::byte> storage(rawStorage.size());
    std::memcpy(storage.data(), rawStorage.c_str(), rawStorage.size());
    Shape shape(std::move(dimensions));
    Layout layout = Layout::contiguousLastDimensionFastest(shape);
    if (tensorStrides || offset != 0) {
        std::vector<ptrdiff_t> selectedStrides =
            tensorStrides ? std::move(*tensorStrides) : strides(layout);
        layout = Layout(std::move(shape), std::move(selectedStrides), offset);
    }
    return Tensor::takeOwnershipOfEncodedBackingStorage(type, std::move(layout),
                                                        std::move(storage));
}

}  // namespace

namespace roc::host_numerics::python_bindings {
Scalar scalarFromPython(nb::handle value) {
    if (nb::isinstance<Tensor>(value)) return nb::cast<Tensor>(value).item();
    return Scalar(nb::cast<std::complex<double>>(value));
}
}  // namespace roc::host_numerics::python_bindings

NB_MODULE(_roc_host_numerics, module) {
    nb::enum_<ScalarCategory>(module, "ScalarCategory")
        .value("Boolean", ScalarCategory::Boolean)
        .value("SignedInteger", ScalarCategory::SignedInteger)
        .value("UnsignedInteger", ScalarCategory::UnsignedInteger)
        .value("FloatingPoint", ScalarCategory::FloatingPoint)
        .value("Complex", ScalarCategory::Complex)
        .value("Scale", ScalarCategory::Scale);

    nb::enum_<ScalarType>(module, "ScalarType")
        .value("Boolean", ScalarType::Boolean)
        .value("UInt8", ScalarType::UInt8)
        .value("Int8", ScalarType::Int8)
        .value("UInt16", ScalarType::UInt16)
        .value("Int16", ScalarType::Int16)
        .value("UInt32", ScalarType::UInt32)
        .value("Int32", ScalarType::Int32)
        .value("UInt64", ScalarType::UInt64)
        .value("Int64", ScalarType::Int64)
        .value("Float16", ScalarType::Float16)
        .value("BFloat16", ScalarType::BFloat16)
        .value("Float32", ScalarType::Float32)
        .value("Float64", ScalarType::Float64)
        .value("ComplexFloat32", ScalarType::ComplexFloat32)
        .value("ComplexFloat64", ScalarType::ComplexFloat64)
        .value("Float8E4M3", ScalarType::Float8E4M3)
        .value("Float8E5M2", ScalarType::Float8E5M2)
        .value("Float8E4M3Fnuz", ScalarType::Float8E4M3Fnuz)
        .value("Float8E5M2Fnuz", ScalarType::Float8E5M2Fnuz)
        .value("Float6E2M3", ScalarType::Float6E2M3)
        .value("Float6E3M2", ScalarType::Float6E3M2)
        .value("Float4E2M1", ScalarType::Float4E2M1)
        .value("Int4", ScalarType::Int4)
        .value("E8M0", ScalarType::E8M0)
        .value("E8M0Zero", ScalarType::E8M0Zero)
        .value("E5M3", ScalarType::E5M3)
        .value("E4M3", ScalarType::E4M3);

    nb::enum_<MathMode>(module, "MathMode")
        .value("Default", MathMode::Default)
        .value("XFloat32", MathMode::XFloat32);

    nb::enum_<AccumulationRounding>(module, "AccumulationRounding")
        .value("TypeDefault", AccumulationRounding::TypeDefault)
        .value("FullPrecision", AccumulationRounding::FullPrecision)
        .value("AfterProductAndSum", AccumulationRounding::AfterProductAndSum);

    nb::enum_<GemmBackend>(module, "GemmBackend")
        .value("Automatic", GemmBackend::Automatic)
        .value("Pointwise", GemmBackend::Pointwise)
        .value("Blocked", GemmBackend::Blocked);

    nb::enum_<OutputConversion>(module, "OutputConversion")
        .value("Default", OutputConversion::Default)
        .value("SaturatingInt8", OutputConversion::SaturatingInt8);

    nb::enum_<Activation>(module, "Activation")
        .value("None_", Activation::None)
        .value("Absolute", Activation::Absolute)
        .value("ClippedRelu", Activation::ClippedRelu)
        .value("Relu", Activation::Relu)
        .value("Gelu", Activation::Gelu)
        .value("GeluDerivative", Activation::GeluDerivative)
        .value("GeluScaling", Activation::GeluScaling)
        .value("LeakyRelu", Activation::LeakyRelu)
        .value("ReluDerivative", Activation::ReluDerivative)
        .value("Sigmoid", Activation::Sigmoid)
        .value("Tanh", Activation::Tanh)
        .value("Silu", Activation::Silu)
        .value("Swish", Activation::Swish)
        .value("Clamp", Activation::Clamp);

    python_bindings::registerGenerationBindings(module);
    python_bindings::registerMxBindings(module);

    nb::enum_<StructuredSparsitySelection>(module, "StructuredSparsitySelection")
        .value("Fixed", StructuredSparsitySelection::Fixed)
        .value("Random", StructuredSparsitySelection::Random);

    nb::enum_<ActivationApplication>(module, "ActivationApplication")
        .value("Forward", ActivationApplication::Forward)
        .value("Gradient", ActivationApplication::Gradient);

    nb::enum_<ReductionOperation>(module, "ReductionOperation")
        .value("Sum", ReductionOperation::Sum)
        .value("MaximumAbsolute", ReductionOperation::MaximumAbsolute);

    nb::enum_<OutputSelectionKind>(module, "OutputSelectionKind")
        .value("All", OutputSelectionKind::All)
        .value("Strided", OutputSelectionKind::Strided)
        .value("Explicit", OutputSelectionKind::Explicit);

    nb::class_<OutputSelection>(module, "OutputSelection")
        .def_static("all", &OutputSelection::all,
                    "index_order"_a = IndexOrder::LastDimensionFastest)
        .def_static("strided", &OutputSelection::strided, "first"_a, "stride"_a,
                    "max_elements"_a = std::numeric_limits<size_t>::max(),
                    "index_order"_a = IndexOrder::LastDimensionFastest)
        .def_static("explicit_indices", &OutputSelection::explicitIndices, "indices"_a,
                    "index_order"_a = IndexOrder::LastDimensionFastest)
        .def_static("prime_stride", &OutputSelection::primeStride, "logical_elements"_a,
                    "allocated_elements"_a, "requested_elements"_a,
                    "index_order"_a = IndexOrder::LastDimensionFastest)
        .def_prop_ro("kind", &OutputSelection::kind)
        .def_prop_ro("selects_all", &OutputSelection::selectsAll)
        .def_prop_ro("first", &OutputSelection::first)
        .def_prop_ro("stride", &OutputSelection::stride)
        .def_prop_ro("max_elements", &OutputSelection::maxElements)
        .def_prop_ro("index_order", &OutputSelection::indexOrder)
        .def("indices", &OutputSelection::indices, "logical_elements"_a);

    nb::class_<ScalarTypeInfo>(module, "ScalarTypeInfo")
        .def_prop_ro("name", [](const ScalarTypeInfo& info) { return std::string(info.name); })
        .def_ro("category", &ScalarTypeInfo::category)
        .def_ro("storage_bits", &ScalarTypeInfo::storageBits)
        .def_ro("exponent_bits", &ScalarTypeInfo::exponentBits)
        .def_ro("mantissa_bits", &ScalarTypeInfo::mantissaBits)
        .def_ro("exponent_bias", &ScalarTypeInfo::exponentBias)
        .def_ro("supports_nan", &ScalarTypeInfo::supportsNaN)
        .def_ro("supports_infinity", &ScalarTypeInfo::supportsInfinity)
        .def("is_packed", &ScalarTypeInfo::isPacked);

    nb::class_<Shape>(module, "Shape")
        .def(nb::init<std::vector<size_t>>())
        .def_prop_ro("rank", &Shape::rank)
        .def_prop_ro("dimensions", &dimensions)
        .def_prop_ro("element_count", [](const Shape& shape) { return shape.elementCount(); });

    nb::class_<Layout>(module, "Layout")
        .def(nb::init<Shape, std::vector<ptrdiff_t>, ptrdiff_t>(), "shape"_a, "strides"_a,
             "offset"_a = 0)
        .def_static("contiguous_last_dimension_fastest", &Layout::contiguousLastDimensionFastest)
        .def_prop_ro("shape", &Layout::shape, nb::rv_policy::reference_internal)
        .def_prop_ro("strides", &strides)
        .def_prop_ro("offset", &Layout::offset);

    nb::class_<Tensor>(module, "Tensor")
        .def(nb::init<ScalarType, Shape>())
        .def(nb::init<ScalarType, Layout>())
        .def_static("from_numpy", &tensorFromNumpy, "array"_a,
                    "scalar_type"_a = std::optional<ScalarType>{})
        .def_static(
            "from_values",
            [](ScalarType type, std::vector<size_t> shape, const std::vector<double>& values) {
                return Tensor::copyValuesWithConversion(type, Shape(std::move(shape)),
                                                        std::span<const double>(values));
            })
        .def_static(
            "from_signed_values",
            [](ScalarType type, std::vector<size_t> shape, const std::vector<int64_t>& values) {
                return Tensor::copyValuesWithConversion(type, Shape(std::move(shape)),
                                                        std::span<const int64_t>(values));
            })
        .def_static(
            "from_unsigned_values",
            [](ScalarType type, std::vector<size_t> shape, const std::vector<uint64_t>& values) {
                return Tensor::copyValuesWithConversion(type, Shape(std::move(shape)),
                                                        std::span<const uint64_t>(values));
            })
        .def_static("from_complex_values",
                    [](ScalarType type, std::vector<size_t> shape,
                       const std::vector<std::complex<double>>& values) {
                        return Tensor::copyValuesWithConversion(
                            type, Shape(std::move(shape)),
                            std::span<const std::complex<double>>(values));
                    })
        .def_static("from_storage", &tensorFromStorage, "type"_a, "shape"_a, "storage"_a,
                    "strides"_a = std::optional<std::vector<ptrdiff_t>>{}, "offset"_a = 0)
        .def_prop_ro("type", &Tensor::type)
        .def_prop_ro("shape", [](const Tensor& tensor) { return dimensions(tensor.shape()); })
        .def_prop_ro("strides", [](const Tensor& tensor) { return strides(tensor.layout()); })
        .def_prop_ro("offset", [](const Tensor& tensor) { return tensor.layout().offset(); })
        .def_prop_ro("size", &Tensor::elementCount)
        .def_prop_ro("storage",
                     [](const Tensor& tensor) {
                         const auto storage = tensor.rawEncodedBackingStorage();
                         return nb::bytes(reinterpret_cast<const char*>(storage.data()),
                                          storage.size());
                     })
        .def_prop_ro("values", [](const Tensor& tensor) { return tensorValues(tensor); })
        .def("item", &tensorItem)
        .def(
            "broadcast_to",
            [](const Tensor& tensor, std::vector<size_t> shape) {
                return tensor.broadcastTo(Shape(std::move(shape)));
            },
            "shape"_a)
        .def("expand_dims", &Tensor::expandDims, "axis"_a)
        .def("clone", [](const Tensor& tensor) { return tensor.deepCopy(); })
        .def("to", static_cast<Tensor (Tensor::*)(ScalarType) const>(&Tensor::copyConvertedTo),
             "type"_a);

    python_bindings::registerGemmBindings(module);

    python_bindings::registerComparisonBindings(module);

    python_bindings::registerOperationBindings(module);

    module.def("scalar_type_info", [](ScalarType type) { return scalarTypeInfo(type); });
}
