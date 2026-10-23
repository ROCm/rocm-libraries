// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
#include <roc/hostnumerics/validation.hpp>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::hostnumerics;

namespace {
std::vector<size_t> dimensions(const Shape& shape) {
    return {shape.dimensions().begin(), shape.dimensions().end()};
}

std::vector<ptrdiff_t> strides(const Layout& layout) {
    return {layout.strides().begin(), layout.strides().end()};
}

template <typename Value, typename Tag>
Value loadTensorValue(std::span<const std::byte> storage, ptrdiff_t logicalOffset) {
    return detail::decodeScalarKnown<Tag, Value>(storage, logicalOffset);
}

template <typename Value>
void appendTensorValues(nb::list& result, Tensor tensor) {
    using LoadFunction = Value (*)(std::span<const std::byte>, ptrdiff_t);
    const LoadFunction load = visitScalarType(
        tensor.type(), []<typename Tag>() -> LoadFunction { return &loadTensorValue<Value, Tag>; });
    const auto storage = tensor.rawEncodedBackingStorage();
    const auto& layout = tensor.layout();
    detail::forEachIndex(tensor.shape(), [&](std::span<const size_t> indices, size_t) {
        const auto offset = layout.elementOffset(indices);
        if (!offset) throw std::runtime_error(detail::message(offset.error()));
        result.append(load(storage, *offset));
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
            appendTensorValues<double>(result, tensor);
            break;
        case ScalarCategory::Count:
            throw std::runtime_error("Tensor values have an invalid scalar type.");
    }
    return result;
}

nb::object tensorItem(const Tensor& tensor) {
    switch (scalarTypeInfo(tensor.type()).category) {
        case ScalarCategory::Boolean:
            return nb::cast(tensor.item<bool>());
        case ScalarCategory::SignedInteger:
            return nb::cast(tensor.item<int64_t>());
        case ScalarCategory::UnsignedInteger:
            return nb::cast(tensor.item<uint64_t>());
        case ScalarCategory::Complex:
            return nb::cast(tensor.item<std::complex<double>>());
        case ScalarCategory::FloatingPoint:
            return nb::cast(tensor.item<double>());
        case ScalarCategory::Count:
            break;
    }
    throw std::runtime_error("Tensor item has an invalid scalar type.");
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
        throw std::runtime_error("Tensor.from_numpy requires a native-endian NumPy dtype.");

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

    throw std::runtime_error(
        "Tensor.from_numpy supports only exact native NumPy bool, integer, "
        "float16/32/64, and complex64/128 storage dtypes.");
}

std::optional<ptrdiff_t> elementProduct(ptrdiff_t left, ptrdiff_t right) {
    constexpr ptrdiff_t min = std::numeric_limits<ptrdiff_t>::min();
    constexpr ptrdiff_t max = std::numeric_limits<ptrdiff_t>::max();
    if ((left > 0 && ((right > 0 && left > max / right) || (right < 0 && right < min / left))) ||
        (left < 0 && ((right > 0 && left < min / right) || (right < 0 && left < max / right))))
        return std::nullopt;
    return left * right;
}

ptrdiff_t checkedElementDelta(ptrdiff_t stride, size_t elementCount) {
    if (stride == 0 || elementCount == 0) return 0;
    if (!std::in_range<ptrdiff_t>(elementCount))
        throw std::runtime_error("Tensor stride extent exceeds ptrdiff_t.");
    const auto product = elementProduct(stride, static_cast<ptrdiff_t>(elementCount));
    if (!product) throw std::runtime_error("Tensor stride product exceeds ptrdiff_t.");
    return *product;
}

ptrdiff_t checkedElementOffsetAdd(ptrdiff_t left, ptrdiff_t right) {
    if ((right > 0 && left > std::numeric_limits<ptrdiff_t>::max() - right) ||
        (right < 0 && left < std::numeric_limits<ptrdiff_t>::min() - right))
        throw std::runtime_error("Tensor element offset exceeds ptrdiff_t.");
    return left + right;
}

Tensor tensorGetItem(const Tensor& tensor, nb::handle key) {
    const nb::tuple keys =
        nb::isinstance<nb::tuple>(key) ? nb::borrow<nb::tuple>(key) : nb::make_tuple(key);
    const size_t rank = tensor.shape().rank();
    size_t consumedAxes = 0;
    bool hasEllipsis = false;
    for (nb::handle selection : keys) {
        if (selection.is_none()) continue;
        if (selection.ptr() == Py_Ellipsis) {
            if (hasEllipsis) throw std::runtime_error("Tensor accepts only one ellipsis.");
            hasEllipsis = true;
        } else {
            ++consumedAxes;
        }
    }
    if (consumedAxes > rank)
        throw std::runtime_error("Tensor received " + std::to_string(consumedAxes) +
                                 " indices for rank " + std::to_string(rank) + ".");
    const nb::slice fullSlice{nb::none(), nb::none(), nb::none()};
    nb::list selections;
    const auto appendFullSlices = [&] {
        for (size_t axis = consumedAxes; axis < rank; ++axis) selections.append(fullSlice);
    };
    for (nb::handle selection : keys) {
        if (selection.ptr() == Py_Ellipsis)
            appendFullSlices();
        else
            selections.append(selection);
    }
    if (!hasEllipsis) appendFullSlices();

    std::vector<size_t> viewDimensions;
    std::vector<ptrdiff_t> viewStrides;
    std::vector<size_t> starts(rank, 0);
    bool empty = false;
    bool strideOverflow = false;
    size_t axis = 0;
    for (nb::handle selection : selections) {
        if (selection.is_none()) {
            viewDimensions.push_back(1);
            viewStrides.push_back(0);
            continue;
        }
        const size_t extent = tensor.shape()[axis];
        if (!std::in_range<Py_ssize_t>(extent))
            throw std::runtime_error("Tensor axis " + std::to_string(axis) +
                                     " extent exceeds Python's index range.");

        if (nb::isinstance<nb::slice>(selection)) {
            const auto [start, stop, step, length] =
                nb::borrow<nb::slice>(selection).compute(extent);
            starts[axis] = length == 0 ? 0 : static_cast<size_t>(start);
            viewDimensions.push_back(length);
            const ptrdiff_t stride = tensor.layout().stride(axis);
            const auto product = elementProduct(stride, static_cast<ptrdiff_t>(step));
            strideOverflow = strideOverflow || (!product && length > 1);
            // Empty/singleton axes never advance, so their stride need not grow.
            viewStrides.push_back(product.value_or(stride));
            empty = empty || length == 0;
        } else {
            const nb::module_ numpy = nb::module_::import_("numpy");
            if (!PyIndex_Check(selection.ptr()) || PyBool_Check(selection.ptr()) ||
                nb::isinstance(selection, numpy.attr("bool_")) ||
                nb::isinstance(selection, numpy.attr("ndarray")))
                throw std::runtime_error(
                    "Tensor indexing accepts integer positions and slices, None, and ellipsis.");
            const nb::object integer = nb::steal<nb::object>(PyNumber_Index(selection.ptr()));
            if (!integer.is_valid()) throw nb::python_error();
            Py_ssize_t index = PyLong_AsSsize_t(integer.ptr());
            if (index == -1 && PyErr_Occurred()) {
                if (!PyErr_ExceptionMatches(PyExc_OverflowError)) throw nb::python_error();
                PyErr_Clear();
                throw std::runtime_error("Tensor index exceeds Python's index range for axis " +
                                         std::to_string(axis) + " with extent " +
                                         std::to_string(extent) + ".");
            }
            if (index < 0) index += static_cast<Py_ssize_t>(extent);
            if (index < 0 || static_cast<size_t>(index) >= extent)
                throw std::runtime_error("Tensor index " + nb::cast<std::string>(nb::str(integer)) +
                                         " is outside axis " + std::to_string(axis) +
                                         " with extent " + std::to_string(extent) + ".");
            starts[axis] = static_cast<size_t>(index);
        }
        ++axis;
    }

    if (strideOverflow && !empty)
        throw std::runtime_error("Tensor slice stride exceeds ptrdiff_t.");
    ptrdiff_t offset = tensor.layout().offset();
    if (!empty) {
        for (size_t axis = 0; axis < rank; ++axis)
            offset = checkedElementOffsetAdd(
                offset, checkedElementDelta(tensor.layout().stride(axis), starts[axis]));
    }
    return tensor.shareStorageWithLayout(
        Layout(Shape(std::move(viewDimensions)), std::move(viewStrides), offset));
}

size_t checkedStorageBytes(size_t elementCount, size_t itemSize) {
    if (itemSize != 0 && elementCount > std::numeric_limits<size_t>::max() / itemSize)
        throw std::runtime_error("NumPy Tensor storage byte count overflow.");
    return elementCount * itemSize;
}

Tensor tensorFromNumpy(nb::object array, std::optional<ScalarType> requestedType) {
    const nb::object numpyArrayType = nb::module_::import_("numpy").attr("ndarray");
    if (!nb::isinstance(array, numpyArrayType))
        throw nb::type_error("Tensor.from_numpy requires a NumPy ndarray.");

    const NumpyStorageType storageType = numpyStorageType(array);
    if (requestedType && *requestedType != storageType.type)
        throw std::runtime_error(
            "Tensor.from_numpy scalar_type must exactly match the NumPy storage dtype.");
    const ScalarType type = requestedType.value_or(storageType.type);

    const ReadOnlyPythonBuffer buffer(array);
    const Py_buffer& view = buffer.view();
    if (view.ndim < 0 || (view.ndim > 0 && (!view.shape || !view.strides)))
        throw std::runtime_error("NumPy Tensor buffer geometry is invalid.");
    if (view.itemsize <= 0 || static_cast<size_t>(view.itemsize) != storageType.itemSize)
        throw std::runtime_error("NumPy Tensor buffer item size does not match its dtype.");
    if (!std::in_range<ptrdiff_t>(storageType.itemSize))
        throw std::runtime_error("NumPy Tensor item size exceeds ptrdiff_t.");
    const ptrdiff_t signedItemSize = static_cast<ptrdiff_t>(storageType.itemSize);

    std::vector<size_t> tensorDimensions;
    std::vector<ptrdiff_t> tensorStrides;
    tensorDimensions.reserve(static_cast<size_t>(view.ndim));
    tensorStrides.reserve(static_cast<size_t>(view.ndim));
    bool empty = false;
    for (Py_ssize_t dimension = 0; dimension < view.ndim; ++dimension) {
        if (view.shape[dimension] < 0) throw std::runtime_error("NumPy Tensor extent is negative.");
        if (!std::in_range<ptrdiff_t>(view.strides[dimension]))
            throw std::runtime_error("NumPy Tensor byte stride exceeds ptrdiff_t.");
        const size_t extent = static_cast<size_t>(view.shape[dimension]);
        const ptrdiff_t byteStride = static_cast<ptrdiff_t>(view.strides[dimension]);
        if (byteStride % signedItemSize != 0)
            throw std::runtime_error(
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
    if (!view.buf) throw std::runtime_error("Nonempty NumPy Tensor has a null data pointer.");

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
        throw std::runtime_error("NumPy Tensor base offset overflow.");
    const ptrdiff_t normalizedOffset = -lowerOffset;
    const ptrdiff_t normalizedUpper = checkedElementOffsetAdd(upperOffset, normalizedOffset);
    if (!std::in_range<size_t>(normalizedOffset) || !std::in_range<size_t>(normalizedUpper))
        throw std::runtime_error("NumPy Tensor addressed range exceeds size_t.");

    const size_t prefixBytes =
        checkedStorageBytes(static_cast<size_t>(normalizedOffset), storageType.itemSize);
    const size_t normalizedUpperSize = static_cast<size_t>(normalizedUpper);
    if (normalizedUpperSize == std::numeric_limits<size_t>::max())
        throw std::runtime_error("NumPy Tensor addressed range overflow.");
    const size_t storageBytes = checkedStorageBytes(normalizedUpperSize + 1, storageType.itemSize);

    const uintptr_t logicalAddress = reinterpret_cast<uintptr_t>(view.buf);
    if (prefixBytes > logicalAddress)
        throw std::runtime_error("NumPy Tensor base address underflow.");
    const uintptr_t storageAddress = logicalAddress - prefixBytes;
    if (storageBytes > std::numeric_limits<uintptr_t>::max() - storageAddress)
        throw std::runtime_error("NumPy Tensor storage address overflow.");

    return Tensor::copyEncodedBackingStorage(
        type, Layout(std::move(shape), std::move(tensorStrides), normalizedOffset),
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(storageAddress),
                                   storageBytes));
}

Tensor tensorFromStorage(ScalarType type, std::vector<size_t> dimensions, nb::bytes rawStorage,
                         std::optional<std::vector<ptrdiff_t>> tensorStrides, ptrdiff_t offset) {
    std::vector<std::byte> storage(rawStorage.size());
    if (!storage.empty()) std::memcpy(storage.data(), rawStorage.c_str(), rawStorage.size());
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

namespace roc::hostnumerics::python_bindings {
Tensor scalarFromPython(nb::handle value) {
    if (nb::isinstance<Tensor>(value)) {
        Tensor tensor = nb::cast<Tensor>(value);
        (void)tensor.item<std::complex<double>>();
        return tensor.deepCopy();
    }
    nb::object array = nb::module_::import_("numpy").attr("asarray")(value);
    Tensor tensor = tensorFromNumpy(std::move(array), std::nullopt);
    (void)tensor.item<std::complex<double>>();
    return tensor;
}

Tensor tensorOperand(nb::handle value, ScalarType scalarType) {
    if (nb::isinstance<Tensor>(value)) return nb::cast<Tensor>(value);
    return scalarFromPython(value).copyConvertedTo(scalarType);
}
}  // namespace roc::hostnumerics::python_bindings

namespace roc::hostnumerics::python_bindings {
void registerTensorBindings(nb::module_& module) {
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
        .def("__bool__",
             [](const Tensor& tensor) {
                 if (tensor.elementCount() != 1)
                     throw std::runtime_error(
                         "Tensor truth value is ambiguous unless it contains exactly one element.");
                 // The sole element is at the layout offset, regardless of its rank.
                 return nb::bool_(tensorItem(
                     tensor.shareStorageWithLayout(Layout(Shape{}, {}, tensor.layout().offset()))));
             })
        .def("__getitem__", &tensorGetItem, "key"_a.none(),
             "Select positions, slices, None, and ellipsis as a Tensor sharing storage.")
        .def("__len__",
             [](const Tensor& tensor) {
                 if (tensor.shape().rank() == 0)
                     throw nb::type_error("Length requires a tensor with at least one axis.");
                 const size_t length = tensor.shape()[0];
                 if (!std::in_range<Py_ssize_t>(length))
                     throw std::runtime_error("Tensor length exceeds Python's index range.");
                 return length;
             })
        .def("__iter__",
             [](nb::object self) {
                 const auto& tensor = nb::cast<const Tensor&>(self);
                 if (tensor.shape().rank() == 0)
                     throw nb::type_error("Iteration requires a tensor with at least one axis.");
                 const nb::module_ builtins = nb::module_::import_("builtins");
                 // A bound method retains the tensor while map selects rows lazily.
                 return builtins.attr("map")(self.attr("__getitem__"),
                                             builtins.attr("range")(tensor.shape()[0]));
             })
        .def(
            "broadcast_to",
            [](const Tensor& tensor, std::vector<size_t> shape) {
                return tensor.broadcastTo(Shape(std::move(shape)));
            },
            "shape"_a)
        .def(
            "expand_dims",
            [](const Tensor& tensor, Py_ssize_t axis) {
                const Py_ssize_t destinationRank =
                    static_cast<Py_ssize_t>(tensor.shape().rank() + 1);
                if (axis < 0) axis += destinationRank;
                if (axis < 0 || axis >= destinationRank)
                    throw std::runtime_error(
                        "Tensor expansion axis is outside the destination rank.");
                return tensor.expandDims(static_cast<size_t>(axis));
            },
            "axis"_a)
        .def("clone", [](const Tensor& tensor) { return tensor.deepCopy(); })
        .def("to", static_cast<Tensor (Tensor::*)(ScalarType) const>(&Tensor::copyConvertedTo),
             "type"_a)
        .def("absolute", &Tensor::absolute)
        .def("relu", &Tensor::relu)
        .def("gelu", &Tensor::gelu)
        .def("sigmoid", &Tensor::sigmoid)
        .def("tanh", &Tensor::tanh, "input_scale"_a = 1.0, "output_scale"_a = 1.0)
        .def("silu", &Tensor::silu)
        .def("swish", &Tensor::swish, "beta"_a = 1.0)
        .def("clip", &Tensor::clip, "minimum"_a, "maximum"_a)
        .def(
            "__add__",
            [](const Tensor& tensor, nb::handle value) {
                return tensor + python_bindings::tensorOperand(value, tensor.type());
            },
            nb::is_operator())
        .def(
            "__radd__",
            [](const Tensor& tensor, nb::handle value) {
                return python_bindings::tensorOperand(value, tensor.type()) + tensor;
            },
            nb::is_operator())
        .def(
            "__sub__",
            [](const Tensor& tensor, nb::handle value) {
                return tensor - python_bindings::tensorOperand(value, tensor.type());
            },
            nb::is_operator())
        .def(
            "__rsub__",
            [](const Tensor& tensor, nb::handle value) {
                return python_bindings::tensorOperand(value, tensor.type()) - tensor;
            },
            nb::is_operator())
        .def(
            "__neg__", [](const Tensor& tensor) { return -tensor; }, nb::is_operator())
        .def(
            "__mul__",
            [](const Tensor& tensor, nb::handle value) {
                return tensor * python_bindings::tensorOperand(value, tensor.type());
            },
            nb::is_operator())
        .def(
            "__rmul__",
            [](const Tensor& tensor, nb::handle value) {
                return python_bindings::tensorOperand(value, tensor.type()) * tensor;
            },
            nb::is_operator());
}
}  // namespace roc::hostnumerics::python_bindings
