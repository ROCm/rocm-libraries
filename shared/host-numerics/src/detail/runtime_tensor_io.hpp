// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <roc/host_numerics/operation_types.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "reference_arithmetic.hpp"

namespace roc::host_numerics::detail {
inline bool isComplexScalarType(ScalarType type) {
    return scalarTypeInfo(type).category == ScalarCategory::Complex;
}

inline bool isScaleScalarType(ScalarType type) {
    return scalarTypeInfo(type).category == ScalarCategory::Scale;
}

template <typename Accumulator>
using RuntimeLoadFunction = Accumulator (*)(std::span<const std::byte>, ptrdiff_t);

template <typename Accumulator>
using RuntimeLoadMatrixBlockFunction = void (*)(std::span<const std::byte>, ptrdiff_t, ptrdiff_t,
                                                ptrdiff_t, size_t, size_t, size_t, size_t,
                                                std::span<Accumulator>);

template <typename Accumulator>
using RuntimeStoreFunction = void (*)(std::span<std::byte>, ptrdiff_t, Accumulator);

template <typename Accumulator, typename Tag>
Accumulator runtimeLoadScalar(std::span<const std::byte> storage, ptrdiff_t logicalOffset) {
    return decodeScalarKnown<Tag, Accumulator>(storage, logicalOffset);
}

template <typename Accumulator, typename Tag>
void runtimeLoadMatrixBlock(std::span<const std::byte> storage, ptrdiff_t offset,
                            ptrdiff_t rowStride, ptrdiff_t columnStride, size_t rowBase,
                            size_t columnBase, size_t rows, size_t columns,
                            std::span<Accumulator> destination) {
    const auto decode = [&](ptrdiff_t logicalOffset) {
        if constexpr (!std::is_void_v<typename Tag::Storage>) {
            typename Tag::Storage encoded;
            std::memcpy(&encoded,
                        storage.data() + static_cast<size_t>(logicalOffset) * sizeof(encoded),
                        sizeof(encoded));
            return decodeScalarValueKnown<Tag, Accumulator>(encoded,
                                                            implicitNativeConversionOptions());
        } else {
            return decodeScalarKnown<Tag, Accumulator>(storage, logicalOffset);
        }
    };
    const auto strideMagnitude = [](ptrdiff_t stride) {
        using Unsigned = std::make_unsigned_t<ptrdiff_t>;
        const Unsigned value = static_cast<Unsigned>(stride);
        return stride >= 0 ? value : Unsigned(0) - value;
    };
    if (strideMagnitude(rowStride) < strideMagnitude(columnStride)) {
        for (size_t column = 0; column < columns; ++column) {
            const ptrdiff_t sourceColumn =
                offset + static_cast<ptrdiff_t>(columnBase + column) * columnStride;
            for (size_t row = 0; row < rows; ++row) {
                const ptrdiff_t sourceOffset =
                    sourceColumn + static_cast<ptrdiff_t>(rowBase + row) * rowStride;
                destination[row * columns + column] = decode(sourceOffset);
            }
        }
        return;
    }
    for (size_t row = 0; row < rows; ++row) {
        const ptrdiff_t sourceRow = offset + static_cast<ptrdiff_t>(rowBase + row) * rowStride;
        for (size_t column = 0; column < columns; ++column) {
            const ptrdiff_t sourceOffset =
                sourceRow + static_cast<ptrdiff_t>(columnBase + column) * columnStride;
            destination[row * columns + column] = decode(sourceOffset);
        }
    }
}

template <typename Accumulator, typename Tag>
void runtimeStoreScalar(std::span<std::byte> storage, ptrdiff_t logicalOffset, Accumulator value) {
    encodeScalarKnown<Tag>(storage, logicalOffset, value);
}

template <typename Accumulator>
RuntimeLoadFunction<Accumulator> runtimeLoadFunction(ScalarType type) {
    return visitScalarType(type,
                           []<typename Tag>() { return &runtimeLoadScalar<Accumulator, Tag>; });
}

template <typename Accumulator>
RuntimeLoadMatrixBlockFunction<Accumulator> runtimeLoadMatrixBlockFunction(ScalarType type) {
    return visitScalarType(
        type, []<typename Tag>() { return &runtimeLoadMatrixBlock<Accumulator, Tag>; });
}

template <typename Accumulator>
RuntimeStoreFunction<Accumulator> runtimeStoreFunction(ScalarType type) {
    return visitScalarType(type,
                           []<typename Tag>() { return &runtimeStoreScalar<Accumulator, Tag>; });
}

template <typename Accumulator>
class RuntimeMatrixReader {
   public:
    explicit RuntimeMatrixReader(const Tensor& view)
        : m_storage(view.rawEncodedBackingStorage()),
          m_offset(view.layout().offset()),
          m_rowStride(view.layout().strides()[0]),
          m_columnStride(view.layout().strides()[1]),
          m_load(runtimeLoadFunction<Accumulator>(view.type())) {}

    Accumulator operator()(size_t row, size_t column) const {
        return m_load(m_storage, m_offset + static_cast<ptrdiff_t>(row) * m_rowStride +
                                     static_cast<ptrdiff_t>(column) * m_columnStride);
    }

   private:
    std::span<const std::byte> m_storage;
    ptrdiff_t m_offset;
    ptrdiff_t m_rowStride;
    ptrdiff_t m_columnStride;
    RuntimeLoadFunction<Accumulator> m_load;
};

template <typename Accumulator>
class RuntimeMatrixBlockReader {
   public:
    // Resolve the encoded storage type once so a whole tile can be decoded without an indirect
    // function call for every scalar.
    explicit RuntimeMatrixBlockReader(const Tensor& view)
        : m_storage(view.rawEncodedBackingStorage()),
          m_offset(view.layout().offset()),
          m_rowStride(view.layout().strides()[0]),
          m_columnStride(view.layout().strides()[1]),
          m_rows(view.shape()[0]),
          m_columns(view.shape()[1]),
          m_load(runtimeLoadMatrixBlockFunction<Accumulator>(view.type())) {}

    void load(size_t rowBase, size_t columnBase, size_t rows, size_t columns,
              std::span<Accumulator> destination) const {
        if (rowBase > m_rows || rows > m_rows - rowBase || columnBase > m_columns ||
            columns > m_columns - columnBase)
            throw std::out_of_range("Runtime matrix block exceeds the source tensor.");
        if (rows != 0 && columns > destination.size() / rows)
            throw std::invalid_argument("Runtime matrix block destination is too small.");
        m_load(m_storage, m_offset, m_rowStride, m_columnStride, rowBase, columnBase, rows, columns,
               destination);
    }

   private:
    std::span<const std::byte> m_storage;
    ptrdiff_t m_offset;
    ptrdiff_t m_rowStride;
    ptrdiff_t m_columnStride;
    size_t m_rows;
    size_t m_columns;
    RuntimeLoadMatrixBlockFunction<Accumulator> m_load;
};

template <typename Accumulator>
class RuntimeMatrixWriter {
   public:
    explicit RuntimeMatrixWriter(const Tensor& view)
        : m_storage(view.rawEncodedBackingStorage()),
          m_offset(view.layout().offset()),
          m_rowStride(view.layout().strides()[0]),
          m_columnStride(view.layout().strides()[1]),
          m_store(runtimeStoreFunction<Accumulator>(view.type())) {}

    void store(size_t row, size_t column, Accumulator value) const {
        m_store(m_storage,
                m_offset + static_cast<ptrdiff_t>(row) * m_rowStride +
                    static_cast<ptrdiff_t>(column) * m_columnStride,
                value);
    }

   private:
    std::span<std::byte> m_storage;
    ptrdiff_t m_offset;
    ptrdiff_t m_rowStride;
    ptrdiff_t m_columnStride;
    RuntimeStoreFunction<Accumulator> m_store;
};

template <typename Accumulator>
class RuntimeOutputConverter {
   public:
    RuntimeOutputConverter(ScalarType outputType, OutputConversion conversion)
        : m_outputType(outputType),
          m_conversion(conversion),
          m_load(runtimeLoadFunction<Accumulator>(outputType)),
          m_store(runtimeStoreFunction<Accumulator>(outputType)) {
        if (m_conversion == OutputConversion::SaturatingInt8 && m_outputType != ScalarType::Int8)
            throw std::invalid_argument(
                "Saturating output conversion requires an Int8 output tensor.");
    }

    Accumulator operator()(Accumulator value) const {
        if (m_conversion == OutputConversion::Default) {
            std::array<std::byte, 16> storage{};
            m_store(storage, 0, value);
            return m_load(storage, 0);
        }

        if constexpr (IsComplex<Accumulator>::value) {
            throw std::invalid_argument(
                "Saturating output conversion does not accept complex values.");
        } else if constexpr (std::is_integral_v<Accumulator>) {
            const Accumulator clamped =
                std::clamp(value, static_cast<Accumulator>(-128), static_cast<Accumulator>(127));
            return static_cast<Accumulator>(static_cast<int8_t>(clamped));
        } else {
            const long double rounded = std::nearbyint(static_cast<long double>(value));
            const long double clamped =
                std::clamp(rounded, static_cast<long double>(-128), static_cast<long double>(127));
            return static_cast<Accumulator>(static_cast<int8_t>(clamped));
        }
    }

   private:
    ScalarType m_outputType;
    OutputConversion m_conversion;
    RuntimeLoadFunction<Accumulator> m_load;
    RuntimeStoreFunction<Accumulator> m_store;
};

template <typename Accumulator>
class RuntimeMatrixOutputWriter {
   public:
    RuntimeMatrixOutputWriter(const Tensor& output, OutputConversion conversion)
        : m_defaultWriter(output),
          m_converter(output.type(), conversion),
          m_conversion(conversion) {}

    void store(size_t row, size_t column, Accumulator value) const {
        if (m_conversion == OutputConversion::Default)
            m_defaultWriter.store(row, column, value);
        else
            m_defaultWriter.store(row, column, m_converter(value));
    }

   private:
    RuntimeMatrixWriter<Accumulator> m_defaultWriter;
    RuntimeOutputConverter<Accumulator> m_converter;
    OutputConversion m_conversion;
};

template <typename Accumulator>
class RuntimeVectorReader {
   public:
    explicit RuntimeVectorReader(const Tensor& view)
        : m_storage(view.rawEncodedBackingStorage()),
          m_offset(view.layout().offset()),
          m_stride(view.layout().strides()[0]),
          m_load(runtimeLoadFunction<Accumulator>(view.type())) {}

    Accumulator operator[](size_t index) const {
        return m_load(m_storage, m_offset + static_cast<ptrdiff_t>(index) * m_stride);
    }

   private:
    std::span<const std::byte> m_storage;
    ptrdiff_t m_offset;
    ptrdiff_t m_stride;
    RuntimeLoadFunction<Accumulator> m_load;
};

template <typename Accumulator>
class RuntimeTensorReader {
   public:
    explicit RuntimeTensorReader(const Tensor& view)
        : m_storage(view.rawEncodedBackingStorage()),
          m_layout(view.layout()),
          m_load(runtimeLoadFunction<Accumulator>(view.type())) {}

    Accumulator operator()(std::span<const size_t> indices) const {
        return m_load(m_storage, m_layout.elementOffset(indices));
    }

   private:
    std::span<const std::byte> m_storage;
    Layout m_layout;
    RuntimeLoadFunction<Accumulator> m_load;
};

template <typename Accumulator>
class RuntimeTensorWriter {
   public:
    explicit RuntimeTensorWriter(const Tensor& view)
        : m_storage(view.rawEncodedBackingStorage()),
          m_layout(view.layout()),
          m_store(runtimeStoreFunction<Accumulator>(view.type())) {}

    void store(std::span<const size_t> indices, Accumulator value) const {
        m_store(m_storage, m_layout.elementOffset(indices), value);
    }

   private:
    std::span<std::byte> m_storage;
    Layout m_layout;
    RuntimeStoreFunction<Accumulator> m_store;
};

template <typename Accumulator>
class RuntimeQuantizer {
   public:
    RuntimeQuantizer() = default;

    explicit RuntimeQuantizer(std::optional<ScalarType> type) {
        if (!type || *type == nativeScalarType<Accumulator>) return;
        m_load = runtimeLoadFunction<Accumulator>(*type);
        m_store = runtimeStoreFunction<Accumulator>(*type);
    }

    Accumulator operator()(Accumulator value) const {
        if (m_load == nullptr) return value;
        std::array<std::byte, 16> storage{};
        m_store(storage, 0, value);
        return m_load(storage, 0);
    }

   private:
    RuntimeLoadFunction<Accumulator> m_load = nullptr;
    RuntimeStoreFunction<Accumulator> m_store = nullptr;
};
}  // namespace roc::host_numerics::detail
