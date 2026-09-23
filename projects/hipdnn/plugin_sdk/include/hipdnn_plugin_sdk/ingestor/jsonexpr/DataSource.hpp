// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// DataSource.hpp - the type-erased data source a compiled tree evaluates against.
//
// The compiled node tree evaluates against IDataSource instead of a concrete
// DataT, so the tree carries no template parameter and no per-DataT virtuals
// are instantiated. Expression<DataT> wraps the caller's object in a
// DataSourceAdapter at evaluation time.

#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Value.hpp>

#include <string>
#include <type_traits>
#include <utility>

namespace hipdnn_plugin_sdk::ingestor::jsonexpr::detail
{
// ---- data-source capability detection ------------------------------------
template <class T, class = void>
struct HasGetData : std::false_type
{
};
template <class T>
struct HasGetData<
    T,
    std::void_t<decltype(std::declval<const T&>().getData(std::declval<std::string>()))>>
    : std::true_type
{
};

// ---- type-erased data source ---------------------------------------------

/// The contract a compiled tree evaluates against.
///
/// getData resolves a variable path to a Value, or returns null when the path
/// does not resolve. Two obligations fall on the implementation, because
/// nothing downstream can take them back:
///
///   - An unresolved path is null, not a substitute value. Null is the
///     language's unresolved marker and it propagates; a stand-in 0 or false
///     would let a narrowing predicate pass on a field that was never read.
///   - The returned Value must nest no deeper than MAX_VALUE_DEPTH. Every
///     consumer of a Value walks it recursively, so a deeper one overflows the
///     stack inside the language rather than inside the accessor. A source
///     backed by an on-disk document must stop at the bound and yield null for
///     the subtree beneath it, as JsonDataSource does; the language reads that
///     as unresolved and the enclosing predicate declines.
///
/// The bound is not re-checked here. Doing so would cost a full traversal of
/// every value read, on every evaluation, to catch a mistake in one accessor.
struct IDataSource
{
    virtual ~IDataSource() = default;
    virtual Value getData(const std::string& path) const = 0;
};

template <class DataT>
struct DataSourceAdapter final : IDataSource
{
    static_assert(HasGetData<DataT>::value, "Data source must provide Value getData(std::string).");
    const DataT& data;
    explicit DataSourceAdapter(const DataT& d)
        : data(d)
    {
    }
    Value getData(const std::string& path) const override
    {
        return data.getData(path);
    }
};
} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr::detail

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
