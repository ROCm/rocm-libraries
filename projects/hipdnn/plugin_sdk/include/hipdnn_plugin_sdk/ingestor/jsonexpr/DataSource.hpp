// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// DataSource.hpp - the type-erased data source a compiled tree evaluates against.
//
// The node tree targets IDataSource rather than a concrete DataT, so the tree
// carries no template parameter and instantiates no per-DataT virtuals.
// Expression<DataT> wraps the caller's object in a DataSourceAdapter at
// evaluation time.

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
// The compiled node tree evaluates against this abstract source rather than a
// concrete DataT, so the tree itself carries no template parameter (and thus
// no per-DataT virtual member instantiation). Expression<DataT> wraps the
// caller's data object in a DataSourceAdapter at evaluation time.
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
