// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// JsonExpression.hpp - the JSON Expression Language's public surface.
//
// All names below live in namespace hipdnn_plugin_sdk::ingestor::jsonexpr;
// these examples assume `namespace jexpr = hipdnn_plugin_sdk::ingestor::jsonexpr;`.
//
// An expression (an nlohmann::json value) is *compiled once* into a reusable
// jexpr::Expression<Data>, then evaluated many times against different data
// sources:
//
//     struct MyData {                       // your data source
//         jexpr::Value getData(const std::string& path) const;
//     };
//
//     auto expr = jexpr::compile<MyData>(rule);   // parse + build tree once
//     jexpr::Value r1 = expr(dataA);              // evaluate, no re-parse
//     jexpr::Value r2 = expr(dataB);              // reuse for other data
//
// Two conventions drive most of the design, and both are documented where they
// are implemented rather than restated here:
//
//   - A string prefixed with '$' is a variable reference and the only way to
//     read data; there is no `var` operator. See Compiler.hpp.
//   - Null is "unresolved", not a value: an unresolved path means a field is
//     absent, so operators propagate null rather than coercing it to
//     false/0/not-equal and silently passing a narrowing predicate. `and`/`or`
//     are correspondingly three-valued. See Operators.hpp.
//
// This header is the entry point; the implementation is split by layer under
// jsonexpr/, each header documenting its own piece:
//
//     jsonexpr/Error.hpp           JsonExpressionCompileError
//     jsonexpr/Value.hpp           the runtime value type
//     jsonexpr/DataSource.hpp      the type-erased data-source contract
//     jsonexpr/Node.hpp            compiled tree nodes
//     jsonexpr/Operators.hpp       one function per operator
//     jsonexpr/OperatorTable.hpp   the operator table, and OpNode
//     jsonexpr/LayoutAliases.hpp   the `stride_order` layout-name pre-pass
//     jsonexpr/Compiler.hpp        json -> node tree
//     jsonexpr/VarIterator.hpp     iteration over referenced variables
//
// Full reference: docs/JsonExpression.md.

#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Compiler.hpp>
#include <hipdnn_plugin_sdk/ingestor/jsonexpr/DataSource.hpp>
#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Error.hpp>
#include <hipdnn_plugin_sdk/ingestor/jsonexpr/LayoutAliases.hpp>
#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Node.hpp>
#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Value.hpp>
#include <hipdnn_plugin_sdk/ingestor/jsonexpr/VarIterator.hpp>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>

namespace hipdnn_plugin_sdk::ingestor::jsonexpr
{
// ===========================================================================
// Expression - a compiled, reusable expression
// ===========================================================================
template <class DataT>
class Expression
{
public:
    Expression() = default;
    explicit Expression(detail::NodePtr root)
        : _root(std::move(root))
    {
    }

    /// Evaluate against a data source. Cheap: walks the pre-compiled tree.
    Value operator()(const DataT& data) const
    {
        if(!_root)
        {
            return {};
        }
        const detail::DataSourceAdapter<DataT> source(data);
        return _root->eval(source);
    }
    Value evaluate(const DataT& data) const
    {
        return (*this)(data);
    }

    explicit operator bool() const
    {
        return static_cast<bool>(_root);
    }

    /// A lazy, pre-order range over every variable path referenced in the
    /// expression. References point into the live tree, so the range must not
    /// outlive this Expression. Duplicates are yielded as they occur;
    /// construct a std::set from the range for the unique, sorted set.
    detail::VarRange variables() const
    {
        return detail::VarRange(_root.get());
    }

private:
    detail::NodePtr _root;
};

/// Compile a rule into a reusable Expression bound to data source
/// type DataT. Throws JsonExpressionCompileError on malformed rules.
///
/// Layout aliases ("nhwc" and friends) opposite a `stride_order` reference are
/// expanded to their canonical integer arrays first, so the compiled tree and
/// evaluation see only arrays.
template <class DataT>
Expression<DataT> compile(const nlohmann::json& rule)
{
    std::map<std::string, std::int64_t> rankPins;
    detail::collectRankPins(rule, rankPins);
    const nlohmann::json expanded = detail::expandLayoutAliases(rule, rankPins);
    return Expression<DataT>(detail::compileNode(expanded));
}

/// True when any variable referenced by `expr` has `root` as its first path
/// token (the segment before the first '.'/'[' separator). The paths yielded by
/// Expression::variables() are already sigil-stripped, so `root` is given
/// without the sigil (e.g. "kernel"). Short-circuits on the first match.
template <class DataT>
bool referencesVariableRoot(const Expression<DataT>& expr, std::string_view root)
{
    for(const std::string& path : expr.variables())
    {
        const std::size_t end = path.find_first_of(".[");
        const std::string_view first(path.data(), end == std::string::npos ? path.size() : end);
        if(first == root)
        {
            return true;
        }
    }
    return false;
}

} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
