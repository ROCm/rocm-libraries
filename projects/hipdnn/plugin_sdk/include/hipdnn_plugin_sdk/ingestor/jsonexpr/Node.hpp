// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// Node.hpp - the compiled expression tree's node types.
//
// A rule compiles once into this tree, which is then evaluated many times.
// OpNode lives in OperatorTable.hpp, beside the table that gives it meaning.

#include <hipdnn_plugin_sdk/ingestor/jsonexpr/DataSource.hpp>
#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Value.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor::jsonexpr::detail
{
// ---- compiled node tree ---------------------------------------------------

struct Node
{
    virtual ~Node() = default;
    virtual Value eval(const IDataSource& data) const = 0;
    virtual const std::string* variable() const
    {
        return nullptr;
    }
    virtual void pushChildren(std::vector<const Node*>& /*unused*/) const {}
};

using NodePtr = std::unique_ptr<Node>;

struct LiteralNode final : Node
{
    Value value;
    explicit LiteralNode(Value v)
        : value(std::move(v))
    {
    }
    Value eval(const IDataSource& /*unused*/) const override
    {
        return value;
    }
};

struct ArrayNode final : Node
{
    std::vector<NodePtr> items;
    Value eval(const IDataSource& data) const override
    {
        Value::Array a;
        a.reserve(items.size());
        for(const auto& it : items)
        {
            Value item = it->eval(data);
            if(item.containsUnresolved())
            {
                return {};
            }
            a.push_back(std::move(item));
        }
        return {std::move(a)};
    }
    void pushChildren(std::vector<const Node*>& stack) const override
    {
        for(auto it = items.rbegin(); it != items.rend(); ++it)
        {
            stack.push_back(it->get());
        }
    }
};

struct VarNode final : Node
{
    std::string path;

    Value eval(const IDataSource& data) const override
    {
        return data.getData(path);
    }

    const std::string* variable() const override
    {
        return &path;
    }
};
} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr::detail

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
