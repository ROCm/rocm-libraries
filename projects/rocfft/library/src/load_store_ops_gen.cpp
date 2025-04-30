// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "device/generator/generator.h"
#include "load_store_ops.h"
#include "rtc_kernel.h"
#include "tree_node.h"

Function LoadOps::add_ops(const Function& f) const
{
    return f;
}

struct StoreOpsVisitor : public BaseVisitor
{
    StoreOpsVisitor(const StoreOps& ops)
        : ops(ops)
        , scale_factor("scale_factor", "const real_type_t<scalar_type>")
    {
    }

    Function visit_Function(const Function& x) override
    {
        if(!ops.enabled())
            return x;

        Function y{x};
        if(ops.scale_factor != 1.0)
        {
            Variable arg{"scale_factor", "const real_type_t<scalar_type>"};
            y.arguments.append(scale_factor);
        }
        return BaseVisitor::visit_Function(y);
    }

    template <typename TStatement>
    StatementList visit_Store(const TStatement& x)
    {
        if(!ops.enabled())
            return {x};

        TStatement y{x};
        y.value = y.value * scale_factor;
        return {y};
    }

    StatementList visit_StoreGlobal(const StoreGlobal& x) override
    {
        return visit_Store(x);
    }

    StatementList visit_StoreGlobalPlanar(const StoreGlobalPlanar& x) override
    {
        return visit_Store(x);
    }

    StatementList visit_IntrinsicStore(const IntrinsicStore& x) override
    {
        return visit_Store(x);
    }

    StatementList visit_IntrinsicStorePlanar(const IntrinsicStorePlanar& x) override
    {
        return visit_Store(x);
    }
    const StoreOps& ops;
    Variable        scale_factor;
};

Function StoreOps::add_ops(const Function& f) const
{
    auto visitor = StoreOpsVisitor{*this};
    return visitor(f);
}

std::string load_store_name_suffix(const LoadOps& loadOps, const StoreOps& storeOps)
{
    std::string suffix;
    suffix += loadOps.name_suffix();
    suffix += storeOps.name_suffix();
    return suffix;
}

void make_load_store_ops(Function& f, const LoadOps& loadOps, const StoreOps& storeOps)
{
    if(loadOps.enabled())
        f = loadOps.add_ops(f);
    if(storeOps.enabled())
        f = storeOps.add_ops(f);
}
