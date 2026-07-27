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

#include "../../shared/sha256.h"
#include "device/generator/generator.h"
#include "load_store_ops.h"
#include "rtc_kernel.h"
#include "tree_node.h"
#include <cstring>

std::string rocfft_spirv_cb_t::get_hash() const
{
    std::string ret;
    if(!symbol_name || !bitcode_data || !bitcode_len_bytes)
        return ret;
    // compute sha256 of symbol name + spirv code, output hex string as
    // this will go into the kernel's name

    sha256_buff state;
    sha256_init(&state);
    sha256_update(&state, symbol_name, std::strlen(symbol_name));
    sha256_update(&state, bitcode_data, bitcode_len_bytes);

    ret.resize(64);
    sha256_finalize(&state);
    sha256_read_hex(&state, ret.data());
    return ret;
}

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
        if(ops.scale_factor != 1.0)
        {
            y.value = y.value * scale_factor;
        }
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

std::string load_store_name_suffix(const std::optional<LoadOps>&  loadOps,
                                   const std::optional<StoreOps>& storeOps)
{
    std::string suffix;
    if(loadOps)
        suffix += loadOps->name_suffix();
    if(storeOps)
        suffix += storeOps->name_suffix();
    return suffix;
}

void make_load_store_ops(Function&                      f,
                         const std::optional<LoadOps>&  loadOps,
                         const std::optional<StoreOps>& storeOps)
{
    if(loadOps && loadOps->enabled())
    {
        f = loadOps->add_ops(f);
    }
    if(storeOps && storeOps->enabled())
    {
        f = storeOps->add_ops(f);
    }
}

std::string load_store_decls(const std::optional<LoadOps>&  loadOps,
                             const std::optional<StoreOps>& storeOps,
                             const CallbackType             cbtype,
                             const char*                    load_data_type,
                             const char*                    store_data_type)
{
    // FIXME: need to namespace things so user-chosen symbols can't
    // (easily) collide with our internal syms
    std::string ops_declarations;
    if(loadOps && loadOps->has_spirv())
    {
        ops_declarations += loadOps->forward_decls(cbtype, load_data_type);
    }
    if(storeOps && storeOps->has_spirv())
    {
        ops_declarations += storeOps->forward_decls(cbtype, store_data_type);
    }
    return ops_declarations;
}
