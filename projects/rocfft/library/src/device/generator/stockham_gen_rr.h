// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once
#include "../../../../shared/arithmetic.h"
#include "stockham_gen_base.h"

struct StockhamKernelRR : public StockhamKernel
{
    explicit StockhamKernelRR(const StockhamGeneratorSpecs& specs)
        : StockhamKernel(specs)
    {
    }

    // TODO- check if using uint in device is also better
    Variable thread{"thread", "unsigned int"}; // use type uint in global
    Variable inbound{"inbound", "bool"};

    std::string tiling_name() override
    {
        return "SBRR";
    }

    StatementList calculate_offsets() override
    {
        Variable d{"d", "int"};
        Variable index_along_d{"index_along_d", "size_t"};
        Variable remaining{"remaining", "size_t"};

        StatementList stmts;
        stmts += Declaration{thread};
        stmts += Declaration(remaining);
        stmts += Declaration(index_along_d);
        stmts += Assign{transform,
                        block_id * transforms_per_block + thread_id / threads_per_transform};
        stmts += Assign{remaining, transform};

        stmts += For{d,
                     1,
                     d < dim,
                     1,
                     {Assign{index_along_d, remaining % lengths[d]},
                      Assign{remaining, remaining / lengths[d]},
                      Assign{offset, offset + index_along_d * stride[d]}}};

        stmts += Assign{batch, remaining};
        stmts += Assign{offset, offset + batch * stride[dim]};
        stmts += Assign{stride_lds, (length + get_lds_padding())};
        stmts += Assign{offset_lds, stride_lds * Parens{transform % transforms_per_block}};

        stmts += Declaration{inbound, batch < nbatch};

        return stmts;
    }

    StatementList load_from_global(bool load_registers) override
    {
        StatementList stmts;
        stmts += Assign{thread, thread_id % threads_per_transform};

        if(!load_registers)
        {
            unsigned int width  = threads_per_transform;
            unsigned int height = length / width;

            for(unsigned int h = 0; h < height; ++h)
            {
                auto idx = thread + h * width;
                stmts += Assign{lds_complex[offset_lds + idx],
                                LoadGlobal{buf, offset + idx * stride0}};
            }
            stmts += LineBreak();

            StatementList stmts_c2real_pre;
            stmts_c2real_pre += CommentLines{
                "use the last thread of each transform to load one more element per row"};
            stmts_c2real_pre += If{
                thread == threads_per_transform - 1,
                {Assign{lds_complex[offset_lds + thread + (height - 1) * width + 1],
                        LoadGlobal{buf, offset + (thread + (height - 1) * width + 1) * stride0}}}};
            if(ebtype == EmbeddedType::C2Real_PRE)
            {
                stmts += CommentLines{"append extra global loading for C2Real pre-process only"};
                stmts += stmts_c2real_pre;
            }
        }
        else
        {
            unsigned int width  = factors[0];
            auto         height = static_cast<float>(length) / width / threads_per_transform;

            auto load_global = std::mem_fn(&StockhamKernel::load_global_generator);
            stmts += add_work(std::bind(load_global, this, _1, _2, _3, _4, _5),
                              width,
                              height,
                              ThreadGuardMode::GUARD_BY_IF);
        }

        return {If{inbound, stmts}};
    }

    StatementList store_to_global(bool store_registers) override
    {
        StatementList stmts;

        if(!store_registers)
        {
            auto width  = threads_per_transform;
            auto height = length / width;
            for(unsigned int h = 0; h < height; ++h)
            {
                auto idx = thread + h * width;
                stmts += StoreGlobal{buf, offset + idx * stride0, lds_complex[offset_lds + idx]};
            }

            stmts += LineBreak{};
            StatementList stmts_real2c_post;
            stmts_real2c_post += CommentLines{
                "use the last thread of each transform to write one more element per row"};
            stmts_real2c_post
                += If{Equal{thread, threads_per_transform - 1},
                      {StoreGlobal{buf,
                                   offset + (thread + (height - 1) * width + 1) * stride0,
                                   lds_complex[offset_lds + thread + (height - 1) * width + 1]}}};

            if(ebtype == EmbeddedType::Real2C_POST)
            {
                stmts += CommentLines{"append extra global write for Real2C post-process only"};
                stmts += stmts_real2c_post;
            }
        }
        else
        {
            auto width     = factors.back();
            auto cumheight = product(factors.begin(), factors.begin() + (factors.size() - 1));
            auto height    = static_cast<float>(length) / width / threads_per_transform;

            auto store_global = std::mem_fn(&StockhamKernel::store_global_generator);
            stmts += add_work(std::bind(store_global, this, _1, _2, _3, _4, _5, cumheight),
                              width,
                              height,
                              ThreadGuardMode::GUARD_BY_IF);
        }

        return {If{inbound, stmts}};
    }

    StatementList real_trans_pre_post() override
    {
        if(ebtype == EmbeddedType::NONE)
            return {};

        std::string   pre_post   = (ebtype == EmbeddedType::C2Real_PRE) ? " before " : " after ";
        auto          twd_offset = (length - factors.front());
        StatementList stmts;
        stmts += CommentLines{"handle even-length real to complex pre-process in lds" + pre_post
                              + "transform"};
        stmts += real2cmplx_pre_post(length, threads_per_transform, twd_offset);
        return stmts;
    }
};
