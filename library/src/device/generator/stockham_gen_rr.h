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
            stmts += CommentLines{"append extra global loading for C2Real pre-process only"};

            StatementList stmts_c2real_pre;
            stmts_c2real_pre += CommentLines{
                "use the last thread of each transform to load one more element per row"};
            stmts_c2real_pre += If{
                thread == threads_per_transform - 1,
                {Assign{lds_complex[offset_lds + thread + (height - 1) * width + 1],
                        LoadGlobal{buf, offset + (thread + (height - 1) * width + 1) * stride0}}}};
            stmts += If{embedded_type == Literal{"EmbeddedType::C2Real_PRE"}, stmts_c2real_pre};
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
            stmts += CommentLines{"append extra global write for Real2C post-process only"};
            StatementList stmts_real2c_post;
            stmts_real2c_post += CommentLines{
                "use the last thread of each transform to write one more element per row"};
            stmts_real2c_post
                += If{Equal{thread, threads_per_transform - 1},
                      {StoreGlobal{buf,
                                   offset + (thread + (height - 1) * width + 1) * stride0,
                                   lds_complex[offset_lds + thread + (height - 1) * width + 1]}}};
            stmts += If{Equal{embedded_type, "EmbeddedType::Real2C_POST"}, stmts_real2c_post};
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

    StatementList real_trans_pre_post(ProcessingType type) override
    {
        std::string   pre_post   = (type == ProcessingType::PRE) ? " before " : " after ";
        auto          twd_offset = (length - factors.front());
        StatementList stmts;
        stmts += CommentLines{"handle even-length real to complex pre-process in lds" + pre_post
                              + "transform"};
        stmts += real2cmplx_pre_post(length, type, threads_per_transform, twd_offset);
        return stmts;
    }

    Function generate_device_function_with_bank_shift()
    {
        std::string function_name
            = "forward_length" + std::to_string(length) + "_" + tiling_name() + "_device";

        Function f{function_name};
        f.arguments = device_arguments();
        f.templates = device_templates();
        f.qualifier = "__device__";
        if(length == 1)
        {
            return f;
        }

        StatementList& body = f.body;
        body += Declaration{W};
        body += Declaration{t};
        body += Declaration{
            lstride, Ternary{Parens{stride_type == "SB_UNIT"}, Parens{1}, Parens{stride_lds}}};

        body += Declaration{l_offset};

        for(unsigned int npass = 0; npass < factors.size(); ++npass)
        {
            // width is the butterfly width, Radix-n. Mostly is used as dt in add_work()
            unsigned int width = factors[npass];
            // height is how many butterflies per thread will do on average
            float height = static_cast<float>(length) / width / threads_per_transform;

            unsigned int cumheight
                = product(factors.begin(),
                          factors.begin() + npass); // cumheight is irrelevant to the above height,
            // is used for twiddle multiplication and lds writing.

            body += LineBreak{};
            body += CommentLines{
                "pass " + std::to_string(npass) + ", width " + std::to_string(width),
                "using " + std::to_string(threads_per_transform) + " threads we need to do "
                    + std::to_string(length / width) + " radix-" + std::to_string(width)
                    + " butterflies",
                "therefore each thread will do " + std::to_string(height) + " butterflies"};

            auto load_lds  = std::mem_fn(&StockhamKernel::load_lds_generator);
            auto store_lds = std::mem_fn(&StockhamKernel::store_lds_generator);

            if(npass > 0)
            {
                // internal full lds2reg (both linear/nonlinear variants)
                StatementList lds2reg_full;
                lds2reg_full += SyncThreads();

                // NB:
                //   When lds conflict becomes significant enough, we can apply lds bank shift to reduce it.
                //   We enable it for small pow of 2 cases on all supported archs for now.
                if(length == 64)
                {
                    lds2reg_full += add_work(
                        std::bind(load_lds, this, _1, _2, _3, _4, _5, Component::BOTH, true),
                        width,
                        height,
                        ThreadGuardMode::NO_GUARD);
                }
                else
                {
                    lds2reg_full += add_work(
                        std::bind(load_lds, this, _1, _2, _3, _4, _5, Component::BOTH, false),
                        width,
                        height,
                        ThreadGuardMode::GUARD_BY_IF);
                }
                body += If{Not{lds_is_real}, lds2reg_full};

                auto apply_twiddle = std::mem_fn(&StockhamKernel::apply_twiddle_generator);
                body += add_work(
                    std::bind(apply_twiddle, this, _1, _2, _3, _4, _5, cumheight, factors.front()),
                    width,
                    height,
                    ThreadGuardMode::NO_GUARD);
            }

            auto butterfly = std::mem_fn(&StockhamKernel::butterfly_generator);
            body += add_work(std::bind(butterfly, this, _1, _2, _3, _4, _5),
                             width,
                             height,
                             ThreadGuardMode::NO_GUARD);

            if(npass == factors.size() - 1)
                body += large_twiddles_multiply(width, height, cumheight);

            // internal lds store (half-with-linear and full-with-linear/nonlinear)
            StatementList reg2lds_full;
            StatementList reg2lds_half;
            if(npass < factors.size() - 1)
            {
                // linear variant store (half) and load (half)
                for(auto component : {Component::REAL, Component::IMAG})
                {
                    bool isFirstStore = (npass == 0) && (component == Component::REAL);
                    auto half_width   = factors[npass];
                    auto half_height
                        = static_cast<float>(length) / half_width / threads_per_transform;
                    // minimize sync as possible
                    if(!isFirstStore)
                        reg2lds_half += SyncThreads();

                    if(length == 64)
                    {
                        reg2lds_half += add_work(
                            std::bind(
                                store_lds, this, _1, _2, _3, _4, _5, component, cumheight, true),
                            half_width,
                            half_height,
                            ThreadGuardMode::GUARD_BY_IF);
                    }
                    else
                    {
                        reg2lds_half += add_work(
                            std::bind(
                                store_lds, this, _1, _2, _3, _4, _5, component, cumheight, false),
                            half_width,
                            half_height,
                            ThreadGuardMode::GUARD_BY_IF);
                    }

                    half_width  = factors[npass + 1];
                    half_height = static_cast<float>(length) / half_width / threads_per_transform;
                    reg2lds_half += SyncThreads();
                    if(length == 64)
                    {
                        reg2lds_half += add_work(
                            std::bind(load_lds, this, _1, _2, _3, _4, _5, component, true),
                            half_width,
                            half_height,
                            ThreadGuardMode::GUARD_BY_IF);
                    }
                    else
                    {
                        reg2lds_half += add_work(
                            std::bind(load_lds, this, _1, _2, _3, _4, _5, component, false),
                            half_width,
                            half_height,
                            ThreadGuardMode::GUARD_BY_IF);
                    }
                }

                // internal full lds store (both linear/nonlinear variants)
                if(npass == 0)
                    reg2lds_full += If{!direct_load_to_reg, {SyncThreads()}};
                else
                    reg2lds_full += SyncThreads();

                if(length == 64)
                {
                    reg2lds_full += add_work(
                        std::bind(
                            store_lds, this, _1, _2, _3, _4, _5, Component::BOTH, cumheight, true),
                        width,
                        height,
                        ThreadGuardMode::GUARD_BY_IF);
                }
                else
                {
                    reg2lds_full += add_work(
                        std::bind(
                            store_lds, this, _1, _2, _3, _4, _5, Component::BOTH, cumheight, false),
                        width,
                        height,
                        ThreadGuardMode::GUARD_BY_IF);
                }

                body += If{Not{lds_is_real}, reg2lds_full};
                body += Else{reg2lds_half};
            }
        }

        return f;
    }
};
