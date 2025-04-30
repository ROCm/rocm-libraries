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

// inherits RR for convenient access to variables to generate global
// function with, but contains RR kernels for the device functions
// since they have distinct specs
struct StockhamKernelFused2D : public StockhamKernelRR
{
    StockhamKernelFused2D(const StockhamGeneratorSpecs& specs0,
                          const StockhamGeneratorSpecs& specs1)
        : StockhamKernelRR(specs0)
        , kernel0(specs0)
        , kernel1(specs1)
    {
        threads_per_transform = std::max(specs1.length * specs0.threads_per_transform,
                                         specs0.length * specs1.threads_per_transform);
        // 2D_SINGLE does one 2D slab per workgroup(threadblock)
        workgroup_size       = threads_per_transform;
        transforms_per_block = 1;
        R.size               = std::max(kernel0.nregisters, kernel1.nregisters);
        kernel0.writeGuard   = true;
        kernel1.writeGuard   = true;
        // 2D kernels use kernel0 and kernel1 device functions,
        // so this writeGuard value is not used and irrelevant
        writeGuard = true;
        // // 2D_SINGLEs haven't implemented this (global function)
        // direct_to_from_reg = false;
    }

    StockhamKernelRR kernel0;
    StockhamKernelRR kernel1;

    std::vector<unsigned int> launcher_lengths() override
    {
        return {kernel0.length, kernel1.length};
    }
    std::vector<unsigned int> launcher_factors() override
    {
        std::vector<unsigned int> ret;
        std::copy(kernel0.factors.begin(), kernel0.factors.end(), std::back_inserter(ret));
        std::copy(kernel1.factors.begin(), kernel1.factors.end(), std::back_inserter(ret));
        return ret;
    }

    std::vector<Expression> device_lds_reg_inout_device_call_arguments() override
    {
        return {R, lds_complex, stride_lds, offset_lds, thread_in_device, write};
    }

    std::vector<Expression> device_call_arguments(unsigned int call_iter) override
    {
        return {R,
                lds_real,
                lds_complex,
                twiddles,
                stride_lds,
                call_iter ? Expression{offset_lds + call_iter * stride_lds * transforms_per_block}
                          : Expression{offset_lds},
                thread_in_device,
                write};
    }

    StatementList load_from_global(bool load_registers) override
    {
        StatementList stmts;

        if(!load_registers)
        {
            auto length0  = kernel0.length;
            auto length1  = kernel1.length;
            auto rw_iters = length0 * length1 / workgroup_size;

            StatementList load_stmts;
            StatementList load_trans_stmts;

            // normal load: length0 is fastest dim (row-length), length1 is strided (column-length)
            {
                load_stmts += CommentLines{
                    "load length-" + std::to_string(length0) + " rows using all threads.",
                    "need " + std::to_string(rw_iters) + " iterations to load all "
                        + std::to_string(length1) + " rows in the slab"};
                // just use rw_iters * workgroup_size threads total, break
                // it down into row/column accesses to fill LDS
                for(unsigned int i = 0; i < rw_iters; ++i)
                {
                    auto row_offset = Parens{(i * workgroup_size + thread_id) / length0};
                    auto col_offset = Parens{(i * workgroup_size + thread_id) % length0};
                    load_stmts += Assign{
                        lds_complex[row_offset * stride_lds + col_offset],
                        LoadGlobal{buf, offset + col_offset * stride[0] + row_offset * stride[1]}};
                }
            }

            // load when shape is transposed (used for C2Real_PRE)
            // length0 is strided dim (column-length), length1 is fastest (row-length)
            {
                load_trans_stmts += CommentLines{
                    "load length-" + std::to_string(length1) + " rows using all threads.",
                    "need " + std::to_string(rw_iters) + " iterations to load all "
                        + std::to_string(length0) + " rows in the slab"};
                for(unsigned int i = 0; i < rw_iters; ++i)
                {
                    auto row_offset = Parens{(i * workgroup_size + thread_id) / length1};
                    auto col_offset = Parens{(i * workgroup_size + thread_id) % length1};
                    load_trans_stmts += Assign{
                        lds_complex[row_offset * stride_lds + col_offset],
                        LoadGlobal{buf, offset + col_offset * stride[1] + row_offset * stride[0]}};
                }

                load_trans_stmts
                    += CommentLines{"append extra global loading for C2Real pre-process only"};
                load_trans_stmts
                    += If{Less{thread_id, length0},
                          {Assign{lds_complex[thread_id * stride_lds + length1],
                                  LoadGlobal{buf, offset + thread_id * stride[0] + length1}}}};
            }

            stmts += If{embedded_type != "EmbeddedType::C2Real_PRE", load_stmts};
            stmts += Else{load_trans_stmts};
        }
        else
        {
            // haven't supported yet
        }

        return stmts;
    }

    StatementList store_to_global(bool store_registers) override
    {
        StatementList stmts;

        if(!store_registers)
        {
            auto length0  = kernel0.length;
            auto length1  = kernel1.length;
            auto rw_iters = length0 * length1 / workgroup_size;

            StatementList store_stmts;
            StatementList store_trans_stmts;

            // normal store: length0 is fastest dim (row-length), length1 is strided (column-length)
            {
                store_stmts += CommentLines{
                    "store length-" + std::to_string(length0) + " rows using all threads.",
                    "need " + std::to_string(rw_iters) + " iterations to store all "
                        + std::to_string(length1) + " rows in the slab"};

                // just use rw_iters * workgroup_size threads total, break
                // it down into row/column accesses to fill LDS
                for(unsigned int i = 0; i < rw_iters; ++i)
                {
                    auto row_offset = Parens{(i * workgroup_size + thread_id) / length0};
                    auto col_offset = Parens{(i * workgroup_size + thread_id) % length0};
                    store_stmts
                        += StoreGlobal{buf,
                                       offset + col_offset * stride[0] + row_offset * stride[1],
                                       lds_complex[row_offset * stride_lds + col_offset]};
                }

                store_stmts += LineBreak{};
                store_stmts += CommentLines{"append extra global store for Real2C "
                                            "post-process only, one more element per row."};
                store_stmts
                    += If{Equal{embedded_type, "EmbeddedType::Real2C_POST"},
                          {If{Less{thread_id, length1},
                              {StoreGlobal{buf,
                                           offset + (thread_id * stride[1]) + length0,
                                           lds_complex[(thread_id * stride_lds) + length0]}}}}};
            }

            // store when shape is transposed (for C2Real_PRE)
            // length0 is strided dim (column-length), length1 is fastest (row-length)
            {
                store_trans_stmts += CommentLines{
                    "store length-" + std::to_string(length1) + " rows using all threads.",
                    "need " + std::to_string(rw_iters) + " iterations to store all "
                        + std::to_string(length0) + " rows in the slab"};
                for(unsigned int i = 0; i < rw_iters; ++i)
                {
                    auto row_offset = Parens{(i * workgroup_size + thread_id) / length1};
                    auto col_offset = Parens{(i * workgroup_size + thread_id) % length1};
                    store_trans_stmts
                        += StoreGlobal{buf,
                                       offset + col_offset * stride[1] + row_offset * stride[0],
                                       lds_complex[row_offset * stride_lds + col_offset]};
                }
            }

            stmts += If{embedded_type != "EmbeddedType::C2Real_PRE", store_stmts};
            stmts += Else{store_trans_stmts};
        }
        else
        {
            // haven't supported yet
        }

        return stmts;
    }

    StatementList real_trans_pre_post(ProcessingType type) override
    {
        std::string pre_post     = (type == ProcessingType::PRE) ? " before " : " after ";
        auto        pre_post_len = (type == ProcessingType::PRE) ? kernel1.length : kernel0.length;
        auto        pre_post_tpt = (type == ProcessingType::PRE) ? kernel1.threads_per_transform
                                                                 : kernel0.threads_per_transform;

        auto twd_offset = (kernel0.length - kernel0.factors.front());
        // if two twd tables are different, then we advance the offset to the end of table2
        // else, we actually save only one twd table
        if(kernel0.factors != kernel1.factors)
            twd_offset += (kernel1.length - kernel1.factors.front());

        StatementList stmts;
        stmts += CommentLines{"handle even-length real to complex pre-process in lds" + pre_post
                              + "transform"};
        stmts += real2cmplx_pre_post(pre_post_len, type, pre_post_tpt, twd_offset);
        return stmts;
    }

    Function generate_global_function() override
    {
        auto is_pow2 = [](unsigned int n) { return n != 0 && (n & (n - 1)) == 0; };

        auto length0 = kernel0.length;
        auto length1 = kernel1.length;

        // for pow2, add padding to avoid bank conflict
        auto length0_padded = is_pow2(length0) ? (length0 + 1) : length0;

        // for ebtype::post case, add an extra padding first and then check if pow2
        auto length0_ebtype_post = length0 + 1;
        auto length0_ebtype_post_padded
            = is_pow2(length0_ebtype_post) ? (length0_ebtype_post + 1) : length0_ebtype_post;

        // for ebtype::pre case, add an extra padding first and then check if pow2
        auto length1_ebtype_pre = length1 + 1;
        auto length1_ebtype_pre_padded
            = is_pow2(length1_ebtype_pre) ? (length1_ebtype_pre + 1) : length1_ebtype_pre;

        Function f{"forward_length" + std::to_string(length) + "x"
                   + std::to_string(kernel1.length)};

        StatementList& body = f.body;
        body += LineBreak{};
        body += CommentLines{"",
                             "this kernel:",
                             "  uses " + std::to_string(threads_per_transform)
                                 + " threads per 2d transform",
                             "  does 1 2d transforms per thread block",
                             "therefore it should be called with " + std::to_string(workgroup_size)
                                 + " threads per block",
                             ""};

        Variable d{"d", "int"};
        Variable index_along_d{"index_along_d", "size_t"};
        Variable remaining{"remaining", "size_t"};
        Variable plength{"plength", "size_t"};

        Variable batch0{"batch0", "size_t"};

        Variable SB_1ST{"SB_1ST", "const StrideBin"};
        Variable SB_2ND{"SB_2ND", "const StrideBin"};

        body += LDSDeclaration{scalar_type.name};
        body += Declaration{R};
        body += Declaration{transform};
        body += Declaration{offset, 0};
        body += Declaration{offset_lds};
        body += Declaration{stride_lds};
        body += Declaration{write};
        body += Declaration{batch0};
        body += Declaration{remaining};
        body += Declaration{plength, 1};
        body += Declaration{index_along_d};
        body += Declaration{lds_is_real, "false"};
        body += Declaration{lds_linear, "true"};
        body += Declaration{direct_load_to_reg, "false"};
        body += Declaration{direct_store_from_reg, "false"};
        body += CallbackLoadDeclaration{scalar_type.name, callback_type.name};
        body += CallbackStoreDeclaration{scalar_type.name, callback_type.name};
        body += Declaration{
            SB_1ST,
            Ternary{Parens(embedded_type == "EmbeddedType::C2Real_PRE"), "SB_NONUNIT", "SB_UNIT"}};
        body += Declaration{
            SB_2ND,
            Ternary{Parens(embedded_type == "EmbeddedType::C2Real_PRE"), "SB_UNIT", "SB_NONUNIT"}};

        body += LineBreak{};
        body += CommentLines{"transform is: 2D slab number (1 per block)"};
        body += Assign{transform, block_id};
        body += Assign{remaining, transform};
        body += CommentLines{"compute 2D slab offset (start from length/stride index 2)"};

        if(static_dim)
        {
            body += Declaration{dim, static_dim};
        }
        body += For{d,
                    2,
                    d < dim,
                    1,
                    {Assign{plength, plength * lengths[d]},
                     Assign{index_along_d, remaining % lengths[d]},
                     Assign{remaining, remaining / lengths[d]},
                     Assign{offset, offset + index_along_d * stride[d]}}};
        body += Assign{batch0, transform / plength};
        body += CommentLines{"offset is the starting global-mem-ptr of the entire 2D-BLOCK"};
        body += Assign{offset, offset + batch0 * stride[dim]};

        body += Assign{stride_lds,
                       Ternary{Parens(embedded_type == "EmbeddedType::Real2C_POST"),
                               length0_ebtype_post_padded,
                               Ternary{Parens(embedded_type == "EmbeddedType::C2Real_PRE"),
                                       length1_ebtype_pre_padded,
                                       length0_padded}}};

        // load
        body += LineBreak{};
        body += load_from_global(false);

        // -------------
        // length0 part
        // -------------
        body += LineBreak{};
        body += CommentLines{"", "length: " + std::to_string(length0), ""};

        auto height              = kernel0.threads_per_transform;
        auto active_threads_rows = kernel0.threads_per_transform * kernel1.length;

        body += CommentLines{
            "this dim: length-" + std::to_string(length0),
            "  uses " + std::to_string(kernel0.threads_per_transform)
                + " threads per row-transform of length-" + std::to_string(length0),
            "  does " + std::to_string(length1) + " row-transforms per thread block",
            "therefore it uses " + std::to_string(active_threads_rows) + " active threads",
            "row-elem is contigous (SB_UNIT)",
            "NOTE: if this is a fused C2Real_PRE, then the shape is transposed"};

        if(active_threads_rows == workgroup_size)
            body += Assign{write, 1};
        else
            body += Assign{write, thread_id < active_threads_rows};

        body += CommentLines{"offset_lds is the starting lds-ptr of each ROW (or COL if C2R_PRE)"};
        body += Assign{offset_lds,
                       Ternary{Parens(embedded_type != "EmbeddedType::C2Real_PRE"),
                               stride_lds * (thread_id / height),
                               thread_id / height}};
        body += CommentLines{"calc the thread_in_device value once and for all device funcs"};
        body += Declaration{thread_in_device,
                            Ternary{lds_linear,
                                    thread_id % kernel0.threads_per_transform,
                                    thread_id / kernel0.transforms_per_block}};

        body += LineBreak{};

        StatementList dim1_work_stmts;
        dim1_work_stmts += CommentLines{"call a pre-load from lds to registers (if necessary)"};
        auto pre_post_lds_tmpl0 = device_lds_reg_inout_device_call_templates();
        auto pre_post_lds_args0 = device_lds_reg_inout_device_call_arguments();
        pre_post_lds_tmpl0.set_value(stride_type.name, "SB_1ST");
        dim1_work_stmts += Call{"lds_to_reg_input_length" + std::to_string(length0) + "_device",
                                pre_post_lds_tmpl0,
                                pre_post_lds_args0};
        dim1_work_stmts += LineBreak{};

        auto templates = device_call_templates();
        templates.set_value(stride_type.name, "SB_1ST");
        dim1_work_stmts += Call{"forward_length" + std::to_string(length0) + "_SBRR_device",
                                templates,
                                device_call_arguments(0)};
        dim1_work_stmts += LineBreak{};

        dim1_work_stmts += CommentLines{"call a post-store from registers to lds (if necessary)"};
        dim1_work_stmts += Call{"lds_from_reg_output_length" + std::to_string(length0) + "_device",
                                pre_post_lds_tmpl0,
                                pre_post_lds_args0};

        // if embedded C2Real_PRE, then dim 1 has to do an extra transform
        {
            StatementList dim1_work_pre_stmts;
            dim1_work_pre_stmts += Assign{offset_lds, length1};
            dim1_work_pre_stmts += Assign{thread_in_device, thread_id};
            dim1_work_pre_stmts += Assign{write, thread_id < kernel0.threads_per_transform};
            dim1_work_pre_stmts += dim1_work_stmts;

            body += dim1_work_stmts;
            body += LineBreak{};
            body += CommentLines{"if C2Real_PRE, then we have to do an extra transform"};
            body += If{Equal{embedded_type, "EmbeddedType::C2Real_PRE"}, dim1_work_pre_stmts};
        }

        // handle even-length complex to real post-process in lds after transform
        body += real_trans_pre_post(ProcessingType::POST);

        // note there is a syncthreads at the start of the next call
        // -------------
        // length1 part
        // -------------
        body += CommentLines{"", "length: " + std::to_string(length1), ""};

        height                   = kernel1.threads_per_transform;
        auto active_threads_cols = kernel1.threads_per_transform * length0;

        body += CommentLines{
            "this dim: length-" + std::to_string(length1),
            "  uses " + std::to_string(kernel1.threads_per_transform)
                + " threads per col-transform of length-" + std::to_string(length1),
            "  does " + std::to_string(length0) + " col-transforms per thread block",
            "therefore it uses " + std::to_string(active_threads_cols) + " active threads",
            "col-elem is non-contigous (SB_NONUNIT), each elem is strided with stride_lds",
            "NOTE: if this is a fused C2Real_PRE, then the shape is transposed"};

        if(active_threads_cols == workgroup_size)
            body += Assign{write, 1};
        else
            body += Assign{write, thread_id < active_threads_cols};
        body += CommentLines{"offset_lds is the starting lds-ptr of each COL (or ROW if R2C_POST)"};
        body += Assign{offset_lds,
                       Ternary{Parens(embedded_type != "EmbeddedType::C2Real_PRE"),
                               thread_id / height,
                               stride_lds * (thread_id / height)}};

        body += CommentLines{"calc the thread_in_device value once and for all device funcs"};
        body += Assign{thread_in_device,
                       Ternary{lds_linear,
                               thread_id % kernel1.threads_per_transform,
                               thread_id / kernel1.transforms_per_block}};

        // handle even-length real to complex pre-process in lds before transform
        body += real_trans_pre_post(ProcessingType::PRE);

        body += LineBreak{};

        StatementList dim2_work_stmts;
        dim2_work_stmts += CommentLines{"call a pre-load from lds to registers (if necessary)"};
        auto pre_post_lds_tmpl1 = device_lds_reg_inout_device_call_templates();
        auto pre_post_lds_args1 = device_lds_reg_inout_device_call_arguments();
        pre_post_lds_tmpl1.set_value(stride_type.name, "SB_2ND");
        dim2_work_stmts += Call{"lds_to_reg_input_length" + std::to_string(length1) + "_device",
                                pre_post_lds_tmpl1,
                                pre_post_lds_args1};
        dim2_work_stmts += LineBreak{};

        auto templates2 = device_call_templates();
        templates2.set_value(stride_type.name, "SB_2ND");
        auto arguments2 = device_call_arguments(0);
        if(factors != kernel1.factors)
            arguments2[3] = twiddles + length0 - factors.front();
        dim2_work_stmts += Call{
            "forward_length" + std::to_string(length1) + "_SBRR_device", templates2, arguments2};
        dim2_work_stmts += LineBreak{};

        dim2_work_stmts += CommentLines{"call a post-store from registers to lds (if necessary)"};
        dim2_work_stmts += Call{"lds_from_reg_output_length" + std::to_string(length1) + "_device",
                                pre_post_lds_tmpl1,
                                pre_post_lds_args1};

        // if embedded post, then dim 2 has to do an extra column
        {
            StatementList dim2_work_post_stmts;
            dim2_work_post_stmts += Assign{offset_lds, length0};
            dim2_work_post_stmts += Assign{thread_in_device, thread_id};
            dim2_work_post_stmts += Assign{write, thread_id < kernel1.threads_per_transform};
            dim2_work_post_stmts += dim2_work_stmts;

            body += dim2_work_stmts;
            body += LineBreak{};
            body += CommentLines{"if Real2C_POST, then we have to do an extra transform"};
            body += If{Equal{embedded_type, "EmbeddedType::Real2C_POST"}, dim2_work_post_stmts};
        }

        // store
        body += SyncThreads{};
        body += store_to_global(false);

        f.qualifier     = "__global__";
        f.templates     = global_templates();
        f.arguments     = global_arguments();
        f.launch_bounds = workgroup_size;
        return f;
    }
};
