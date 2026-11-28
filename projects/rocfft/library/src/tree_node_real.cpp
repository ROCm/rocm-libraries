// Copyright (C) 2021 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "tree_node_real.h"
#include "../../shared/arithmetic.h"
#include "../../shared/precision_type.h"
#include "function_pool.h"
#include "node_factory.h"
#include <algorithm>
#include <array>

namespace
{
    constexpr auto INPUT  = io_data_label::INPUT;
    constexpr auto OUTPUT = io_data_label::OUTPUT;

    constexpr std::array<size_t, 2> swap_xy{1, 0};
    constexpr std::array<size_t, 3> swap_yz{0, 2, 1};
    constexpr std::array<size_t, 3> left_xyz{1, 2, 0};
    constexpr std::array<size_t, 3> right_xyz{2, 0, 1};

    data_set_t real_as_complex(const data_set_t& real)
    {
        if(real.has_array_type() && real.get_array_type() == rocfft_array_type_real)
            return real.reinterpret_as(rocfft_array_type_complex_interleaved);

        // Provisional children can be untyped; buffer assignment may also give
        // the real side an alias type without changing its real-element strides.
        return real.with_element_type(rocfft_array_type_real, real.get_precision())
            .reinterpret_as(rocfft_array_type_complex_interleaved);
    }

    data_set_t untyped(const data_set_t& layout)
    {
        return layout.with_element_type(rocfft_array_type_unset, layout.get_precision());
    }

    data_set_t reorder_lengths(data_set_t layout, std::span<const size_t> order)
    {
        layout.reorder_axes(order, axis_scope::length_only);
        return layout;
    }

    std::unique_ptr<TreeNode> make_real_row(TreeNode* parent)
    {
        return NodeFactory::CreateNodeFromScheme(
            CS_REAL_TRANSFORM_EVEN,
            parent,
            untyped(parent->input_layout().get_data_set_for_len_axes({0})),
            untyped(parent->output_layout().get_data_set_for_len_axes({0})));
    }

    // SBCR and the inverse fused-XY path use a single batch axis for each
    // transverse plane, separate from the outermost batch.
    data_set_t fold_transverse_plane(const data_set_t& column)
    {
        return data_set_t::full_range(
                   {column.axis_span(0), column.axis_span(1) * column.axis_span(2)},
                   {column.axis_stride(0), column.axis_stride(1)},
                   column.axis_span(3),
                   column.axis_stride(3),
                   2)
            .with_element_type(rocfft_array_type_unset, column.get_precision());
    }

    void set_io_strides(TreeNode& node, const data_set_t& input, const data_set_t& output)
    {
        node.set_strides(INPUT, input);
        node.set_strides(OUTPUT, output);
    }

    template <typename Strides>
    void set_inplace_strides(TreeNode& node, const Strides& strides)
    {
        node.set_strides(INPUT, strides);
        node.set_strides(OUTPUT, strides);
    }

    // check if we have an SBCC kernel along the specified dimension
    bool SBCC_dim_available(const function_pool& pool, size_t length, rocfft_precision precision)
    {
        // Check the C part.
        // The first R is built recursively with 2D_FFT, leave the check part to themselves
        size_t numTrans = 0;
        // do we have a purpose-built sbcc kernel
        bool  have_sbcc = false;
        FMKey sbcc_key(length, precision, CS_KERNEL_STOCKHAM_BLOCK_CC);
        if(pool.has_function(sbcc_key))
        {
            numTrans  = pool.get_kernel(sbcc_key).transforms_per_block;
            have_sbcc = true;
        }
        else
        {
            FMKey normal_key(length, precision);
            if(!pool.has_function(normal_key))
                return false;
            numTrans = pool.get_kernel(normal_key).transforms_per_block;
        }

        // for regular stockham kernels, ensure we are doing enough rows
        // to coalesce properly. 4 seems to be enough for
        // double-precision, whereas some sizes that do 7 rows seem to be
        // slower for single.
        if(!have_sbcc)
        {
            size_t minRows = precision == rocfft_precision_single ? 8 : 4;
            if(numTrans < minRows)
                return false;
        }

        return true;
    }

    // True if a fused real-Stockham kernel of the given cfft length / precision
    // fits in this device's per-workgroup LDS.  Fused mode disables half-LDS and
    // adds 1 padding element per batch, so
    //   lds_bytes = (cfftLength + 1) * transforms_per_block * bytes_per_complex
    // Budget matches the runtime check in LeafNode::SetupGridParam (tree_node.cpp).
    // Returns false if no Stockham kernel is registered for this length.
    bool fused_real_stockham_fits_lds(const function_pool&   pool,
                                      size_t                 cfftLength,
                                      rocfft_precision       precision,
                                      const hipDeviceProp_t& deviceProp)
    {
        FMKey key(cfftLength, precision, CS_KERNEL_STOCKHAM);
        if(!pool.has_function(key))
            return false;
        const size_t batch_width = std::max<size_t>(pool.get_kernel(key).transforms_per_block, 1);
        const size_t fused_lds_bytes
            = (cfftLength + 1) * batch_width * complex_type_size(precision);
        return fused_lds_bytes <= deviceProp.sharedMemPerBlock;
    }

    bool fused_real_2d_available(const function_pool& pool,
                                 const data_set_t&    real,
                                 rocfft_precision     precision,
                                 rocfft_array_type    in,
                                 rocfft_array_type    out)
    {
        const auto x       = real.axis_span(0) / 2;
        const auto y       = real.axis_span(1);
        const bool forward = in == rocfft_array_type_real;
        return NodeFactory::use_CS_2D_SINGLE(
            pool, forward ? x : y, forward ? y : x, precision, in, out);
    }
}

/*****************************************************
 * CS_REAL_TRANSFORM_USING_CMPLX
 *****************************************************/
void RealTransCmplxNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    // Embed the data into a full-length complex array, perform
    // complex transform, and then extract the relevant output.
    bool          r2c            = input_array_type() == rocfft_array_type_real;
    ComputeScheme copyHeadScheme = r2c ? CS_KERNEL_COPY_R_TO_CMPLX : CS_KERNEL_COPY_HERM_TO_CMPLX;
    ComputeScheme copyTailScheme = r2c ? CS_KERNEL_COPY_CMPLX_TO_HERM : CS_KERNEL_COPY_CMPLX_TO_R;

    const auto& real_layout = r2c ? input_layout() : output_layout();
    const auto  packed      = untyped(real_layout.get_packed_data_set());

    // check schemes from solution map
    ComputeScheme determined_scheme = CS_NONE;
    if(!noSolution)
    {
        if((child_scheme_trees.size() != 3) || (child_scheme_trees[0]->curScheme != copyHeadScheme)
           || (child_scheme_trees[2]->curScheme != copyTailScheme))
        {
            throw std::runtime_error(
                "RealTransCmplxNode: Unexpected child scheme from solution map");
        }
        determined_scheme = child_scheme_trees[1]->curScheme;
    }

    auto copyHeadPlan
        = NodeFactory::CreateNodeFromScheme(copyHeadScheme, this, untyped(input_layout()), packed);
    childNodes.emplace_back(std::move(copyHeadPlan));

    // complex fft
    NodeMetaData fftPlanData(packed, packed, this);

    auto fftPlan = NodeFactory::CreateExplicitNode(fftPlanData, this, determined_scheme);
    fftPlan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[1].get());

    // NB:
    // The tail copy kernel accepts only complex-interleaved input.
    fftPlan->GetLastLeaf()->allowedOutArrayTypes = {rocfft_array_type_complex_interleaved};
    childNodes.emplace_back(std::move(fftPlan));

    auto copyTailPlan
        = NodeFactory::CreateNodeFromScheme(copyTailScheme, this, packed, untyped(output_layout()));

    childNodes.emplace_back(std::move(copyTailPlan));
}

void RealTransCmplxNode::AssignParams_internal()
{
    assert(childNodes.size() == 3);
    auto& copyHeadPlan = childNodes[0];
    auto& fftPlan      = childNodes[1];
    auto& copyTailPlan = childNodes[2];

    copyHeadPlan->set_strides(INPUT, input_layout());
    const auto packed = copyHeadPlan->output_layout().get_packed_data_set();
    copyHeadPlan->set_strides(OUTPUT, packed);
    set_inplace_strides(*fftPlan, packed);

    fftPlan->AssignParams();

    set_io_strides(*copyTailPlan, fftPlan->output_layout(), output_layout());
}

/*****************************************************
 * CS_REAL_TRANSFORM_EVEN
 *****************************************************/
void RealTransEvenNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    const bool  forward   = direction == -1;
    const auto& real      = forward ? input_layout() : output_layout();
    const auto& hermitian = forward ? output_layout() : input_layout();

    // Fastest moving dimension must be even:
    if(real.axis_span(0) % 2 != 0)
        throw std::runtime_error("fastest dimension is not even in RealTransEvenNode");

    // check schemes from solution map
    ComputeScheme determined_scheme = CS_NONE;
    size_t        fft_node_id       = 0;
    if(!noSolution)
    {
        if(child_scheme_trees.size() != 1 && child_scheme_trees.size() != 2)
            throw std::runtime_error(
                "RealTransEvenNode: Unexpected child scheme from solution map");
        try_fuse_pre_post_processing = (child_scheme_trees.size() == 1) ? true : false;

        // fwd
        if(direction == -1)
        {
            fft_node_id = 0;
            if(try_fuse_pre_post_processing == false) // not fused
                assert(child_scheme_trees[1]->curScheme == CS_KERNEL_R_TO_CMPLX);
        }
        // bwd
        else
        {
            fft_node_id = try_fuse_pre_post_processing ? 0 : 1;
            if(try_fuse_pre_post_processing == false) // not fused
                assert(child_scheme_trees[0]->curScheme == CS_KERNEL_CMPLX_TO_R);
        }
        determined_scheme = child_scheme_trees[fft_node_id]->curScheme;
    }

    // FFTs work on the real buffer reinterpreted in complex units. Decide the
    // scheme and fusion before constructing the actual FFT child.
    const auto complex       = untyped(real_as_complex(real));
    size_t     length_factor = 0;
    if(noSolution)
    {
        NodeMetaData probe(complex, complex, this);
        determined_scheme = NodeFactory::DecideNodeScheme(pool, probe, this);
        if(determined_scheme == CS_NONE)
            throw std::runtime_error("DecideNodeScheme Failed!: CS_NONE");
        length_factor = probe.length_factor;
        // Stockham is the only leaf chosen by the 1D scheme heuristic.
        try_fuse_pre_post_processing
            = determined_scheme == CS_KERNEL_STOCKHAM
              && (try_fuse_pre_post_processing
                  || (data_rank() == 1
                      && fused_real_stockham_fits_lds(
                          pool, complex.axis_span(0), get_precision(), deviceProp)
                      && input_array_type() != rocfft_array_type_hermitian_planar
                      && output_array_type() != rocfft_array_type_hermitian_planar));
    }

    // Embedded pre-processing reads one extra bin; the FFT ABI still
    // describes C bins, with the original Hermitian strides.
    auto fft_input  = !forward && try_fuse_pre_post_processing
                          ? untyped(hermitian.with_axis_span(0, complex.axis_span(0)))
                          : complex;
    auto fft_output = forward && try_fuse_pre_post_processing ? untyped(hermitian) : complex;
    std::unique_ptr<TreeNode> cfftPlan;
    if(noSolution)
    {
        cfftPlan = NodeFactory::CreateNodeFromScheme(
            determined_scheme, this, std::move(fft_input), std::move(fft_output));
        cfftPlan->length_factor = length_factor;
    }
    else
    {
        NodeMetaData final_data(std::move(fft_input), std::move(fft_output), this);
        cfftPlan = NodeFactory::CreateExplicitNode(final_data, this, determined_scheme);
    }
    cfftPlan->RecursiveBuildTree(noSolution ? nullptr : child_scheme_trees[fft_node_id].get());

    switch(direction)
    {
    case -1:
    {
        // real-to-complex transform: in-place complex transform then post-process

        if(try_fuse_pre_post_processing)
            cfftPlan->ebtype = EmbeddedType::Real2C_POST;

        childNodes.emplace_back(std::move(cfftPlan));

        // add separate post-processing if we couldn't fuse
        if(!try_fuse_pre_post_processing)
        {
            // NB:
            //   input of CS_KERNEL_R_TO_CMPLX allows single-ptr-buffer type only (can't be planar),
            //   so we set the allowed-out-type of the previous kernel to follow the rule.
            //   Precisely, it should be {real, interleaved}, but CI is enough since we only use
            //   CI/CP internally during assign-buffer.
            childNodes.back()->GetLastLeaf()->allowedOutArrayTypes
                = {rocfft_array_type_complex_interleaved};

            childNodes.emplace_back(NodeFactory::CreateNodeFromScheme(
                CS_KERNEL_R_TO_CMPLX, this, complex, untyped(hermitian)));
        }
        break;
    }
    case 1:
    {
        // complex-to-real transform: pre-process followed by in-place complex transform

        if(!try_fuse_pre_post_processing)
        {
            // add separate pre-processing if we couldn't fuse
            childNodes.emplace_back(NodeFactory::CreateNodeFromScheme(
                CS_KERNEL_CMPLX_TO_R, this, untyped(hermitian), complex));
        }
        else
        {
            cfftPlan->ebtype = EmbeddedType::C2Real_PRE;
        }

        childNodes.emplace_back(std::move(cfftPlan));
        break;
    }
    default:
    {
        std::cerr << "invalid direction: plan creation failed!\n";
    }
    }
}

void RealTransEvenNode::AssignParams_internal()
{
    // definitely will have FFT.  pre/post processing
    // might be fused into the FFT or separate.
    assert(childNodes.size() == 1 || childNodes.size() == 2);

    const bool forward = direction == -1;
    const auto complex = real_as_complex(forward ? input_layout() : output_layout());
    if(forward)
    {
        auto& fftPlan = childNodes[0];
        set_inplace_strides(*fftPlan, complex);
        fftPlan->AssignParams();

        if(childNodes.size() == 2)
        {
            auto& postPlan = childNodes[1];
            assert(postPlan->scheme == CS_KERNEL_R_TO_CMPLX
                   || postPlan->scheme == CS_KERNEL_R_TO_CMPLX_TRANSPOSE);
            set_io_strides(*postPlan, complex, output_layout());
        }
        else
        {
            // The fused FFT writes in Hermitian units.
            fftPlan->set_strides(OUTPUT, output_layout());
        }
    }
    else
    {
        const bool fused = childNodes[0]->ebtype == EmbeddedType::C2Real_PRE;
        if(!fused)
        {
            auto& prePlan = childNodes[0];
            assert(prePlan->scheme == CS_KERNEL_CMPLX_TO_R);
            set_io_strides(*prePlan, input_layout(), complex);
        }

        auto& fftPlan = childNodes[fused ? 0 : 1];
        set_io_strides(*fftPlan, fused ? input_layout() : complex, complex);
        fftPlan->AssignParams();
    }
}

/*****************************************************
 * CS_REAL_2D_EVEN
 *****************************************************/
void Real2DEvenNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    const auto& real = direction == -1 ? input_layout() : output_layout();

    // Fastest moving dimension must be even:
    if(real.axis_span(0) % 2 != 0)
        throw std::runtime_error("fastest dimension is not even in RealTransEvenNode");

    //
    // TODO- we need to replace the solution-decision part with plan-solution.
    //

    // if we have SBCC for the higher dimension, use that and avoid transpose
    solution = SBCC_dim_available(pool, input_length(1), get_precision()) ? INPLACE_SBCC : TR_PAIR;

    // but if we have 2D_SINGLE available, then we use it
    // NB: use the check function in NodeFactory to make sure lds limit
    // TODO- this part should be done in offline-tuning
    if(fused_real_2d_available(
           pool, real, get_precision(), input_array_type(), output_array_type()))
        solution = REAL_2D_SINGLE;

    switch(solution)
    {
    case REAL_2D_SINGLE:
        BuildTree_internal_2D_SINGLE();
        break;
    case INPLACE_SBCC:
        BuildTree_internal_SBCC(child_scheme_trees);
        break;
    case TR_PAIR:
        BuildTree_internal_TR_pair(child_scheme_trees);
        break;
    }
}

void Real2DEvenNode::BuildTree_internal_2D_SINGLE()
{
    const bool   forward = input_array_type() == rocfft_array_type_real;
    const auto   complex = untyped(real_as_complex(forward ? input_layout() : output_layout()));
    NodeMetaData data(forward ? complex
                              : untyped(input_layout().with_axis_span(0, complex.axis_span(0))),
                      forward ? untyped(output_layout()) : complex,
                      this);
    auto cfftPlan = NodeFactory::CreateExplicitNode(data, this);
    cfftPlan->RecursiveBuildTree();
    cfftPlan->ebtype          = forward ? EmbeddedType::Real2C_POST : EmbeddedType::C2Real_PRE;
    cfftPlan->allowOutofplace = true;
    childNodes.emplace_back(std::move(cfftPlan));
}

void Real2DEvenNode::BuildTree_internal_SBCC(SchemeTreeVec& child_scheme_trees)
{
    const bool  forward    = input_array_type() == rocfft_array_type_real;
    const auto& hermitian  = forward ? output_layout() : input_layout();
    const auto  y          = untyped(hermitian.get_data_set_for_len_axes({1}));
    bool        haveSBCC   = pool.has_SBCC_kernel(y.axis_span(0), get_precision());
    auto        sbccScheme = haveSBCC ? CS_KERNEL_STOCKHAM_BLOCK_CC : CS_KERNEL_STOCKHAM;
    bool        noSolution = child_scheme_trees.empty();

    if(forward)
    {
        // check schemes from solution map
        ComputeScheme determined_scheme_1 = sbccScheme;
        if(!noSolution)
        {
            if((child_scheme_trees.size() != 2)
               || (child_scheme_trees[0]->curScheme != CS_REAL_TRANSFORM_EVEN))
            {
                throw std::runtime_error(
                    "Real2DEvenNode: Unexpected child scheme from solution map");
            }
            determined_scheme_1 = child_scheme_trees[1]->curScheme;
        }

        // first row fft + postproc is mandatory for fastest dimension
        auto rcplan = make_real_row(this);
        static_cast<RealTransEvenNode*>(rcplan.get())->try_fuse_pre_post_processing
            = fused_real_stockham_fits_lds(pool, input_length(0) / 2, get_precision(), deviceProp);

        rcplan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[0].get());
        childNodes.emplace_back(std::move(rcplan));

        auto sbccY = NodeFactory::CreateNodeFromScheme(determined_scheme_1, this, y, y);
        sbccY->legacyOutputOrder = OutputAxisOrder::SWAP_01;
        childNodes.emplace_back(std::move(sbccY));
    }
    else
    {
        // check schemes from solution map
        ComputeScheme determined_scheme_0 = sbccScheme;
        if(!noSolution)
        {
            if((child_scheme_trees.size() != 2)
               || (child_scheme_trees[1]->curScheme != CS_REAL_TRANSFORM_EVEN))
            {
                throw std::runtime_error(
                    "Real2DEvenNode: Unexpected child scheme from solution map");
            }
            determined_scheme_0 = child_scheme_trees[0]->curScheme;
        }

        auto sbccY = NodeFactory::CreateNodeFromScheme(determined_scheme_0, this, y, y);
        sbccY->legacyOutputOrder = OutputAxisOrder::SWAP_01;
        childNodes.emplace_back(std::move(sbccY));

        // c2r
        auto crplan = make_real_row(this);
        static_cast<RealTransEvenNode*>(crplan.get())->try_fuse_pre_post_processing
            = fused_real_stockham_fits_lds(pool, output_length(0) / 2, get_precision(), deviceProp);

        crplan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[1].get());
        childNodes.emplace_back(std::move(crplan));
    }
}

void Real2DEvenNode::BuildTree_internal_TR_pair(SchemeTreeVec& child_scheme_trees)
{
    const bool noSolution = child_scheme_trees.empty();
    const bool forward    = input_array_type() == rocfft_array_type_real;
    const auto xy         = untyped(forward ? output_layout() : input_layout());
    const auto yx         = reorder_lengths(xy, swap_xy);
    const auto y          = yx.get_data_set_for_len_axes({0});

    if(forward)
    {
        // check schemes from solution map
        ComputeScheme determined_scheme_2 = CS_NONE;
        if(!noSolution)
        {
            if((child_scheme_trees.size() != 4)
               || (child_scheme_trees[0]->curScheme != CS_REAL_TRANSFORM_EVEN)
               || (child_scheme_trees[1]->curScheme != CS_KERNEL_TRANSPOSE)
               || (child_scheme_trees[3]->curScheme != CS_KERNEL_TRANSPOSE))
            {
                throw std::runtime_error(
                    "Real2DEvenNode: Unexpected child scheme from solution map");
            }
            determined_scheme_2 = child_scheme_trees[2]->curScheme;
        }
        // RTRT
        // first row fft
        auto row1Plan = make_real_row(this);
        row1Plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[0].get());

        // first transpose
        auto trans1Plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this, xy, xy);

        // second row fft
        NodeMetaData row2PlanData(y, y, this);
        auto row2Plan = NodeFactory::CreateExplicitNode(row2PlanData, this, determined_scheme_2);
        row2Plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[2].get());

        // second transpose
        auto trans2Plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this, yx, yx);

        // --------------------------------
        // Fuse Shims:
        // 1-1. Try (stockham + r2c)(from real even) + transpose
        // 1-2. else, try r2c (from real even) + transpose
        // 2. row2 and trans2: RTFuse
        // --------------------------------
        auto STK_R2CTrans = NodeFactory::CreateFuseShim(FT_STOCKHAM_R2C_TRANSPOSE,
                                                        {row1Plan.get(), trans1Plan.get()});
        if(STK_R2CTrans->IsSchemeFusable())
        {
            fuseShims.emplace_back(std::move(STK_R2CTrans));
        }
        else
        {
            auto R2CTrans = NodeFactory::CreateFuseShim(
                FT_R2C_TRANSPOSE, {row1Plan.get(), trans1Plan.get(), row2Plan.get()});
            if(R2CTrans->IsSchemeFusable())
                fuseShims.emplace_back(std::move(R2CTrans));
        }

        auto RT = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS,
                                              {row2Plan.get(), trans2Plan.get()});
        if(RT->IsSchemeFusable())
            fuseShims.emplace_back(std::move(RT));

        // --------------------------------
        // RTRT
        // --------------------------------
        // Fuse r2c trans
        childNodes.emplace_back(std::move(row1Plan));
        childNodes.emplace_back(std::move(trans1Plan));
        // Fuse RT
        childNodes.emplace_back(std::move(row2Plan));
        childNodes.emplace_back(std::move(trans2Plan));
    }
    else
    {
        // check schemes from solution map
        ComputeScheme determined_scheme_1 = CS_NONE;
        if(!noSolution)
        {
            if((child_scheme_trees.size() != 4)
               || (child_scheme_trees[0]->curScheme != CS_KERNEL_TRANSPOSE)
               || (child_scheme_trees[2]->curScheme != CS_KERNEL_TRANSPOSE)
               || (child_scheme_trees[3]->curScheme != CS_REAL_TRANSFORM_EVEN))
            {
                throw std::runtime_error(
                    "Real2DEvenNode: Unexpected child scheme from solution map");
            }
            determined_scheme_1 = child_scheme_trees[1]->curScheme;
        }

        // TRTR
        // first transpose
        auto trans1Plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this, xy, xy);

        // c2c row transform
        NodeMetaData c2cPlanData(y, y, this);
        auto c2cPlan = NodeFactory::CreateExplicitNode(c2cPlanData, this, determined_scheme_1);
        c2cPlan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[1].get());

        // second transpose
        auto trans2plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this, yx, yx);

        // c2r row transform
        if(!noSolution && (child_scheme_trees[3]->curScheme != CS_REAL_TRANSFORM_EVEN))
            throw std::runtime_error("Real2DEvenNode: Unexpected child scheme from solution map");
        auto c2rPlan = make_real_row(this);
        c2rPlan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[3].get());

        // --------------------------------
        // Fuse Shims:
        // 1. trans1 and c2c
        // 2. transpose + c2r (first child of real even)
        // --------------------------------
        auto TR = NodeFactory::CreateFuseShim(FT_TRANS_WITH_STOCKHAM,
                                              {trans1Plan.get(), c2cPlan.get()});
        if(TR->IsSchemeFusable())
            fuseShims.emplace_back(std::move(TR));

        auto TransC2R
            = NodeFactory::CreateFuseShim(FT_TRANSPOSE_C2R, {trans2plan.get(), c2rPlan.get()});
        if(TransC2R->IsSchemeFusable())
            fuseShims.emplace_back(std::move(TransC2R));

        // --------------------------------
        // TRTR
        // --------------------------------
        childNodes.emplace_back(std::move(trans1Plan));
        childNodes.emplace_back(std::move(c2cPlan));
        //
        childNodes.emplace_back(std::move(trans2plan));
        childNodes.emplace_back(std::move(c2rPlan));
    }
}

void Real2DEvenNode::AssignParams_internal()
{
    switch(solution)
    {
    case REAL_2D_SINGLE:
        AssignParams_internal_2D_SINGLE();
        break;
    case INPLACE_SBCC:
        AssignParams_internal_SBCC();
        break;
    case TR_PAIR:
        AssignParams_internal_TR_pair();
        break;
    }
}

void Real2DEvenNode::AssignParams_internal_2D_SINGLE()
{
    const bool forward  = input_array_type() == rocfft_array_type_real;
    const auto complex  = real_as_complex(forward ? input_layout() : output_layout());
    auto&      fused_2d = childNodes[0];
    set_io_strides(
        *fused_2d, forward ? complex : input_layout(), forward ? output_layout() : complex);
    fused_2d->AssignParams();
}

void Real2DEvenNode::AssignParams_internal_SBCC()
{
    const bool forward = direction == -1;
    const auto y = (forward ? output_layout() : input_layout()).get_data_set_for_len_axes({1});
    auto&      rowPlan = childNodes[forward ? 0 : 1];
    auto&      sbccY   = childNodes[forward ? 1 : 0];
    if(forward)
    {
        set_io_strides(*rowPlan, input_layout(), output_layout());
        rowPlan->AssignParams();
        set_inplace_strides(*sbccY, y);
    }
    else
    {
        set_inplace_strides(*sbccY, y);
        sbccY->AssignParams();
        set_io_strides(*rowPlan, input_layout(), output_layout());
        rowPlan->AssignParams();
    }
}

void Real2DEvenNode::AssignParams_internal_TR_pair()
{
    const bool  forward   = input_array_type() == rocfft_array_type_real;
    const auto& hermitian = forward ? output_layout() : input_layout();
    auto        yx        = reorder_lengths(hermitian, swap_xy);
    const auto  packed_yx = yx.get_packed_data_set();
    const auto  packed_xy = reorder_lengths(packed_yx, swap_xy);

    if(forward)
    {
        auto& row1Plan = childNodes[0];
        set_io_strides(*row1Plan, input_layout(), output_layout());
        row1Plan->AssignParams();

        set_io_strides(*childNodes[1], row1Plan->output_layout(), packed_xy);

        auto& row2Plan = childNodes[2];
        set_inplace_strides(*row2Plan, packed_yx);
        row2Plan->AssignParams();

        set_io_strides(*childNodes[3], row2Plan->output_layout(), yx);
    }
    else
    {
        set_io_strides(*childNodes[0], input_layout(), packed_xy);

        auto& c2cPlan = childNodes[1];
        set_inplace_strides(*c2cPlan, packed_yx);
        c2cPlan->AssignParams();

        // The second transpose retains the packed temporary's batch distance.
        yx.set_axis_stride(data_rank(), packed_yx.axis_stride(data_rank()));
        set_io_strides(*childNodes[2], packed_yx, yx);

        auto& c2rPlan = childNodes[3];
        set_io_strides(*c2rPlan, reorder_lengths(std::move(yx), swap_xy), output_layout());
        c2rPlan->AssignParams();
    }
}

/*****************************************************
 * CS_REAL_3D_EVEN
 *****************************************************/
void Real3DEvenNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    Build_solution();

    switch(solution)
    {
    case REAL_2D_SINGLE_SBCC:
        BuildTree_internal_2D_SINGLE_CC();
        break;
    case INPLACE_SBCC:
        BuildTree_internal_SBCC(child_scheme_trees);
        break;
    case SBCR:
        BuildTree_internal_SBCR(child_scheme_trees);
        break;
    case TR_PAIRS:
        BuildTree_internal_TR_pairs(child_scheme_trees);
        break;
    default:
        throw std::runtime_error("3D R2C/C2R build tree failure: " + PrintScheme(scheme));
        break;
    }
}

// Check if we can use 2D_SINGLE kernel for the 3D real-even case.
// The check is based on the input/output length and strides, precision
// and kernel availability in the pool.
bool Real3DEvenNode::use_real_2D_single_SBCC(const NodeMetaData&  nodeData,
                                             const function_pool& pool,
                                             const bool&          is_default_contiguous_layout)
{
    const auto  in  = nodeData.input_array_type();
    const auto  out = nodeData.output_array_type();
    const auto& real
        = in == rocfft_array_type_real ? nodeData.input_layout() : nodeData.output_layout();
    const auto p = nodeData.get_precision();
    return fused_real_2d_available(pool, real, p, in, out)
           && SBCC_dim_available(pool, real.axis_span(2), p) && is_default_contiguous_layout;
}

void Real3DEvenNode::Build_solution()
{
    const bool  forward = input_array_type() == rocfft_array_type_real;
    const auto& real    = forward ? input_layout() : output_layout();

    // Fastest moving dimension must be even:
    if(real.axis_span(0) % 2 != 0)
        throw std::runtime_error("fastest dimension is not even in RealTransEvenNode");

    // but if we have 2D_SINGLE available, then we use it
    // NB: use the check function in NodeFactory to make sure lds limit
    // TODO- this part should be done in offline-tuning
    NodeMetaData nodeData(input_layout(), output_layout(), this);

    const bool is_default_contiguous_layout
        = this->GetPlanRoot()->inStrideUnit && this->GetPlanRoot()->outStrideUnit;
    if(use_real_2D_single_SBCC(nodeData, pool, is_default_contiguous_layout))
    {
        solution = REAL_2D_SINGLE_SBCC;
        return;
    }

    // NB:
    //   - We need better general mechanism to choose In-place SBCC, SBCR and SBRC solution.
    if(!forward)
    {
        // FIXME:
        //    1. Currently, BuildTree_internal_SBCR and AssignParams_internal_SBCR
        //       support unit strides only. We might want to differentiate
        //       implementation for unit/non-unit strides cases both on host and
        //       device side.
        //    2. Enable for gfx908 and gfx90a only. Need more tuning for Navi arch.
        const auto p = get_precision();
        if((is_device_gcn_arch(deviceProp, "gfx908") || is_device_gcn_arch(deviceProp, "gfx90a"))
           && pool.has_SBCR_kernel(output_length(0) / 2, p)
           && pool.has_SBCR_kernel(output_length(1), p) && pool.has_SBCR_kernel(output_length(2), p)
           && (placement
               == rocfft_placement_notinplace) // In-place SBCC is faster than SBCR solution for in-place
           && (input_stride(0) == 1 && output_stride(0) == 1
               && input_stride(1) == input_length(0) // unit strides
               && output_stride(1) == output_length(0)
               && input_stride(2) == input_stride(1) * input_length(1)
               && output_stride(2) == output_stride(1) * output_length(1)))
        {
            solution = SBCR;
            return;
        }
    }

    // if we have SBCC kernels for the other two dimensions, transform them using SBCC and avoid transposes.
    const bool sbcc_inplace = SBCC_dim_available(pool, input_length(1), get_precision())
                              && SBCC_dim_available(pool, input_length(2), get_precision());

    solution = (sbcc_inplace) ? INPLACE_SBCC : TR_PAIRS;
}

void Real3DEvenNode::BuildTree_internal_2D_SINGLE_CC()
{
    const bool  forward   = input_array_type() == rocfft_array_type_real;
    const auto& hermitian = forward ? output_layout() : input_layout();
    const auto  complex   = real_as_complex(forward ? input_layout() : output_layout());
    const auto  xy        = untyped(complex.get_data_set_for_len_axes({0, 1}));
    const auto  hxy       = forward ? untyped(hermitian.get_data_set_for_len_axes({0, 1}))
                                    : untyped(hermitian.with_axis_span(0, complex.axis_span(0))
                                           .get_data_set_for_len_axes({0, 1}));
    auto        xyPlan    = NodeFactory::CreateNodeFromScheme(
        CS_KERNEL_2D_SINGLE, this, forward ? xy : hxy, forward ? hxy : xy);
    xyPlan->RecursiveBuildTree();
    xyPlan->ebtype = forward ? EmbeddedType::Real2C_POST : EmbeddedType::C2Real_PRE;

    const auto z             = hermitian.get_data_set_for_len_axes({2});
    const auto column        = forward ? untyped(z) : fold_transverse_plane(z);
    const auto column_scheme = pool.has_SBCC_kernel(z.axis_span(0), get_precision())
                                   ? CS_KERNEL_STOCKHAM_BLOCK_CC
                                   : CS_KERNEL_STOCKHAM;
    auto       sbccZ = NodeFactory::CreateNodeFromScheme(column_scheme, this, column, column);
    if(forward)
    {
        sbccZ->legacyOutputOrder = OutputAxisOrder::ROTATE_LEFT_3;
        childNodes.emplace_back(std::move(xyPlan));
        childNodes.emplace_back(std::move(sbccZ));
    }
    else
    {
        childNodes.emplace_back(std::move(sbccZ));
        childNodes.emplace_back(std::move(xyPlan));
    }
}

void Real3DEvenNode::BuildTree_internal_SBCC(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();
    // from solution map
    ComputeScheme determined_scheme_dimZ = CS_NONE;
    ComputeScheme determined_scheme_dimY = CS_NONE;

    auto add_sbcc_children = [this](const data_set_t& stage,
                                    ComputeScheme     determined_scheme_dimZ,
                                    ComputeScheme     determined_scheme_dimY) {
        const auto    y = stage.axis_span(1);
        const auto    z = stage.axis_span(2);
        const auto    p = get_precision();
        ComputeScheme scheme;

        // Performance improvements for (192,192,192) with SBCC.
        auto use_SBCC_192 = (z == 192 && y == 192) && (p == rocfft_precision_single);

        // A special case (192,200,XX), (168,192,XX) on gfx908, we eventually need to remove these
        if(is_device_gcn_arch(deviceProp, "gfx908"))
        {
            if(((z == 192 && y == 200) || (z == 168 && y == 192)) && (p == rocfft_precision_single))
                use_SBCC_192 = true;
        }

        // SBCC along Z dimension
        if(z == 192)
            scheme = use_SBCC_192 ? CS_KERNEL_STOCKHAM_BLOCK_CC : CS_KERNEL_STOCKHAM;
        else
        {
            bool haveSBCC_Z = pool.has_SBCC_kernel(z, p);
            scheme          = haveSBCC_Z ? CS_KERNEL_STOCKHAM_BLOCK_CC : CS_KERNEL_STOCKHAM;
        }
        scheme              = (determined_scheme_dimZ == CS_NONE) ? scheme : determined_scheme_dimZ;
        const auto column_z = untyped(stage.get_data_set_for_len_axes({2}));
        auto       sbccZ    = NodeFactory::CreateNodeFromScheme(scheme, this, column_z, column_z);
        sbccZ->legacyOutputOrder = OutputAxisOrder::ROTATE_LEFT_3;
        childNodes.emplace_back(std::move(sbccZ));

        // SBCC along Y dimension
        if(y == 192)
            scheme = use_SBCC_192 ? CS_KERNEL_STOCKHAM_BLOCK_CC : CS_KERNEL_STOCKHAM;
        else
        {
            bool haveSBCC_Y = pool.has_SBCC_kernel(y, p);
            scheme          = haveSBCC_Y ? CS_KERNEL_STOCKHAM_BLOCK_CC : CS_KERNEL_STOCKHAM;
        }
        scheme              = (determined_scheme_dimY == CS_NONE) ? scheme : determined_scheme_dimY;
        const auto column_y = untyped(stage.get_data_set_for_len_axes({1}));
        auto       sbccY    = NodeFactory::CreateNodeFromScheme(scheme, this, column_y, column_y);
        sbccY->legacyOutputOrder = OutputAxisOrder::SWAP_01;
        childNodes.emplace_back(std::move(sbccY));
    };

    if(input_array_type() == rocfft_array_type_real) // forward
    {
        // check schemes from solution map
        if(!noSolution)
        {
            if((child_scheme_trees.size() != 3)
               || (child_scheme_trees[0]->curScheme != CS_REAL_TRANSFORM_EVEN))
            {
                throw std::runtime_error(
                    "Real3DEvenNode: Unexpected child scheme from solution map");
            }
            determined_scheme_dimZ = child_scheme_trees[1]->curScheme;
            determined_scheme_dimY = child_scheme_trees[2]->curScheme;
        }

        // first row fft + postproc is mandatory for fastest dimension
        auto rcplan = make_real_row(this);
        static_cast<RealTransEvenNode*>(rcplan.get())->try_fuse_pre_post_processing
            = fused_real_stockham_fits_lds(pool, input_length(0) / 2, get_precision(), deviceProp);

        rcplan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[0].get());

        // if we have SBCC kernels for the other two dimensions, transform them using SBCC and avoid transposes
        childNodes.emplace_back(std::move(rcplan));
        add_sbcc_children(output_layout(), determined_scheme_dimZ, determined_scheme_dimY);
    }
    else
    {
        // check schemes from solution map
        if(!noSolution)
        {
            if((child_scheme_trees.size() != 3)
               || (child_scheme_trees[2]->curScheme != CS_REAL_TRANSFORM_EVEN))
            {
                throw std::runtime_error(
                    "Real3DEvenNode: Unexpected child scheme from solution map");
            }
            determined_scheme_dimZ = child_scheme_trees[0]->curScheme;
            determined_scheme_dimY = child_scheme_trees[1]->curScheme;
        }

        add_sbcc_children(input_layout(), determined_scheme_dimZ, determined_scheme_dimY);

        // c2r
        auto crplan = make_real_row(this);
        static_cast<RealTransEvenNode*>(crplan.get())->try_fuse_pre_post_processing
            = fused_real_stockham_fits_lds(pool, output_length(0) / 2, get_precision(), deviceProp);

        crplan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[2].get());
        childNodes.emplace_back(std::move(crplan));

        // --------------------------------
        // Fuse Shims:
        // 2. trans1 + c2r (first child of real even)
        // note the CheckSchemeFusable will check if the first one is transpose
        // --------------------------------
        auto TransC2R = NodeFactory::CreateFuseShim(
            FT_TRANSPOSE_C2R, {childNodes[childNodes.size() - 2].get(), childNodes.back().get()});
        if(TransC2R->IsSchemeFusable())
            fuseShims.emplace_back(std::move(TransC2R));
    }
}

void Real3DEvenNode::BuildTree_internal_SBCR(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();
    // check schemes from solution map
    if(!noSolution)
    {
        if((child_scheme_trees.size() != 3)
           || (child_scheme_trees[0]->curScheme != CS_KERNEL_STOCKHAM_BLOCK_CR)
           || (child_scheme_trees[1]->curScheme != CS_KERNEL_STOCKHAM_BLOCK_CR)
           || (child_scheme_trees[2]->curScheme != CS_KERNEL_STOCKHAM_BLOCK_CR))
        {
            throw std::runtime_error("Real3DEvenNode: Unexpected child scheme from solution map");
        }
    }

    const auto z     = fold_transverse_plane(input_layout().get_data_set_for_len_axes({2}));
    auto       sbcrZ = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CR, this, z, z);
    childNodes.emplace_back(std::move(sbcrZ));

    const auto y     = fold_transverse_plane(input_layout().get_data_set_for_len_axes({1}));
    auto       sbcrY = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CR, this, y, y);
    childNodes.emplace_back(std::move(sbcrY));

    const auto complex    = real_as_complex(output_layout());
    const auto x          = fold_transverse_plane(complex.get_data_set_for_len_axes({0}));
    auto       x_input    = x;
    const auto transverse = x.axis_span(1);
    x_input.set_strides_and_distances(
        std::array<size_t, 3>{transverse, 1, input_length(0) * transverse});
    auto sbcrX = NodeFactory::CreateNodeFromScheme(
        CS_KERNEL_STOCKHAM_BLOCK_CR, this, std::move(x_input), x);
    childNodes.emplace_back(std::move(sbcrX));
}

void Real3DEvenNode::BuildTree_internal_TR_pairs(SchemeTreeVec& child_scheme_trees)
{
    const bool noSolution = child_scheme_trees.empty();
    const bool forward    = input_array_type() == rocfft_array_type_real;
    const auto xyz        = untyped(forward ? output_layout() : input_layout());
    const auto yzx        = reorder_lengths(xyz, left_xyz);
    const auto zxy        = reorder_lengths(xyz, right_xyz);
    const auto y          = yzx.get_data_set_for_len_axes({0});
    const auto z          = zxy.get_data_set_for_len_axes({0});

    if(forward)
    {
        // check schemes from solution map
        ComputeScheme determined_scheme_2 = CS_NONE;
        ComputeScheme determined_scheme_4 = CS_NONE;
        if(!noSolution)
        {
            if((child_scheme_trees.size() != 6)
               || (child_scheme_trees[0]->curScheme != CS_REAL_TRANSFORM_EVEN)
               || (child_scheme_trees[1]->curScheme != CS_KERNEL_TRANSPOSE_Z_XY)
               || (child_scheme_trees[3]->curScheme != CS_KERNEL_TRANSPOSE_Z_XY)
               || (child_scheme_trees[5]->curScheme != CS_KERNEL_TRANSPOSE_Z_XY))
            {
                throw std::runtime_error(
                    "Real3DEvenNode: Unexpected child scheme from solution map");
            }
            determined_scheme_2 = child_scheme_trees[2]->curScheme;
            determined_scheme_4 = child_scheme_trees[4]->curScheme;
        }

        // first row fft + postproc is mandatory for fastest dimension
        auto rcplan = make_real_row(this);
        rcplan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[0].get());

        // first transpose
        auto trans1 = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_Z_XY, this, xyz, xyz);

        // first column
        NodeMetaData c1planData(y, y, this);
        auto c1plan = NodeFactory::CreateExplicitNode(c1planData, this, determined_scheme_2);
        c1plan->allowOutofplace = false; // let it be inplace
        c1plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[2].get());

        // second transpose
        auto trans2 = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_Z_XY, this, yzx, yzx);

        // second column
        NodeMetaData c2planData(z, z, this);
        auto c2plan = NodeFactory::CreateExplicitNode(c2planData, this, determined_scheme_4);
        c2plan->allowOutofplace = false; // let it be inplace
        c2plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[4].get());

        // third transpose
        auto trans3 = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_Z_XY, this, zxy, zxy);

        // --------------------------------
        // Fuse Shims: [RealEven + T][RT][RT]
        // 1-1. Try (stockham + r2c)(from real even) + transp
        // 1-2. else, try r2c (from real even) + transp
        // 2. RT1 = trans1 check + c1plan + trans2
        // 3. RT2 = trans2 check + c2plan + trans3
        // --------------------------------
        auto STK_R2CTrans
            = NodeFactory::CreateFuseShim(FT_STOCKHAM_R2C_TRANSPOSE, {rcplan.get(), trans1.get()});
        if(STK_R2CTrans->IsSchemeFusable())
        {
            fuseShims.emplace_back(std::move(STK_R2CTrans));
        }
        else
        {
            auto R2CTrans = NodeFactory::CreateFuseShim(FT_R2C_TRANSPOSE,
                                                        {rcplan.get(), trans1.get(), c1plan.get()});
            if(R2CTrans->IsSchemeFusable())
                fuseShims.emplace_back(std::move(R2CTrans));
        }

        auto RT1 = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS_Z_XY,
                                               {trans1.get(), c1plan.get(), trans2.get()});
        if(RT1->IsSchemeFusable())
        {
            fuseShims.emplace_back(std::move(RT1));
        }
        else
        {
            auto RTStride1
                = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS, {c1plan.get(), trans2.get()});
            if(RTStride1->IsSchemeFusable())
                fuseShims.emplace_back(std::move(RTStride1));
        }

        auto RT2 = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS_Z_XY,
                                               {trans2.get(), c2plan.get(), trans3.get()});
        if(RT2->IsSchemeFusable())
        {
            fuseShims.emplace_back(std::move(RT2));
        }
        else
        {
            auto RTStride2
                = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS, {c2plan.get(), trans3.get()});
            if(RTStride2->IsSchemeFusable())
                fuseShims.emplace_back(std::move(RTStride2));
        }

        // --------------------------------
        // 1DEven + TRTRT
        // --------------------------------
        childNodes.emplace_back(std::move(rcplan));
        childNodes.emplace_back(std::move(trans1));
        // Fuse R + TRANSPOSE_Z_XY
        childNodes.emplace_back(std::move(c1plan));
        childNodes.emplace_back(std::move(trans2));
        // Fuse R + TRANSPOSE_Z_XY
        childNodes.emplace_back(std::move(c2plan));
        childNodes.emplace_back(std::move(trans3));
    }
    else
    {
        // check schemes from solution map
        ComputeScheme determined_scheme_1 = CS_NONE;
        ComputeScheme determined_scheme_3 = CS_NONE;
        if(!noSolution)
        {
            if((child_scheme_trees.size() != 6)
               || (child_scheme_trees[0]->curScheme != CS_KERNEL_TRANSPOSE_XY_Z)
               || (child_scheme_trees[2]->curScheme != CS_KERNEL_TRANSPOSE_XY_Z)
               || (child_scheme_trees[4]->curScheme != CS_KERNEL_TRANSPOSE_XY_Z)
               || (child_scheme_trees[5]->curScheme != CS_REAL_TRANSFORM_EVEN))
            {
                throw std::runtime_error("RC3DNode: Unexpected child scheme from solution map");
            }
            determined_scheme_1 = child_scheme_trees[1]->curScheme;
            determined_scheme_3 = child_scheme_trees[3]->curScheme;
        }

        // transpose
        const auto xzy = reorder_lengths(xyz, swap_yz);
        auto trans3 = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, this, xzy, xzy);

        // column
        NodeMetaData c2planData(z, z, this);
        auto c2plan = NodeFactory::CreateExplicitNode(c2planData, this, determined_scheme_1);
        c2plan->allowOutofplace = false; // let it be inplace
        c2plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[1].get());

        // transpose
        const auto zyx = reorder_lengths(zxy, swap_yz);
        auto trans2 = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, this, zyx, zyx);

        // column
        NodeMetaData c1planData(y, y, this);
        auto c1plan = NodeFactory::CreateExplicitNode(c1planData, this, determined_scheme_3);
        c1plan->allowOutofplace = false; // let it be inplace
        c1plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[3].get());

        // transpose
        const auto yxz = reorder_lengths(yzx, swap_yz);
        auto trans1 = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, this, yxz, yxz);

        // --------------------------------
        // Fuse Shims:
        // 1. RT = c2plan + trans2 + c1plan(check-stockham)
        // --------------------------------
        auto RT = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS_XY_Z,
                                              {c2plan.get(), trans2.get(), c1plan.get()});
        if(RT->IsSchemeFusable())
            fuseShims.emplace_back(std::move(RT));

        // --------------------------------
        // TRTRT + 1DEven
        // TODO- eventually we should fuse two TR (TRANSPOSE_XY_Z_STOCKHAM)
        // --------------------------------
        childNodes.emplace_back(std::move(trans3));
        // Fuse R + TRANSPOSE_XY_Z
        childNodes.emplace_back(std::move(c2plan));
        childNodes.emplace_back(std::move(trans2));
        childNodes.emplace_back(std::move(c1plan));
        // Fuse this trans and pre-kernel-c2r of 1D-even
        childNodes.emplace_back(std::move(trans1));

        // c2r
        auto crplan = make_real_row(this);
        crplan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[5].get());
        childNodes.emplace_back(std::move(crplan));

        // --------------------------------
        // Fuse Shims:
        // 2. trans1 + c2r (first child of real even)
        // note the CheckSchemeFusable will check if the first one is transpose
        // --------------------------------
        auto TransC2R = NodeFactory::CreateFuseShim(
            FT_TRANSPOSE_C2R, {childNodes[childNodes.size() - 2].get(), childNodes.back().get()});
        if(TransC2R->IsSchemeFusable())
            fuseShims.emplace_back(std::move(TransC2R));
    }
}

void Real3DEvenNode::AssignParams_internal()
{
    switch(solution)
    {
    case REAL_2D_SINGLE_SBCC:
        AssignParams_internal_2D_SINGLE_CC();
        break;
    case INPLACE_SBCC:
        AssignParams_internal_SBCC();
        break;
    case SBCR:
        AssignParams_internal_SBCR();
        break;
    case TR_PAIRS:
        AssignParams_internal_TR_pairs();
        break;
    default:
        throw std::runtime_error("3D R2C/C2R assign params failure: " + PrintScheme(scheme));
        break;
    }
}

void Real3DEvenNode::AssignParams_internal_2D_SINGLE_CC()
{
    const bool forward  = input_array_type() == rocfft_array_type_real;
    const auto complex  = real_as_complex(forward ? input_layout() : output_layout());
    auto&      fused_2d = childNodes[forward ? 0 : 1];
    auto&      sbccZ    = childNodes[forward ? 1 : 0];
    if(forward)
    {
        set_io_strides(*fused_2d, complex, output_layout());
        fused_2d->AssignParams();

        set_inplace_strides(
            *sbccZ,
            std::array{output_stride(2), output_stride(0), output_stride(1), output_distance()});
        sbccZ->AssignParams();
    }
    else
    {
        // The Z column kernel folds the two transverse axes into one.
        set_inplace_strides(*sbccZ, std::array{input_stride(2), input_stride(0), input_distance()});
        sbccZ->AssignParams();

        // The XY kernel derives its outer distance from Z, not the caller's distance.
        const auto z = input_length(2);
        fused_2d->set_strides(
            INPUT,
            std::array{input_stride(0), input_stride(1), input_stride(2), z * input_stride(2)});
        fused_2d->set_strides(OUTPUT,
                              std::array{complex.axis_stride(0),
                                         complex.axis_stride(1),
                                         complex.axis_stride(2),
                                         z * complex.axis_stride(2)});
        fused_2d->AssignParams();
    }
}

void Real3DEvenNode::AssignParams_internal_SBCC()
{
    assert(childNodes.size() == 3);

    const bool  forward   = input_array_type() == rocfft_array_type_real;
    const auto& hermitian = forward ? output_layout() : input_layout();
    auto&       row       = childNodes[forward ? 0 : 2];
    auto&       sbccZ     = childNodes[forward ? 1 : 0];
    auto&       sbccY     = childNodes[forward ? 2 : 1];
    if(forward)
    {
        set_io_strides(*row, input_layout(), output_layout());
        row->AssignParams();
    }

    const auto x        = hermitian.axis_stride(0);
    const auto y        = hermitian.axis_stride(1);
    const auto z        = hermitian.axis_stride(2);
    const auto distance = hermitian.axis_stride(3);
    set_inplace_strides(*sbccZ, std::array{z, x, y, distance});
    sbccZ->AssignParams();
    set_inplace_strides(*sbccY, std::array{y, x, z, distance});
    sbccY->AssignParams();

    if(!forward)
    {
        set_io_strides(*row, input_layout(), output_layout());
        row->AssignParams();
    }
}

void Real3DEvenNode::AssignParams_internal_SBCR()
{
    if(childNodes.size() != 3)
        throw std::runtime_error("Require SBCR childNodes.size() == 3");

    auto& sbcrZ = childNodes[0];
    sbcrZ->set_strides(INPUT, std::array{input_stride(2), input_stride(0), input_distance()});
    sbcrZ->set_strides(OUTPUT, std::array<size_t, 3>{1, sbcrZ->input_length(0), input_distance()});
    sbcrZ->AssignParams();

    auto&      sbcrY   = childNodes[1];
    const auto y       = sbcrY->input_length(0);
    const auto y_batch = sbcrY->input_length(1);
    sbcrY->set_strides(INPUT, std::array<size_t, 3>{y_batch, 1, y * y_batch});
    sbcrY->set_strides(OUTPUT, std::array<size_t, 3>{1, y, y * y_batch});
    sbcrY->AssignParams();

    auto&      sbcrX   = childNodes[2];
    const auto x       = sbcrX->input_length(0);
    const auto x_batch = sbcrX->input_length(1);
    sbcrX->ebtype      = EmbeddedType::C2Real_PRE;
    sbcrX->set_array_type(OUTPUT, rocfft_array_type_complex_interleaved);
    // The input plane contains H bins; the kernel shape describes C bins.
    sbcrX->set_strides(INPUT, std::array<size_t, 3>{x_batch, 1, (x + 1) * x_batch});
    sbcrX->set_strides(OUTPUT, std::array<size_t, 3>{1, x, x * x_batch});
    sbcrX->AssignParams();
}

void Real3DEvenNode::AssignParams_internal_TR_pairs()
{
    if(input_array_type() == rocfft_array_type_real)
    {
        auto& row = childNodes[0];
        set_io_strides(*row, input_layout(), output_layout());
        row->AssignParams();

        // Each transpose writes a packed plane, retaining the batch distance.
        for(size_t i = 1; i < 5; i += 2)
        {
            auto&       trans    = childNodes[i];
            const auto& previous = *childNodes[i - 1];
            const auto  y        = trans->input_length(1);
            const auto  yz       = y * trans->input_length(2);
            const auto  distance = previous.output_distance();
            trans->set_strides(INPUT, previous.output_layout());
            trans->set_strides(OUTPUT, std::array<size_t, 4>{yz, 1, y, distance});

            auto& column = childNodes[i + 1];
            set_inplace_strides(*column, std::array<size_t, 4>{1, y, yz, distance});
            column->AssignParams();
        }

        auto& trans = childNodes.back();
        trans->set_strides(INPUT, childNodes[4]->output_layout());
        trans->set_strides(
            OUTPUT,
            std::array{output_stride(2), output_stride(0), output_stride(1), output_distance()});
    }
    else
    {
        for(size_t i = 0; i < 5; i += 2)
        {
            auto&       trans    = childNodes[i];
            const auto& previous = i == 0 ? input_layout() : childNodes[i - 1]->output_layout();
            const auto  distance = previous.axis_stride(3);
            // XY_Z has its last two axes swapped in each stage frame.
            trans->set_strides(INPUT,
                               std::array{previous.axis_stride(0),
                                          previous.axis_stride(2),
                                          previous.axis_stride(1),
                                          distance});
            const auto y  = trans->input_length(1);
            const auto xy = trans->input_length(0) * y;
            trans->set_strides(OUTPUT, std::array<size_t, 4>{y, 1, xy, distance});

            auto&      column  = childNodes[i + 1];
            const auto strides = std::array<size_t, 4>{1, y, xy, distance};
            if(i < 4)
                set_inplace_strides(*column, strides);
            else
            {
                column->set_strides(INPUT, strides);
                column->set_strides(OUTPUT, output_layout());
            }
            column->AssignParams();
        }
    }
}

/*****************************************************
 * CS_REAL_3D_PP
 *****************************************************/
size_t Real3DPPNode::GetPPOffDim() const
{
    const auto& real = direction == -1 ? input_layout() : output_layout();
    auto        key  = PPFMKey(real.axis_span(0),
                       real.axis_span(1),
                       real.axis_span(2),
                       get_precision(),
                       GetRootPlanTransformType(),
                       scheme);
    if(!pool.has_function(key, batch_count()))
        throw std::runtime_error("GetPPOffDim failed to find a valid kernel");

    // CS_REAL_3D_PP will have two corresponding kernels in
    // the function pool, both will have the same off-dim
    // value, and at least of one them must be an SBCC PP
    auto child_scheme = CS_KERNEL_STOCKHAM_PP_BLOCK_CC;
    auto kernel       = pool.get_kernel(key, child_scheme, batch_count());

    return kernel.pp_params.off_dim;
}

void Real3DPPNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    ppOffDim = GetPPOffDim();

    switch(ppOffDim)
    {
    case 0: // work along x will be split between y and z
    {
        // y col fft + partial pass along x
        // partial pass along x + z col fft
        throw std::runtime_error(
            "Real3DPPNode::BuildTree_internal: partial-passes along x not currently supported");
        break;
    }
    case 1: // work along y will be split between x and z
    {
        const bool  forward   = direction == -1;
        const auto& hermitian = forward ? output_layout() : input_layout();
        const auto  complex   = real_as_complex(forward ? input_layout() : output_layout());
        const auto  x         = untyped(complex.get_data_set_for_len_axes({0}));
        const auto  h         = untyped(hermitian.get_data_set_for_len_axes({0}));
        const auto  z         = untyped(hermitian.get_data_set_for_len_axes({2}));

        // X row FFT and partial pass along Y, with embedded real processing.
        auto xPartialPassPlan
            = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_PP,
                                                this,
                                                forward ? x : h.with_axis_span(0, x.axis_span(0)),
                                                forward ? h : x);
        xPartialPassPlan->ppOffDim = ppOffDim;
        xPartialPassPlan->ebtype   = forward ? EmbeddedType::Real2C_POST : EmbeddedType::C2Real_PRE;
        xPartialPassPlan->allowInplace = true;
        xPartialPassPlan->comments.push_back("partial-pass enabled for second dimension.");

        // Z column FFT and the other partial pass along Y. Keep [Z,X,Y,B].
        auto zPartialPassPlan
            = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_PP_BLOCK_CC, this, z, z);
        zPartialPassPlan->ppOffDim     = ppOffDim;
        zPartialPassPlan->allowInplace = false;

        if(forward)
        {
            zPartialPassPlan->comments.push_back("partial-pass enabled for second dimension.");
            childNodes.emplace_back(std::move(xPartialPassPlan));
            childNodes.emplace_back(std::move(zPartialPassPlan));
        }
        else
        {
            zPartialPassPlan->legacyOutputOrder = OutputAxisOrder::ROTATE_LEFT_3;
            zPartialPassPlan->comments.push_back("partial-pass enabled for first dimension.");
            childNodes.emplace_back(std::move(zPartialPassPlan));
            childNodes.emplace_back(std::move(xPartialPassPlan));
        }

        break;
    }
    case 2: // work along z will be split between x and y
    {
        // x row fft + partial pass along z
        // partial pass along z + y col fft
        throw std::runtime_error(
            "Real3DPPNode::BuildTree_internal: partial-passes along z not currently supported");
        break;
    }
    default:
        throw std::runtime_error("Real3DPPNode::BuildTree_internal:: Unexpected ppOffDim");
    }
}

void Real3DPPNode::AssignParams_internal()
{
    switch(ppOffDim)
    {
    case 0: // work along x will be split between y and z
    {
        // y col fft + partial pass along x
        // partial pass along x + z col fft
        throw std::runtime_error(
            "Real3DPPNode::AssignParams_internal: partial-passes along x not currently supported");
        break;
    }

    case 1: // work along y will be split between x and z
    {
        const bool  forward   = direction == -1;
        const auto  complex   = real_as_complex(forward ? input_layout() : output_layout());
        const auto& hermitian = forward ? output_layout() : input_layout();
        auto&       xyPlan    = childNodes[forward ? 0 : 1];
        auto&       yzPlan    = childNodes[forward ? 1 : 0];
        if(forward)
        {
            set_io_strides(*xyPlan, complex, hermitian);
            xyPlan->AssignParams();

            set_inplace_strides(*yzPlan,
                                std::array{output_stride(2),
                                           output_stride(0),
                                           output_stride(1),
                                           xyPlan->output_distance()});
            yzPlan->AssignParams();
        }
        else
        {
            set_inplace_strides(
                *yzPlan,
                std::array{input_stride(2), input_stride(0), input_stride(1), input_distance()});
            yzPlan->AssignParams();

            set_io_strides(*xyPlan, hermitian, complex);
            xyPlan->AssignParams();
        }

        break;
    }
    case 2: // work along z will be split between x and y
    {
        // x row fft + partial pass along z
        // partial pass along z + y col fft
        throw std::runtime_error(
            "Real3DPPNode::AssignParams_internal: partial-passes along z not currently supported");
        break;
    }
    default:
        throw std::runtime_error("Real3DPPNode::AssignParams_internal: Unexpected ppOffDim");
    }
}

/*****************************************************
 * CS_KERNEL_R_TO_CMPLX
 * CS_KERNEL_R_TO_CMPLX_TRANSPOSE
 * CS_KERNEL_CMPLX_TO_R
 * CS_KERNEL_TRANSPOSE_CMPLX_TO_R
 *****************************************************/

size_t PrePostKernelNode::GetTwiddleTableLength()
{
    if(scheme == CS_KERNEL_R_TO_CMPLX || scheme == CS_KERNEL_R_TO_CMPLX_TRANSPOSE)
        return 2 * input_length(0);
    if(scheme == CS_KERNEL_CMPLX_TO_R)
        return 2 * (input_length(0) - 1);
    else if(scheme == CS_KERNEL_TRANSPOSE_CMPLX_TO_R)
        return 2 * (input_length(length_rank() - 1) - 1);

    throw std::runtime_error("GetTwiddleTableLength: Unexpected scheme in PrePostKernelNode: "
                             + PrintScheme(scheme));
}

size_t PrePostKernelNode::GetTwiddleTableLengthLimit()
{
    // The kernel only uses 1/4th of the real length twiddle table
    return DivRoundingUp<size_t>(GetTwiddleTableLength(), 4);
}
