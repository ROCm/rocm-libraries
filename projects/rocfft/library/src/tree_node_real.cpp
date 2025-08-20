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

#include "tree_node_real.h"
#include "../../shared/arithmetic.h"
#include "function_pool.h"
#include "node_factory.h"
#include "real2complex.h"

// work out the real and complex lengths on a real-complex plan, and
// return pointers to those lengths
static void set_complex_length(TreeNode&                   node,
                               const std::vector<size_t>*& realLength,
                               const std::vector<size_t>*& complexLength)
{
    // if output is not set yet, then node.length counts in real
    // units.  compute number of complex units and make both input
    // and output lengths correct
    if(node.outputLength.empty())
    {
        node.outputLength = node.length;
        node.outputLength.front() /= 2;
        node.outputLength.front() += 1;

        if(node.direction == 1)
            std::swap(node.length, node.outputLength);
    }

    if(node.direction == -1)
    {
        // forward transform
        realLength    = &node.length;
        complexLength = &node.outputLength;
    }
    else
    {
        // complex
        complexLength = &node.length;
        realLength    = &node.outputLength;
    }
}

// check if we have an SBCC kernel along the specified dimension
static bool SBCC_dim_available(function_pool&             pool,
                               const std::vector<size_t>& length,
                               size_t                     sbcc_dim,
                               rocfft_precision           precision)
{
    // Check the C part.
    // The first R is built recursively with 2D_FFT, leave the check part to themselves
    size_t numTrans = 0;
    // do we have a purpose-built sbcc kernel
    bool  have_sbcc = false;
    FMKey sbcc_key(length[sbcc_dim], precision, CS_KERNEL_STOCKHAM_BLOCK_CC);
    if(pool.has_function(sbcc_key))
    {
        numTrans  = pool.get_kernel(sbcc_key).transforms_per_block;
        have_sbcc = true;
    }
    else
    {
        FMKey normal_key(length[sbcc_dim], precision);
        if(!pool.has_function(normal_key))
            return false;
        numTrans = pool.get_kernel(normal_key).transforms_per_block;
    }

    // NB:
    //  we can remove this limitation if we are using sbcc instead of stockham 1d,
    //  (especially for sbcc with load-to-reg, the numTrans is increased)
    // if(length[0] < numTrans && !have_sbcc)
    //     return false;

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

// check if we have an SBCR kernel along the specified dimension
static bool SBCR_dim_available(const function_pool&       pool,
                               const std::vector<size_t>& length,
                               size_t                     sbcr_dim,
                               rocfft_precision           precision)
{
    return pool.has_SBCR_kernel(length[sbcr_dim], precision);
}

/*****************************************************
 * CS_REAL_TRANSFORM_USING_CMPLX
 *****************************************************/
void RealTransCmplxNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    // Embed the data into a full-length complex array, perform a
    // complex transform, and then extract the relevant output.
    bool          r2c            = inArrayType == rocfft_array_type_real;
    ComputeScheme copyHeadScheme = r2c ? CS_KERNEL_COPY_R_TO_CMPLX : CS_KERNEL_COPY_HERM_TO_CMPLX;
    ComputeScheme copyTailScheme = r2c ? CS_KERNEL_COPY_CMPLX_TO_HERM : CS_KERNEL_COPY_CMPLX_TO_R;

    const std::vector<size_t>* realLength    = nullptr;
    const std::vector<size_t>* complexLength = nullptr;
    set_complex_length(*this, realLength, complexLength);

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

    auto copyHeadPlan = NodeFactory::CreateNodeFromScheme(copyHeadScheme, this);
    // head copy plan
    copyHeadPlan->dimension = dimension;
    copyHeadPlan->length    = length;
    if(!r2c)
        copyHeadPlan->outputLength = *realLength;
    childNodes.emplace_back(std::move(copyHeadPlan));

    // complex fft
    NodeMetaData fftPlanData(this);
    fftPlanData.dimension = dimension;
    fftPlanData.length    = *realLength;

    auto fftPlan = NodeFactory::CreateExplicitNode(fftPlanData, this, determined_scheme);
    fftPlan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[1].get());

    // NB:
    //   The tail copy kernel allows only CI type, so the previous kernel should output CI type
    // TODO: make it more elegant..
    //   for example, simply set allowedOutArrayTypes to fftPlan without GetLastLeaf() (propagate)
    //   or add a allowedInArrayType..
    fftPlan->GetLastLeaf()->allowedOutArrayTypes = {rocfft_array_type_complex_interleaved};
    childNodes.emplace_back(std::move(fftPlan));

    auto copyTailPlan       = NodeFactory::CreateNodeFromScheme(copyTailScheme, this);
    copyTailPlan->dimension = dimension;
    copyTailPlan->length    = *realLength;
    if(r2c)
        copyTailPlan->outputLength = *complexLength;

    childNodes.emplace_back(std::move(copyTailPlan));
}

void RealTransCmplxNode::AssignParams_internal()
{
    assert(childNodes.size() == 3);
    auto& copyHeadPlan = childNodes[0];
    auto& fftPlan      = childNodes[1];
    auto& copyTailPlan = childNodes[2];

    copyHeadPlan->inStride = inStride;
    copyHeadPlan->iDist    = iDist;

    copyHeadPlan->outStride.push_back(1);
    copyHeadPlan->oDist = copyHeadPlan->outputLength.empty() ? copyHeadPlan->length[0]
                                                             : copyHeadPlan->outputLength[0];
    for(size_t index = 1; index < length.size(); index++)
    {
        copyHeadPlan->outStride.push_back(copyHeadPlan->oDist);
        copyHeadPlan->oDist *= length[index];
    }

    fftPlan->inStride  = copyHeadPlan->outStride;
    fftPlan->iDist     = copyHeadPlan->oDist;
    fftPlan->outStride = fftPlan->inStride;
    fftPlan->oDist     = fftPlan->iDist;

    fftPlan->AssignParams();

    copyTailPlan->inStride = fftPlan->outStride;
    copyTailPlan->iDist    = fftPlan->oDist;

    copyTailPlan->outStride = outStride;
    copyTailPlan->oDist     = oDist;
}

/*****************************************************
 * CS_REAL_TRANSFORM_EVEN
 *****************************************************/
void RealTransEvenNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    const std::vector<size_t>* realLength    = nullptr;
    const std::vector<size_t>* complexLength = nullptr;
    set_complex_length(*this, realLength, complexLength);

    // Fastest moving dimension must be even:
    if(realLength->at(0) % 2 != 0)
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

    // NB:
    // immediate FFT children of CS_REAL_TRANSFORM_EVEN must be
    // working directly on the real buffer, but pretending it's
    // complex

    NodeMetaData cfftPlanData(this);
    cfftPlanData.dimension = dimension;
    cfftPlanData.length    = *realLength;
    cfftPlanData.length[0] = cfftPlanData.length[0] / 2;
    auto cfftPlan          = NodeFactory::CreateExplicitNode(cfftPlanData, this, determined_scheme);
    // NB: the buffer is real, but we treat it as complex
    cfftPlan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[fft_node_id].get());

    if(noSolution)
    {
        // fuse pre/post-processing into fft if it's single-kernel
        if(try_fuse_pre_post_processing)
            try_fuse_pre_post_processing = cfftPlan->isLeafNode();

        // Enable fusion for small simple 1D cases only.
        // TODO: remove it after full solution done.
        if((cfftPlan->scheme == CS_KERNEL_STOCKHAM) && // simple decomposition
           (length.size() == 1) && // 1D
           (length[0] < 512) && // < small case < 512
           (inArrayType != rocfft_array_type_hermitian_planar) && // no planar
           (outArrayType != rocfft_array_type_hermitian_planar))
        {
            try_fuse_pre_post_processing = true;
        }
    }

    switch(direction)
    {
    case -1:
    {
        // real-to-complex transform: in-place complex transform then post-process

        if(try_fuse_pre_post_processing)
        {
            cfftPlan->ebtype       = EmbeddedType::Real2C_POST;
            cfftPlan->outputLength = cfftPlan->length;
            cfftPlan->outputLength.front() += 1;
        }

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

            auto postPlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_R_TO_CMPLX, this);
            postPlan->dimension = 1;
            postPlan->length    = *realLength;
            postPlan->length[0] /= 2;
            postPlan->outputLength = *complexLength;
            childNodes.emplace_back(std::move(postPlan));
        }
        break;
    }
    case 1:
    {
        // complex-to-real transform: pre-process followed by in-place complex transform

        if(!try_fuse_pre_post_processing)
        {
            // add separate pre-processing if we couldn't fuse
            auto prePlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_CMPLX_TO_R, this);
            prePlan->dimension = 1;
            prePlan->length    = *complexLength;
            // output of the prePlan is in complex units
            prePlan->outputLength = outputLength;
            prePlan->outputLength.front() /= 2;
            childNodes.emplace_back(std::move(prePlan));
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

    if(direction == -1)
    {
        // forward transform, r2c

        auto& fftPlan     = childNodes[0];
        fftPlan->inStride = inStride;
        for(unsigned int i = 1; i < fftPlan->inStride.size(); ++i)
        {
            fftPlan->inStride[i] /= 2;
        }
        fftPlan->iDist     = iDist / 2;
        fftPlan->outStride = inStride;
        for(unsigned int i = 1; i < fftPlan->outStride.size(); ++i)
        {
            fftPlan->outStride[i] /= 2;
        }
        fftPlan->oDist = iDist / 2;
        fftPlan->AssignParams();
        assert(fftPlan->length.size() == fftPlan->inStride.size());
        assert(fftPlan->length.size() == fftPlan->outStride.size());

        if(childNodes.size() == 2)
        {
            auto& postPlan = childNodes[1];
            assert(postPlan->scheme == CS_KERNEL_R_TO_CMPLX
                   || postPlan->scheme == CS_KERNEL_R_TO_CMPLX_TRANSPOSE);
            postPlan->inStride = inStride;
            for(unsigned int i = 1; i < postPlan->inStride.size(); ++i)
            {
                postPlan->inStride[i] /= 2;
            }
            postPlan->iDist     = iDist / 2;
            postPlan->outStride = outStride;
            postPlan->oDist     = oDist;

            assert(postPlan->length.size() == postPlan->inStride.size());
            assert(postPlan->length.size() == postPlan->outStride.size());
        }
        else
        {
            // we fused post-proc into the FFT kernel, so give the correct out strides
            fftPlan->outStride = outStride;
            fftPlan->oDist     = oDist;
        }
    }
    else
    {
        // backward transform, c2r
        bool fusedPreProcessing = childNodes[0]->ebtype == EmbeddedType::C2Real_PRE;

        // oDist is in reals, subplan->oDist is in complexes

        if(!fusedPreProcessing)
        {
            auto& prePlan = childNodes[0];
            assert(prePlan->scheme == CS_KERNEL_CMPLX_TO_R);

            prePlan->iDist = iDist;
            prePlan->oDist = oDist / 2;

            // Strides are actually distances for multimensional transforms.
            // Only the first value is used, but we require dimension values.
            prePlan->inStride  = inStride;
            prePlan->outStride = outStride;
            // Strides are in complex types
            for(unsigned int i = 1; i < prePlan->outStride.size(); ++i)
            {
                prePlan->outStride[i] /= 2;
            }
            assert(prePlan->length.size() == prePlan->inStride.size());
            assert(prePlan->length.size() == prePlan->outStride.size());
        }

        auto& fftPlan = fusedPreProcessing ? childNodes[0] : childNodes[1];
        // Transform the strides from real to complex.

        fftPlan->inStride  = fusedPreProcessing ? inStride : outStride;
        fftPlan->iDist     = fusedPreProcessing ? iDist : oDist / 2;
        fftPlan->outStride = outStride;
        fftPlan->oDist     = oDist / 2;
        // The strides must be translated from real to complex.
        for(unsigned int i = 1; i < fftPlan->inStride.size(); ++i)
        {
            if(!fusedPreProcessing)
                fftPlan->inStride[i] /= 2;
            fftPlan->outStride[i] /= 2;
        }

        fftPlan->AssignParams();
        assert(fftPlan->length.size() == fftPlan->inStride.size());
        assert(fftPlan->length.size() == fftPlan->outStride.size());

        // we apply callbacks on the root plan's output
        TreeNode* rootPlan = this;
        while(rootPlan->parent != nullptr)
            rootPlan = rootPlan->parent;
    }
}

/*****************************************************
 * CS_REAL_2D_EVEN
 *****************************************************/
void Real2DEvenNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    const std::vector<size_t>* realLength    = nullptr;
    const std::vector<size_t>* complexLength = nullptr;
    set_complex_length(*this, realLength, complexLength);

    // Fastest moving dimension must be even:
    if(realLength->at(0) % 2 != 0)
        throw std::runtime_error("fastest dimension is not even in RealTransEvenNode");

    //
    // TODO- we need to replace the solution-decision part with plan-solution.
    //

    // if we have SBCC for the higher dimension, use that and avoid transpose
    solution = SBCC_dim_available(pool, length, 1, precision) ? INPLACE_SBCC : TR_PAIR;

    // but if we have 2D_SINGLE available, then we use it
    // NB: use the check function in NodeFactory to make sure lds limit
    // TODO- this part should be done in offline-tuning
    NodeMetaData nodeData(this);
    if(inArrayType == rocfft_array_type_real) //forward
    {
        nodeData.length = {length[0] / 2, length[1]};
        if(NodeFactory::use_CS_2D_SINGLE(pool, nodeData, inArrayType, outArrayType))
            solution = REAL_2D_SINGLE;
    }
    else
    {
        nodeData.length = {outputLength[1], outputLength[0] / 2};
        if(NodeFactory::use_CS_2D_SINGLE(pool, nodeData, inArrayType, outArrayType))
            solution = REAL_2D_SINGLE;
    }

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
    if(inArrayType == rocfft_array_type_real)
    {
        NodeMetaData cfftPlanData(this);
        cfftPlanData.dimension = dimension;
        cfftPlanData.length    = length;
        cfftPlanData.length[0] = cfftPlanData.length[0] / 2;

        auto cfftPlan = NodeFactory::CreateExplicitNode(cfftPlanData, this);
        cfftPlan->RecursiveBuildTree();
        cfftPlan->ebtype          = EmbeddedType::Real2C_POST;
        cfftPlan->allowOutofplace = true;
        cfftPlan->outputLength    = cfftPlan->length;
        cfftPlan->outputLength.front() += 1;

        childNodes.emplace_back(std::move(cfftPlan));
    }
    else
    {
        NodeMetaData cfftPlanData(this);
        cfftPlanData.dimension = dimension;
        cfftPlanData.length    = {outputLength[1], outputLength[0] / 2};

        auto cfftPlan = NodeFactory::CreateExplicitNode(cfftPlanData, this);
        cfftPlan->RecursiveBuildTree();
        cfftPlan->ebtype          = EmbeddedType::C2Real_PRE;
        cfftPlan->allowOutofplace = true;

        childNodes.emplace_back(std::move(cfftPlan));
    }
}

void Real2DEvenNode::BuildTree_internal_SBCC(SchemeTreeVec& child_scheme_trees)
{
    bool haveSBCC   = pool.has_SBCC_kernel(length[1], precision);
    auto sbccScheme = haveSBCC ? CS_KERNEL_STOCKHAM_BLOCK_CC : CS_KERNEL_STOCKHAM;
    bool noSolution = child_scheme_trees.empty();

    if(inArrayType == rocfft_array_type_real) //forward
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
        auto rcplan = NodeFactory::CreateNodeFromScheme(CS_REAL_TRANSFORM_EVEN, this);
        // for length > 2048, don't try pre/post because LDS usage is too high
        static_cast<RealTransEvenNode*>(rcplan.get())->try_fuse_pre_post_processing
            = length[0] <= 2048;

        rcplan->length    = length;
        rcplan->dimension = 1;
        rcplan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[0].get());
        childNodes.emplace_back(std::move(rcplan));

        auto sbccY          = NodeFactory::CreateNodeFromScheme(determined_scheme_1, this);
        sbccY->length       = childNodes.back()->outputLength;
        sbccY->outputLength = sbccY->length;
        std::swap(sbccY->length[0], sbccY->length[1]);
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

        auto sbccY          = NodeFactory::CreateNodeFromScheme(determined_scheme_0, this);
        sbccY->outputLength = length;
        sbccY->length       = length;
        std::swap(sbccY->length[0], sbccY->length[1]);
        childNodes.emplace_back(std::move(sbccY));

        // c2r
        auto crplan = NodeFactory::CreateNodeFromScheme(CS_REAL_TRANSFORM_EVEN, this);
        // for length > 2048, don't try pre/post because LDS usage is too high
        static_cast<RealTransEvenNode*>(crplan.get())->try_fuse_pre_post_processing
            = length[0] <= 2048;

        crplan->length    = outputLength;
        crplan->dimension = 1;
        crplan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[1].get());
        childNodes.emplace_back(std::move(crplan));
    }
}

void Real2DEvenNode::BuildTree_internal_TR_pair(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    if(inArrayType == rocfft_array_type_real) //forward
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
        auto row1Plan       = NodeFactory::CreateNodeFromScheme(CS_REAL_TRANSFORM_EVEN, this);
        row1Plan->length    = length;
        row1Plan->dimension = 1;
        row1Plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[0].get());

        // first transpose
        auto trans1Plan    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this);
        trans1Plan->length = row1Plan->outputLength;
        trans1Plan->SetTransposeOutputLength();

        // second row fft
        NodeMetaData row2PlanData(this);
        row2PlanData.length    = trans1Plan->outputLength;
        row2PlanData.dimension = 1;
        auto row2Plan = NodeFactory::CreateExplicitNode(row2PlanData, this, determined_scheme_2);
        row2Plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[2].get());

        // second transpose
        auto trans2Plan    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this);
        trans2Plan->length = trans1Plan->outputLength;
        trans2Plan->SetTransposeOutputLength();

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
        auto trans1Plan    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this);
        trans1Plan->length = length;
        trans1Plan->SetTransposeOutputLength();
        trans1Plan->dimension = 2;

        // c2c row transform
        NodeMetaData c2cPlanData(this);
        c2cPlanData.dimension = 1;
        c2cPlanData.length    = trans1Plan->outputLength;
        auto c2cPlan = NodeFactory::CreateExplicitNode(c2cPlanData, this, determined_scheme_1);
        c2cPlan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[1].get());

        // second transpose
        auto trans2plan    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this);
        trans2plan->length = trans1Plan->outputLength;
        trans2plan->SetTransposeOutputLength();
        trans2plan->dimension = 2;
        // NOTE

        // c2r row transform
        if(!noSolution && (child_scheme_trees[3]->curScheme != CS_REAL_TRANSFORM_EVEN))
            throw std::runtime_error("Real2DEvenNode: Unexpected child scheme from solution map");
        auto c2rPlan    = NodeFactory::CreateNodeFromScheme(CS_REAL_TRANSFORM_EVEN, this);
        c2rPlan->length = outputLength;
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
    const bool forward = inArrayType == rocfft_array_type_real;
    if(forward)
    {
        auto& fused_2d     = childNodes[0];
        fused_2d->inStride = inStride;
        for(unsigned int i = 1; i < fused_2d->inStride.size(); ++i)
        {
            fused_2d->inStride[i] /= 2;
        }
        fused_2d->iDist     = iDist / 2;
        fused_2d->outStride = outStride;
        fused_2d->oDist     = oDist;
        fused_2d->AssignParams();
    }
    else
    {
        auto& fused_2d      = childNodes[0];
        fused_2d->inStride  = inStride;
        fused_2d->iDist     = iDist;
        fused_2d->outStride = outStride;
        fused_2d->oDist     = oDist / 2;
        // The strides must be translated from real to complex.
        for(unsigned int i = 1; i < fused_2d->inStride.size(); ++i)
        {
            fused_2d->outStride[i] /= 2;
        }
        std::swap(fused_2d->inStride[0], fused_2d->inStride[1]);
        std::swap(fused_2d->outStride[0], fused_2d->outStride[1]);

        fused_2d->AssignParams();
    }
}

void Real2DEvenNode::AssignParams_internal_SBCC()
{
    const bool forward = direction == -1;
    if(forward)
    {
        auto& rowPlan = childNodes[0];

        rowPlan->inStride = inStride;
        rowPlan->iDist    = iDist;

        rowPlan->outStride = outStride;
        rowPlan->oDist     = oDist;

        rowPlan->AssignParams();

        auto& sbccY     = childNodes[1];
        sbccY->inStride = rowPlan->outStride;
        std::swap(sbccY->inStride[0], sbccY->inStride[1]);
        sbccY->iDist = rowPlan->oDist;

        sbccY->outStride = sbccY->inStride;
        sbccY->oDist     = sbccY->iDist;
    }
    else
    {
        // input strides for last c2r node
        std::vector<size_t> c2r_inStride = inStride;
        size_t              c2r_iDist    = iDist;
        auto&               sbccY        = childNodes[0];
        sbccY->inStride                  = inStride;
        // SBCC along Y dim
        std::swap(sbccY->inStride[0], sbccY->inStride[1]);
        sbccY->iDist     = iDist;
        sbccY->outStride = sbccY->inStride;
        sbccY->oDist     = iDist;
        sbccY->AssignParams();

        auto& crplan = childNodes.back();
        {
            crplan->inStride  = c2r_inStride;
            crplan->iDist     = c2r_iDist;
            crplan->outStride = outStride;
            crplan->oDist     = oDist;
            crplan->dimension = 1;
            crplan->AssignParams();
        }
    }
}

void Real2DEvenNode::AssignParams_internal_TR_pair()
{
    const bool forward = inArrayType == rocfft_array_type_real;
    if(forward)
    {
        auto& row1Plan = childNodes[0];
        {
            // The first sub-plan changes type in real/complex transforms.
            row1Plan->inStride = inStride;
            row1Plan->iDist    = iDist;

            row1Plan->outStride = outStride;
            row1Plan->oDist     = oDist;

            row1Plan->AssignParams();
        }

        auto& trans1Plan = childNodes[1];
        {
            // B -> T
            trans1Plan->inStride = row1Plan->outStride;
            trans1Plan->iDist    = row1Plan->oDist;

            trans1Plan->outStride.push_back(trans1Plan->length[1]);
            trans1Plan->outStride.push_back(1);
            trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->outStride[0];
        }

        auto& row2Plan = childNodes[2];
        {
            // T -> T
            row2Plan->inStride = trans1Plan->outStride;
            std::swap(row2Plan->inStride[0], row2Plan->inStride[1]);
            row2Plan->iDist = trans1Plan->oDist;

            row2Plan->outStride = row2Plan->inStride;
            row2Plan->oDist     = row2Plan->iDist;

            row2Plan->AssignParams();
        }

        auto& trans2Plan = childNodes[3];
        {
            // T -> B
            trans2Plan->inStride = row2Plan->outStride;
            trans2Plan->iDist    = row2Plan->oDist;

            trans2Plan->outStride = outStride;
            std::swap(trans2Plan->outStride[0], trans2Plan->outStride[1]);
            trans2Plan->oDist = oDist;
        }
    }
    else
    {
        auto& trans1Plan = childNodes[0];
        {
            trans1Plan->inStride = inStride;
            trans1Plan->iDist    = iDist;

            trans1Plan->outStride.push_back(trans1Plan->length[1]);
            trans1Plan->outStride.push_back(1);
            trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->outStride[0];
        }
        auto& c2cPlan = childNodes[1];
        {
            c2cPlan->inStride = trans1Plan->outStride;
            std::swap(c2cPlan->inStride[0], c2cPlan->inStride[1]);
            c2cPlan->iDist = trans1Plan->oDist;

            c2cPlan->outStride = c2cPlan->inStride;
            c2cPlan->oDist     = c2cPlan->iDist;

            c2cPlan->AssignParams();
        }
        auto& trans2Plan = childNodes[2];
        {
            trans2Plan->inStride = trans1Plan->outStride;
            std::swap(trans2Plan->inStride[0], trans2Plan->inStride[1]);
            trans2Plan->iDist = trans1Plan->oDist;

            trans2Plan->outStride = trans1Plan->inStride;
            std::swap(trans2Plan->outStride[0], trans2Plan->outStride[1]);
            trans2Plan->oDist = trans2Plan->length[0] * trans2Plan->outStride[0];
        }
        auto& c2rPlan = childNodes[3];
        {
            c2rPlan->inStride = trans2Plan->outStride;
            std::swap(c2rPlan->inStride[0], c2rPlan->inStride[1]);
            c2rPlan->iDist = trans2Plan->oDist;

            c2rPlan->outStride = outStride;
            c2rPlan->oDist     = oDist;

            c2rPlan->AssignParams();
        }
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

void Real3DEvenNode::Build_solution()
{
    const std::vector<size_t>* realLength    = nullptr;
    const std::vector<size_t>* complexLength = nullptr;
    set_complex_length(*this, realLength, complexLength);

    // Fastest moving dimension must be even:
    if(realLength->at(0) % 2 != 0)
        throw std::runtime_error("fastest dimension is not even in RealTransEvenNode");

    const bool forward = inArrayType == rocfft_array_type_real;

    const bool planInStrideUnit  = this->GetPlanRoot()->inStrideUnit;
    const bool planOutStrideUnit = this->GetPlanRoot()->outStrideUnit;

    // but if we have 2D_SINGLE available, then we use it
    // NB: use the check function in NodeFactory to make sure lds limit
    // TODO- this part should be done in offline-tuning
    NodeMetaData nodeData(this);
    if(forward)
    {
        nodeData.length = {length[0] / 2, length[1]};
        if(NodeFactory::use_CS_2D_SINGLE(pool, nodeData, inArrayType, outArrayType)
           && SBCC_dim_available(pool, length, 2, precision)
           && (planInStrideUnit && planOutStrideUnit))
        {
            solution = REAL_2D_SINGLE_SBCC;
            return;
        }
    }
    else
    {
        nodeData.length = {outputLength[1], outputLength[0] / 2};
        if(NodeFactory::use_CS_2D_SINGLE(pool, nodeData, inArrayType, outArrayType)
           && SBCC_dim_available(pool, outputLength, 2, precision)
           && (planInStrideUnit && planOutStrideUnit))
        {
            solution = REAL_2D_SINGLE_SBCC;
            return;
        }
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
        std::vector<size_t> c2r_length = {outputLength[0] / 2, outputLength[1], outputLength[2]};
        if((is_device_gcn_arch(deviceProp, "gfx908") || is_device_gcn_arch(deviceProp, "gfx90a"))
           && (SBCR_dim_available(pool, c2r_length, 0, precision))
           && (SBCR_dim_available(pool, c2r_length, 1, precision))
           && (SBCR_dim_available(pool, c2r_length, 2, precision))
           && (placement
               == rocfft_placement_notinplace) // In-place SBCC is faster than SBCR solution for in-place
           && (inStride[0] == 1 && outStride[0] == 1
               && inStride[1] == complexLength->at(0) // unit strides
               && outStride[1] == realLength->at(0)
               && inStride[2] == inStride[1] * complexLength->at(1)
               && outStride[2] == outStride[1] * realLength->at(1)))
        {
            solution = SBCR;
            return;
        }
    }

    // if we have SBCC kernels for the other two dimensions, transform them using SBCC and avoid transposes.
    bool sbcc_inplace = SBCC_dim_available(pool, length, 1, precision)
                        && SBCC_dim_available(pool, length, 2, precision);
#if 0
    // ensure the fastest dimensions are big enough to get enough
    // column tiles to perform well
    if(length[0] <= 52 || length[1] <= 52)
        sbcc_inplace = false;
    // also exclude particular problematic sizes for higher dims
    if(length[1] == 168 || length[2] == 168)
        sbcc_inplace = false;
    // if all 3 lengths are SBRC-able, then R2C will already be 3
    // kernel.  SBRC should be slightly better since row accesses
    // should be a bit nicer in general than column accesses.
    if(pool.has_SBRC_kernel( length[0] / 2, precision)
       && pool.has_SBRC_kernel( length[1], precision)
       && pool.has_SBRC_kernel( length[2], precision))
    {
        sbcc_inplace = false;
    }
#endif

    solution = (sbcc_inplace) ? INPLACE_SBCC : TR_PAIRS;
}

void Real3DEvenNode::BuildTree_internal_2D_SINGLE_CC()
{
    const bool forward = inArrayType == rocfft_array_type_real;

    if(forward)
    {
        auto xyPlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_2D_SINGLE, this);
        xyPlan->length    = {length[0] / 2, length[1], length[2]};
        xyPlan->dimension = 2;
        xyPlan->RecursiveBuildTree();
        xyPlan->ebtype       = EmbeddedType::Real2C_POST;
        xyPlan->outputLength = xyPlan->length;
        xyPlan->outputLength.front() += 1;

        bool haveSBCC_Z     = pool.has_SBCC_kernel(length[2], precision);
        auto scheme         = haveSBCC_Z ? CS_KERNEL_STOCKHAM_BLOCK_CC : CS_KERNEL_STOCKHAM;
        auto sbccZ          = NodeFactory::CreateNodeFromScheme(scheme, this);
        sbccZ->length       = {length[2], (length[0] / 2 + 1), length[1]};
        sbccZ->outputLength = outputLength;
        sbccZ->dimension    = 1;

        childNodes.emplace_back(std::move(xyPlan));
        childNodes.emplace_back(std::move(sbccZ));
    }
    else
    {
        bool haveSBCC_Z  = pool.has_SBCC_kernel(outputLength[2], precision);
        auto scheme      = haveSBCC_Z ? CS_KERNEL_STOCKHAM_BLOCK_CC : CS_KERNEL_STOCKHAM;
        auto sbccZ       = NodeFactory::CreateNodeFromScheme(scheme, this);
        sbccZ->length    = {outputLength[2], (outputLength[0] / 2 + 1) * outputLength[1]};
        sbccZ->dimension = 1;

        auto xyPlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_2D_SINGLE, this);
        xyPlan->length    = {outputLength[1], outputLength[0] / 2, outputLength[2]};
        xyPlan->dimension = 2;
        xyPlan->RecursiveBuildTree();
        xyPlan->ebtype = EmbeddedType::C2Real_PRE;

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

    auto add_sbcc_children = [this](const std::vector<size_t>& remainingLength,
                                    ComputeScheme              determined_scheme_dimZ,
                                    ComputeScheme              determined_scheme_dimY) {
        ComputeScheme scheme;

        // Performance improvements for (192,192,192) with SBCC.
        auto use_SBCC_192 = (remainingLength[2] == 192 && remainingLength[1] == 192)
                            && (precision == rocfft_precision_single);

        // A special case (192,200,XX), (168,192,XX) on gfx908, we eventually need to remove these
        if(is_device_gcn_arch(deviceProp, "gfx908"))
        {
            if(((remainingLength[2] == 192 && remainingLength[1] == 200)
                || (remainingLength[2] == 168 && remainingLength[1] == 192))
               && (precision == rocfft_precision_single))
                use_SBCC_192 = true;
        }

        // SBCC along Z dimension
        if(remainingLength[2] == 192)
            scheme = use_SBCC_192 ? CS_KERNEL_STOCKHAM_BLOCK_CC : CS_KERNEL_STOCKHAM;
        else
        {
            bool haveSBCC_Z = pool.has_SBCC_kernel(remainingLength[2], precision);
            scheme          = haveSBCC_Z ? CS_KERNEL_STOCKHAM_BLOCK_CC : CS_KERNEL_STOCKHAM;
        }
        scheme        = (determined_scheme_dimZ == CS_NONE) ? scheme : determined_scheme_dimZ;
        auto sbccZ    = NodeFactory::CreateNodeFromScheme(scheme, this);
        sbccZ->length = remainingLength;
        std::swap(sbccZ->length[1], sbccZ->length[2]);
        std::swap(sbccZ->length[0], sbccZ->length[1]);
        sbccZ->outputLength = remainingLength;
        childNodes.emplace_back(std::move(sbccZ));

        // SBCC along Y dimension
        if(remainingLength[1] == 192)
            scheme = use_SBCC_192 ? CS_KERNEL_STOCKHAM_BLOCK_CC : CS_KERNEL_STOCKHAM;
        else
        {
            bool haveSBCC_Y = pool.has_SBCC_kernel(remainingLength[1], precision);
            scheme          = haveSBCC_Y ? CS_KERNEL_STOCKHAM_BLOCK_CC : CS_KERNEL_STOCKHAM;
        }
        scheme        = (determined_scheme_dimY == CS_NONE) ? scheme : determined_scheme_dimY;
        auto sbccY    = NodeFactory::CreateNodeFromScheme(scheme, this);
        sbccY->length = remainingLength;
        std::swap(sbccY->length[0], sbccY->length[1]);
        sbccY->outputLength = remainingLength;
        childNodes.emplace_back(std::move(sbccY));
    };

    std::vector<size_t> remainingLength = direction == -1 ? outputLength : length;

    if(inArrayType == rocfft_array_type_real) // forward
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
        auto rcplan = NodeFactory::CreateNodeFromScheme(CS_REAL_TRANSFORM_EVEN, this);
        // for length > 2048, don't try pre/post because LDS usage is too high
        static_cast<RealTransEvenNode*>(rcplan.get())->try_fuse_pre_post_processing
            = length[0] <= 2048;

        rcplan->length    = length;
        rcplan->dimension = 1;
        rcplan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[0].get());

        // if we have SBCC kernels for the other two dimensions, transform them using SBCC and avoid transposes
        childNodes.emplace_back(std::move(rcplan));
        add_sbcc_children(remainingLength, determined_scheme_dimZ, determined_scheme_dimY);
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

        add_sbcc_children(remainingLength, determined_scheme_dimZ, determined_scheme_dimY);

        // c2r
        auto crplan = NodeFactory::CreateNodeFromScheme(CS_REAL_TRANSFORM_EVEN, this);
        // for length > 2048, don't try pre/post because LDS usage is too high
        static_cast<RealTransEvenNode*>(crplan.get())->try_fuse_pre_post_processing
            = length[0] <= 2048;

        crplan->length    = outputLength;
        crplan->dimension = 1;
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

    auto sbcrZ       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CR, this);
    sbcrZ->length    = {outputLength[2], (outputLength[0] / 2 + 1) * outputLength[1]};
    sbcrZ->dimension = 1;
    childNodes.emplace_back(std::move(sbcrZ));

    auto sbcrY       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CR, this);
    sbcrY->length    = {outputLength[1], outputLength[2] * (outputLength[0] / 2 + 1)};
    sbcrY->dimension = 1;
    childNodes.emplace_back(std::move(sbcrY));

    auto sbcrX       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CR, this);
    sbcrX->length    = {outputLength[0] / 2, outputLength[1] * outputLength[2]};
    sbcrX->dimension = 1;
    childNodes.emplace_back(std::move(sbcrX));
}

void Real3DEvenNode::BuildTree_internal_TR_pairs(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    if(inArrayType == rocfft_array_type_real) // forward
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
        auto rcplan = NodeFactory::CreateNodeFromScheme(CS_REAL_TRANSFORM_EVEN, this);

        rcplan->length    = length;
        rcplan->dimension = 1;
        rcplan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[0].get());

        // first transpose
        auto trans1    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_Z_XY, this);
        trans1->length = rcplan->outputLength;
        trans1->SetTransposeOutputLength();
        trans1->dimension = 2;

        // first column
        NodeMetaData c1planData(this);
        c1planData.length    = trans1->outputLength;
        c1planData.dimension = 1;
        auto c1plan = NodeFactory::CreateExplicitNode(c1planData, this, determined_scheme_2);
        c1plan->allowOutofplace = false; // let it be inplace
        c1plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[2].get());

        // second transpose
        auto trans2    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_Z_XY, this);
        trans2->length = trans1->outputLength;
        trans2->SetTransposeOutputLength();
        trans2->dimension = 2;

        // second column
        NodeMetaData c2planData(this);
        c2planData.length    = trans2->outputLength;
        c2planData.dimension = 1;
        auto c2plan = NodeFactory::CreateExplicitNode(c2planData, this, determined_scheme_4);
        c2plan->allowOutofplace = false; // let it be inplace
        c2plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[4].get());

        // third transpose
        auto trans3    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_Z_XY, this);
        trans3->length = trans2->outputLength;
        trans3->SetTransposeOutputLength();
        trans3->dimension = 2;

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
        auto trans3    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, this);
        trans3->length = length;
        trans3->SetTransposeOutputLength();
        std::swap(trans3->length[1], trans3->length[2]);
        trans3->dimension = 2;

        // column
        NodeMetaData c2planData(this);
        c2planData.length    = trans3->outputLength;
        c2planData.dimension = 1;
        auto c2plan = NodeFactory::CreateExplicitNode(c2planData, this, determined_scheme_1);
        c2plan->allowOutofplace = false; // let it be inplace
        c2plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[1].get());

        // transpose
        auto trans2    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, this);
        trans2->length = trans3->outputLength;
        trans2->SetTransposeOutputLength();
        std::swap(trans2->length[1], trans2->length[2]);
        trans2->dimension = 2;

        // column
        NodeMetaData c1planData(this);
        c1planData.length    = trans2->outputLength;
        c1planData.dimension = 1;
        auto c1plan = NodeFactory::CreateExplicitNode(c1planData, this, determined_scheme_3);
        c1plan->allowOutofplace = false; // let it be inplace
        c1plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[3].get());

        // transpose
        auto trans1    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, this);
        trans1->length = trans2->outputLength;
        trans1->SetTransposeOutputLength();
        std::swap(trans1->length[1], trans1->length[2]);
        trans1->dimension = 2;

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
        auto crplan = NodeFactory::CreateNodeFromScheme(CS_REAL_TRANSFORM_EVEN, this);

        crplan->length    = outputLength;
        crplan->dimension = 1;
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
    const bool forward = inArrayType == rocfft_array_type_real;

    if(forward)
    {
        auto& fused_2d     = childNodes[0];
        fused_2d->inStride = inStride;
        for(unsigned int i = 1; i < fused_2d->inStride.size(); ++i)
        {
            fused_2d->inStride[i] /= 2;
        }
        fused_2d->iDist     = iDist / 2;
        fused_2d->outStride = outStride;
        fused_2d->oDist     = oDist;
        fused_2d->AssignParams();

        // SBCC along Z dim
        auto& sbccZ      = childNodes[1];
        sbccZ->inStride  = {outStride[2], outStride[0], outStride[1]};
        sbccZ->outStride = sbccZ->inStride;
        sbccZ->iDist     = oDist;
        sbccZ->oDist     = oDist;
        sbccZ->AssignParams();
    }
    else
    {
        auto& sbccZ      = childNodes[0];
        sbccZ->inStride  = {inStride[2], inStride[0]};
        sbccZ->iDist     = iDist;
        sbccZ->outStride = sbccZ->inStride;
        sbccZ->oDist     = iDist;
        sbccZ->AssignParams();

        auto& fused_2d      = childNodes[1];
        fused_2d->inStride  = inStride;
        fused_2d->outStride = outStride;
        // The strides must be translated from real to complex.
        for(unsigned int i = 1; i < fused_2d->inStride.size(); ++i)
        {
            fused_2d->outStride[i] /= 2;
        }
        std::swap(fused_2d->inStride[0], fused_2d->inStride[1]);
        std::swap(fused_2d->outStride[0], fused_2d->outStride[1]);

        fused_2d->iDist = fused_2d->length.back() * fused_2d->inStride.back();
        fused_2d->oDist = fused_2d->length.back() * fused_2d->outStride.back();
        fused_2d->AssignParams();
    }
}

void Real3DEvenNode::AssignParams_internal_SBCC()
{
    assert(childNodes.size() == 3);

    const bool forward = inArrayType == rocfft_array_type_real;
    if(forward)
    {
        auto& rcplan = childNodes[0];
        {
            // The first sub-plan changes type in real/complex transforms.
            rcplan->inStride  = inStride;
            rcplan->iDist     = iDist;
            rcplan->outStride = outStride;
            rcplan->oDist     = oDist;
            rcplan->dimension = 1;
            rcplan->AssignParams();
        }

        // in-place SBCC for higher dims
        {
            auto& sbccZ     = childNodes[1];
            sbccZ->inStride = outStride;
            // SBCC along Z dim
            std::swap(sbccZ->inStride[1], sbccZ->inStride[2]);
            std::swap(sbccZ->inStride[0], sbccZ->inStride[1]);
            sbccZ->iDist     = oDist;
            sbccZ->outStride = sbccZ->inStride;
            sbccZ->oDist     = oDist;
            sbccZ->AssignParams();

            auto& sbccY     = childNodes[2];
            sbccY->inStride = outStride;
            // SBCC along Y dim
            std::swap(sbccY->inStride[0], sbccY->inStride[1]);
            sbccY->iDist     = oDist;
            sbccY->outStride = sbccY->inStride;
            sbccY->oDist     = oDist;
            sbccY->AssignParams();
        }
    }
    else
    {
        // input strides for last c2r node
        std::vector<size_t> c2r_inStride = inStride;
        size_t              c2r_iDist    = iDist;

        // in-place SBCC for higher dimensions
        {
            auto& sbccZ     = childNodes[0];
            sbccZ->inStride = inStride;
            // SBCC along Z dim
            std::swap(sbccZ->inStride[1], sbccZ->inStride[2]);
            std::swap(sbccZ->inStride[0], sbccZ->inStride[1]);
            sbccZ->iDist     = iDist;
            sbccZ->outStride = sbccZ->inStride;
            sbccZ->oDist     = iDist;
            sbccZ->AssignParams();

            auto& sbccY     = childNodes[1];
            sbccY->inStride = inStride;
            // SBCC along Y dim
            std::swap(sbccY->inStride[0], sbccY->inStride[1]);
            sbccY->iDist     = iDist;
            sbccY->outStride = sbccY->inStride;
            sbccY->oDist     = iDist;
            sbccY->AssignParams();
        }

        auto& crplan = childNodes.back();
        {
            crplan->inStride  = c2r_inStride;
            crplan->iDist     = c2r_iDist;
            crplan->outStride = outStride;
            crplan->oDist     = oDist;
            crplan->dimension = 1;
            crplan->AssignParams();
        }
    }
}

void Real3DEvenNode::AssignParams_internal_SBCR()
{
    if(childNodes.size() != 3)
        throw std::runtime_error("Require SBCR childNodes.size() == 3");

    auto& sbcrZ      = childNodes[0];
    sbcrZ->inStride  = {inStride[2], inStride[0]};
    sbcrZ->iDist     = iDist;
    sbcrZ->outStride = {1, sbcrZ->length[0]};
    sbcrZ->oDist     = iDist;

    sbcrZ->AssignParams();

    auto& sbcrY      = childNodes[1];
    sbcrY->inStride  = {sbcrY->length[1], 1};
    sbcrY->iDist     = sbcrY->length[0] * sbcrY->length[1];
    sbcrY->outStride = {1, sbcrY->length[0]};
    sbcrY->oDist     = sbcrY->iDist;
    sbcrY->AssignParams();

    auto& sbcrX         = childNodes[2];
    sbcrX->ebtype       = EmbeddedType::C2Real_PRE;
    sbcrX->outArrayType = rocfft_array_type_complex_interleaved;
    sbcrX->inStride     = {sbcrX->length[1], 1};
    sbcrX->iDist        = (sbcrX->length[0] + 1) * sbcrX->length[1];
    sbcrX->outStride    = {1, sbcrX->length[0]};
    sbcrX->oDist = sbcrX->length[0] * sbcrX->length[1]; // TODO: refactor for non-unit strides

    sbcrX->AssignParams();

    // we apply callbacks on the root plan's output
    TreeNode* rootPlan = this;
    while(rootPlan->parent != nullptr)
        rootPlan = rootPlan->parent;
}

void Real3DEvenNode::AssignParams_internal_TR_pairs()
{
    const bool forward = inArrayType == rocfft_array_type_real;
    if(forward)
    {
        auto& rcplan = childNodes[0];
        {
            // The first sub-plan changes type in real/complex transforms.
            rcplan->inStride  = inStride;
            rcplan->iDist     = iDist;
            rcplan->outStride = outStride;
            rcplan->oDist     = oDist;
            rcplan->dimension = 1;
            rcplan->AssignParams();
        }

        auto& trans1 = childNodes[1];
        {
            trans1->inStride = rcplan->outStride;
            trans1->iDist    = rcplan->oDist;
            trans1->outStride.push_back(trans1->length[2] * trans1->length[1]);
            trans1->outStride.push_back(1);
            trans1->outStride.push_back(trans1->length[1]);
            trans1->oDist = trans1->iDist;
        }

        auto& c1plan = childNodes[2];
        {
            c1plan->inStride = trans1->outStride;
            std::swap(c1plan->inStride[0], c1plan->inStride[1]);
            std::swap(c1plan->inStride[1], c1plan->inStride[2]);
            c1plan->iDist     = trans1->oDist;
            c1plan->outStride = c1plan->inStride;
            c1plan->oDist     = c1plan->iDist;
            c1plan->dimension = 1;
            c1plan->AssignParams();
        }

        auto& trans2 = childNodes[3];
        {
            trans2->inStride = c1plan->outStride;
            trans2->iDist    = c1plan->oDist;
            trans2->outStride.push_back(trans2->length[2] * trans2->length[1]);
            trans2->outStride.push_back(1);
            trans2->outStride.push_back(trans2->length[1]);
            trans2->oDist = trans2->iDist;
        }

        auto& c2plan = childNodes[4];
        {
            c2plan->inStride = trans2->outStride;
            std::swap(c2plan->inStride[0], c2plan->inStride[1]);
            std::swap(c2plan->inStride[1], c2plan->inStride[2]);
            c2plan->iDist     = trans2->oDist;
            c2plan->outStride = c2plan->inStride;
            c2plan->oDist     = c2plan->iDist;
            c2plan->dimension = 1;
            c2plan->AssignParams();
        }

        auto& trans3 = childNodes[5];
        {
            trans3->inStride  = c2plan->outStride;
            trans3->iDist     = c2plan->oDist;
            trans3->outStride = outStride;
            std::swap(trans3->outStride[1], trans3->outStride[2]);
            std::swap(trans3->outStride[0], trans3->outStride[1]);
            trans3->oDist = oDist;
        }
    }
    else
    {
        // input strides for last c2r node
        std::vector<size_t> c2r_inStride = inStride;
        size_t              c2r_iDist    = iDist;

        {
            auto& trans3     = childNodes[0];
            trans3->inStride = inStride;
            std::swap(trans3->inStride[1], trans3->inStride[2]);
            trans3->iDist = iDist;
            trans3->outStride.push_back(trans3->length[1]);
            trans3->outStride.push_back(1);
            trans3->outStride.push_back(trans3->outStride[0] * trans3->length[0]);
            trans3->oDist = trans3->iDist;
        }

        {
            auto& ccplan      = childNodes[1];
            ccplan->inStride  = {childNodes[0]->outStride[1],
                                childNodes[0]->outStride[0],
                                childNodes[0]->outStride[2]};
            ccplan->iDist     = childNodes[0]->oDist;
            ccplan->outStride = ccplan->inStride;
            ccplan->oDist     = ccplan->iDist;
            ccplan->dimension = 1;
            ccplan->AssignParams();
        }

        {
            auto& trans2     = childNodes[2];
            trans2->inStride = childNodes[1]->outStride;
            std::swap(trans2->inStride[1], trans2->inStride[2]);
            trans2->iDist = childNodes[1]->oDist;
            trans2->outStride.push_back(trans2->length[1]);
            trans2->outStride.push_back(1);
            trans2->outStride.push_back(trans2->outStride[0] * trans2->length[0]);
            trans2->oDist = trans2->iDist;
        }

        {
            auto& ccplan      = childNodes[3];
            ccplan->inStride  = {childNodes[2]->outStride[1],
                                childNodes[2]->outStride[0],
                                childNodes[2]->outStride[2]};
            ccplan->iDist     = childNodes[2]->oDist;
            ccplan->outStride = ccplan->inStride;
            ccplan->oDist     = ccplan->iDist;
            ccplan->dimension = 1;
            ccplan->AssignParams();
        }

        {
            auto& trans1     = childNodes[4];
            trans1->inStride = childNodes[3]->outStride;
            std::swap(trans1->inStride[1], trans1->inStride[2]);
            trans1->iDist = childNodes[3]->oDist;
            trans1->outStride.push_back(trans1->length[1]);
            trans1->outStride.push_back(1);
            trans1->outStride.push_back(trans1->outStride[0] * trans1->length[0]);
            trans1->oDist = trans1->iDist;
            c2r_inStride  = {trans1->outStride[1], trans1->outStride[0], trans1->outStride[2]};
            c2r_iDist     = trans1->oDist;
        }

        auto& crplan = childNodes.back();
        {
            crplan->inStride  = c2r_inStride;
            crplan->iDist     = c2r_iDist;
            crplan->outStride = outStride;
            crplan->oDist     = oDist;
            crplan->dimension = 1;
            crplan->AssignParams();
        }
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
        return 2 * length[0];
    if(scheme == CS_KERNEL_CMPLX_TO_R)
        return 2 * (length[0] - 1);
    else if(scheme == CS_KERNEL_TRANSPOSE_CMPLX_TO_R)
        return 2 * (length.back() - 1);

    throw std::runtime_error("GetTwiddleTableLength: Unexpected scheme in PrePostKernelNode: "
                             + PrintScheme(scheme));
}

size_t PrePostKernelNode::GetTwiddleTableLengthLimit()
{
    // The kernel only uses 1/4th of the real length twiddle table
    return DivRoundingUp<size_t>(GetTwiddleTableLength(), 4);
}
