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

#include "assignment_policy.h"
#include "../../shared/arithmetic.h"
#include "../../shared/ptrdiff.h"
#include "./device/kernels/array_format.h"
#include "enum_printer.h"
#include "logging.h"
#include "node_factory.h"
#include <numeric>
#include <optional>
#include <set>

void PlacementTrace::Print(rocfft_ostream& os)
{
    if(parent->curNode)
    {
        parent->Print(os);
        os << " --> ";
    }

    os << "[ " << PrintScheme(curNode->scheme).c_str();
    os << ": " << PrintOperatingBufferCode(inBuf) << "->" << PrintOperatingBufferCode(outBuf);
    os << " ]";

    if(branches.empty())
    {
        os << ": num-fused-kernels= " << parent->numFusedNodes;
        os << ", num-inplace-kernels= " << parent->numInplace;
        os << std::endl;
    }
}

size_t PlacementTrace::BackwardCalcFusions(ExecPlan&       execPlan,
                                           int             curFuseShimID,
                                           PlacementTrace* shimLastNode)
{
    numFusedNodes = 0;
    if(curFuseShimID < 0)
        return 0;

    auto& shim = execPlan.fuseShims[curFuseShimID];

    // if this node is the last node of the fuseShim, pass self as shimLastNode and continue going back...
    if(curNode == shim->LastFuseNode())
    {
        // should not have shimLastNode set, and should not have a null parent
        if(shimLastNode != nullptr || parent == nullptr)
            throw std::runtime_error(
                "Tracing FusedShimsNode error when backtracking assignment path");
        numFusedNodes = parent->BackwardCalcFusions(execPlan, curFuseShimID, this);
    }
    // if this node is the first node of the fuseShim, check if fusion can be done with the placement
    else if(curNode == shim->FirstFuseNode())
    {
        // we should already have a shimLastNode
        if(!shimLastNode)
            throw std::runtime_error(
                "Tracing FusedShimsNode error when backtracking assignment path");
        size_t numFusion
            = shim->PlacementFusable(this->inBuf, this->outBuf, shimLastNode->outBuf) ? 1 : 0;
        numFusedNodes
            = parent ? parent->BackwardCalcFusions(execPlan, curFuseShimID - 1, nullptr) + numFusion
                     : numFusion;
    }
    // this node is either outside of a shim (shimLastNode == nullptr)
    // or inside of a shim(shimLastNode != nullptr), simply keep on going back...
    else
    {
        numFusedNodes
            = parent ? parent->BackwardCalcFusions(execPlan, curFuseShimID, shimLastNode) : 0;
    }

    return numFusedNodes;
}

size_t PlacementTrace::NumUsedBuffers() const
{
    return usedBuffers.size();
}

void PlacementTrace::Backtracking(ExecPlan& execPlan, int execSeqID)
{
    const auto& execSeq = execPlan.execSeq;

    if((execSeqID < 0) || (curNode != execSeq[execSeqID]))
        throw std::runtime_error("Backtracking error: accessing invalid resource");

    auto node          = execSeq[execSeqID];
    node->placement    = this->isInplace ? rocfft_placement_inplace : rocfft_placement_notinplace;
    node->obIn         = this->inBuf;
    node->obOut        = this->outBuf;
    node->inArrayType  = this->iType;
    node->outArrayType = this->oType;

    // Even-length internal nodes have real data for input or output but
    // child nodes treat it as complex interleaved
    auto realEvenAncestor = node->GetRealEvenAncestor();
    if(realEvenAncestor)
    {
        // forward transform, first node (if it's a leaf) needs to treat real input as complex
        if(node->direction == -1 && node == realEvenAncestor->GetFirstLeaf())
            node->inArrayType = rocfft_array_type_complex_interleaved;
        // inverse transform, last node (if it's a leaf) needs to treat real output as complex
        if(node->direction == 1 && node == realEvenAncestor->GetLastLeaf())
            node->outArrayType = rocfft_array_type_complex_interleaved;
    }

    // Ensure that all nodes that take real input are declared as
    // such.  This is particularly important for leaf nodes, since
    // kernelio debugging depends on knowing the correct type of the
    // array to print.
    if(node->scheme == CS_KERNEL_COPY_R_TO_CMPLX)
        node->inArrayType = rocfft_array_type_real;

    // for nodes that uses bluestein buffer
    auto setBluesteinOffset = [node](size_t& offset) {
        for(auto p = node->parent; p != nullptr; p = p->parent)
        {
            if(p->iOffset)
            {
                offset = p->iOffset;
                break;
            }
            else if(p->oOffset)
            {
                offset = p->oOffset;
                break;
            }
        }
    };
    if(node->obIn == OB_TEMP_BLUESTEIN)
        setBluesteinOffset(node->iOffset);
    else
        node->iOffset = 0;
    if(node->obOut == OB_TEMP_BLUESTEIN)
        setBluesteinOffset(node->oOffset);
    else
        node->oOffset = 0;

    // keep going backward to next node, skipping over chirp setup nodes
    int nextExecSeqID = execSeqID;
    while(nextExecSeqID > 0)
    {
        --nextExecSeqID;
        auto nextNode = execSeq[nextExecSeqID];
        if(!nextNode->IsBluesteinChirpSetup() || execPlan.IsChirpPlan)
        {
            parent->Backtracking(execPlan, nextExecSeqID);
            return;
        }
    }

    // if we're here, then 'node' must have already been the first
    // node (and had its input set properly), or it's preceded only
    // by chirp setup nodes.  input type we set earlier is already
    // right.
}

// test if rootArrayType == testArrayType,
// if testAryType == rootAryType, return true (since they're definitely equivalent)
// but if root is real or HI, the equivalent internal type could be CI (1-ptr)
// or if root is HP, the equivalent internal type could be CP (2-ptr)
bool AssignmentPolicy::EquivalentArrayType(rocfft_array_type rootAryType,
                                           rocfft_array_type testAryType)
{
    if(rootAryType == rocfft_array_type_real
       || rootAryType == rocfft_array_type_hermitian_interleaved)
        return (testAryType == rootAryType)
               || (testAryType == rocfft_array_type_complex_interleaved);

    if(rootAryType == rocfft_array_type_hermitian_planar)
        return (testAryType == rootAryType) || (testAryType == rocfft_array_type_complex_planar);

    return (testAryType == rootAryType);
}

// return true if OB_TEMP_BLUESTEIN is a valid output buffer for the node
static bool ValidOutBufferBluestein(TreeNode& node)
{
    // nodes may only write to bluestein if:
    // - they are setting up the chirp buffer
    // - they are the internal steps of multi-kernel bluestein, and
    //   FFT steps may be further decomposed into separate kernels.
    // - sbrc nodes in multi-kernel fused bluestein that do:
    //   (1) chirp + padding + forward fft, or
    //   (2) chirp / input Hadamard product + padding + forward fft

    // First check multi-kernel fused Bluestein implementation
    if((node.fuseBlue == BFT_FWD_CHIRP || node.fuseBlue == BFT_FWD_CHIRP_MUL)
       && node.scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
    {
        return true;
    }

    // go up the tree, looking for a bluestein parent node
    for(auto n = &node; n != nullptr; n = n->parent)
    {
        auto p = n->parent;

        if(p == nullptr)
            break;

        if(p->scheme != CS_BLUESTEIN)
        {
            // keep going, can't decide if we're under bluestein yet
            continue;
        }
        // bluestein could have 3 children (in which case the first
        // two are setup and the third only uses the chirp buffer but
        // does not write to it)
        if(p->childNodes.size() == 3)
            return n->IsBluesteinChirpSetup();
        // or it could have 6 children, in which case all but the
        // last must write to bluestein
        else if(p->childNodes.size() == 6)
        {
            return n != p->childNodes.back().get();
        }
    }
    // if we're here, we must be at the root node, so we're not
    // under bluestein and a bluestein write is invalid
    return false;
}

bool AssignmentPolicy::ValidOutBuffer(ExecPlan&           execPlan,
                                      NodeBufTestCacheKey cacheMapKey,
                                      TreeNode&           node,
                                      OperatingBuffer     buffer,
                                      rocfft_array_type   arrayType)
{
    auto cacheMapIter = node_buf_test_cache.find(cacheMapKey);
    if(cacheMapIter != node_buf_test_cache.end())
        return cacheMapIter->second;

    // define a local function to decide if a node's output fits into
    // OB_USER_IN or OB_USER_OUT.  Temp buffers are dynamically sized
    // to always fit.  This function accepts OB_USER_IN also to mean
    // the input side of an in-place R2C transform (which the plan
    // would normally call OB_USER_OUT).
    auto dataFits = [&execPlan](const TreeNode& node, OperatingBuffer buffer) {
        auto outLengthBlueN = {node.lengthBlueN};
        auto nodeLen        = (node.fuseBlue == BFT_NONE) ? node.GetOutputLength() : outLengthBlueN;
        auto bufLen         = buffer == OB_USER_OUT ? execPlan.rootPlan->GetOutputLength()
                                                    : execPlan.rootPlan->length;

        // if node's output is complex and buffer's format is real,
        // adjust output length to be 2x to make the units of
        // comparison match
        bool kernelOutputIsReal = node.scheme == CS_KERNEL_COPY_CMPLX_TO_R;
        bool outBufferIsReal
            = (buffer == OB_USER_OUT && execPlan.rootPlan->outArrayType == rocfft_array_type_real)
              || (buffer == OB_USER_IN && execPlan.rootPlan->inArrayType == rocfft_array_type_real);
        if(outBufferIsReal)
        {
            if(!kernelOutputIsReal)
                nodeLen.front() *= 2;
        }

        if(BufferIsUnitStride(execPlan, buffer))
        {
            // just check if there's enough space
            return product(nodeLen.begin(), nodeLen.end()) <= product(bufLen.begin(), bufLen.end());
        }

        // ensure that the node's dimensions fit exactly into the
        // buffer's dimensions.  e.g. if the node wants XxYxZ and the
        // buffer is AxZ, this is ok so long as X*Y == A
        for(auto len : nodeLen)
        {
            // not decomposing evenly
            if(bufLen.empty() || bufLen.front() % len != 0)
                return false;
            bufLen.front() /= len;
            if(bufLen.front() == 1)
                bufLen.erase(bufLen.begin());
        }
        return true;
    };

    bool test_result = true;

    // an initial elimination to reject those illegal out-buffer
    if(node.isOutBufAllowed(buffer) == false)
    {
        test_result = false;
    }
    // an initial elimination to reject those illegal out-arrayType
    else if(node.isOutArrayTypeAllowed(arrayType) == false)
    {
        test_result = false;
    }
    // Bluestein and CMPLX buffer must be CI
    else if((buffer == OB_TEMP_BLUESTEIN || buffer == OB_TEMP_CMPLX_FOR_REAL)
            && arrayType != rocfft_array_type_complex_interleaved)
    {
        test_result = false;
    }
    // bluestein nodes must write to temp bluestein buffer
    else if(buffer == OB_TEMP_BLUESTEIN)
    {
        test_result = ValidOutBufferBluestein(node);
    }
    // second node in multi-kernel fused Bluestein must write only to
    // two specific buffers
    else if((buffer == OB_USER_OUT || buffer == OB_TEMP_CMPLX_FOR_REAL)
            && node.fuseBlue == BFT_FWD_CHIRP_MUL)
    {
        test_result = false;
    }
    // third node in multi-kernel fused Bluestein must not write to OB_TEMP_CMPLX_FOR_REAL
    else if(buffer == OB_TEMP_CMPLX_FOR_REAL && node.fuseBlue == BFT_INV_CHIRP_MUL)
    {
        test_result = false;
    }
    // if output goes to a temp buffer, that will be dynamically sized
    // to be big enough so it's always ok but if output is in/out, we
    // have to fit into whatever the user gave us
    else if(buffer == OB_USER_IN || buffer == OB_USER_OUT)
    {
        auto fitArrayType = buffer == OB_USER_IN ? execPlan.rootPlan->inArrayType
                                                 : execPlan.rootPlan->outArrayType;
        test_result       = dataFits(node, buffer) && EquivalentArrayType(fitArrayType, arrayType);
        if(!test_result)
        {
            // if that didn't fit, and we're writing to OB_USER_OUT, and
            // this is an in-place R2C/C2R transform, then we could also
            // try fitting into the shape of the input.
            if(buffer == OB_USER_OUT && execPlan.rootPlan->placement == rocfft_placement_inplace
               && (execPlan.rootPlan->inArrayType == rocfft_array_type_real
                   || execPlan.rootPlan->outArrayType == rocfft_array_type_real))
                test_result = dataFits(node, OB_USER_IN)
                              && EquivalentArrayType(execPlan.rootPlan->inArrayType, arrayType);
        }
    }

    node_buf_test_cache[cacheMapKey] = test_result;
    return test_result;
}

static void RecursiveTraverse(TreeNode* node, const std::function<void(TreeNode*)>& func)
{
    func(node);
    for(auto& n : node->childNodes)
        RecursiveTraverse(n.get(), func);
}

bool AssignmentPolicy::CheckAssignmentValid(ExecPlan& execPlan)
{
    auto getBufSize = [](TreeNode* node, bool input) {
        auto lengthBlueN = {node->lengthBlueN};
        auto outputLen   = node->fuseBlue == BFT_NONE ? node->GetOutputLength() : lengthBlueN;

        if(input)
            return compute_ptrdiff(node->length, node->inStride, node->batch, node->iDist);
        else
        {
            return compute_ptrdiff(node->UseOutputLengthForPadding() ? outputLen : node->length,
                                   node->outStride,
                                   node->batch,
                                   node->oDist);
        }
    };

    size_t sizeBufIn  = 0;
    size_t sizeBufOut = 0;
    if(execPlan.rootPlan->placement == rocfft_placement_notinplace)
    {
        sizeBufIn  = getBufSize(execPlan.rootPlan.get(), true);
        sizeBufOut = getBufSize(execPlan.rootPlan.get(), false);
    }
    else
        sizeBufOut = std::max(getBufSize(execPlan.rootPlan.get(), true),
                              getBufSize(execPlan.rootPlan.get(), false));

    for(auto& curr : execPlan.execSeq)
    {
        auto currSizeBufOut = getBufSize(curr, false);
        if((curr->obOut == OB_USER_IN && currSizeBufOut > sizeBufIn)
           || (curr->obOut == OB_USER_OUT && currSizeBufOut > sizeBufOut))
        {
            // std::cout << "buffer access violation, re-assign" << std::endl;
            return false;
        }

        if(curr->placement == rocfft_placement_inplace)
        {
            const int infact  = curr->inArrayType == rocfft_array_type_real ? 1 : 2;
            const int outfact = curr->outArrayType == rocfft_array_type_real ? 1 : 2;
            for(size_t i = 0; i < curr->inStride.size(); i++)
            {
                if(outfact * curr->inStride[i] != infact * curr->outStride[i])
                {
                    return false;
                }
            }
        }

        // assignment already respects allowInplace and
        // allowOutofplace flags on leaf nodes.  now, check that
        // internal nodes also respect those flags.
        //
        // we're only iterating over leaf nodes in execSeq.  to
        // ensure we only check parent flags once, check the parent's
        // flags for any node that's the last child of its parent.
        auto isLastChildOfParent = [](TreeNode* node) {
            return node->parent && node->parent->childNodes.back().get() == node;
        };
        auto ptr = curr;
        while(isLastChildOfParent(ptr))
        {
            auto parent = ptr->parent;
            if(!parent->isPlacementAllowed(parent->placement))
                return false;
            ptr = parent;
        }
    }

    return true;
}

void AssignmentPolicy::UpdateWinnerFromValidPaths(ExecPlan& execPlan)
{
    if(winnerCandidates.empty())
        return;

    //std::cout << "total candidates: " << winnerCandidates.size() << std::endl;

    // sort the candidate, front is the best
    std::sort(
        winnerCandidates.begin(), winnerCandidates.end(), [](const auto& lhs, const auto& rhs) {
            // compare numFusedNodes (more is better)
            if(lhs->numFusedNodes > rhs->numFusedNodes)
                return true;
            if(lhs->numFusedNodes < rhs->numFusedNodes)
                return false;

            // if tie, we still choose the one with less buffers
            if(lhs->NumUsedBuffers() < rhs->NumUsedBuffers())
                return true;
            if(lhs->NumUsedBuffers() > rhs->NumUsedBuffers())
                return false;

            // once we do have temp buffers, more temp ops that have
            // better opportunities for padding are generally better,
            // since we can avoid more bad memory access patterns
            auto leftTempOps  = lhs->NumPaddableTempOps();
            auto rightTempOps = rhs->NumPaddableTempOps();
            if(leftTempOps != rightTempOps)
                return leftTempOps > rightTempOps;

            // if tie, we still choose the one with more inplace
            if(lhs->numInplace > rhs->numInplace)
                return true;
            if(lhs->numInplace < rhs->numInplace)
                return false;

            // if tie, compare numTypeSwitching (less is better)
            return lhs->numTypeSwitching < rhs->numTypeSwitching;
        });

    for(auto& winner : winnerCandidates)
    {
        // fill the assignment to tree-node from the PlacementTrace path
        winner->Backtracking(execPlan, execPlan.execSeq.size() - 1);

        // assign the stride things. remember to refresh for the internal nodes
        execPlan.rootPlan->RefreshTree();
        // TODO- Next big thing to generalize
        execPlan.rootPlan->AssignParams();

        // Act as a final guard to check the stride and dist
        // Ideally, all the valid-tests were handled in the AssignBuffers,
        // So the first candidate is the result.
        // But some inplace r2c/c2r are tricky and not easy to handle (most of them are dist and stride)
        // This final guard somehow is the "error-detector"...
        // TODO- Eventually we should make the AssignBuffer more robust
        if(CheckAssignmentValid(execPlan))
        {
            // std::cout << "num Fused: " << winner->numFusedNodes;
            // std::cout << ", num IP:" << winner->numInplace;
            // std::cout << std::endl;
            numCurWinnerFusions = winner->numFusedNodes;
            return;
        }
    }
    return;
}

void AssignmentPolicy::FindBluesteinFusedNodes(ExecPlan&               execPlan,
                                               std::vector<TreeNode*>& fusedNodes)
{
    std::vector<TreeNode*> blueNodes;
    execPlan.rootPlan->RecursiveFindChildNodes(CS_BLUESTEIN, blueNodes);
    execPlan.rootPlan->AssignParams();

    for(const auto& node : blueNodes)
        if(node->typeBlue == BT_MULTI_KERNEL_FUSED)
            fusedNodes.emplace_back(node);
}

void AssignmentPolicy::AssignChirpBuffers(ExecPlan& execPlan)
{
    execPlan.IsChirpPlan = false;

    std::vector<TreeNode*> blueMultiFusedNodes;
    FindBluesteinFusedNodes(execPlan, blueMultiFusedNodes);

    for(auto& node : blueMultiFusedNodes)
    {
        auto& chirpFwdNode            = node->childNodes[0];
        chirpFwdNode->obIn            = OB_TEMP_BLUESTEIN;
        chirpFwdNode->inArrayType     = rocfft_array_type_complex_interleaved;
        chirpFwdNode->obOut           = OB_TEMP_BLUESTEIN;
        chirpFwdNode->outArrayType    = rocfft_array_type_complex_interleaved;
        chirpFwdNode->placement       = rocfft_placement_inplace;
        chirpFwdNode->allowInplace    = true;
        chirpFwdNode->allowOutofplace = false;

        auto& chirpFwdMulNode            = node->childNodes[1];
        chirpFwdMulNode->placement       = rocfft_placement_notinplace;
        chirpFwdMulNode->allowInplace    = true;
        chirpFwdMulNode->allowOutofplace = true;

        NodeMetaData chirpFwdNodePlan(chirpFwdNode.get());
        chirpFwdNodePlan.length    = chirpFwdNode->length;
        chirpFwdNodePlan.dimension = chirpFwdNode->dimension;

        // create a temporary plan that just does chirp - the plan nodes
        // get copied into our actual plan so the comm_rank, mgpuPlan,
        // location don't need to be correct
        ExecPlan execPlanFwdChirp{0, false, {}};
        execPlanFwdChirp.IsChirpPlan = true;
        execPlanFwdChirp.rootPlan    = NodeFactory::CreateExplicitNode(chirpFwdNodePlan, nullptr);

        execPlanFwdChirp.rootPlan->RecursiveBuildTree();
        execPlanFwdChirp.rootPlan->RecursiveCopyNodeData(*chirpFwdNode);
        execPlanFwdChirp.rootPlan->CollectLeaves(execPlanFwdChirp.execSeq,
                                                 execPlanFwdChirp.fuseShims);

        execPlanFwdChirp.assignOptStrategy = rocfft_optimize_balance;

        AssignBuffers_internal(execPlanFwdChirp);

        chirpFwdNode->RecursiveCopyNodeData(*execPlanFwdChirp.rootPlan);
    }

    if(!blueMultiFusedNodes.empty())
    {
        execPlan.execSeq.clear();
        execPlan.fuseShims.clear();
        execPlan.rootPlan->CollectLeaves(execPlan.execSeq, execPlan.fuseShims);
    }
}

void AssignmentPolicy::AssignBuffers(ExecPlan& execPlan)
{
    AssignChirpBuffers(execPlan);

    AssignBuffers_internal(execPlan);
}

void AssignmentPolicy::AssignBuffers_internal(ExecPlan& execPlan)
{
    int maxFusions      = execPlan.fuseShims.size();
    numCurWinnerFusions = -1; // no winner yet
    mustUseTBuffer      = false;
    mustUseCBuffer      = false;

    // remember to clear the container either in the beginning or at the end
    winnerCandidates.clear();
    availableBuffers.clear();
    availableArrayTypes.clear();
    node_buf_test_cache.clear();

    // Start from a minimal requirement; // out buffer
    availableBuffers.insert(execPlan.rootPlan->obOut);

    // real-inverse transforms are allowed to modify input
    if(execPlan.rootPlan->direction == 1
       && execPlan.rootPlan->outArrayType == rocfft_array_type_real)
    {
        availableBuffers.insert(execPlan.rootPlan->obIn);
    }

    // Insert the valid ArrayTypes to use internally
    rocfft_array_type aliasInType  = execPlan.rootPlan->inArrayType;
    rocfft_array_type aliasOutType = execPlan.rootPlan->outArrayType;

    // if HP/CP -> treated as CP INTERNALLY, else (CI,HI,real) -> treated as CI INTERNALLY
    // if no interleaved in in/out, no need to try, if no planar in in/out, no need to try
    aliasInType  = is_complex_planar(aliasInType) ? rocfft_array_type_complex_planar
                                                  : rocfft_array_type_complex_interleaved;
    aliasOutType = is_complex_planar(aliasOutType) ? rocfft_array_type_complex_planar
                                                   : rocfft_array_type_complex_interleaved;
    availableArrayTypes.insert(aliasInType);
    availableArrayTypes.insert(aliasOutType);

    // look for nodes that imply presence of other buffers (bluestein)
    RecursiveTraverse(execPlan.rootPlan.get(), [this](TreeNode* n) {
        if(n->scheme == CS_BLUESTEIN)
        {
            availableBuffers.insert(OB_TEMP_BLUESTEIN);
            availableArrayTypes.insert(rocfft_array_type_complex_interleaved);
        }
    });

    // First try !
    PlacementTrace dummyRoot;
    dummyRoot.outBuf = execPlan.rootPlan->obIn;
    dummyRoot.oType  = aliasInType;
    Enumerate(&dummyRoot, execPlan, 0, dummyRoot.outBuf, dummyRoot.oType);
    // update num-of-winner's-fusions from winnerCandidates list
    UpdateWinnerFromValidPaths(execPlan);
    if(numCurWinnerFusions != -1)
    {
        // we already satisfy the strategy, so don't need to go further
        if(execPlan.assignOptStrategy <= rocfft_optimize_min_buffer)
            return;
        // we already fulfill all possible fusions
        if(numCurWinnerFusions == maxFusions)
            return;
    }

    // if we are here:
    // 1. we haven't found a winner (working assignment)
    // 2. we found a assignment but we want to try if it's possible
    //    to have more fusions by adding TEMP buffer
    //    (strategy > rocfft_optimize_min_buffer)
    mustUseTBuffer = true;
    availableBuffers.insert(OB_TEMP);
    dummyRoot.branches.clear();
    winnerCandidates.clear();
    Enumerate(&dummyRoot, execPlan, 0, dummyRoot.outBuf, dummyRoot.oType);
    // NB:
    //   in this ABT try, winnerCandidates must contain T-buf (mustUseTBuffer=true)
    //   and it's possible winnerCandidates is empty because there is no new path giving more fusions.
    //   So num-of-winner's-fusions won't be updated, but we may have a winner from prev try
    //   in this case, we should return if the strategy is "balance".
    UpdateWinnerFromValidPaths(execPlan);
    if(numCurWinnerFusions != -1)
    {
        // we already satisfy the strategy, so don't need to go further
        if(execPlan.assignOptStrategy <= rocfft_optimize_balance)
            return;
        // we already fulfill all possible fusions
        if(numCurWinnerFusions == maxFusions)
            return;
    }

    // Same as above: if we are here....
    mustUseCBuffer = true;
    availableBuffers.insert(OB_TEMP_CMPLX_FOR_REAL);
    availableArrayTypes.insert(rocfft_array_type_complex_interleaved);
    dummyRoot.branches.clear();
    winnerCandidates.clear();
    Enumerate(&dummyRoot, execPlan, 0, dummyRoot.outBuf, dummyRoot.oType);
    // NB:
    //   in this ABTC try, winnerCandidates must contain C-buf (mustUseCBuffer=true)
    UpdateWinnerFromValidPaths(execPlan);
    if(numCurWinnerFusions != -1)
        return;

    // else, we can't find any valid buffer assignment !
    throw std::runtime_error("Can't find valid buffer assignment with current buffers.");
}

void AssignmentPolicy::Enumerate(PlacementTrace*   parent,
                                 ExecPlan&         execPlan,
                                 size_t            curSeqID,
                                 OperatingBuffer   startBuf,
                                 rocfft_array_type startType)
{
    auto& execSeq   = execPlan.execSeq;
    auto& fuseShims = execPlan.fuseShims;

    // Terminal Condition
    // we've done all, check if this path works (matches the rootPlan's out)
    if(curSeqID >= execSeq.size())
    {
        auto endBuf       = execPlan.rootPlan->obOut;
        auto endArrayType = execPlan.rootPlan->outArrayType;

        // the out buf and array type must match
        if(parent->outBuf == endBuf && EquivalentArrayType(endArrayType, parent->oType))
        {
            // we are in the second try (adding T Buffer) but we don't have it in the path:
            // this means we've already tried this path in the previous try.
            if(mustUseTBuffer && parent->usedBuffers.count(OB_TEMP) == 0)
                return;

            // we are in the third try (adding C Buffer) but we don't have it in the path:
            // this means we've already tried this path in the previous try.
            if(mustUseCBuffer && parent->usedBuffers.count(OB_TEMP_CMPLX_FOR_REAL) == 0)
                return;

            // See how many fusions can be done in this path
            int numFusions = parent->BackwardCalcFusions(execPlan, fuseShims.size() - 1, nullptr);
            // skip it if this doesn't outdo the winner of prev. try (prev try = fewer buffers)
            if(numCurWinnerFusions >= numFusions)
                return;

            // set the oType to its original type of RootPlan (for example, change internal-CP to HP)
            parent->oType = endArrayType;

            winnerCandidates.emplace_back(parent);
            // debug
            // parent->Print(*LogSingleton::GetInstance().GetTraceOS());
        }
        return;
    }

    TreeNode* curNode = execSeq[curSeqID];

    // For chirp scheme, the startBuf/AType can differ from parent's, so we don't do this
    // auto startBuf  = parent->outBuf;
    // auto startType = parent->oType;

    if(!execPlan.IsChirpPlan && curNode->IsBluesteinChirpSetup())
    {
        auto blueNode = curNode;
        // bluestein setup kernels can input/output bluestein buffer only.
        do
        {
            if(blueNode->typeBlue != BT_MULTI_KERNEL_FUSED)
            {
                // chirp setup nodes must use bluestein buffer, not
                // connected to other nodes, so just set their buffers
                // directly and don't enumerate them with PlacementTraces
                blueNode->obIn         = OB_TEMP_BLUESTEIN;
                blueNode->inArrayType  = rocfft_array_type_complex_interleaved;
                blueNode->obOut        = OB_TEMP_BLUESTEIN;
                blueNode->outArrayType = rocfft_array_type_complex_interleaved;
                blueNode->placement    = rocfft_placement_inplace;
            }

            ++curSeqID;
            blueNode = execSeq[curSeqID];
        } while(blueNode->IsBluesteinChirpSetup());

        // blueNode is now no longer a bluestein setup node

        // advance to next
        // NOTE that it is important we propagate startBuf, startType to next node
        Enumerate(parent, execPlan, curSeqID, startBuf, startType);
        return;
    }

    // Branch of using inplace, any node dis-alllowing inplace will skip this
    if(curNode->isPlacementAllowed(rocfft_placement_inplace))
    {
        // If buffer is not available (when USER_IN is read-only), skip as well.
        if(availableBuffers.count(startBuf))
        {
            NodeBufTestCacheKey cKey{curSeqID, startBuf, startType};
            if(ValidOutBuffer(execPlan, cKey, *curNode, startBuf, startType))
            {
                // Create/Push a PlacementTrace for an Inplace-Operation (others recurs)
                parent->branches.emplace_back(std::make_unique<PlacementTrace>(
                    curNode, startBuf, startBuf, startType, startType, parent));
                // advance to next
                Enumerate(
                    parent->branches.back().get(), execPlan, curSeqID + 1, startBuf, startType);
            }
        }
    }

    // Branch of using out-of-place, any node dis-alllowing notinplace will skip this
    if(curNode->isPlacementAllowed(rocfft_placement_notinplace))
    {
        // try every available output buffer
        for(auto testOutputBuf : availableBuffers)
        {
            // except for startBuf, since this is a out-of-place try
            if(testOutputBuf == startBuf)
                continue;

            // try every available array type
            for(auto testOutType : availableArrayTypes)
            {
                NodeBufTestCacheKey cKey{curSeqID, testOutputBuf, testOutType};
                if(ValidOutBuffer(execPlan, cKey, *curNode, testOutputBuf, testOutType))
                {
                    // Create/Push a PlacementTrace for OuOfPlace-Operation (others recurs)
                    parent->branches.emplace_back(std::make_unique<PlacementTrace>(
                        curNode, startBuf, testOutputBuf, startType, testOutType, parent));
                    // advance to next
                    Enumerate(parent->branches.back().get(),
                              execPlan,
                              curSeqID + 1,
                              testOutputBuf,
                              testOutType);
                }
            } // end of testing each array type
        } // end of testing each out buffer
    } // end of out-of-place
}

// Lengths/strides on tree nodes are usually (but not always) fastest
// dimension first.  Define a structure that can be sorted
// fastest-to-slowest, without actually re-sorting the original
// length/stride vectors on the tree node.
template <typename Tint>
struct LengthStrideSort
{
    LengthStrideSort(const Tint& length, Tint& stride)
        : _length(&length)
        , _stride(&stride)
    {
    }

    // accessors
    const Tint& length() const
    {
        return *_length;
    }
    const Tint& stride() const
    {
        return *_stride;
    }
    Tint& stride()
    {
        return *_stride;
    }

    // sort faster strides first
    bool operator<(const LengthStrideSort<Tint>& other) const
    {
        if(this->stride() != other.stride())
            return this->stride() < other.stride();
        // tiebreak on length.  strides really ought to always be
        // different for each dim, but in edge cases like if length
        // of a dim is 1, you could have a stride value that doesn't
        // matter and matches another dim.
        return this->length() < other.length();
    }

private:
    // store pointers to actual length/stride
    const Tint* _length;
    Tint*       _stride;
};

template <typename T>
static bool StrideIsContiguous(const T& lengthStride)
{
    size_t start1d = 1;
    for(const auto& i : lengthStride)
    {
        if(start1d != i.stride())
            return false;
        start1d *= i.length();
    }
    return true;
}

// Which dimension are we padding for a kernel?
enum class PaddingDim
{
    // some kernels can only pad the highest dim or second highest dim
    HIGHEST,
    SECOND_HIGHEST,
};

// Function to pad strides on a buffer, if the strides would produce
// a problematic access pattern.  Previous stride/dist/length are
// provided in cases where a previous write must be compatible with a
// read from the same buffer.
//
// padSecondHighestDim controls exactly which dimension gets the
// padding - normally we'd try to pad the highest dimension.
static void PadStride(std::vector<size_t>&       stride,
                      size_t&                    dist,
                      const std::vector<size_t>& length,
                      size_t                     batch,
                      const std::vector<size_t>& prevStride,
                      const size_t&              prevDist,
                      const std::vector<size_t>& prevLength,
                      PaddingDim                 paddingDim)
{
    // only consider padding for 2D and higher
    if(stride.size() < 2)
        return;

    // dist is effectively unused if batch == 1 and users are allowed
    // to give us a number that doesn't make sense in that case.  but
    // if so, assume it's contiguous so that all the math works out.
    if(batch == 1)
        dist = compute_ptrdiff(length, stride, 1, 1);

    // sort lengths, treat batch dimension as another dim
    std::vector<LengthStrideSort<size_t>> cur;
    for(unsigned int i = 0; i < length.size(); ++i)
    {
        cur.emplace_back(length[i], stride[i]);
    }
    cur.emplace_back(batch, dist);
    std::sort(cur.begin(), cur.end());

    // if we have a previous stride, we need to respect its padding
    if(!prevStride.empty())
    {
        // sort previous lengths too, again treating batch dimension as
        // another dim
        std::vector<LengthStrideSort<const size_t>> prev;
        for(unsigned int i = 0; i < prevLength.size(); ++i)
        {
            prev.emplace_back(prevLength[i], prevStride[i]);
        }
        prev.emplace_back(batch, prevDist);
        std::sort(prev.begin(), prev.end());

        // If we have fewer previous lengths than current lengths,
        // combine 2 fastest dims.
        //
        // This should only happen with a plan like:
        //
        // B -> T         <-- large 1D parent
        //   B -> B       <-- decomposed child 1
        //   B -> T       <-- decomposed child 2
        //
        // Child 2 starts the temp write, and is part of the 1D
        // decomposition.  It will output more apparent dimensions
        // than the parent.  But it might pad its outputs, so the
        // parent's output needs to be adjusted accordingly.
        if(cur.size() < prev.size())
        {
            if(prev.size() - cur.size() > 1)
                throw std::runtime_error("padding: expected only one fewer cur dim");
            if(prev[0].length() != prev[1].stride() || prev[0].stride() != 1)
                throw std::runtime_error(
                    "padding: expected unit stride for decomposition of fastest dim");
            if(prev[0].length() * prev[1].length() != cur[0].length())
                throw std::runtime_error(
                    "padding: fastest two previous lengths don't combine into fastest cur length");
            // pretend the decomposed length is actually the full large 1D length
            prev.erase(prev.begin(), prev.begin() + 2);
            prev.emplace(prev.begin(), cur[0].length(), cur[0].stride());
        }

        // walk down cur and prev lengths to decompose as we go
        auto curIt  = cur.begin();
        auto prevIt = prev.begin();
        // track how much previous length remains, and how much
        // stride we've accumulated if we've only partially
        // decomposed a length
        size_t remain        = 0;
        size_t partialStride = 0;
        while(curIt != cur.end() && prevIt != prev.end())
        {
            if(remain == 0)
                remain = prevIt->length();
            if(partialStride == 0)
                partialStride = prevIt->stride();

            if(remain % curIt->length() != 0)
                throw std::runtime_error("padding: failed to decompose prev length");

            remain /= curIt->length();
            if(remain == 1)
            {
                // fully decomposed this prev length, assign stride
                curIt->stride() = partialStride;
                ++prevIt;
                ++curIt;
                remain        = 0;
                partialStride = 0;
            }
            else
            {
                // partially decomposed
                curIt->stride() = partialStride;
                partialStride *= curIt->length();
                ++curIt;
            }
        }
        // should get to the end of both lists if the two actually match
        if(curIt != cur.end())
            throw std::runtime_error("padding: didn't get to end of cur");
        if(prevIt != prev.end())
            throw std::runtime_error("padding: didn't get to end of prev");
    }

    // otherwise we can just do the right padding for the stride we have
    else
    {
        // don't pad if it already looks padded
        if(!StrideIsContiguous(cur))
            return;

        // separate out batch dim from the rest
        auto batchDim = cur.rbegin();
        auto highDim  = std::next(batchDim);

        // get highest dim length and lower dim length(s)
        size_t highLength   = highDim->length();
        size_t lowerLengths = 1;
        for(auto i = highDim + 1; i != cur.rend(); ++i)
            lowerLengths *= i->length();

        const size_t biggerDim  = std::max(highLength, lowerLengths);
        const size_t smallerDim = std::min(highLength, lowerLengths);
        bool needsPadding = ((smallerDim % 64 == 0) || (biggerDim % 64 == 0)) && (biggerDim >= 512);

        if(!needsPadding)
            return;

        static const size_t padding = 64;

        // normal case - adjust highest dim
        if(paddingDim == PaddingDim::HIGHEST)
        {
            highDim->stride() += padding;
        }
        // if the kernels involved don't allow padding the highest
        // dim, pad the second highest one instead
        else
        {
            auto nextHighestDim = std::next(highDim);
            nextHighestDim->stride() += padding;
            highDim->stride() += padding * nextHighestDim->length();
        }
        batchDim->stride() = highDim->stride() * highDim->length();
    }
}

// Gather up the length and stride information of a temp buffer
// operation (read or write) into a struct.  We want to be able to
// collect all of the related operations on a buffer to confirm that
// they're paddable, before we actually start changing any strides on
// any nodes.
struct TempBufOp
{
    enum Operation
    {
        BufRead,
        BufWrite,
    };
    TempBufOp(const std::vector<size_t>& _length,
              std::vector<size_t>&       _stride,
              size_t&                    _dist,
              Operation                  _op,
              TreeNode&                  _node)
        : length(_length)
        , stride(_stride)
        , dist(_dist)
        , op(_op)
        , node(_node)
    {
        // HACK: 1D stockham with fused C2R pre-processing will read
        // one extra element.  Padding needs to see this because
        // previous operations would count that extra element.
        if(node.scheme == CS_KERNEL_STOCKHAM && node.ebtype == EmbeddedType::C2Real_PRE
           && op == BufRead)
            const_cast<std::vector<size_t>&>(length).front()++;
    }

    const std::vector<size_t> length;
    std::vector<size_t>&      stride;
    size_t&                   dist;

    Operation op;
    TreeNode& node;

    // Return the forced dimension for padding for this kernel and
    // operation - if the kernel does not support padding on all
    // dimensions.
    std::optional<PaddingDim> GetForcedPaddingDim() const
    {
        // The current implementation of 3D transpose kernels
        // (XY_Z and Z_XY) does not work correctly for
        // arbitrary strides.  Normally we would want to pad
        // the highest dimension.  But XY_Z can only pad the
        // middle dimension on output and Z_XY can only pad
        // the middle dimension on input.
        if(node.scheme == CS_KERNEL_TRANSPOSE_XY_Z)
            return op == TempBufOp::BufWrite ? PaddingDim::SECOND_HIGHEST : PaddingDim::HIGHEST;
        if(node.scheme == CS_KERNEL_TRANSPOSE_Z_XY)
            return op == TempBufOp::BufRead ? PaddingDim::SECOND_HIGHEST : PaddingDim::HIGHEST;
        // other kernels would pad on highest
        return {};
    }
};

// Given a node and a temp buffer, collect the read/write operations
// that continue in that buffer until the data leaves that buffer.
//
// startNode is true when we're starting collection, and false when
// we're recursing into children and siblings.  It controls whether
// we want to look at the given node's input - we would want to do
// that when recursively chasing down subsequent operations on a
// buffer after an initial write.  But we would not want to look at
// input when starting from a parent like:
//
// T -> T         <-- parent node
//   T -> B       <-- child 1
//   B -> T       <-- child 2
//
// If parent is our starting node then the input to parent must have
// already been considered by a previous collection, and stopped when
// the data left T in child 1.  Therefore, what we're interested in
// for this collection is parent writing to T, starting from child 2.
void CollectTempBufOps(TreeNode&               node,
                       OperatingBuffer         buf,
                       std::vector<TempBufOp>& users,
                       bool                    startNode = false)
{
    // helper to insert an op into the vector - checks to make sure
    // we're not duplicating anything
    auto insertOp = [&users](TempBufOp&& newOp) {
        // ensure we're not adding a duplicate
        for(auto& u : users)
        {
            if(u.op == newOp.op && &u.node == &newOp.node)
                return;
        }
        users.emplace_back(newOp);
    };

    // If input is the buffer we're looking at
    if(!startNode && node.obIn == buf)
    {
        // Store this read
        insertOp({node.length, node.inStride, node.iDist, TempBufOp::BufRead, node});

        // If this is a parent node, its children can also continue
        // using the buffer
        for(auto& child : node.childNodes)
        {
            // chirp setup nodes don't connect to anything, so skip them
            if(child->IsBluesteinChirpSetup())
                continue;

            // Once a child stops using this temp buffer, stop
            // looking at children and return so we don't consider
            // this node's output either (since even if obOut is the
            // same node, there's clearly stuff happening in the
            // child nodes that means the data leaves the original
            // obIn).
            if(child->obIn != buf)
                return;
            CollectTempBufOps(*child, buf, users);
            if(child->obOut != buf)
                return;
        }
    }

    // If output is the buffer we're looking at
    if(node.obOut == buf)
    {
        // If this is a parent node, look backwards through the
        // children to collect child nodes that write to the buffer
        //
        // We're collecting the children in backward order so we can
        // add the operations in forward order after.
        std::vector<TreeNode*> backwardChildren;
        for(auto child = node.childNodes.rbegin(); child != node.childNodes.rend(); ++child)
        {
            // Once a child stops using this temp buffer, stop looking
            // at children.
            if(child->get()->obOut != buf)
                break;
            backwardChildren.push_back(child->get());
            if(child->get()->obIn != buf)
                break;
        }
        for(auto child = backwardChildren.rbegin(); child != backwardChildren.rend(); ++child)
            CollectTempBufOps(**child, buf, users);

        // Store this write
        insertOp({node.UseOutputLengthForPadding() ? node.GetOutputLength() : node.length,
                  node.outStride,
                  node.oDist,
                  TempBufOp::BufWrite,
                  node});

        // If we have a parent node, tail-recurse into our following sibling node.
        //
        // e.g.
        // A -> B     <-- parent
        //   A -> T   <-- child 1
        //   T -> T   <-- child 2
        //   T -> B   <-- child 3
        //   B -> B   <-- child 4
        //
        // Child 1 is a new use of a temp buffer.  We want to collect
        // children 2 and 3 also.
        auto parent = node.parent;
        if(parent)
        {
            auto self = std::find_if(
                parent->childNodes.begin(),
                parent->childNodes.end(),
                [&node](std::unique_ptr<TreeNode>& child) { return child.get() == &node; });
            auto nextSibling = std::next(self);
            if(nextSibling != parent->childNodes.end())
                CollectTempBufOps(*nextSibling->get(), buf, users);
        }
    }
}

static bool IsPaddableTempBuffer(OperatingBuffer buf)
{
    // Non-Bluestein temp buffers are candidates for padding.
    // Skip Bluestein because it has non-obvious rules around
    // what size of data is actually in the buffer, and it's a
    // slow fallback path anyway.
    return buf == OB_TEMP || buf == OB_TEMP_CMPLX_FOR_REAL;
}

void AssignmentPolicy::PadPlan(ExecPlan& execPlan)
{
    // for strided FFTs with dist 1, we mess around with dimensions
    // in ways that confuse padding.  don't try.
    if(execPlan.rootPlan->iDist == 1 || execPlan.rootPlan->oDist == 1)
        return;

    RecursiveTraverse(execPlan.rootPlan.get(), [&execPlan](TreeNode* n) {
        // Skip nodes that have partial passes
        if(n->isPartialPassEnabled())
            return;

        // Look for nodes that begin writing to a new temp buffer
        // (i.e. obOut is a paddable temp buffer, and obIn was a
        // different buffer)
        if(IsPaddableTempBuffer(n->obOut) && (n->obIn != n->obOut || !n->childNodes.empty()))
        {
            // collect up the input/output ops that use this temp
            // buffer before the data leaves it
            std::vector<TempBufOp> users;
            CollectTempBufOps(*n, n->obOut, users, true);

            // If the collected users doesn't end with a read, that
            // should mean we got the last children of a parent node.
            // An earlier point of this traversal should have
            // collected these users already when we traversed the
            // parent.
            //
            // e.g.
            //
            // A -> T     <-- parent
            //   A -> B   <-- child 1
            //   B -> T   <-- child 2
            // T -> B     <-- node after parent
            //
            // We'll visit child 2 and think it's a new write to T.
            // But when we traversed parent, we would already have
            // looked at parent's last children because parent writes
            // to T.
            if(users.back().op == TempBufOp::BufWrite)
                return;

            // Ensure that we're not trying to pad a write to a
            // sub-dimension.  e.g. if we're decomposing a large 1D
            // size into smaller subdimensions, we can't pad in the
            // middle of that.
            for(const auto& u : users)
            {
                if(u.op == TempBufOp::BufWrite && u.node.parent
                   && u.length.size() > u.node.parent->length.size())
                    return;
            }

            for(const auto& u : users)
            {
                // Padded Bluestein doesn't work in all cases
                if(u.node.scheme == CS_BLUESTEIN || u.node.scheme == CS_KERNEL_PAD_MUL
                   || u.node.scheme == CS_KERNEL_FFT_MUL || u.node.scheme == CS_KERNEL_RES_MUL)
                    return;
                // SBCR plans combine higher dimensions in ways that confuse padding
                if(u.node.scheme == CS_KERNEL_STOCKHAM_BLOCK_CR)
                    return;
            }

            // Ensure that if we're forced to pad along one dimension
            // that this forced choice isn't illegal later on for
            // this buffer.
            std::optional<PaddingDim> prevPaddingDim;
            for(const auto& u : users)
            {
                auto curPaddingDim = u.GetForcedPaddingDim();
                if(prevPaddingDim.has_value() && curPaddingDim.has_value()
                   && *prevPaddingDim != *curPaddingDim)
                {
                    // conflict - an earlier operation restricted us to
                    // a dimension and a later operation is restricted
                    // differently.
                    return;
                }

                // if this is the first op, assume we padded on highest dim
                if(!prevPaddingDim.has_value())
                    prevPaddingDim = curPaddingDim.value_or(PaddingDim::HIGHEST);
                // otherwise, remember any forced dim this op has
                else if(curPaddingDim.has_value() && u.op == TempBufOp::BufWrite)
                    prevPaddingDim = curPaddingDim;
            }

            // specific case for gfx906 - large double-precision pow2
            // 2D strides are slower with padding
            if(is_device_gcn_arch(execPlan.deviceProp, "gfx906"))
            {
                const auto& stride = users.front().stride;
                if(stride.size() == 3 && IsPo2(stride[0]) && stride[0] > 2048 && IsPo2(stride[1])
                   && stride[1] > 2048)
                    return;
            }

            // standalone real-complex kernels can't handle arbitrary
            // strides at the moment, so exclude them from padding
            if(std::any_of(users.begin(), users.end(), [](TempBufOp& op) {
                   return (op.node.scheme == CS_KERNEL_R_TO_CMPLX
                           || op.node.scheme == CS_KERNEL_CMPLX_TO_R);
               }))
                return;

            // pass previous write's strides to padding logic so it
            // can know how the data was shaped
            TempBufOp* previousWrite = nullptr;
            for(auto& u : users)
            {
                if(previousWrite)
                    PadStride(u.stride,
                              u.dist,
                              u.length,
                              n->batch,
                              previousWrite->stride,
                              previousWrite->dist,
                              previousWrite->length,
                              u.GetForcedPaddingDim().value_or(PaddingDim::HIGHEST));
                else
                    PadStride(u.stride,
                              u.dist,
                              u.length,
                              n->batch,
                              {},
                              0,
                              {},
                              u.GetForcedPaddingDim().value_or(PaddingDim::HIGHEST));
                if(u.op == TempBufOp::BufWrite)
                    previousWrite = &u;
            }
        }
    });
}

size_t PlacementTrace::NumPaddableTempOps() const
{
    auto   trace       = this;
    size_t tempOpCount = 0;
    while(trace != nullptr && trace->curNode != nullptr)
    {
        tempOpCount += IsPaddableTempBuffer(trace->inBuf) && trace->curNode->PaddingBenefitsInput();
        tempOpCount
            += IsPaddableTempBuffer(trace->outBuf) && trace->curNode->PaddingBenefitsOutput();
        trace = trace->parent;
    }
    return tempOpCount;
}
