// Copyright (C) 2016 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TREE_NODE_H
#define TREE_NODE_H

#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "../../../shared/gpubuf.h"
#include "../../../shared/hip_object_wrapper.h"
#include "../../../shared/rocfft_complex.h"
#include "../../shared/device_properties.h"
#include "../device/kernels/callback.h"
#include "../device/kernels/common.h"
#include "compute_scheme.h"
#include "enum_printer.h"
#include "function_map_key.h"
#include "function_pool.h"
#include "kargs.h"
#include "load_store_ops.h"
#include "logging.h"
#include "rocfft_mpi.h"
#include "rtc_kernel.h"
#include <hip/hip_runtime_api.h>

enum NodeType
{
    NT_UNDEFINED, // un init
    NT_INTERNAL, // an internal node contains childrens
    NT_LEAF, // a leaf node represents a kernel and has no childrens
};

enum FuseType
{
    FT_TRANS_WITH_STOCKHAM, // T_R
    FT_STOCKHAM_WITH_TRANS, // R_T
    FT_STOCKHAM_WITH_TRANS_Z_XY, // R_T-Z_XY
    FT_STOCKHAM_WITH_TRANS_XY_Z, // R_T-XY_Z
    FT_R2C_TRANSPOSE, // post-r2c + transpose
    FT_TRANSPOSE_C2R, // transpose + pre-c2r
    FT_STOCKHAM_R2C_TRANSPOSE, // Stokham + post-r2c + transpose (Advance of FT_R2C_TRANSPOSE)
};

struct GridParam
{
    unsigned int b_x, b_y, b_z; // in HIP, the data type of dimensions of work
    // items, work groups is unsigned int
    unsigned int wgs_x, wgs_y, wgs_z;
    unsigned int lds_bytes; // dynamic LDS allocation size

    GridParam()
        : b_x(1)
        , b_y(1)
        , b_z(1)
        , wgs_x(1)
        , wgs_y(1)
        , wgs_z(1)
        , lds_bytes(0)
    {
    }
};

// get the arch name, as a part of key of solution map
static std::string get_arch_name(const hipDeviceProp_t& prop)
{
    static const std::vector<std::string> arch_list = {"gfx803",
                                                       "gfx900",
                                                       "gfx906",
                                                       "gfx908",
                                                       "gfx90a",
                                                       "gfx942",
                                                       "gfx1030",
                                                       "gfx1100",
                                                       "gfx1101",
                                                       "gfx1102",
                                                       "gfx1200",
                                                       "gfx1201"};

    static const std::string anyArch("any");
    std::string              archName(prop.gcnArchName);

    for(const auto& arch : arch_list)
    {
        if(archName.find(arch) != std::string::npos)
            return arch;
    }

    // kind of a fall-back solution
    return anyArch;
}

static bool is_device_gcn_arch(const hipDeviceProp_t& prop, const std::string& cmpTarget)
{
    std::string archName(prop.gcnArchName);
    return archName.find(cmpTarget) != std::string::npos;
}

static bool is_diagonal_sbrc_3D_length(size_t len)
{
    // SBRC diagonal-transpose dimensions are currently 128, 256
    return len == 128 || len == 256;
}

static bool is_cube_size(const std::vector<size_t>& length)
{
    return length.size() == 3 && length[0] == length[1] && length[1] == length[2];
}

// Given a map of precision-length exceptions, check whether the
// length is present.  Assume half-precision has the same exceptions
// as single-precision.
static bool length_excepted(const std::map<rocfft_precision, std::set<size_t>>& exceptions,
                            rocfft_precision                                    precision,
                            size_t                                              length)
{
    if(precision == rocfft_precision_half)
        precision = rocfft_precision_single;
    return exceptions.at(precision).count(length);
}

void get_large_twd_base_steps(size_t large1DLen, bool use3steps, size_t& base, size_t& steps);

struct SchemeTree
{
    ComputeScheme                            curScheme;
    size_t                                   numKernels = 0;
    std::vector<std::unique_ptr<SchemeTree>> children;

    SchemeTree() {}
    SchemeTree(ComputeScheme s)
        : curScheme(s)
    {
    }
};

using SchemeTreeVec = std::vector<std::unique_ptr<SchemeTree>>;

static SchemeTreeVec EmptySchemeTreeVec = {};

using SchemeVec = std::vector<ComputeScheme>;

static SchemeVec EmptySchemeVec = {};

class TreeNode;
class LeafNode;
class function_pool;

// The mininal tree node data needed to decide the scheme
struct NodeMetaData
{
    size_t                  batch     = 1;
    size_t                  dimension = 1;
    std::vector<size_t>     length;
    std::vector<size_t>     outputLength;
    std::vector<size_t>     inStride, outStride;
    std::vector<size_t>     inStrideBlue, outStrideBlue;
    size_t                  iDist = 0, oDist = 0;
    size_t                  iDistBlue = 0, oDistBlue = 0;
    size_t                  iOffset = 0, oOffset = 0;
    int                     direction    = -1;
    rocfft_result_placement placement    = rocfft_placement_inplace;
    rocfft_precision        precision    = rocfft_precision_single;
    rocfft_array_type       inArrayType  = rocfft_array_type_unset;
    rocfft_array_type       outArrayType = rocfft_array_type_unset;
    hipDeviceProp_t         deviceProp   = {};
    bool                    rootIsC2C;

    explicit NodeMetaData(TreeNode* refNode);
};

class rocfft_ostream;

class FuseShim
{
    friend class NodeFactory;

protected:
    FuseShim(const std::vector<TreeNode*>& components, FuseType type)
        : fuseType(type)
        , nodes(components)
    {
        // default
        lastFusedNode = nodes.size() - 1;
    }

    // if these schemes can be fused
    virtual bool CheckSchemeFusable() = 0;

    bool schemeFusable = false;

public:
    FuseType fuseType;

    // nodes that contained in this shim
    std::vector<TreeNode*> nodes;

    // basically all fusion should be effectively-outofplace,
    // but TransC2R and R2CTrans can do some tricks
    bool   allowInplace   = true;
    size_t firstFusedNode = 0;
    size_t lastFusedNode;

public:
    // for the derived class
    virtual ~FuseShim() = default;

    // if the in/out buffer meets the placement requirement
    // the firstOBuffer is optional, used in R2CTrans only
    virtual bool
        PlacementFusable(OperatingBuffer iBuf, OperatingBuffer firstOBuf, OperatingBuffer lastOBuf);

    // return the result of CheckSchemeFusable
    bool IsSchemeFusable() const;

    // NB: Some fusions perform better or worse in different arch.
    //     We mark those exceptions from the execPlan.
    //     (We can only know the arch name from execPlan)
    //     A known case is RTFuse, length 168 in MI50
    void OverwriteFusableFlag(bool fusable);

    void ForEachNode(std::function<void(TreeNode*)> func);

    // the first/last node that to be fused
    // for R_T, T_R, it is pretty simple [0] and [1]
    // but for R_T-Z_XY, we store an extra "pre-node" to test if the RT fuse can be done
    // in this case, the first, last are [1], [2]. [0] doesn't participate the fusion
    virtual TreeNode* FirstFuseNode() const;
    virtual TreeNode* LastFuseNode() const;

    virtual std::unique_ptr<TreeNode> FuseKernels() = 0;
};

class TreeNode
{
    friend class NodeFactory;

protected:
    TreeNode(TreeNode* p)
        : parent(p)
        , deviceProp(p ? p->deviceProp : get_curr_device_prop())
        , pool(deviceProp)
    {
        if(p != nullptr)
        {
            precision = p->precision;
            batch     = p->batch;
            direction = p->direction;
        }

        allowedOutBuf
            = OB_USER_IN | OB_USER_OUT | OB_TEMP | OB_TEMP_CMPLX_FOR_REAL | OB_TEMP_BLUESTEIN;

        allowedOutArrayTypes = {rocfft_array_type_complex_interleaved,
                                rocfft_array_type_complex_planar,
                                rocfft_array_type_real,
                                rocfft_array_type_hermitian_interleaved};
    }

public:
    // node type: internal node or leaf node, or un-defined (un-init)
    NodeType nodeType = NT_UNDEFINED;

    // Batch size
    size_t batch = 1;

    // Transform dimension - note this can be different from data dimension, user
    // provided
    size_t dimension = 1;

    // Length of the FFT in each dimension, internal value
    std::vector<size_t> length;

    // Row-major output lengths, from fastest to slowest.  If empty,
    // output length is assumed to be the same as input length.
    //
    // This is set for nodes that might do non-obvious things with
    // strides (e.g. having fastest dimension not be first), so that
    // buffer assignment can know whether a node's output will fit in
    // a given buffer.
    std::vector<size_t> outputLength;

    // Stride of the FFT in each dimension
    std::vector<size_t> inStride, outStride;

    // Stride of the fused Bluestein FFT in each dimension
    std::vector<size_t> inStrideBlue, outStrideBlue;

    // Distance between consecutive batch members:
    size_t iDist = 0, oDist = 0;

    // Distance between consecutive batch members in fused Bluestein nodes
    size_t iDistBlue = 0, oDistBlue = 0;

    // Offsets to start of data in buffer:
    size_t iOffset = 0, oOffset = 0;

    // Direction of the transform (-1: forward, +1: inverse)
    int direction = -1;

    // Data format parameters:
    rocfft_result_placement placement    = rocfft_placement_inplace;
    rocfft_precision        precision    = rocfft_precision_single;
    rocfft_array_type       inArrayType  = rocfft_array_type_unset;
    rocfft_array_type       outArrayType = rocfft_array_type_unset;

    // Extra twiddle multiplication for large 1D
    size_t large1D = 0;
    // decompose large twiddle to product of 256(8) or 128(7) or 64(6)...or 16(4)
    // default is 8, and sbcc could be dynamically decomposed
    size_t largeTwdBase = 8;
    // flag indicating if using the 3-step decomp. for large twiddle? (16^3, 32^3, 64^3)
    // if false, always use 8 as the base (256*256*256....)
    bool largeTwd3Steps = false;
    // "Steps": how many exact loops we need to decompose the LTWD?
    // if we pass this as a template arg in kernel, should avoid dynamic while-loop
    // We will update this in get_large_twd_base_steps()
    size_t ltwdSteps = 0;
    // true if large twd multiply uses batch as transform count - this
    // is done on strided large 1D FFTs where the batch dimension moves
    // faster than the large 1D subdimension
    bool largeTwdBatchIsTransformCount = false;

    // embedded C2R/R2C pre/post processing
    EmbeddedType ebtype = EmbeddedType::NONE;

    // if the kernel supports/use/not-use dir-to-from-reg
    DirectRegType dir2regMode = DirectRegType::FORCE_OFF_OR_NOT_SUPPORT;

    // sbrc transpose type
    mutable SBRC_TRANSPOSE_TYPE sbrcTranstype = SBRC_TRANSPOSE_TYPE::NONE;

    // specified kernel keys from solution map. (if there are any)
    std::unique_ptr<FMKey>   specified_key;
    std::unique_ptr<PPFMKey> specified_pp_key;

    // Tree structure:
    // non-owning pointer to parent node, may be null
    TreeNode* parent = nullptr;
    // owned pointers to children
    std::vector<std::unique_ptr<TreeNode>> childNodes;

    // one shim is a group of several "possibly" fusable nodes
    std::vector<std::unique_ptr<FuseShim>> fuseShims;

    // FIXME: document
    ComputeScheme   scheme = CS_NONE;
    OperatingBuffer obIn = OB_UNINIT, obOut = OB_UNINIT;

    // Length of the FFT for computing zero-padded linear convolutions
    // in Bluestein's algorithm. If Bluestein is required to compute an
    // FFT of length N, then lengthBlue >= 2N - 1.
    size_t lengthBlue  = 0;
    size_t lengthBlueN = 0;

    // Index of off-dimension in partial-pass nodes
    size_t ppOffDim = 0;

    // Index of current dimension (full pass) in partial-pass nodes
    size_t ppCurrDim = 0;

    //
    BluesteinType     typeBlue   = BluesteinType::BT_NONE;
    BluesteinFuseType fuseBlue   = BluesteinFuseType::BFT_NONE;
    bool              need_chirp = false;

    // Device pointers:
    // twiddle memory is owned by the repo
    void*            twiddles            = nullptr;
    size_t           twiddles_size       = 0;
    void*            twiddles_large      = nullptr;
    size_t           twiddles_large_size = 0;
    void*            twiddles_pp         = nullptr;
    size_t           twiddles_pp_size    = 0;
    void*            chirp               = nullptr;
    size_t           chirp_size          = 0;
    gpubuf_t<size_t> devKernArg;

    // callback parameters
    UserCallbacks callbacks;

    hipDeviceProp_t deviceProp = {};
    function_pool   pool;

    // comments inserted by optimization passes to explain changes done
    // to the node
    std::vector<std::string> comments;

    // runtime-compiled kernels for this node
    std::shared_future<std::unique_ptr<RTCKernel>> compiledKernel;
    std::shared_future<std::unique_ptr<RTCKernel>> compiledKernelWithCallbacks;

    // Does this node allow inplace/not-inplace? default true,
    // each class handles the exception
    // transpose, sbrc, fused stockham only outofplace
    // bluestein component leaf node (multiply) only inplace
    bool allowInplace    = true;
    bool allowOutofplace = true;

    // only root node keeps this info from execPlan
    bool inStrideUnit;
    bool outStrideUnit;

    // if soffset < 2^32 then we can't use it. Check soffset by buffer size
    // separate load and store,
    // in inplace kernels, enabling both is not always a good choice
    IntrinsicAccessType intrinsicMode = IntrinsicAccessType::DISABLE_BOTH;

    size_t                      allowedOutBuf;
    std::set<rocfft_array_type> allowedOutArrayTypes;

    LoadOps  loadOps;
    StoreOps storeOps;

public:
    // Disallow copy constructor:
    TreeNode(const TreeNode&) = delete;

    // for the derived class
    virtual ~TreeNode();

    // Disallow assignment operator:
    TreeNode& operator=(const TreeNode&) = delete;

    // Copy data from another node (to a fused node)
    void CopyNodeData(const TreeNode& srcNode);

    // Copy data from the NodeMetaData (after deciding scheme)
    void CopyNodeData(const NodeMetaData& data);

    bool isPlacementAllowed(rocfft_result_placement) const;
    bool isOutBufAllowed(OperatingBuffer oB) const;
    bool isOutArrayTypeAllowed(rocfft_array_type) const;
    bool isRootNode() const;
    bool isLeafNode() const;
    // Get the current node downcasted to a LeafNode.  Throws bad_cast
    // if this node isn't actually a leaf.
    LeafNode&       getLeafNode();
    const LeafNode& getLeafNode() const;

    // whether or not the input/output access pattern may benefit from padding
    virtual bool PaddingBenefitsInput()
    {
        return false;
    }
    virtual bool PaddingBenefitsOutput()
    {
        return false;
    }

    void RecursiveBuildTree(SchemeTree* solution_scheme = nullptr);

    virtual void SanityCheck(SchemeTree*         solution_scheme = nullptr,
                             std::vector<FMKey>& kernel_keys     = EmptyFMKeyVec);

    // used when RTC kernels, and output to solution map for pre-building to AOT-cache
    // 3D_SBRC will override this
    virtual unsigned int GetStaticDim() const
    {
        return length.size();
    }

    // If high dims are contiguous, we can collapse them to make offset
    // calculation simpler
    void CollapseContiguousDims();
    // Leaf nodes can override this to say what dims can be collapsed.
    // Return values are indexes into the length/stride arrays.
    virtual std::vector<size_t> CollapsibleDims()
    {
        return {};
    }

    // Check node scheme to see if partial pass is enabled
    bool isPartialPassEnabled() const
    {
        return (scheme == CS_3D_PP || scheme == CS_KERNEL_STOCKHAM_PP
                || scheme == CS_KERNEL_STOCKHAM_PP_BLOCK_CC);
    }

    // able to fuse CS_KERNEL_STOCKHAM and CS_KERNEL_TRANSPOSE_Z_XY ?
    bool fuse_CS_KERNEL_TRANSPOSE_Z_XY();
    // able to fuse CS_KERNEL_STOCKHAM and CS_KERNEL_TRANSPOSE_XY_Z ?
    bool fuse_CS_KERNEL_TRANSPOSE_XY_Z();
    // able to fuse STK, r2c, transp to CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY ?
    bool fuse_CS_KERNEL_STK_R2C_TRANSPOSE();

    void ApplyFusion();

    void RefreshTree();

    // Set strides and distances:
    void AssignParams();

    // Collect LeadNodes and FuseShims:
    void CollectLeaves(std::vector<TreeNode*>& seq, std::vector<FuseShim*>& fuseSeq);

    // Determine work memory requirements:
    void DetermineBufferMemory(size_t& tmpBufSize,
                               size_t& cmplxForRealSize,
                               size_t& blueSize,
                               size_t& chirpSize);

    // Output plan information for debug purposes:
    virtual void Print(rocfft_ostream& os, int indent = 0) const;

    // logic B - using in-place transposes, todo
    //void RecursiveBuildTreeLogicB();

    void RecursiveFindChildNodes(const ComputeScheme& scheme, std::vector<TreeNode*>& nodes);
    void RecursiveCopyNodeData(const TreeNode& srcNode);

    void RecursiveRemoveNode(TreeNode* node);

    // insert a newNode before the node "pos"
    void RecursiveInsertNode(TreeNode* pos, std::unique_ptr<TreeNode>& newNode);

    // Get root node of plan
    TreeNode* GetPlanRoot();
    // If 'this' is a leaf, return it.  Otherwise, return the first
    // leaf node under 'this' in the execution sequence.
    TreeNode* GetFirstLeaf();
    // If 'this' is a leaf, return it.  Otherwise, return the last leaf
    // node under 'this' in the execution sequence.
    TreeNode* GetLastLeaf();
    // Return ancestor node of 'this' that is real-even (1D/2D/3D), or
    // nullptr if there is no such ancestor
    TreeNode* GetRealEvenAncestor();
    bool      IsRootPlanC2CTransform();

    // Return ancestor node of 'this' that is partial-pass, or
    // nullptr if there is no such ancestor
    TreeNode* GetPartialPassAncestor() const;

    // Set length of transpose kernel node, since those are easily
    // knowable just by looking at the scheme and they're used in
    // many plans.  Throws an exception if this is not a transpose
    // node.
    void SetTransposeOutputLength();

    // Get row-major output length of this node.
    std::vector<size_t> GetOutputLength() const
    {
        return outputLength.empty() ? length : outputLength;
    }
    // Padding needs matching stride + length to make its decisions.
    // For most nodes, outStride + length can be used together.  For
    // some nodes, outputLength is what matches outStride.
    virtual bool UseOutputLengthForPadding()
    {
        return false;
    }

    virtual bool KernelCheck(std::vector<FMKey>& kernel_keys = EmptyFMKeyVec) = 0;
    virtual bool CreateDevKernelArgs()                                        = 0;
    virtual bool CreateDeviceResources()                                      = 0;
    virtual void SetupGridParam(GridParam& gp)                                = 0;

    // for 3D SBRC kernels, decide the transpose type based on the
    // block width and lengths that the block tiles need to align on.
    // default type is NONE, meaning this isn't a SBRC node
    virtual SBRC_TRANSPOSE_TYPE sbrc_transpose_type(unsigned int blockWidth) const
    {
        return NONE;
    }

    // default implementation of leaf node, for non-sbrc type without sbrc_trans
    virtual FMKey GetKernelKey() const
    {
        if(specified_key)
            return *specified_key.get();

        return (dimension == 1) ? FMKey(length[0], precision, scheme)
                                : FMKey(length[0], length[1], precision, scheme);
    }

    // Partial pass parent nodes, e.g., CS_3D_PP, have
    // two kernels associated with them. The key for
    // querying the function pool is different from the
    // the standard kernel key.
    virtual PPFMKey GetPPKernelsKey() const
    {
        if(specified_pp_key)
            return *specified_pp_key.get();

        auto pp_parent_node = GetPartialPassAncestor();
        if(!pp_parent_node)
            throw std::runtime_error("Invalid parent node for partial pass");

        return PPFMKey(pp_parent_node->length[0],
                       pp_parent_node->length[1],
                       pp_parent_node->length[2],
                       precision,
                       pp_parent_node->scheme);
    }

    // Query the function pool with the right key,
    // and return the kernel linked to this node.
    virtual FFTKernel GetKernel() const
    {
        if(isPartialPassEnabled())
        {
            auto key = GetPPKernelsKey();
            return pool.get_kernel(key, scheme);
        }
        else
        {
            auto key = GetKernelKey();
            return pool.get_kernel(key);
        }
    }

    // Check if the function pool has a kernel,
    // querying it with the key linked to this node.
    virtual bool HasKernel() const
    {
        if(isPartialPassEnabled())
        {
            auto key = GetPPKernelsKey();
            if(!pool.has_function(key))
            {
                if(LOG_TRACE_ENABLED())
                    (*LogSingleton::GetInstance().GetTraceOS()) << PrintMissingKernelInfo(key);

                return false;
            }
        }
        else
        {
            auto key = GetKernelKey();
            if(!pool.has_function(key))
            {
                if(LOG_TRACE_ENABLED())
                    (*LogSingleton::GetInstance().GetTraceOS()) << PrintMissingKernelInfo(key);

                return false;
            }
        }

        return true;
    }

    // Compute the large twd decomposition base
    void set_large_twd_base_steps(size_t largeTWDLength);

    // return true if this node is setting up the Bluestein chirp
    // buffer - those nodes aren't connected to the input/output chain
    // of user data
    bool IsBluesteinChirpSetup();

    // Assuming callbacks need to run on this node, return the
    // specific CallbackType for this node - takes into account
    // whether the node is treating real data as complex
    CallbackType GetCallbackType(bool enable_callbacks) const;

protected:
    virtual void BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) = 0;
    virtual void AssignParams_internal()                                                    = 0;
};

class InternalNode : public TreeNode
{
    friend class NodeFactory;

protected:
    explicit InternalNode(TreeNode* p)
        : TreeNode(p)
    {
        nodeType = NT_INTERNAL;
    }

    bool CreateDevKernelArgs() override
    {
        throw std::runtime_error("Shouldn't call CreateDevKernelArgs in a non-LeafNode");
        return false;
    }

    bool CreateDeviceResources() override
    {
        throw std::runtime_error("Shouldn't call CreateDeviceResources in a non-LeafNode");
        return false;
    }

    void SetupGridParam(GridParam& gp) override
    {
        throw std::runtime_error("Shouldn't call SetupGridParam in a non-LeafNode");
    }

public:
    bool KernelCheck(std::vector<FMKey>& kernel_keys = EmptyFMKeyVec) override
    {
        return true;
    }
};

class LeafNode : public InternalNode
{
    friend class NodeFactory;

protected:
    LeafNode(TreeNode* p, ComputeScheme s)
        : InternalNode(p)
    {
        nodeType = NT_LEAF;
        scheme   = s;
    }

public:
    bool                externalKernel   = false;
    bool                need_twd_table   = false;
    bool                twd_no_radices   = false;
    bool                twd_attach_halfN = false;
    std::vector<size_t> kernelFactors    = {};
    std::vector<size_t> kernelFactorsPP  = {}; // factors for off-direction partial pass(es)
    size_t              bwd              = 1; // bwd, wgs, lds are for grid param lds_bytes
    size_t              wgs              = 0;
    size_t              lds              = 0;

    void BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) final {
    } // nothing to do in leaf node
    void AssignParams_internal() final {} // nothing to do in leaf node
    bool CreateLargeTwdTable();

    virtual size_t GetTwiddleTableLength();
    // Limit length of generated twiddle table.  Default limit is 0,
    // which means to generate the full length of table.
    virtual size_t GetTwiddleTableLengthLimit()
    {
        return 0;
    }
    virtual void SetupGridParam_internal(GridParam& gp) = 0;

public:
    // leaf node would print additional informations about kernel setting
    void         Print(rocfft_ostream& os, int indent = 0) const override;
    bool         KernelCheck(std::vector<FMKey>& kernel_keys = EmptyFMKeyVec) override;
    void         SanityCheck(SchemeTree*         solution_scheme = nullptr,
                             std::vector<FMKey>& kernel_keys     = EmptyFMKeyVec) override;
    virtual bool CreateDevKernelArgs() override;
    bool         CreateDeviceResources() override;
    void         SetupGridParam(GridParam& gp) override;
    FMKey        GetKernelKey() const override;
    virtual void GetKernelFactors();
    virtual void GetKernelPartialPassFactors();
};

/*****************************************************
 * CS_KERNEL_TRANSPOSE
 * CS_KERNEL_TRANSPOSE_XY_Z
 * CS_KERNEL_TRANSPOSE_Z_XY
 *****************************************************/
class TransposeNode : public LeafNode
{
    friend class NodeFactory;

protected:
    TransposeNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
        allowInplace = false;
    }

    void SetupGridParam_internal(GridParam& gp) override;

public:
    // Transpose tiles read more row-ish and write more column-ish.  So
    // assume output benefits more from padding than input.
    bool PaddingBenefitsOutput() override
    {
        // HACK: only assume we benefit if we have no large twiddle multiply.
        //
        // Since large twiddle multiply (i.e. middle T of L1D_TRTRT)
        // cannot be fused with an FFT kernel, we should not try too
        // hard to pad its output.  The other T nodes of that plan can
        // keep their buffer assigments so that padding doesn't upset
        // the current choice of which nodes we fuse.
        return large1D == 0;
    }
};

// Identifier for a location that a buffer lives on, or that a kernel
// will execute on.  this specifies a multi-process rank as well as a
// device ID.
struct rocfft_location_t
{
    rocfft_location_t() = default;
    rocfft_location_t(int _comm_rank, int _device)
        : comm_rank(_comm_rank)
        , device(_device)
    {
    }

    // return a location for the current device on comm rank 0
    static rocfft_location_t rank0_current_device()
    {
        rocfft_location_t id;
        if(hipGetDevice(&id.device) != hipSuccess)
            throw std::runtime_error("hipGetDevice failed");
        return id;
    }

    // allow locations to be sorted
    bool operator<(const rocfft_location_t& other) const
    {
        if(comm_rank != other.comm_rank)
            return comm_rank < other.comm_rank;
        return device < other.device;
    }

    bool operator==(const rocfft_location_t& other) const
    {
        return comm_rank == other.comm_rank && device == other.device;
    }

    int comm_rank = 0;
    int device    = 0;
};

// Internally-allocated temporary buffers (as opposed to
// user-provided work/in/out buffers)
class InternalTempBuffer
{
public:
    InternalTempBuffer(int comm_rank)
        : comm_rank(comm_rank)
    {
    }
    InternalTempBuffer(const InternalTempBuffer&) = delete;
    InternalTempBuffer& operator=(const InternalTempBuffer&) = delete;
    ~InternalTempBuffer()                                    = default;

    void set_size_bytes(size_t in)
    {
        if(buf)
            throw std::runtime_error("cannot set internal buffer size after allocation");
        if(in > size_bytes)
            size_bytes = in;
    }

    size_t get_size_bytes() const
    {
        return size_bytes;
    }

    void alloc(int deviceID)
    {
        rocfft_scoped_device device(deviceID);
        if(buf.alloc(size_bytes) != hipSuccess)
            throw std::runtime_error("internal temp buffer allocation failure");
    }

    void* data()
    {
        return buf.data();
    }

    int get_comm_rank() const
    {
        return comm_rank;
    }

private:
    int    comm_rank  = 0;
    size_t size_bytes = 0;
    gpubuf buf;
};

// Class representing a buffer in a multi-plan item.
//
// An item in a plan can work on inputs or outputs like:
// - a specific temp buffer allocated during plan creation
// - the Nth pointer that the user provided as input at execute time
// - the Mth pointer that the user provided as output at execute time
//
// These buffers need to be set during plan creation.  While temp
// buffers are knowable at that time, user-provided pointers are not.
// So this class just records which logical pointer we will want.
//
// The get() method accepts the user-provided input/output pointers,
// and returns the correct pointer during plan executions.
class BufferPtr
{
public:
    BufferPtr()                 = default;
    BufferPtr(const BufferPtr&) = default;
    BufferPtr& operator=(const BufferPtr&) = default;
    ~BufferPtr()                           = default;

    // return a new BufferPtr that points to a user input
    static BufferPtr user_input(size_t idx, int comm_rank)
    {
        BufferPtr ret;
        ret.type      = PTR_USER_IN;
        ret.idx       = idx;
        ret.comm_rank = comm_rank;
        return ret;
    }

    // return a new BufferPtr that points to a user output
    static BufferPtr user_output(size_t idx, int comm_rank)
    {
        BufferPtr ret;
        ret.type      = PTR_USER_OUT;
        ret.idx       = idx;
        ret.comm_rank = comm_rank;
        return ret;
    }

    // return a new BufferPtr that points to a temp buffer
    static BufferPtr temp(std::shared_ptr<InternalTempBuffer> ptr)
    {
        BufferPtr ret;
        ret.type      = PTR_TEMP;
        ret.temp_ptr  = ptr;
        ret.comm_rank = ptr->get_comm_rank();
        return ret;
    }

    // Get a pointer to the buffer.  The buffer might be an
    // user-provided input or output buffer that's only known at
    // execute time.
    void* get(void* in_buffer[], void* out_buffer[], int local_comm_rank) const
    {
        if(comm_rank != local_comm_rank)
            return nullptr;
        switch(type)
        {
        case PTR_NULL:
            throw std::runtime_error("fetching null item pointer");
        case PTR_USER_IN:
            return in_buffer[idx];
        case PTR_USER_OUT:
            return out_buffer[idx];
        case PTR_TEMP:
            return temp_ptr->data();
        }
    }

    std::string str() const
    {
        switch(type)
        {
        case PTR_NULL:
            return "(null)";
        case PTR_USER_IN:
            if(comm_rank != -1)
                return "user input buffer " + std::to_string(idx) + " on rank "
                       + std::to_string(comm_rank);
            else
                return "user input buffer " + std::to_string(idx);
        case PTR_USER_OUT:
            if(comm_rank != -1)
                return "user output buffer " + std::to_string(idx) + " on rank "
                       + std::to_string(comm_rank);
            else
                return "user output buffer " + std::to_string(idx);
        case PTR_TEMP:
        {
            std::stringstream ss;
            ss << "temp buffer on rank " << comm_rank << " ";
            if(temp_ptr)
                ss << temp_ptr->data();
            else
                ss << "(null)";
            return ss.str();
        }
        }
    }

    operator bool() const
    {
        return type != PTR_NULL;
    }

    bool operator==(const BufferPtr& other) const
    {
        return this->type == other.type && this->idx == other.idx
               && this->temp_ptr == other.temp_ptr;
    }
    bool operator!=(const BufferPtr& other) const
    {
        return !(*this == other);
    }

    enum PtrType
    {
        PTR_NULL,
        PTR_USER_IN,
        PTR_USER_OUT,
        PTR_TEMP,
    };

    PtrType ptr_type() const
    {
        return type;
    }

private:
    PtrType                             type      = PTR_NULL;
    size_t                              idx       = 0;
    int                                 comm_rank = -1;
    std::shared_ptr<InternalTempBuffer> temp_ptr;
};

struct rocfft_mp_request_t;

// Abstract base class for all items in a multi-node/device plan
struct MultiPlanItem
{
    MultiPlanItem();
    virtual ~MultiPlanItem();
    MultiPlanItem(const MultiPlanItem&) = delete;
    MultiPlanItem& operator=(const MultiPlanItem&) = delete;

    // multi-process requests
    std::vector<rocfft_mp_request_t> comm_requests;

    // Allocate this object's stream and queue work onto it.  This
    // object's event is allocated and recorded on the stream when
    // the last piece of work is queued, so callers can wait on that
    // event to know when the work is complete.
    virtual void ExecuteAsync(const rocfft_plan     plan,
                              void*                 in_buffer[],
                              void*                 out_buffer[],
                              rocfft_execution_info info,
                              size_t                multiPlanIdx)
        = 0;

    // wait for async operations to finish
    virtual void Wait() = 0;

    // wait for outstanding communication requests to finish
    void WaitCommRequests();

    // Get work buffer requirements for this item.  Only ExecPlans
    // should need this, as data movement shouldn't need temp buffers.
    virtual size_t WorkBufBytes(size_t base_type_size) const
    {
        return 0;
    }

    // Print a description of this item to the plan log
    virtual void Print(rocfft_ostream& os, const int indent) const = 0;

    // utility function to print a buffer enum with a description of
    // the pointer and an offset
    static std::string PrintBufferPtrOffset(const BufferPtr& ptr, size_t offset);

    // Check if this item writes to the specified BufferPtr
    virtual bool WritesToBuffer(const BufferPtr& ptr) const = 0;

    // Check if the specified rank will execute this item
    virtual bool ExecutesOnRank(int rank) const = 0;
    // Check if local rank will execute this item
    bool ExecutesOnLocalRank() const
    {
        return ExecutesOnRank(local_comm_rank);
    };

    // High-level description of what this item is doing, displayed when logging plan graph
    std::string description;
    // Group to assign this item to (letters, numbers, underscores).
    // Items in the same group are drawn together in the graph
    std::string group;

    // Compute a communication tag for an operation, for an item with
    // multiple operations in it.  The multi-plan index uniquely
    // identifies an item and works for this purpose for an item with a
    // single operation in it.  This function produces unique tags for
    // items with multiple operations.
    static int GetOperationCommTag(size_t multiPlanIdx, size_t opIdx);

    // This process's rank relative to the plan communicator.
    int local_comm_rank;

    // Sub-communicator for this operation, which is only set if
    // we know we're only talking to a subset of the ranks
    std::optional<MPI_Comm_wrapper_t> subcomm;
    // Helper to get the sub-communicator if it's set, or the plan's
    // communicator.
    MPI_Comm ActiveMPIComm(const rocfft_plan plan) const;
};

// Communication operations
struct CommPointToPoint : public MultiPlanItem
{
    CommPointToPoint(int               local_comm_rank,
                     rocfft_precision  _precision,
                     rocfft_array_type _arrayType,
                     size_t            _numElems,
                     rocfft_location_t _srcLocation,
                     BufferPtr         _srcPtr,
                     size_t            _srcOffset,
                     rocfft_location_t _destLocation,
                     BufferPtr         _destPtr,
                     size_t            _destOffset)
        : precision(_precision)
        , arrayType(_arrayType)
        , numElems(_numElems)
        , srcLocation(_srcLocation)
        , srcPtr(_srcPtr)
        , srcOffset(_srcOffset)
        , destLocation(_destLocation)
        , destPtr(_destPtr)
        , destOffset(_destOffset)
    {
        // same-rank communication needs a stream and event to signal completion
        if(local_comm_rank == srcLocation.comm_rank
           && srcLocation.comm_rank == destLocation.comm_rank)
        {
            rocfft_scoped_device dev(srcLocation.device);
            stream.alloc();
            event.alloc();
        }
    }

    void ExecuteAsync(const rocfft_plan     plan,
                      void*                 in_buffer[],
                      void*                 out_buffer[],
                      rocfft_execution_info info,
                      size_t                multiPlanIdx) override;
    void Wait() override;

    void Print(rocfft_ostream& os, const int indent) const override;

    bool WritesToBuffer(const BufferPtr& ptr) const override
    {
        return ptr == destPtr;
    }

    bool ExecutesOnRank(int comm_rank) const override
    {
        return srcLocation.comm_rank == comm_rank || destLocation.comm_rank == comm_rank;
    }

private:
    const rocfft_precision  precision;
    const rocfft_array_type arrayType;

    // number of elements to copy
    const size_t numElems;

    const rocfft_location_t srcLocation;
    const BufferPtr         srcPtr;
    const size_t            srcOffset = 0;

    const rocfft_location_t destLocation;
    const BufferPtr         destPtr;
    const size_t            destOffset = 0;
    // Stream to run the async operation in
    hipStream_wrapper_t stream;
    // Event to signal when the async operations are finished.
    hipEvent_wrapper_t event;
};

// This struct has a vector of ranks to scatter to.  Executing can
// create an MPI group with those ranks.
struct CommScatter : public MultiPlanItem
{
    CommScatter(rocfft_precision  _precision,
                rocfft_array_type _arrayType,

                rocfft_location_t _srcLocation,
                BufferPtr         _srcPtr)
        : precision(_precision)
        , arrayType(_arrayType)
        , srcLocation(_srcLocation)
        , srcPtr(_srcPtr)
    {
    }

    // one or more ranks to send data to
    struct ScatterOp
    {
        ScatterOp(rocfft_location_t destLocation,
                  BufferPtr         destPtr,
                  size_t            srcOffset,
                  size_t            destOffset,
                  size_t            numElems)
            : destLocation(destLocation)
            , destPtr(destPtr)
            , srcOffset(srcOffset)
            , destOffset(destOffset)
            , numElems(numElems)
        {
        }

        rocfft_location_t destLocation;
        BufferPtr         destPtr;

        size_t srcOffset;
        size_t destOffset;

        // Number of elements to copy
        size_t numElems;
    };

    // add an operation to scatter to a destination
    void AddOperation(int local_comm_rank, ScatterOp&& op)
    {
        // same-rank communication needs a stream and event to signal completion
        if(srcLocation.comm_rank == local_comm_rank
           && op.destLocation.comm_rank == srcLocation.comm_rank)
        {
            rocfft_scoped_device dev(srcLocation.device);
            if(!stream)
                stream.alloc();
            if(!event)
                event.alloc();
        }
        ops.emplace_back(std::move(op));
    }

    void ExecuteAsync(const rocfft_plan     plan,
                      void*                 in_buffer[],
                      void*                 out_buffer[],
                      rocfft_execution_info info,
                      size_t                multiPlanIdx) override;
    void Wait() override;

    void Print(rocfft_ostream& os, const int indent) const override;

    bool WritesToBuffer(const BufferPtr& ptr) const override
    {
        for(const auto& op : ops)
        {
            if(ptr == op.destPtr)
                return true;
        }
        return false;
    }

    bool ExecutesOnRank(int comm_rank) const override
    {
        return srcLocation.comm_rank == comm_rank
               || std::any_of(ops.begin(), ops.end(), [comm_rank](const ScatterOp& op) {
                      return op.destLocation.comm_rank == comm_rank;
                  });
    }

private:
    const rocfft_precision  precision;
    const rocfft_array_type arrayType;
    const rocfft_location_t srcLocation;
    const BufferPtr         srcPtr;

    std::vector<ScatterOp> ops;

    // Stream to run the async operations in
    hipStream_wrapper_t stream;
    // Event to signal when the async operations are finished.
    hipEvent_wrapper_t event;
};

// This struct has a vector of ranks to gather from.  Executing can
// create an MPI group with those ranks.
struct CommGather : public MultiPlanItem
{
    CommGather(int               local_comm_rank,
               rocfft_precision  _precision,
               rocfft_array_type _arrayType,
               rocfft_location_t _destLocation,
               BufferPtr         _destPtr)
        : precision(_precision)
        , arrayType(_arrayType)
        , destLocation(_destLocation)
        , destPtr(_destPtr)
    {
    }

    // one or more ranks to get data from
    struct GatherOp
    {
        GatherOp(rocfft_location_t srcLocation,
                 BufferPtr         srcPtr,
                 size_t            srcOffset,
                 size_t            destOffset,
                 size_t            numElems)
            : srcLocation(srcLocation)
            , srcPtr(srcPtr)
            , srcOffset(srcOffset)
            , destOffset(destOffset)
            , numElems(numElems)
        {
        }

        rocfft_location_t srcLocation;
        BufferPtr         srcPtr;

        size_t srcOffset;
        size_t destOffset;

        // Number of elements to copy
        size_t numElems;
    };

    // add an operation to gather from a source
    void AddOperation(int local_comm_rank, GatherOp&& op)
    {
        // allocate space to allow each gather operation to have a
        // stream/event, though we only allocate streams/events for
        // rank-local communication
        streams.resize(streams.size() + 1);
        events.resize(events.size() + 1);

        // look at the actual operation and allocate stream/event for
        // rank-local communication, so we can get completion information
        if(destLocation.comm_rank == op.srcLocation.comm_rank
           && op.srcLocation.comm_rank == local_comm_rank)
        {
            rocfft_scoped_device dev(op.srcLocation.device);
            streams.back().alloc();
            events.back().alloc();
        }
        ops.emplace_back(std::move(op));
    }

    void ExecuteAsync(const rocfft_plan     plan,
                      void*                 in_buffer[],
                      void*                 out_buffer[],
                      rocfft_execution_info info,
                      size_t                multiPlanIdx) override;
    void Wait() override;

    void Print(rocfft_ostream& os, const int indent) const override;

    bool WritesToBuffer(const BufferPtr& ptr) const override
    {
        return ptr == destPtr;
    }

    bool ExecutesOnRank(int comm_rank) const override
    {
        return destLocation.comm_rank == comm_rank
               || std::any_of(ops.begin(), ops.end(), [comm_rank](const GatherOp& op) {
                      return op.srcLocation.comm_rank == comm_rank;
                  });
    }

private:
    const rocfft_precision  precision;
    const rocfft_array_type arrayType;

    const rocfft_location_t destLocation;
    const BufferPtr         destPtr;

    std::vector<GatherOp> ops;

    // Streams to run the async operations in - since each memcpy is
    // coming from a different source device, each needs a separate
    // stream
    std::vector<hipStream_wrapper_t> streams;
    // Events to signal when the async operations are finished.
    std::vector<hipEvent_wrapper_t> events;
};

// Send data from all ranks to all ranks in the plan.  Each rank must
// send from/to a single buffer (with different read/write offsets
// for each other rank).
// The all-to-all communication is performed using MPI_Ialltoall or MPI_Ialltoallv.
// The former is preferable, as it is usually more optimized.
struct CommAllToAll : public MultiPlanItem
{

    CommAllToAll(rocfft_precision                    _precision,
                 rocfft_array_type                   _arrayType,
                 const std::vector<size_t>&          _sendOffsets,
                 const std::vector<size_t>&          _sendCounts,
                 const std::vector<size_t>&          _recvOffsets,
                 const std::vector<size_t>&          _recvCounts,
                 BufferPtr                           _sendBuf,
                 BufferPtr                           _recvBuf,
                 std::optional<MPI_Comm_wrapper_t>&& _subcomm)
        : precision(_precision)
        , arrayType(_arrayType)
        , sendOffsets(_sendOffsets)
        , sendCounts(_sendCounts)
        , recvOffsets(_recvOffsets)
        , recvCounts(_recvCounts)
        , sendBuf(_sendBuf)
        , recvBuf(_recvBuf)
        // check if uniform exchange to use MPI_Alltoall
        , uniform_counts(std::all_of(sendCounts.begin(),
                                     sendCounts.end(),
                                     [&](size_t c) { return c == sendCounts[0]; })
                         && std::all_of(recvCounts.begin(), recvCounts.end(), [&](size_t c) {
                                return c == recvCounts[0];
                            }))
    {
        subcomm = std::move(_subcomm);

        // Currently MPI interface uses 32-bit signed ints, so assert
        // that our counts/offsets don't overflow that type
        auto checkArray = [](const std::vector<size_t>& arr) {
            if(std::any_of(arr.begin(), arr.end(), [](size_t elem) {
                   return elem > std::numeric_limits<int>::max();
               }))
                throw std::runtime_error("count/offset exceeds int max");
        };

        checkArray(sendOffsets);
        checkArray(sendCounts);
        checkArray(recvOffsets);
        checkArray(recvCounts);
    }

    // enum for error handling for different multi-process communication
    // libraries such as MPI, RCCL, rocSHMEM, etc.
    enum CommStatus
    {
        COMM_SUCCESS,
        COMM_MPI_ERROR
    };
    CommStatus  comm_status = COMM_SUCCESS;
    std::string error_message;

    void ExecuteAsync(const rocfft_plan     plan,
                      void*                 in_buffer[],
                      void*                 out_buffer[],
                      rocfft_execution_info info,
                      size_t                multiPlanIdx) override;

    void Wait() override;

    void Print(rocfft_ostream& os, const int indent) const override;

    bool WritesToBuffer(const BufferPtr& ptr) const override
    {
        // only writes to receive buffer
        return ptr == recvBuf;
    }

    bool ExecutesOnRank(int comm_rank) const override
    {
        // runs on all ranks
        return true;
    }

private:
    const rocfft_precision  precision;
    const rocfft_array_type arrayType;

    // counts and offsets are all in elements (where element size is
    // knowable from precision + array type), for the current rank
    const std::vector<size_t> sendOffsets;
    const std::vector<size_t> sendCounts;
    const std::vector<size_t> recvOffsets;
    const std::vector<size_t> recvCounts;

    // send/receive buffers
    const BufferPtr sendBuf;
    const BufferPtr recvBuf;

    // check uniform counts for using AlltoAll instead of AlltoAllv
    const bool uniform_counts = false;
};

// Tree-structured FFT plan.  This is specific to a single device on
// a single rank, since the TreeNodes inside here will have device
// memory allocated for things like kernel arguments and twiddles.
struct ExecPlan : public MultiPlanItem
{
    ExecPlan(int local_comm_rank, bool _mgpuPlan, rocfft_location_t _location)
        : location(_location)
        , mgpuPlan(_mgpuPlan)
    {
        // in a multi-GPU plan, this plan is just one piece of the overall
        // distributed FFT.  need to allocate stream/event for completion
        // signalling
        if(mgpuPlan && location.comm_rank == local_comm_rank)
        {
            // if the user provides a stream for us at execute time, we
            // might not use this stream.  but allocate it during plan
            // creation because it can be slow to do it during execution
            rocfft_scoped_device dev(location.device);
            stream.alloc();
            event.alloc();
        }
    }
    // device where this work will be executed
    rocfft_location_t location;

    // In a multi-device plan, this flag is set to true to allow
    // the recording and synchronization of events, as well as
    // the creation of required temporary buffers
    bool mgpuPlan = false;

    // Normally, input/output are provided by users.  In a multi-device
    // plan, we might use temp buffers for input/output.  If so, these
    // are pointers to those temp buffers.
    BufferPtr inputPtr;
    BufferPtr outputPtr;

    void ExecuteAsync(const rocfft_plan     plan,
                      void*                 in_buffer[],
                      void*                 out_buffer[],
                      rocfft_execution_info info,
                      size_t                multiPlanIdx) override;

    void Wait() override;

    void Print(rocfft_ostream& os, const int indent) const override;

    // shared pointer allows for ExecPlans to be copyable
    std::shared_ptr<TreeNode> rootPlan;

    // non-owning pointers to the leaf-node children of rootPlan, which
    // are the nodes that do actual work
    std::vector<TreeNode*> execSeq;

    // kernels that extracted from solution map
    std::vector<FMKey> solution_kernels;

    // Keep the references of kernel-solution-nodes during tuning-process.
    // We have to assign back some values after  buffer-assignment /
    // colapse-batch-dim , which are not called when enumerating
    // kernel-configurations.
    std::vector<KernelConfig*> sol_kernel_configs;

    // scheme decompositions from solution map
    std::unique_ptr<SchemeTree> rootScheme;

    // flattened potentially-fusable shims of rootPlan
    std::vector<FuseShim*> fuseShims;

    std::vector<GridParam> gridParam;

    hipDeviceProp_t deviceProp;

    std::vector<size_t> iLength;
    std::vector<size_t> oLength;

    // Indicates whether this is a standalone chirp plan
    // in multi-kernel Bluestein implementations (buffers
    // in the standalone plan are not connected with the
    // rest of the nodes in the fft plan).
    bool IsChirpPlan;

    // default: starting from ABT, balance buffers and fusions
    // we could allow users to set in the later PR
    rocfft_optimize_strategy assignOptStrategy = rocfft_optimize_balance;

    // these sizes count in complex elements
    size_t workBufSize      = 0;
    size_t tmpWorkBufSize   = 0;
    size_t copyWorkBufSize  = 0;
    size_t blueWorkBufSize  = 0;
    size_t chirpWorkBufSize = 0;

    // OB_IN refers to iStride, OB_OUT refers to oStride
    std::map<OperatingBuffer, bool> isUnitStride;

    size_t WorkBufBytes(size_t base_type_size) const override
    {
        // base type is the size of one real, work buf counts in
        // complex numbers
        return workBufSize * 2 * base_type_size;
    }

    // for callbacks, work out which nodes of the plan are loading data
    // from global memory, and storing data to global memory
    std::pair<TreeNode*, TreeNode*> get_load_store_nodes() const;

    bool WritesToBuffer(const BufferPtr& ptr) const override
    {
        return ptr == outputPtr;
    }

    bool ExecutesOnRank(int comm_rank) const override
    {
        return location.comm_rank == comm_rank;
    }

private:
    // Stream to run the async operations in - might be unallocated
    // if the user gave us a stream to use
    hipStream_wrapper_t stream;
    // Event to signal when the async operations are finished.
    hipEvent_wrapper_t event;
};

std::unique_ptr<SchemeTree> ApplySolution(ExecPlan& execPlan);

// get a min_token (without batch, stride, offset...) of a node, for generating a prob-key
void GetNodeToken(const TreeNode& probNode, std::string& min_token, std::string& full_token);
void ProcessNode(ExecPlan& execPlan);
void PrintNode(rocfft_ostream& os, const ExecPlan& execPlan, const int indent = 0);
bool BufferIsUnitStride(ExecPlan& execPlan, OperatingBuffer buf);

#endif // TREE_NODE_H
