// Copyright (C) 2020 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "tree_node.h"
#include "../../shared/precision_type.h"
#include "function_pool.h"
#include "kernel_launch.h"
#include "logging.h"
#include "plan.h"
#include "repo.h"
#include "rocfft_mpi.h"
#include "twiddles.h"

#include <limits>
#include <sstream>

struct rocfft_mp_request_t
{
#ifdef ROCFFT_MPI_ENABLE
    rocfft_mp_request_t(const MPI_Request& req)
        : mpi_request(req)
    {
    }
    MPI_Request mpi_request;
#endif
};

TreeNode::~TreeNode()
{
    if(twiddles)
    {
        if(scheme == CS_KERNEL_2D_SINGLE)
            Repo::ReleaseTwiddle2D(twiddles);
        else
            Repo::ReleaseTwiddle1D(twiddles);
        twiddles = nullptr;
    }
    if(twiddles_large)
    {
        Repo::ReleaseTwiddle1D(twiddles_large);
        twiddles_large = nullptr;
    }
    if(chirp)
    {
        Repo::ReleaseChirp(chirp);
        chirp = nullptr;
    }
}

NodeMetaData::NodeMetaData(TreeNode* refNode)
{
    if(refNode != nullptr)
    {
        precision  = refNode->precision;
        batch      = refNode->batch;
        direction  = refNode->direction;
        rootIsC2C  = refNode->IsRootPlanC2CTransform();
        deviceProp = refNode->deviceProp;
    }
}

bool LeafNode::CreateLargeTwdTable()
{
    if(large1D != 0)
    {
        std::tie(twiddles_large, twiddles_large_size)
            = Repo::GetTwiddles1D(large1D, 0, precision, deviceProp, largeTwdBase, false, {});
    }

    return true;
}

size_t LeafNode::GetTwiddleTableLength()
{
    // length used by twiddle table is length[0] by default
    // could be override by some special schemes
    return length[0];
}

FMKey LeafNode::GetKernelKey() const
{
    if(!externalKernel)
        return FMKey::EmptyFMKey();

    return TreeNode::GetKernelKey();
}

void LeafNode::GetKernelFactors()
{
    auto kernel   = GetKernel();
    kernelFactors = kernel.factors;
}

void LeafNode::GetKernelPartialPassFactors()
{
    auto kernel     = GetKernel();
    kernelFactorsPP = std::vector<size_t>(kernel.pp_params.pp_factors_curr.begin(),
                                          kernel.pp_params.pp_factors_curr.end());

    switch(ppOffDim)
    {
    case 0: // work along x will be split between y and z
    {
        throw std::runtime_error(
            "GetKernelPartialPassFactors: partial-passes along x not currently supported");
        break;
    }
    case 1: // work along y will be split between x and z
    {
        if(scheme == CS_KERNEL_STOCKHAM_PP)
        {
            std::stringstream msg;
            msg << "work in the off-dimension:" << std::endl;
            msg << "\t     radix: [";
            for(const auto factor : kernelFactorsPP)
                msg << " " << factor;
            msg << " ] pass(es) + Hadamard product with twiddle factors. \n";
            comments.push_back(msg.str());
        }
        if(scheme == CS_KERNEL_STOCKHAM_PP_BLOCK_CC)
        {
            std::stringstream msg;
            msg << "work in the off-dimension:" << std::endl;
            msg << "\t     local data transposition + radix: [";
            for(const auto factor : kernelFactorsPP)
                msg << " " << factor;
            msg << " ] pass(es). \n";
            comments.push_back(msg.str());
        }

        break;
    }
    case 2: // work along z will be split between x and y
    {
        // x row fft + partial pass along z
        // partial pass along z + y col fft
        throw std::runtime_error(
            "GetKernelPartialPassFactors: partial-passes along z not currently supported");
        break;
    }
    default:
        throw std::runtime_error("Invalid off-dimension for partial pass");
    }
}

bool LeafNode::KernelCheck(std::vector<FMKey>& kernel_keys)
{
    if(!externalKernel)
    {
        // such as solutions kernels for 2D_RTRT or 1D_CRT, the "T" kernel is not an external one
        // so in the solution map we will keep it as a empty key. By storing and checking the emptykey,
        // we can increase the reilability of solution map.
        if(!kernel_keys.empty())
        {
            if(LOG_TRACE_ENABLED())
                (*LogSingleton::GetInstance().GetTraceOS())
                    << "solution kernel is an built-in kernel" << std::endl;

            // kernel_key from solution map should be an EmptyFMKey for a built-in kernel
            if(kernel_keys.front() != FMKey::EmptyFMKey())
                return false;
            kernel_keys.erase(kernel_keys.begin());
        }
        return true;
    }

    specified_key = nullptr;
    if(!kernel_keys.empty())
    {
        FMKey assignedKey = kernel_keys.front();
        kernel_keys.erase(kernel_keys.begin());

        // check if the assigned key is consistent with the node information
        if((length[0] != assignedKey.lengths[0])
           || (dimension == 2 && length[1] != assignedKey.lengths[1])
           || (precision != assignedKey.precision) || (scheme != assignedKey.scheme)
           || (ebtype != assignedKey.kernel_config.ebType))
        {
            if(LOG_TRACE_ENABLED())
                (*LogSingleton::GetInstance().GetTraceOS())
                    << "solution kernel keys are invalid: key properties != node's properties"
                    << std::endl;
            return false;
        }
        else
        {
            // get sbrc_trans_type from assignedKey (for sbrc)
            sbrcTranstype = assignedKey.sbrcTrans;

            pool.add_new_kernel(assignedKey);
            specified_key = std::make_unique<FMKey>(assignedKey);
        }
    }

    // get the final key and check if we have the kernel.
    // Note that the check is trivial if we are using "specified_key"
    // since we definitly have the kernel, but not trivial if it's the auto-gen key
    if(!HasKernel())
        return false;

    GetKernelFactors();

    if(isPartialPassEnabled())
        GetKernelPartialPassFactors();

    auto kernel = GetKernel();
    dir2regMode = (kernel.direct_to_from_reg) ? DirectRegType::TRY_ENABLE_IF_SUPPORT
                                              : DirectRegType::FORCE_OFF_OR_NOT_SUPPORT;

    return true;
}

void LeafNode::SanityCheck(SchemeTree* solution_scheme, std::vector<FMKey>& kernels_keys)
{
    if(!KernelCheck(kernels_keys))
        throw std::runtime_error("Kernel not found or mismatches node (solution map issue)");

    TreeNode::SanityCheck(solution_scheme, kernels_keys);
}

void LeafNode::Print(rocfft_ostream& os, int indent) const
{
    TreeNode::Print(os, indent);

    std::string indentStr;
    while(indent--)
        indentStr += "    ";

    os << indentStr.c_str() << "Leaf-Node: external-kernel configuration: ";
    indentStr += "    ";
    os << "\n" << indentStr.c_str() << "workgroup_size: " << wgs;
    os << "\n" << indentStr.c_str() << "trans_per_block: " << bwd;
    os << "\n" << indentStr.c_str() << "radices: [ ";
    for(size_t i = 0; i < kernelFactors.size(); i++)
    {
        os << kernelFactors[i] << " ";
    }
    os << "]\n";
}

bool LeafNode::CreateDevKernelArgs()
{
    devKernArg = kargs_create(length, inStride, outStride, iDist, oDist);
    return (devKernArg != nullptr);
}

bool LeafNode::CreateDeviceResources()
{
    if(need_chirp)
    {
        std::tie(chirp, chirp_size) = Repo::GetChirp(lengthBlueN, precision, deviceProp);
    }

    if(need_twd_table)
    {
        if(!twd_no_radices)
            GetKernelFactors();
        size_t twd_len                    = GetTwiddleTableLength();
        std::tie(twiddles, twiddles_size) = Repo::GetTwiddles1D(twd_len,
                                                                GetTwiddleTableLengthLimit(),
                                                                precision,
                                                                deviceProp,
                                                                0,
                                                                twd_attach_halfN,
                                                                kernelFactors);
    }

    return CreateLargeTwdTable();
}

void LeafNode::SetupGridParam(GridParam& gp)
{
    // derived classes setup the gp (bwd, wgs, lds, padding), funPtr
    SetupGridParam_internal(gp);

    auto key = GetKernelKey();

    // common: sum up the value;
    gp.lds_bytes = lds * complex_type_size(precision);
    if(scheme == CS_KERNEL_STOCKHAM && ebtype == EmbeddedType::NONE)
    {
        if(pool.has_function(key))
        {
            // NB:
            // Special case on specific arch:
            // For some cases using hald_lds, finer tuning(enlarge) dynamic
            // lds allocation size affects occupancy without changing the
            // kernel code. It is a middle solution between perf and code
            // consistency. Eventually, we need better solution arch
            // specific.
            bool double_half_lds_alloc = false;
            if(is_device_gcn_arch(deviceProp, "gfx90a") && (length[0] == 343 || length[0] == 49))
            {
                double_half_lds_alloc = true;
            }

            auto kernel = pool.get_kernel(key);
            if(kernel.half_lds && (!double_half_lds_alloc))
                gp.lds_bytes /= 2;
        }
    }
    if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
    {
        // SBCC support half-lds conditionally
        if((dir2regMode == DirectRegType::TRY_ENABLE_IF_SUPPORT) && (ebtype == EmbeddedType::NONE)
           && pool.has_function(key))
        {
            auto kernel = pool.get_kernel(key);
            if(kernel.half_lds)
                gp.lds_bytes /= 2;
        }
    }

    if(scheme == CS_KERNEL_STOCKHAM_BLOCK_CC || scheme == CS_KERNEL_STOCKHAM_PP_BLOCK_CC)
    {
        auto apply_large_twd = (largeTwdBase > 0 && ltwdSteps > 0);
        if(apply_large_twd && largeTwdBase < 8)
        {
            // append twiddle table to dynamic lds
            gp.lds_bytes += twiddles_large_size;
        }
    }
    // NB:
    //   SBCR / SBRC are not able to use half-lds due to both of them can't satisfy dir-to/from-registers at them same time.

    // Confirm that the requested LDS bytes will fit into what the
    // device can provide.  If it can't, we've made a mistake in our
    // computation somewhere.
    if(gp.lds_bytes > deviceProp.sharedMemPerBlock)
        throw std::runtime_error(std::to_string(gp.lds_bytes)
                                 + " bytes of LDS requested, but device only provides "
                                 + std::to_string(deviceProp.sharedMemPerBlock));
}

/*****************************************************
 * CS_KERNEL_TRANSPOSE
 * CS_KERNEL_TRANSPOSE_XY_Z
 * CS_KERNEL_TRANSPOSE_Z_XY
 *****************************************************/

// grid params are set up by RTC
void TransposeNode::SetupGridParam_internal(GridParam& gp) {}

void TreeNode::SetTransposeOutputLength()
{
    switch(scheme)
    {
    case CS_KERNEL_TRANSPOSE:
    {
        outputLength = length;
        std::swap(outputLength[0], outputLength[1]);
        break;
    }
    case CS_KERNEL_TRANSPOSE_XY_Z:
    case CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z:
    {
        outputLength = length;
        std::swap(outputLength[1], outputLength[2]);
        std::swap(outputLength[0], outputLength[1]);
        break;
    }
    case CS_KERNEL_TRANSPOSE_Z_XY:
    case CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY:
    {
        outputLength = length;
        std::swap(outputLength[0], outputLength[1]);
        std::swap(outputLength[1], outputLength[2]);
        break;
    }
    default:
        throw std::runtime_error("can't set transpose output length on non-transpose node");
    }
}

void TreeNode::CollapseContiguousDims()
{
    // collapse children
    for(auto& child : childNodes)
        child->CollapseContiguousDims();

    const auto collapsibleDims = CollapsibleDims();
    if(collapsibleDims.empty())
        return;

    // utility function to collect the dims to collapse
    auto collectCollapse = [&collapsibleDims](const size_t               dist,
                                              size_t&                    newBatch,
                                              const std::vector<size_t>& length,
                                              const std::vector<size_t>& stride) {
        std::vector<size_t> dimsToCollapse;
        // start with batch dim and go backwards through collapsible dims
        // so we can collapse them without invalidating remaining indexes
        auto curStride = dist;
        for(auto i = collapsibleDims.rbegin(); i != collapsibleDims.rend(); ++i)
        {
            if(stride[*i] == 0)
                break;
            if(curStride % stride[*i] != 0)
                break;
            if(curStride / stride[*i] != length[*i])
                break;
            dimsToCollapse.push_back(*i);
            newBatch *= length[*i];
            curStride = stride[*i];
        }
        return dimsToCollapse;
    };

    // utility function to actually do the collapsing -
    // dimsToCollapse must be in reverse order so we erase dims from
    // highest to lowest
    auto doCollapse = [](size_t&                    dist,
                         const std::vector<size_t>& dimsToCollapse,
                         std::vector<size_t>&       lengthToCollapse,
                         std::vector<size_t>&       strideToCollapse) {
        for(auto i : dimsToCollapse)
        {
            dist /= lengthToCollapse[i];
            lengthToCollapse.erase(lengthToCollapse.begin() + i);
            strideToCollapse.erase(strideToCollapse.begin() + i);
        }
    };

    size_t              newInputBatch = batch;
    std::vector<size_t> inputDimsToCollapse
        = collectCollapse(iDist, newInputBatch, length, inStride);
    auto                outputLengthTemp = GetOutputLength();
    size_t              newOutputBatch   = batch;
    std::vector<size_t> outputDimsToCollapse
        = collectCollapse(oDist, newOutputBatch, outputLengthTemp, outStride);
    if(inputDimsToCollapse != outputDimsToCollapse || newInputBatch != newOutputBatch)
        return;

    if(!inputDimsToCollapse.empty())
    {
        std::stringstream msg;
        msg << "collapsed contiguous high length(s)";
        for(auto i = inputDimsToCollapse.rbegin(); i != inputDimsToCollapse.rend(); ++i)
            msg << " " << length[*i];
        msg << " into batch";
        comments.push_back(msg.str());
    }

    doCollapse(iDist, inputDimsToCollapse, length, inStride);
    doCollapse(oDist, outputDimsToCollapse, outputLengthTemp, outStride);
    batch = newInputBatch;

    if(!outputLength.empty())
        outputLength = outputLengthTemp;
}

bool TreeNode::IsBluesteinChirpSetup()
{
    // setup nodes must be under a bluestein parent. multi-kernel fused
    // bluestein is an exception to this rule as the first two chirp + padding
    // nodes are under an L1D_CC node.
    if(typeBlue != BT_MULTI_KERNEL_FUSED && (parent == nullptr || parent->scheme != CS_BLUESTEIN))
        return false;
    // bluestein could either be 3-kernel plan (so-called single kernel Bluestein),
    // meaning the first two are setup kernels, or multi-kernel bluestein (fused or non-fused)
    // where only the first is setup
    switch(parent->typeBlue)
    {
    case BluesteinType::BT_NONE:
        return false;
    case BluesteinType::BT_SINGLE_KERNEL:
        return this == parent->childNodes[0].get() || this == parent->childNodes[1].get();
    case BluesteinType::BT_MULTI_KERNEL:
        return this == parent->childNodes[0].get();
    case BluesteinType::BT_MULTI_KERNEL_FUSED:
        return (fuseBlue == BFT_FWD_CHIRP) ? true : false;
    }

    throw std::runtime_error("unexpected bluestein plan shape");
}

MultiPlanItem::MultiPlanItem() {}

MultiPlanItem::~MultiPlanItem() {}

std::string MultiPlanItem::PrintBufferPtrOffset(const BufferPtr& ptr, size_t offset)
{
    std::stringstream ss;
    ss << ptr.str() << " offset " << offset << " elems";
    return ss.str();
}

int MultiPlanItem::GetOperationCommTag(size_t multiPlanIdx, size_t opIdx)
{
    // use top half of int for multiPlan index, bottom half for
    // operation index
    int tag = multiPlanIdx;
    tag <<= 16;
    tag |= static_cast<uint16_t>(opIdx);
    return tag;
}

void MultiPlanItem::WaitCommRequests()
{
#ifdef ROCFFT_MPI_ENABLE
    if(comm_requests.empty())
        return;

    std::vector<MPI_Request> mpi_requests;
    mpi_requests.reserve(comm_requests.size());
    for(auto& comm_req : comm_requests)
        mpi_requests.push_back(comm_req.mpi_request);

    std::vector<MPI_Status> mpi_status(mpi_requests.size());
    auto rcmpi = MPI_Waitall(mpi_requests.size(), mpi_requests.data(), mpi_status.data());
    if(rcmpi != MPI_SUCCESS)
        throw std::runtime_error("MPI_Waitall failed: " + std::to_string(rcmpi));
    comm_requests.clear();
#endif
}

#ifdef ROCFFT_MPI_ENABLE
MPI_Comm MultiPlanItem::ActiveMPIComm(rocfft_plan plan) const
{
    if(subcomm)
        return *subcomm;
    else
        return plan->mpi_comm;
}
#endif

void CommPointToPoint::ExecuteAsync(const rocfft_plan     plan,
                                    void*                 in_buffer[],
                                    void*                 out_buffer[],
                                    rocfft_execution_info info,
                                    size_t                multiPlanIdx)
{
    rocfft_scoped_device dev(srcLocation.device);

    if(LOG_PLAN_ENABLED())
    {
        log_plan("CommPointToPoint\n");
    }

    auto srcWithOffset = ptr_offset(
        srcPtr.get(in_buffer, out_buffer, local_comm_rank), srcOffset, precision, arrayType);
    auto destWithOffset = ptr_offset(
        destPtr.get(in_buffer, out_buffer, local_comm_rank), destOffset, precision, arrayType);

    if(srcLocation.comm_rank == destLocation.comm_rank)
    {
        const auto memSize = numElems * element_size(precision, arrayType);
        auto       hiprt   = hipSuccess;
        if(srcLocation.device == destLocation.device)
        {
            hiprt = hipMemcpyAsync(
                destWithOffset, srcWithOffset, memSize, hipMemcpyDeviceToDevice, stream);
        }
        else
        {
            hiprt = hipMemcpyPeerAsync(destWithOffset,
                                       destLocation.device,
                                       srcWithOffset,
                                       srcLocation.device,
                                       memSize,
                                       stream);
        }

        if(hiprt != hipSuccess)
            throw std::runtime_error("hipMemcpy failed");

        // all work is enqueued to the stream, record the event on
        // the stream
        if(hipEventRecord(event, stream) != hipSuccess)
            throw std::runtime_error("hipEventRecord failed");
    }
    else
    {
#if !defined ROCFFT_MPI_ENABLE
        throw std::runtime_error("MPI communication not enabled");
#else
        if(srcLocation.comm_rank == local_comm_rank)
        {
            MPI_Request request;
            const auto  mpiret = MPI_Isend(srcWithOffset,
                                          numElems,
                                          rocfft_type_to_mpi_type(precision, arrayType),
                                          destLocation.comm_rank,
                                          multiPlanIdx,
                                          plan->mpi_comm,
                                          &request);
            if(mpiret != MPI_SUCCESS)
            {
                throw std::runtime_error("MPI_Isend PointToPoint failed on rank "
                                         + std::to_string(local_comm_rank));
            }
            comm_requests.push_back(request);
        }
        else if(destLocation.comm_rank == local_comm_rank)
        {
            MPI_Request request;
            const auto  mpiret = MPI_Irecv(destWithOffset,
                                          numElems,
                                          rocfft_type_to_mpi_type(precision, arrayType),
                                          srcLocation.comm_rank,
                                          multiPlanIdx,
                                          plan->mpi_comm,
                                          &request);
            if(mpiret != MPI_SUCCESS)
            {
                throw std::runtime_error("MPI_Irecv PointToPoint failed on rank "
                                         + std::to_string(local_comm_rank));
            }
            comm_requests.push_back(request);
        }
#endif
    }
}

void CommPointToPoint::Wait()
{
    WaitCommRequests();

    if(event && hipEventSynchronize(event) != hipSuccess)
        throw std::runtime_error("hipEventSynchronize failed");
}

void CommPointToPoint::Print(rocfft_ostream& os, const int indent) const
{
    const std::string indentStr("    ", indent);

    os << indentStr << "CommPointToPoint " << precision_name(precision) << " "
       << PrintArrayType(arrayType) << ":"
       << "\n";
    os << indentStr << "  srcCommRank: " << srcLocation.comm_rank << "\n";
    os << indentStr << "  srcDeviceID: " << srcLocation.device << "\n";
    os << indentStr << "  srcBuf: " << PrintBufferPtrOffset(srcPtr, srcOffset) << "\n";
    os << indentStr << "  destCommRank: " << destLocation.comm_rank << "\n";
    os << indentStr << "  destDeviceID: " << destLocation.device << "\n";
    os << indentStr << "  destBuf: " << PrintBufferPtrOffset(destPtr, destOffset) << "\n";
    os << indentStr << "  numElems: " << numElems << "\n";
    os << std::endl;
}

void CommScatter::ExecuteAsync(const rocfft_plan     plan,
                               void*                 in_buffer[],
                               void*                 out_buffer[],
                               rocfft_execution_info info,
                               size_t                multiPlanIdx)
{
    rocfft_scoped_device dev(srcLocation.device);

    if(LOG_PLAN_ENABLED())
    {
        log_plan("CommScatter\n");
    }

    for(unsigned int opIdx = 0; opIdx < ops.size(); ++opIdx)
    {
        const auto& op = ops[opIdx];

        auto srcWithOffset = ptr_offset(
            srcPtr.get(in_buffer, out_buffer, local_comm_rank), op.srcOffset, precision, arrayType);
        auto destWithOffset = ptr_offset(op.destPtr.get(in_buffer, out_buffer, local_comm_rank),
                                         op.destOffset,
                                         precision,
                                         arrayType);

        hipError_t err = hipSuccess;
        if(op.destLocation.comm_rank == srcLocation.comm_rank)
        {
            const auto memSize = op.numElems * element_size(precision, arrayType);
            if(local_comm_rank == op.destLocation.comm_rank)
            {
                if(srcLocation.device == op.destLocation.device)
                    err = hipMemcpyAsync(
                        destWithOffset, srcWithOffset, memSize, hipMemcpyDeviceToDevice, stream);
                else
                    err = hipMemcpyPeerAsync(destWithOffset,
                                             op.destLocation.device,
                                             srcWithOffset,
                                             srcLocation.device,
                                             memSize,
                                             stream);

                if(err != hipSuccess)
                    throw std::runtime_error("hipMemcpy failed");
            }
        }
        else
        {
            // Inter-proccess communication
#if !defined ROCFFT_MPI_ENABLE
            throw std::runtime_error("MPI communication not enabled");
#else
            if(local_comm_rank == srcLocation.comm_rank)
            {
                MPI_Request request;
                const auto  mpiret = MPI_Isend(srcWithOffset,
                                              op.numElems,
                                              rocfft_type_to_mpi_type(precision, arrayType),
                                              op.destLocation.comm_rank,
                                              GetOperationCommTag(multiPlanIdx, opIdx),
                                              plan->mpi_comm,
                                              &request);
                if(mpiret != MPI_SUCCESS)
                {
                    throw std::runtime_error("MPI_Isend failed on rank"
                                             + std::to_string(local_comm_rank));
                }
                comm_requests.push_back(request);
            }
            else if(local_comm_rank == op.destLocation.comm_rank)
            {
                MPI_Request request;
                const auto  mpiret = MPI_Irecv(destWithOffset,
                                              op.numElems,
                                              rocfft_type_to_mpi_type(precision, arrayType),
                                              srcLocation.comm_rank,
                                              GetOperationCommTag(multiPlanIdx, opIdx),
                                              plan->mpi_comm,
                                              &request);
                if(mpiret != MPI_SUCCESS)
                {
                    throw std::runtime_error("MPI_Irecv failed on rank"
                                             + std::to_string(local_comm_rank) + " for op index "
                                             + std::to_string(opIdx));
                }
                comm_requests.push_back(request);
            }

#endif
        }
    }
    // All rank-local work is enqueued to the stream, record the
    // event on the stream
    if(event && hipEventRecord(event, stream) != hipSuccess)
        throw std::runtime_error("hipEventRecord failed");
}

void CommScatter::Wait()
{
    WaitCommRequests();

    if(event && hipEventSynchronize(event) != hipSuccess)
        throw std::runtime_error("hipEventSynchronize failed");
}

void CommScatter::Print(rocfft_ostream& os, const int indent) const
{
    std::string indentStr;
    int         i = indent;
    while(i--)
        indentStr += "    ";

    os << indentStr << "CommScatter " << precision_name(precision) << " "
       << PrintArrayType(arrayType) << ":\n";
    os << indentStr << "  srcCommRank: " << srcLocation.comm_rank << "\n";
    os << indentStr << "  srcDeviceID: " << srcLocation.device << "\n";

    for(const auto& op : ops)
    {
        os << indentStr << "    destCommRank: " << op.destLocation.comm_rank << "\n";
        os << indentStr << "    destDeviceID: " << op.destLocation.device << "\n";
        os << indentStr << "    srcBuf: " << PrintBufferPtrOffset(srcPtr, op.srcOffset) << "\n";
        os << indentStr << "    destBuf: " << PrintBufferPtrOffset(op.destPtr, op.destOffset)
           << "\n";
        os << indentStr << "    numElems: " << op.numElems << "\n";
        os << "\n";
    }
}

void CommGather::ExecuteAsync(const rocfft_plan     plan,
                              void*                 in_buffer[],
                              void*                 out_buffer[],
                              rocfft_execution_info info,
                              size_t                multiPlanIdx)
{
    if(LOG_PLAN_ENABLED())
    {
        log_plan("CommGather\n");
    }

    for(unsigned int opIdx = 0; opIdx < ops.size(); ++opIdx)
    {
        const auto& op     = ops[opIdx];
        auto&       stream = streams[opIdx];
        auto&       event  = events[opIdx];

        rocfft_scoped_device dev(op.srcLocation.device);

        auto srcWithOffset  = ptr_offset(op.srcPtr.get(in_buffer, out_buffer, local_comm_rank),
                                        op.srcOffset,
                                        precision,
                                        arrayType);
        auto destWithOffset = ptr_offset(destPtr.get(in_buffer, out_buffer, local_comm_rank),
                                         op.destOffset,
                                         precision,
                                         arrayType);

        hipError_t err = hipSuccess;
        if(destLocation.comm_rank == op.srcLocation.comm_rank)
        {
            const auto memSize = op.numElems * element_size(precision, arrayType);

            if(local_comm_rank == destLocation.comm_rank)
            {
                if(op.srcLocation.device == destLocation.device)
                {
                    err = hipMemcpyAsync(
                        destWithOffset, srcWithOffset, memSize, hipMemcpyDeviceToDevice, stream);
                }
                else
                {
                    err = hipMemcpyPeerAsync(destWithOffset,
                                             destLocation.device,
                                             srcWithOffset,
                                             op.srcLocation.device,
                                             memSize,
                                             stream);
                }
                if(err != hipSuccess)
                    throw std::runtime_error("hipMemcpy failed");
            }
        }
        else
        {
            // Inter-proccess communication
#if !defined ROCFFT_MPI_ENABLE
            throw std::runtime_error("MPI communication not enabled");
#else

            if(local_comm_rank == op.srcLocation.comm_rank)
            {
                MPI_Request request;
                auto        rcmpi = MPI_Isend(srcWithOffset,
                                       op.numElems,
                                       rocfft_type_to_mpi_type(precision, arrayType),
                                       destLocation.comm_rank,
                                       GetOperationCommTag(multiPlanIdx, opIdx),
                                       plan->mpi_comm,
                                       &request);
                if(rcmpi != MPI_SUCCESS)
                    throw std::runtime_error("MPI_Isend failed: " + std::to_string(rcmpi));
                comm_requests.push_back(request);
            }
            else if(local_comm_rank == destLocation.comm_rank)
            {
                MPI_Request request;
                auto        rcmpi = MPI_Irecv(destWithOffset,
                                       op.numElems,
                                       rocfft_type_to_mpi_type(precision, arrayType),
                                       op.srcLocation.comm_rank,
                                       GetOperationCommTag(multiPlanIdx, opIdx),
                                       plan->mpi_comm,
                                       &request);
                if(rcmpi != MPI_SUCCESS)
                    throw std::runtime_error("MPI_Irecv failed: " + std::to_string(rcmpi));
                comm_requests.push_back(request);
            }
#endif
        }

        // All work for this stream is enqueued, record the event on
        // the stream.  The event is only allocated if necessary, so
        // check it before recording.
        if(event && hipEventRecord(event, stream) != hipSuccess)
            throw std::runtime_error("hipEventRecord failed");
    }
}

void CommGather::Wait()
{
    WaitCommRequests();

    for(const auto& event : events)
    {
        if(event && hipEventSynchronize(event) != hipSuccess)
            throw std::runtime_error("hipEventSynchronize failed");
    }
}

void CommGather::Print(rocfft_ostream& os, const int indent) const
{
    std::string indentStr;
    int         i = indent;
    while(i--)
        indentStr += "    ";

    os << indentStr << "CommGather " << precision_name(precision) << " "
       << PrintArrayType(arrayType) << ":"
       << "\n";
    os << indentStr << "  destCommRank: " << destLocation.comm_rank << "\n";
    os << indentStr << "  destDeviceID: " << destLocation.device << "\n";

    for(const auto& op : ops)
    {
        os << indentStr << "    srcCommRank: " << op.srcLocation.comm_rank << "\n";
        os << indentStr << "    srcDeviceID: " << op.srcLocation.device << "\n";
        os << indentStr << "    srcBuf: " << PrintBufferPtrOffset(op.srcPtr, op.srcOffset) << "\n";
        os << indentStr << "    destBuf: " << PrintBufferPtrOffset(destPtr, op.destOffset) << "\n";
        os << indentStr << "    numElems: " << op.numElems << "\n";
        os << "\n";
    }
}

void CommAllToAll::ExecuteAsync(const rocfft_plan     plan,
                                void*                 in_buffer[],
                                void*                 out_buffer[],
                                rocfft_execution_info info,
                                size_t                multiPlanIdx)
{
    if(LOG_PLAN_ENABLED())
    {
        log_plan("CommAllToAll: deciding between MPI_Ialltoall and MPI_Ialltoallv\n");
    }

#ifdef ROCFFT_MPI_ENABLE
    auto mpi_comm = ActiveMPIComm(plan);

    MPI_Request  request;
    MPI_Datatype elem_type       = rocfft_type_to_mpi_type(precision, arrayType);
    const int    local_comm_rank = plan->get_local_comm_rank();

    if(uniform_counts)
    {
        if(LOG_PLAN_ENABLED())
            log_plan("Using MPI_Ialltoall\n");

        size_t send_count_elems = sendCounts[0];
        if(send_count_elems > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            comm_status   = COMM_MPI_ERROR;
            error_message = "send count too large to fit in MPI int on rank "
                            + std::to_string(local_comm_rank);
            return;
        }

        const int send_count = static_cast<int>(send_count_elems);

        const auto mpiret = MPI_Ialltoall(sendBuf.get(in_buffer, out_buffer, local_comm_rank),
                                          send_count,
                                          elem_type,
                                          recvBuf.get(in_buffer, out_buffer, local_comm_rank),
                                          send_count,
                                          elem_type,
                                          mpi_comm,
                                          &request);

        if(mpiret != MPI_SUCCESS)
        {
            char errmsg[MPI_MAX_ERROR_STRING];
            int  errlen = 0;
            MPI_Error_string(mpiret, errmsg, &errlen);

            comm_status   = COMM_MPI_ERROR;
            error_message = "MPI_Ialltoall failed on rank " + std::to_string(local_comm_rank) + ": "
                            + std::string(errmsg);

            return;
        }
    }
    else
    {
        if(LOG_PLAN_ENABLED())
            log_plan("Using MPI_Ialltoallv\n");

        // non-uniform exchange case (default)

        // MPI takes ints for everything, safely convert our size_t elements to int bytes
        // prevent overflow for large batched transforms
        auto convertToInt = [&](const std::vector<size_t>& src, std::vector<int>& dest) -> bool {
            dest.clear();
            dest.reserve(src.size());
            for(size_t v : src)
            {
                if(v > static_cast<size_t>(std::numeric_limits<int>::max()))
                    return false; // overflow
                dest.push_back(static_cast<int>(v));
            }
            return true;
        };

        std::vector<int> intSendOffsets;
        std::vector<int> intSendCounts;
        std::vector<int> intRecvOffsets;
        std::vector<int> intRecvCounts;

        if(!convertToInt(sendOffsets, intSendOffsets) || !convertToInt(sendCounts, intSendCounts)
           || !convertToInt(recvOffsets, intRecvOffsets)
           || !convertToInt(recvCounts, intRecvCounts))
        {
            comm_status   = COMM_MPI_ERROR;
            error_message = "send/recv counts or offsets too large to fit in MPI int on rank "
                            + std::to_string(local_comm_rank);
            return;
        }

        const auto mpiret = MPI_Ialltoallv(sendBuf.get(in_buffer, out_buffer, local_comm_rank),
                                           intSendCounts.data(),
                                           intSendOffsets.data(),
                                           elem_type,
                                           recvBuf.get(in_buffer, out_buffer, local_comm_rank),
                                           intRecvCounts.data(),
                                           intRecvOffsets.data(),
                                           elem_type,
                                           mpi_comm,
                                           &request);

        if(mpiret != MPI_SUCCESS)
        {
            char errmsg[MPI_MAX_ERROR_STRING];
            int  errlen = 0;
            MPI_Error_string(mpiret, errmsg, &errlen);

            comm_status   = COMM_MPI_ERROR;
            error_message = "MPI_Ialltoallv failed on rank " + std::to_string(local_comm_rank)
                            + ": " + std::string(errmsg);

            return;
        }
    }

    comm_requests.push_back(request);

#else
    throw std::runtime_error("CommAllToAll not implemented");
#endif
}

void CommAllToAll::Wait()
{
    WaitCommRequests();

    if(comm_status == COMM_MPI_ERROR)
    {
        // Now it's safe to throw or return error
        throw std::runtime_error(error_message);
    }
}

void CommAllToAll::Print(rocfft_ostream& os, const int indent) const
{
    std::string indentStr;
    int         i = indent;
    while(i--)
        indentStr += "    ";

    auto printVec = [&os](const char* prefix, const std::vector<size_t>& vec) {
        os << prefix << ": ";
        for(auto val : vec)
            os << val << " ";
        os << "\n";
    };

    os << indentStr << "CommAllToAll " << precision_name(precision) << " "
       << PrintArrayType(arrayType) << (uniform_counts ? " (MPI_Ialltoall)" : " (MPI_Ialltoallv)")
       << ":\n";

    if(uniform_counts)
    {
        // alltoall: just print the count once
        os << indentStr << " count_per_rank: " << sendCounts[0] << "\n";
    }
    else
    {
        // alltoallv: print full arrays
        printVec("sendOffsets", sendOffsets);
        printVec("sendCounts", sendCounts);
        printVec("recvOffsets", recvOffsets);
        printVec("recvCounts", recvCounts);
    }
}

void ExecPlan::Print(rocfft_ostream& os, const int indent) const
{
    std::string indentStr;
    int         i = indent;
    while(i--)
        indentStr += "    ";
    os << indentStr << "MPI rank: " << local_comm_rank << "\n";
    os << indentStr << "ExecPlan:" << std::endl;
    os << indentStr << "  deviceID: " << location.device << std::endl;
    os << indentStr << "  local_comm_rank:" << local_comm_rank << "\n";
    os << indentStr << "  commRanks:" << location.comm_rank << std::endl;
    if(inputPtr)
        os << indentStr << "  inputPtr: " << inputPtr.str() << std::endl;
    if(outputPtr)
        os << indentStr << "  outputPtr: " << outputPtr.str() << std::endl;

    PrintNode(os, *this, indent);
}