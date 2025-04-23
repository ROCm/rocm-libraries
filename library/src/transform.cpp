/******************************************************************************
* Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../shared/array_predicate.h"
#include "../../shared/precision_type.h"
#include "logging.h"
#include "plan.h"
#include "rocfft/rocfft.h"
#include "rocfft_exception.h"
#include "transform.h"

rocfft_status rocfft_execution_info_create(rocfft_execution_info* info)
try
{
    rocfft_execution_info einfo = new rocfft_execution_info_t;
    *info                       = einfo;
    log_trace(__func__, "info", *info);

    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_execution_info_destroy(rocfft_execution_info info)
try
{
    log_trace(__func__, "info", info);
    if(info != nullptr)
        delete info;

    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_execution_info_set_work_buffer(rocfft_execution_info info,
                                                    void*                 work_buffer,
                                                    const size_t          size_in_bytes)
try
{
    log_trace(__func__, "info", info, "work_buffer", work_buffer, "size_in_bytes", size_in_bytes);
    if(!work_buffer)
        return rocfft_status_invalid_work_buffer;
    info->workBufferSize = size_in_bytes;
    info->workBuffer     = work_buffer;

    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_execution_info_set_stream(rocfft_execution_info info, void* stream)
try
{
    log_trace(__func__, "info", info, "stream", stream);
    info->rocfft_stream = (hipStream_t)stream;
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_execution_info_set_load_callback(rocfft_execution_info info,
                                                      void**                cb_functions,
                                                      void**                cb_data,
                                                      size_t                shared_mem_bytes)
try
{
    // currently, we're not allocating LDS for callbacks, so fail
    // if any was requested
    if(shared_mem_bytes)
        return rocfft_status_invalid_arg_value;

    info->callbacks.load_cb_fn        = cb_functions ? cb_functions[0] : nullptr;
    info->callbacks.load_cb_data      = cb_data ? cb_data[0] : nullptr;
    info->callbacks.load_cb_lds_bytes = shared_mem_bytes;
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

rocfft_status rocfft_execution_info_set_store_callback(rocfft_execution_info info,
                                                       void**                cb_functions,
                                                       void**                cb_data,
                                                       size_t                shared_mem_bytes)
try
{
    // currently, we're not allocating LDS for callbacks, so fail
    // if any was requested
    if(shared_mem_bytes)
        return rocfft_status_invalid_arg_value;

    info->callbacks.store_cb_fn        = cb_functions ? cb_functions[0] : nullptr;
    info->callbacks.store_cb_data      = cb_data ? cb_data[0] : nullptr;
    info->callbacks.store_cb_lds_bytes = shared_mem_bytes;
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

std::vector<size_t> rocfft_plan_t::MultiPlanTopologicalSort() const
{
    std::vector<size_t> ret;
    std::vector<bool>   visited(multiPlan.size(), false);

    for(size_t idx = 0; idx < visited.size(); ++idx)
    {
        if(!visited[idx])
        {
            TopologicalSortDFS(idx, visited, ret);
        }
    }
    return ret;
}

void rocfft_plan_t::TopologicalSortDFS(size_t               idx,
                                       std::vector<bool>&   visited,
                                       std::vector<size_t>& sorted) const
{
    visited[idx] = true;
    for(auto adjacent : multiPlanAntecedents[idx])
    {
        if(!visited[adjacent])
        {
            TopologicalSortDFS(adjacent, visited, sorted);
        }
    }
    sorted.push_back(idx);
}

void rocfft_plan_t::LogFields(const char* description, const std::vector<rocfft_field_t>& fields)
{
    if(!LOG_PLAN_ENABLED())
        return;

    auto& os = *LogSingleton::GetInstance().GetPlanOS();

    for(size_t fieldIdx = 0; fieldIdx < fields.size(); ++fieldIdx)
    {
        const auto& f = fields[fieldIdx];

        os << description << " field " << fieldIdx << ":" << std::endl;
        for(size_t brickIdx = 0; brickIdx < f.bricks.size(); ++brickIdx)
        {
            const auto& b = f.bricks[brickIdx];
            os << "  brick " << brickIdx << ":" << std::endl;
            os << "    comm_rank: " << b.location.comm_rank << std::endl;
            os << "    device: " << b.location.device << std::endl;
            os << "    lower bound:";
            for(auto i : b.lower)
                os << " " << i;
            os << std::endl;
            os << "    upper bound:";
            for(auto i : b.upper)
                os << " " << i;
            os << std::endl;

            os << "    stride:";
            for(auto i : b.stride)
                os << " " << i;
            os << std::endl;

            auto len = b.length();
            os << "    length:";
            for(auto i : len)
                os << " " << i;
            os << std::endl;

            os << "    elements: " << b.count_elems() << std::endl;
        }
    }
}

void rocfft_plan_t::LogSortedPlan(const std::vector<size_t>& sortedIdx) const
{
    // If we have a single-node plan, just log that without any extra indenting
    if(multiPlan.size() == 1)
    {
        if(LOG_PLAN_ENABLED())
        {
            auto& os = *LogSingleton::GetInstance().GetPlanOS();
            multiPlan.front()->Print(os, 0);
        }
        return;
    }

    if(LOG_PLAN_ENABLED())
    {
        auto& os = *LogSingleton::GetInstance().GetPlanOS();
        os << "multiPlans: " << multiPlan.size() << "\n";
        for(size_t midx = 0; midx < multiPlan.size(); ++midx)
        {
            if(multiPlan[midx]->ExecutesOnLocalRank())
            {
                os << "multiPlan: " << midx << "\n";
                os << "\tantecedents:";
                const auto& antecedents = multiPlanAntecedents[midx];
                for(auto antecedentIdx : antecedents)
                {
                    os << " " << antecedentIdx;
                }
                os << "\n";
                multiPlan[midx]->Print(os, 1);
            }
        }
    }

    if(LOG_GRAPH_ENABLED())
    {
        // multi-device plan, log that with dependency graph
        auto& os = *LogSingleton::GetInstance().GetGraphOS();
        os << "digraph plan {\n";
        os << "ranksep=2;\n";

        // gather up all of the groups for the nodes so they can turn into clustered subgraphs
        std::multimap<std::string, size_t> groups;

        for(auto i = sortedIdx.begin(); i != sortedIdx.end(); ++i)
        {
            auto idx = *i;

            const auto& antecedents = multiPlanAntecedents[idx];
            for(auto antecedentIdx : antecedents)
            {
                os << antecedentIdx << " -> " << idx << ";\n";
            }

            rocfft_ostream item;
            if(!multiPlan[idx])
                item << "(null)";
            else
                multiPlan[idx]->Print(item, 1);
            item << std::endl;
            std::string itemStr = item.str();
            // escape \n and "
            std::string itemStrEscaped;
            for(auto c : itemStr)
            {
                if(c == '\"')
                {
                    itemStrEscaped.push_back('\\');
                    itemStrEscaped.push_back('\"');
                }
                else if(c == '\n')
                {
                    itemStrEscaped.push_back('\\');
                    itemStrEscaped.push_back('n');
                }
                else
                    itemStrEscaped.push_back(c);
            }
            os << idx << " [label=\"" << idx << "\\n"
               << multiPlan[idx]->description << "\" tooltip=\"" << itemStrEscaped << "\"];\n";

            groups.insert(std::make_pair(multiPlan[idx]->group, idx));
        }
        auto giter = groups.begin();
        while(giter != groups.end())
        {
            auto gend = groups.upper_bound(giter->first);
            os << "subgraph cluster_" << giter->first << " {\n";
            os << "\nlabel=\"" << giter->first << "\";\n";
            for(; giter != gend; ++giter)
            {
                os << giter->second << ";";
            }
            os << "}\n";
        }
        os << "}\n" << std::endl;
    }
}

void rocfft_plan_t::Execute(void* in_buffer[], void* out_buffer[], rocfft_execution_info info)
{
    // Vector of topologically sorted indexes to the items in multiPlan
    auto sortedIdx = MultiPlanTopologicalSort();

    const auto local_comm_rank = get_local_comm_rank();

    // Log input/output pointers
    if(LOG_PLAN_ENABLED())
    {
        auto& os         = *LogSingleton::GetInstance().GetPlanOS();
        auto  inPtrCount = desc.count_pointers(desc.inFields, desc.inArrayType, local_comm_rank);
        for(size_t i = 0; i < inPtrCount; ++i)
        {
            os << "user input " << i << ": " << in_buffer[i] << std::endl;
        }
        if(placement == rocfft_placement_notinplace)
        {
            auto outPtrCount
                = desc.count_pointers(desc.outFields, desc.outArrayType, local_comm_rank);
            for(size_t i = 0; i < outPtrCount; ++i)
            {
                os << "user output " << i << ": " << out_buffer[i] << std::endl;
            }
        }
    }

    LogFields("input", desc.inFields);
    LogFields("output", desc.outFields);

    LogSortedPlan(sortedIdx);

    for(auto i = sortedIdx.begin(); i != sortedIdx.end(); ++i)
    {
        const auto idx = *i;

        if(!multiPlan[idx])
            continue;

        auto& item = *multiPlan[idx];

        for(auto antecedentIdx : multiPlanAntecedents[idx])
        {
            // The multiPlan item may not be created on this rank (in which case it's not a
            // rank-local operation):
            if(!multiPlan[antecedentIdx])
                continue;

            // Check if antecedent involved us:
            auto& antecedent = *multiPlan[antecedentIdx];

            // The antecedent involved us somehow, wait for it:
            if(antecedent.ExecutesOnRank(local_comm_rank))
            {
                antecedent.Wait();
            }
        }

        // Done waiting for all our antecedents, so this item can now proceed.

        // Launch this item async:
        if(item.ExecutesOnRank(local_comm_rank))
        {
            item.ExecuteAsync(this, in_buffer, out_buffer, info, idx);
        }
    }

    // finished executing all items, wait for outstanding work to complete
    for(auto i = sortedIdx.begin(); i != sortedIdx.end(); ++i)
    {
        auto idx = *i;

        if(!multiPlan[idx])
            continue;

        auto& item = *multiPlan[idx];
        if(item.ExecutesOnRank(local_comm_rank))
        {
            item.Wait();
        }
    }
}

rocfft_status rocfft_execute(const rocfft_plan     plan,
                             void*                 in_buffer[],
                             void*                 out_buffer[],
                             rocfft_execution_info info)
try
{
    log_trace(
        __func__, "plan", plan, "in_buffer", in_buffer, "out_buffer", out_buffer, "info", info);

    if(!plan)
        return rocfft_status_failure;

    try
    {
        plan->Execute(in_buffer, out_buffer, info);
    }
    catch(std::exception& e)
    {
        if(LOG_TRACE_ENABLED())
        {
            (*LogSingleton::GetInstance().GetTraceOS()) << e.what() << std::endl;
        }
        return rocfft_status_failure;
    }
    return rocfft_status_success;
}
catch(...)
{
    return rocfft_handle_exception();
}

void ExecPlan::ExecuteAsync(const rocfft_plan     plan,
                            void*                 in_buffer[],
                            void*                 out_buffer[],
                            rocfft_execution_info info,
                            size_t                multiPlanIdx)
{
    rocfft_scoped_device dev(location.device);

    // tolerate user not providing an execution_info
    rocfft_execution_info_t exec_info;
    if(info)
        exec_info = *info;

    // use the local stream if user didn't provide one
    if(mgpuPlan && !exec_info.rocfft_stream)
    {
        exec_info.rocfft_stream = this->stream;
    }

    // TransformPowX below needs in_buffer, out_buffer to work with.
    // But we need to potentially override pointers in those arrays.
    // So copy them to temporary vectors.
    // This is not necessary for single-device plans.
    std::vector<void*> in_buffer_copy;
    std::vector<void*> out_buffer_copy;

    if(mgpuPlan)
    {
        auto local_comm_rank = plan->get_local_comm_rank();
        std::copy_n(
            in_buffer,
            plan->desc.count_pointers(plan->desc.inFields, plan->desc.inArrayType, local_comm_rank),
            std::back_inserter(in_buffer_copy));

        // if input/output are overridden, override now
        if(inputPtr)
            in_buffer_copy[0] = inputPtr.get(in_buffer, out_buffer, local_comm_rank);

        if(rootPlan->placement == rocfft_placement_notinplace)
        {
            std::copy_n(out_buffer,
                        plan->desc.count_pointers(
                            plan->desc.outFields, plan->desc.outArrayType, local_comm_rank),
                        std::back_inserter(out_buffer_copy));
            if(outputPtr)
                out_buffer_copy[0] = outputPtr.get(in_buffer, out_buffer, local_comm_rank);
        }
    }

    // select the input and output buffers based on whether
    // we have a single or multi device plan.
    auto in_transform_ptrs  = mgpuPlan ? in_buffer_copy.data() : in_buffer;
    auto out_transform_ptrs = mgpuPlan ? out_buffer_copy.data() : out_buffer;

    gpubuf autoAllocWorkBuf;

    if(workBufSize > 0)
    {
        auto requiredWorkBufBytes = WorkBufBytes(real_type_size(rootPlan->precision));
        if(!exec_info.workBuffer)
        {
            // user didn't provide a buffer, alloc one now
            if(autoAllocWorkBuf.alloc(requiredWorkBufBytes) != hipSuccess)
                throw std::runtime_error("work buffer allocation failure");
            exec_info.workBufferSize = requiredWorkBufBytes;
            exec_info.workBuffer     = autoAllocWorkBuf.data();
        }
        // otherwise user provided a buffer, but complain if it's too small
        else if(exec_info.workBufferSize < requiredWorkBufBytes)
        {
            if(LOG_TRACE_ENABLED())
                (*LogSingleton::GetInstance().GetTraceOS())
                    << "user work buffer too small" << std::endl;
            throw rocfft_status_invalid_work_buffer;
        }
    }

    // Callbacks do not currently support planar format
    if((array_type_is_planar(rootPlan->inArrayType) || array_type_is_planar(rootPlan->outArrayType))
       && (exec_info.callbacks.load_cb_fn || exec_info.callbacks.store_cb_fn))
        throw std::runtime_error("callbacks not supported with planar format");

    try
    {
        TransformPowX(*this,
                      in_transform_ptrs,
                      (rootPlan->placement == rocfft_placement_inplace) ? in_transform_ptrs
                                                                        : out_transform_ptrs,
                      &exec_info,
                      multiPlanIdx);
        // all work is enqueued to the stream, record the event on
        // the stream. Not needed for single-device plans.
        if(mgpuPlan)
        {
            if(hipEventRecord(event, exec_info.rocfft_stream) != hipSuccess)
                throw std::runtime_error("hipEventRecord failed");
        }
    }
    catch(std::exception& e)
    {
        if(LOG_TRACE_ENABLED())
        {
            (*LogSingleton::GetInstance().GetTraceOS()) << e.what() << std::endl;
        }
        throw;
    }
}

void ExecPlan::Wait()
{
    // for a single-device plan, we don't need to synchronize
    // events
    if(mgpuPlan && event)
    {
        if(hipEventSynchronize(event) != hipSuccess)
            throw std::runtime_error("hipEventSynchronize failed");
    }
}
