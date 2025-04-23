/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#include "ago_internal.h"

static int agoOptimizeDramaAllocRemoveUnusedData(AgoGraph * agraph)
{
    for (;;)
    {
        bool doRepeat = false;

        // check and mark data usage
        agoOptimizeDramaMarkDataUsage(agraph);

        // find and remove virtual nodes that are not used
        for (AgoData * adata = agraph->dataList.head; adata;) {
            bool relatedToDelayElement = false;
            if (adata->ref.type == VX_TYPE_DELAY) {
                // object can't be removed since it is a delay element
                relatedToDelayElement = true;
            }
            else {
                // object can't be removed since it is part of a delay element
                relatedToDelayElement = agoIsPartOfDelay(adata);
            }
            if (!relatedToDelayElement && adata->isVirtual && (adata->outputUsageCount == 0) && (adata->inputUsageCount == 0) && (adata->inoutUsageCount == 0)) {
                AgoData * next = adata->next;
                agoRemoveDataInGraph(agraph, adata);
                adata = next;
                doRepeat = true; // to repeat the removal process again
                continue;
            }
            adata = adata->next;
        }
        if (doRepeat)
            continue;

        break;
    }
    return 0;
}

#if (ENABLE_OPENCL || ENABLE_HIP)
int agoGpuAllocBuffers(AgoGraph * graph)
{
    // get default target
    vx_uint32 bufferMergeFlags = 0;
    char textBuffer[1024];
    if (agoGetEnvironmentVariable("AGO_BUFFER_MERGE_FLAGS", textBuffer, sizeof(textBuffer))) {
        bufferMergeFlags = atoi(textBuffer);
    }

    // mark hierarchical level (start,end) of all data in the graph
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        for (vx_uint32 i = 0; i < node->paramCount; i++) {
            AgoData * data = node->paramList[i];
            if (data) {
                data->hierarchical_life_start = INT_MAX;
                data->hierarchical_life_end = 0;
                data->initialization_flags = 0;
                for (vx_uint32 j = 0; j < data->numChildren; j++) {
                    data->children[j]->hierarchical_life_start = INT_MAX;
                    data->children[j]->hierarchical_life_end = 0;
                    data->children[j]->initialization_flags = 0;
                }
            }
        }
    }
    for (AgoSuperNode * supernode = graph->supernodeList; supernode; supernode = supernode->next) {
        for (AgoData * data : supernode->dataList) {
            data->hierarchical_life_start = min(data->hierarchical_life_start, supernode->hierarchical_level_start);
            data->hierarchical_life_end = max(data->hierarchical_life_end, supernode->hierarchical_level_end);
        }
    }
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        if (!node->supernode) {
            for (vx_uint32 i = 0; i < node->paramCount; i++) {
                AgoData * data = node->paramList[i];
                if (data) {
                    data->hierarchical_life_start = min(data->hierarchical_life_start, node->hierarchical_level);
                    data->hierarchical_life_end = max(data->hierarchical_life_end, node->hierarchical_level);
                    for (vx_uint32 j = 0; j < data->numChildren; j++) {
                        data->children[j]->hierarchical_life_start = min(data->children[j]->hierarchical_life_start, node->hierarchical_level);
                        data->children[j]->hierarchical_life_end = max(data->children[j]->hierarchical_life_end, node->hierarchical_level);
                    }
                }
            }
        }
    }

    // get the list of virtual data (D) that need GPU buffers and mark if CPU access is not needed for virtual buffers
    auto isDataValidForGd = [=](AgoData * data) -> bool {
        return data && data->isVirtual;
    };
    std::vector<AgoData *> D;
    for (AgoSuperNode * supernode = graph->supernodeList; supernode; supernode = supernode->next) {
        for (size_t i = 0; i < supernode->dataList.size(); i++) {
            AgoData * data = supernode->dataList[i];
            if (supernode->dataInfo[i].needed_as_a_kernel_argument && isDataValidForGd(data) && (data->initialization_flags & 1) == 0) {
                data->initialization_flags |= 1;
                if (!(bufferMergeFlags & 2)) {
                    data->device_type_unused = AGO_TARGET_AFFINITY_CPU;
                }
                D.push_back(data);
            }
        }
    }
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        if (!node->supernode) {
            if (node->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_GPU ||
                node->akernel->opencl_buffer_access_enable)
            {
                for (vx_uint32 i = 0; i < node->paramCount; i++) {
                    AgoData * data = node->paramList[i];
                    if (isDataValidForGd(data) && (data->initialization_flags & 1) == 0) {
                        data->initialization_flags |= 1;
                        if (!(bufferMergeFlags & 2)) {
                            data->device_type_unused = AGO_TARGET_AFFINITY_CPU;
                        }
                        D.push_back(data);
                    }
                }
            }
        }
    }
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        if (!node->supernode) {
            if (node->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_CPU &&
                !node->akernel->opencl_buffer_access_enable)
            {
                for (vx_uint32 i = 0; i < node->paramCount; i++) {
                    AgoData * data = node->paramList[i];
                    if (isDataValidForGd(data)) {
                        data->device_type_unused &= ~AGO_TARGET_AFFINITY_CPU;
                        for (vx_uint32 j = 0; j < data->numChildren; j++) {
                            data->children[j]->device_type_unused &= ~AGO_TARGET_AFFINITY_CPU;
                        }
                    }
                }
            }
        }
    }

#if ENABLE_OPENCL
    // get data groups (Gd)
    auto getMemObjectType = [=](AgoData * data) -> cl_mem_object_type {
        cl_mem_object_type obj_type = CL_MEM_OBJECT_BUFFER;
        return (vx_uint32)obj_type;
    };
#elif ENABLE_HIP
    auto getMemObjectType = [=](AgoData * data) -> vx_uint32 {
        vx_uint32 obj_type = HIP_MEM_KIND_BUFFER;
        return (vx_uint32)obj_type;
    };
#endif
    auto getMemObjectSize = [=](AgoData * data) -> size_t {
        return data->gpu_buffer_offset + data->size;
    };
    auto isMergePossible = [=](std::vector<AgoData *>& G, AgoData * data) -> bool {
        bool possible = false;
        for (auto d : G) {
            if(d->alias_data == data || d == data->alias_data) {
                possible = true;
                break;
            }
        }
        if (!possible) {
            possible = true;
            vx_uint32 s = data->hierarchical_life_start;
            vx_uint32 e = data->hierarchical_life_end;
            vx_uint32 dataMemType = getMemObjectType(data);
            for (auto d : G) {
                vx_uint32 dMemType = getMemObjectType(d);
                if((dataMemType != dMemType) ||
                   (s >= d->hierarchical_life_start && s <= d->hierarchical_life_end) ||
                   (e >= d->hierarchical_life_start && e <= d->hierarchical_life_end))
                {
                    possible = false;
                    break;
                }
            }
        }
        return possible;
    };
    auto calcMergedCost = [=](std::vector<AgoData *>& G, AgoData * data) -> size_t {
        size_t size = getMemObjectSize(data);
        for (auto d : G) {
            size = max(size, getMemObjectSize(d));
        }
        return size;
    };
    std::vector< std::vector<AgoData *> > Gd;
    std::vector< size_t > Gsize;
    for (AgoData * data : D) {
        if (data->alias_data) {
            size_t bestj = INT_MAX;
            for (size_t j = 0; j < Gd.size(); j++) {
                for (size_t k = 0; k < Gd[j].size(); k++) {
                    if(data->alias_data == Gd[j][k] || data == Gd[j][k]->alias_data) {
                        bestj = j;
                        break;
                    }
                }
                if(bestj != INT_MAX)
                    break;
            }
            if(bestj == INT_MAX) {
                bestj = Gd.size();
                Gd.push_back(std::vector<AgoData *>());
            }
            Gd[bestj].push_back(data);
        }
    }
    for (AgoData * data : D) {
        size_t bestj = INT_MAX, bestCost = INT_MAX;
        if (!(bufferMergeFlags & 1)) {
            for (size_t j = 0; j < Gd.size(); j++) {
                if(isMergePossible(Gd[j], data)) {
                    size_t cost = calcMergedCost(Gd[j], data);
                    if(cost < bestCost) {
                        bestj = j;
                        bestCost = cost;
                    }
                }
            }
        }
        if(bestj == INT_MAX) {
            bestj = Gd.size();
            bestCost = getMemObjectSize(data);
            Gd.push_back(std::vector<AgoData *>());
        }
        Gd[bestj].push_back(data);
    }

    // allocate one GPU buffer per group
    for (size_t j = 0; j < Gd.size(); j++) {
        size_t k = 0;
        for (size_t i = 1; i < Gd[j].size(); i++) {
            if(getMemObjectSize(Gd[j][i]) > getMemObjectSize(Gd[j][k]))
                k = i;
        }
#if ENABLE_OPENCL
        if (agoGpuOclAllocBuffer(Gd[j][k]) < 0) {
            return -1;
        }
        for (size_t i = 0; i < Gd[j].size(); i++) {
            if(i != k) {
                if(Gd[j][i]->alias_offset > 0) {
                    cl_buffer_region region = {
                        Gd[j][i]->alias_offset,
                        Gd[j][k]->size + Gd[j][k]->gpu_buffer_offset - Gd[j][i]->alias_offset
                    };
                    Gd[j][i]->opencl_buffer = Gd[j][i]->opencl_buffer_allocated =
                        clCreateSubBuffer(Gd[j][k]->opencl_buffer, CL_MEM_READ_WRITE,
                                CL_BUFFER_CREATE_TYPE_REGION, &region, NULL);
                }
                else {
                    Gd[j][i]->opencl_buffer = Gd[j][k]->opencl_buffer;
                }
            }
        }
#else
        if (agoGpuHipAllocBuffer(Gd[j][k]) < 0) {
            return -1;
        }
        for (size_t i = 0; i < Gd[j].size(); i++) {
            if(i != k) {
                if(Gd[j][i]->alias_offset > 0) {
                    Gd[j][i]->hip_memory = Gd[j][k]->hip_memory + Gd[j][i]->alias_offset;
                    Gd[j][i]->hip_memory_allocated = Gd[j][k]->hip_memory;
                }
                else {
                    Gd[j][i]->hip_memory = Gd[j][k]->hip_memory;
                }
            }
        }
#endif
    }

    // allocate GPU buffers if node scheduled on GPU using OpenCL or using opencl_buffer_access_enable
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        if (node->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_GPU ||
            node->akernel->opencl_buffer_access_enable)
        {
            for (vx_uint32 i = 0; i < node->paramCount; i++) {
                AgoData * data = node->paramList[i];
#if ENABLE_OPENCL
                if (data && !data->opencl_buffer && !data->isVirtual) {
#elif ENABLE_HIP
                if (data && !data->hip_memory && !data->isVirtual) {
#endif
                    if (agoIsPartOfDelay(data)) {
                        int siblingTrace[AGO_MAX_DEPTH_FROM_DELAY_OBJECT], siblingTraceCount = 0;
                        data = agoGetSiblingTraceToDelayForUpdate(data, siblingTrace, siblingTraceCount);
                        if (!data) return -1;
                    }
#if ENABLE_OPENCL
                    if (agoGpuOclAllocBuffer(data) < 0) {
                        return -1;
                    }
#elif ENABLE_HIP
                    if (agoGpuHipAllocBuffer(data) < 0) {
                        return -1;
                    }
#endif
                }
            }
        }
    }

   // The AGO_BUFFER_MERGE_FLAGS is set for HarrisCorner as buffer merging is not working for this node.
   // At the end of the GPU buffer allocation routine, unset the AGO_BUFFER_MERGE_FLAGS environment variable
   // if the HarrisCorner is present in the graph.
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        if (strstr(node->akernel->name, "Harris") != NULL) {
            agoUnsetEnvironmentVariable("AGO_BUFFER_MERGE_FLAGS");
            break;
        }
    }

    return 0;
}

static int agoOptimizeDramaAllocGpuResources(AgoGraph * graph)
{
    // check to make sure that GPU resources are needed
    bool gpuNeeded = false;
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        if (node->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_GPU || node->akernel->opencl_buffer_access_enable) {
            gpuNeeded = true;
            break;
        }
    }
    if (gpuNeeded) {
        // make sure to allocate context and command queue for OPENCL or HIP
#if ENABLE_OPENCL
        if (!graph->opencl_cmdq) {
            // make sure that the context has been created
            vx_context context = graph->ref.context;
            if (!context->opencl_context) {
                if (agoGpuOclCreateContext(context, nullptr) < 0) {
                    return -1;
                }
            }
            // create command queue: for now use device#0 -- TBD: this needs to be changed in future
            cl_int err = -1;
            graph->opencl_device = context->opencl_device_list[0];
#if defined(CL_VERSION_2_0)
            cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, context->opencl_cmdq_properties, 0 };
            graph->opencl_cmdq = clCreateCommandQueueWithProperties(context->opencl_context, graph->opencl_device, properties, &err);
#else
            graph->opencl_cmdq = clCreateCommandQueue(context->opencl_context, graph->opencl_device, context->opencl_cmdq_properties, &err);
#endif
            if (err) {
                agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: clCreateCommandQueueWithProperties(%p,%p,0,*) => %d\n", context->opencl_context, graph->opencl_device, err);
                return -1;
            }
        }
#elif ENABLE_HIP
        vx_context context = graph->ref.context;
        if (context->hip_device_id < 0) {
            if (agoGpuHipCreateContext(context, -1) < 0) {
                return -1;
            }
        }
        // create a seperate stream for graph 
        hipError_t err = hipStreamCreate(&graph->hip_stream0);
        if (err != hipSuccess) {
            agoAddLogEntry(NULL, VX_FAILURE, "ERROR: hipStreamCreate(%p) => %d (failed) for graph => dev%d\n", graph->hip_stream0, err, context->hip_device_id);
            return -1;
        }
        //Force creation of the underlying HW queue associated with the HIP stream created above here;
        // otherwise, the HW queue creation will be delayed until this stream is used in the graph
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            agoAddLogEntry(NULL, VX_FAILURE, "ERROR: hipDeviceSynchronize => %d (failed)\n", err);
            return -1;
        }
#endif
    }

    // identify GPU groups and make sure that they all have same affinity
    std::map<vx_uint32, AgoTargetAffinityInfo_> groupMap;
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        if (node->attr_affinity.group > 0) {
            if (groupMap.find(node->attr_affinity.group) == groupMap.end()) {
                groupMap.insert(std::pair<vx_uint32, AgoTargetAffinityInfo_>(node->attr_affinity.group, node->attr_affinity));
            }
            if (memcmp(&groupMap[node->attr_affinity.group], &node->attr_affinity, sizeof(node->attr_affinity)) != 0) {
                agoAddLogEntry(&node->ref, VX_FAILURE, "ERROR: agoOptimizeDramaAllocGpuResources: mismatched affinity in nodes of group#%d\n", node->attr_affinity.group);
                return -1;
            }
        }
        else if (node->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_GPU) {
#if ENABLE_OPENCL
            node->opencl_build_options = node->ref.context->opencl_build_options;
            if (node->akernel->func) {
                // generate kernel function code
                int status = node->akernel->func(node, ago_kernel_cmd_opencl_codegen);
                if (status == VX_SUCCESS) {
                    if (node->opencl_type & NODE_OPENCL_TYPE_FULL_KERNEL) {
                        strcpy(node->opencl_name, NODE_OPENCL_KERNEL_NAME);
                        for(vx_size dim = node->opencl_work_dim; dim < 3; dim++) {
                            node->opencl_global_work[dim] = 1;
                            node->opencl_local_work[dim] = 1;
                        }
                        node->opencl_work_dim = 3;
                    }
                    else {
                        agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: agoOptimizeDramaAllocGpuResources: doesn't support kernel %s as a standalone OpenCL kernel\n", node->akernel->name);
                        return -1;
                    }
                }
                else if (status != AGO_ERROR_KERNEL_NOT_IMPLEMENTED) {
                    agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: agoOptimizeDramaAllocGpuResources: kernel %s failed to generate OpenCL code (error %d)\n", node->akernel->name, status);
                    return -1;
                }
            }
            else if (node->akernel->opencl_codegen_callback_f) {
                // generate kernel function
                node->opencl_name[0] = 0;
                node->opencl_work_dim = 0;
                node->opencl_global_work[0] = 0;
                node->opencl_global_work[1] = 0;
                node->opencl_global_work[2] = 0;
                node->opencl_local_work[0] = 0;
                node->opencl_local_work[1] = 0;
                node->opencl_local_work[2] = 0;
                node->opencl_param_mem2reg_mask = 0;
                node->opencl_param_discard_mask = 0;
                node->opencl_param_atomic_mask = 0;
                node->opencl_compute_work_multiplier = 0;
                node->opencl_compute_work_param_index = 0;
                node->opencl_output_array_param_index_plus1 = 0;
                node->opencl_local_buffer_usage_mask = 0;
                node->opencl_local_buffer_size_in_bytes = 0;
                node->opencl_code = "";
                int status = node->akernel->opencl_codegen_callback_f(node, (vx_reference *)node->paramList, node->paramCount,
                    false, node->opencl_name, node->opencl_code, node->opencl_build_options, node->opencl_work_dim, node->opencl_global_work,
                    node->opencl_local_work, node->opencl_local_buffer_usage_mask, node->opencl_local_buffer_size_in_bytes);
                if (status == VX_SUCCESS) {
                    node->opencl_type = NODE_OPENCL_TYPE_FULL_KERNEL;
                    for(vx_size dim = node->opencl_work_dim; dim < 3; dim++) {
                        node->opencl_global_work[dim] = 1;
                        node->opencl_local_work[dim] = 1;
                    }
                    node->opencl_work_dim = 3;
                }
                else if (status != AGO_ERROR_KERNEL_NOT_IMPLEMENTED) {
                    agoAddLogEntry(&node->akernel->ref, status, "ERROR: agoOptimizeDramaAllocGpuResources: kernel %s failed to generate OpenCL code (error %d)\n", node->akernel->name, status);
                    return -1;
                }
            }
            else {
                agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: agoOptimizeDramaAllocGpuResources: doesn't support kernel %s on GPU\n", node->akernel->name);
                return -1;
            }
#elif ENABLE_HIP
            node->hip_stream0 = graph->hip_stream0;
#endif
        }
    }
#if (ENABLE_OPENCL || ENABLE_HIP)
    // create a supernode for each group
    for (auto itgroup = groupMap.begin(); itgroup != groupMap.end(); itgroup++) {
        AgoSuperNode * supernode = NULL;
        // add individual nodes into supernode
        for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
            if (node->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_GPU && node->attr_affinity.group == itgroup->first) {
            // add supernode only for OPenCL flow. Not supported in HIP for now
#if ENABLE_OPENCL
                // make sure supernode is created for GPU
                if (!supernode) {
                    supernode = new AgoSuperNode; if (!supernode) return -1;
                    supernode->group = itgroup->first;
                }
                // link supernode into node
                node->supernode = supernode;
                // initialize supernode with OpenCL information
                supernode->isGpuOclSuperNode = true;
                // add node functionality into supernode
                supernode->opencl_cmdq = graph->opencl_cmdq;
                if (agoGpuOclSuperNodeMerge(graph, supernode, node) < 0) {
                    return -1;
                }
#else
                if (supernode) {
                    supernode->hip_stream0 = graph->hip_stream0;
                    if (agoGpuHipSuperNodeMerge(graph, supernode, node) < 0) {
                        return -1;
                    }
                }
#endif
            }
        }
        if (supernode) {
            // add supernode to the master list
            supernode->next = graph->supernodeList;
            graph->supernodeList = supernode;
        }
    }

    // update supernodes for buffer usage and hierarchical levels
    for (AgoSuperNode * supernode = graph->supernodeList; supernode; supernode = supernode->next) {
#if ENABLE_OPENCL
        if (agoGpuOclSuperNodeUpdate(graph, supernode) < 0) {
#else
        if (agoGpuHipSuperNodeUpdate(graph, supernode) < 0) {
#endif
            return -1;
        }
    }
#endif
    // allocate GPU buffers if node scheduled on GPU using OpenCL or using opencl_buffer_access_enable
    if (agoGpuAllocBuffers(graph) < 0) {
        return -1;
    }

#if ENABLE_OPENCL
    // finalize all GPU supernodes and single nodes
    for (AgoSuperNode * supernode = graph->supernodeList; supernode; supernode = supernode->next) {
        if (agoGpuOclSuperNodeFinalize(graph, supernode) < 0) {
            return -1;
        }
    }
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        if (node->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_GPU && node->attr_affinity.group == 0) {
            if (agoGpuOclSingleNodeFinalize(graph, node) < 0) {
                return -1;
            }
        }
    }
#elif ENABLE_HIP
    for (AgoSuperNode * supernode = graph->supernodeList; supernode; supernode = supernode->next) {
        if (agoGpuHipSuperNodeFinalize(graph, supernode) < 0) {
            return -1;
        }
    }
#endif

    return 0;
}
#endif

static int agoOptimizeDramaAllocSetDefaultTargets(AgoGraph * agraph)
{
    // get unused GPU group ID
    vx_uint32 nextAvailGroupId = 1;
    for (AgoNode * node = agraph->nodeList.head; node; node = node->next) {
        if (node->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_GPU) {
            if (node->attr_affinity.group >= nextAvailGroupId) {
                nextAvailGroupId = node->attr_affinity.group + 1;
            }
        }
    }

    // get default target
    vx_uint32 default_target = AGO_KERNEL_TARGET_DEFAULT;
    char textBuffer[1024];
    if (agoGetEnvironmentVariable("AGO_DEFAULT_TARGET", textBuffer, sizeof(textBuffer))) {
        if (!strcmp(textBuffer, "GPU")) {
            default_target = AGO_KERNEL_FLAG_DEVICE_GPU;
        }
        else if (!strcmp(textBuffer, "CPU")) {
            default_target = AGO_KERNEL_FLAG_DEVICE_CPU;
        }
    }

    for (AgoNode * node = agraph->nodeList.head; node; node = node->next) {
        // get target support info
        node->target_support_flags = 0;
        if (node->akernel->func) {
            node->akernel->func(node, ago_kernel_cmd_query_target_support);
        }
        else if (node->akernel->query_target_support_f) {
            vx_uint32 supported_target_affinity = 0;
#if ENABLE_OPENCL
            vx_bool use_opencl_1_2 = (agraph->ref.context->opencl_config_flags & CONFIG_OPENCL_USE_1_2) ? vx_true_e : vx_false_e;
            vx_status status = node->akernel->query_target_support_f(agraph, node, use_opencl_1_2, supported_target_affinity);
            if (status) {
                agoAddLogEntry(&node->akernel->ref, status, "ERROR: kernel %s: query_target_support_f(*,*,%d,*) => %d\n", node->akernel->name, use_opencl_1_2, status);
                return -1;
            }
#else
            vx_status status = node->akernel->query_target_support_f(agraph, node, vx_false_e, supported_target_affinity);
            if (status) {
                agoAddLogEntry(&node->akernel->ref, status, "ERROR: kernel %s: query_target_support_f(*,*,%d,*) => %d\n", node->akernel->name, vx_false_e, status);
                return -1;
            }
#endif
            node->target_support_flags = 0;
            if (supported_target_affinity & AGO_KERNEL_FLAG_DEVICE_CPU) {
                // mark that CPU target affinity is supported
                node->target_support_flags |= AGO_KERNEL_FLAG_DEVICE_CPU;
                supported_target_affinity &= ~AGO_KERNEL_FLAG_DEVICE_CPU;
            }
            if (supported_target_affinity & AGO_KERNEL_FLAG_DEVICE_GPU) {
                // mark that GPU target affinity is supported with full kernels
                node->target_support_flags |= AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL;
                supported_target_affinity &= ~AGO_KERNEL_FLAG_DEVICE_GPU;
            }
            if (supported_target_affinity) {
                agoAddLogEntry(&node->akernel->ref, status, "ERROR: kernel %s: query_target_support_f returned unsupported affinity flags: 0x%08x\n", node->akernel->name, supported_target_affinity);
                return -1;
            }
        }
        else {
            // default: only CPU is supported
            node->target_support_flags = AGO_KERNEL_FLAG_DEVICE_CPU;
        }

        // check to make sure that kernel supports CPU and/or GPU
        if (!(node->target_support_flags & (AGO_KERNEL_FLAG_DEVICE_CPU | AGO_KERNEL_FLAG_DEVICE_GPU))) {
            agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: kernel %s not supported yet\n", node->akernel->name);
            return -1;
        }

        // set default targets
        if (node->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_CPU) {
            if (node->target_support_flags & AGO_KERNEL_FLAG_DEVICE_CPU) {
                // reset group
                node->attr_affinity.device_info = 0;
                node->attr_affinity.group = 0;
            }
            else {
                // fall back to GPU
                vxAddLogEntry((vx_reference)node, VX_SUCCESS, "WARNING: kernel %s not supported on CPU -- falling back to GPU\n", node->akernel->name);
                // set default target as GPU
                node->attr_affinity.device_type = AGO_KERNEL_FLAG_DEVICE_GPU;
                node->attr_affinity.device_info = 0;
                node->attr_affinity.group = 0;
                if (node->target_support_flags & (AGO_KERNEL_FLAG_GPU_INTEG_R2R | AGO_KERNEL_FLAG_GPU_INTEG_M2R)) {
                    // use an unsed group Id
                    node->attr_affinity.group = nextAvailGroupId++;
                }
            }
        }
        else if (node->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_GPU) {
            if (node->target_support_flags & AGO_KERNEL_FLAG_DEVICE_GPU) {
                if (node->target_support_flags & AGO_KERNEL_FLAG_GPU_INTEG_FULL) {
                    if (node->attr_affinity.group != 0) {
                        agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: kernel %s can't be grouped with other kernels on GPU\n", node->akernel->name);
                        return -1;
                    }
                }
                else if (node->target_support_flags & (AGO_KERNEL_FLAG_GPU_INTEG_R2R | AGO_KERNEL_FLAG_GPU_INTEG_M2R)) {
                        if (node->attr_affinity.group == 0) {
                            // use an unsed group Id
                            node->attr_affinity.group = nextAvailGroupId++;
                        }
                }
                // set default target as GPU
                node->attr_affinity.device_type = AGO_KERNEL_FLAG_DEVICE_GPU;
                node->attr_affinity.device_info = 0;
            }
            else {
                // fall back to CPU
                vxAddLogEntry((vx_reference)node, VX_SUCCESS, "WARNING: kernel %s not supported on GPU -- falling back to CPU\n", node->akernel->name);
                // set default target as CPU
                node->attr_affinity.device_type = AGO_KERNEL_FLAG_DEVICE_CPU;
                node->attr_affinity.device_info = 0;
                node->attr_affinity.group = 0;
            }
        }
        else {
            if (default_target == AGO_KERNEL_FLAG_DEVICE_GPU) {
                // choose GPU as default if supported
                if (node->target_support_flags & AGO_KERNEL_FLAG_DEVICE_GPU) {
                    // set default target as GPU
                    node->attr_affinity.device_type = AGO_KERNEL_FLAG_DEVICE_GPU;
                    node->attr_affinity.device_info = 0;
                    node->attr_affinity.group = 0;
                    if (node->target_support_flags & (AGO_KERNEL_FLAG_GPU_INTEG_R2R | AGO_KERNEL_FLAG_GPU_INTEG_M2R)) {
                        // use an unsed group Id
                        node->attr_affinity.group = nextAvailGroupId++;
                    }
                }
                else {
                    // set default target as CPU
                    node->attr_affinity.device_type = AGO_KERNEL_FLAG_DEVICE_CPU;
                    node->attr_affinity.device_info = 0;
                    node->attr_affinity.group = 0;
                }
            }
            else {
                // choose CPU as default if supported
                if (node->target_support_flags & AGO_KERNEL_FLAG_DEVICE_CPU) {
                    // set default target as CPU
                    node->attr_affinity.device_type = AGO_KERNEL_FLAG_DEVICE_CPU;
                    node->attr_affinity.device_info = 0;
                    node->attr_affinity.group = 0;
                }
                else {
                    // set default target as GPU
                    node->attr_affinity.device_type = AGO_KERNEL_FLAG_DEVICE_GPU;
                    node->attr_affinity.device_info = 0;
                    node->attr_affinity.group = 0;
                    if (node->target_support_flags & (AGO_KERNEL_FLAG_GPU_INTEG_R2R | AGO_KERNEL_FLAG_GPU_INTEG_M2R)) {
                        // use an unsed group Id
                        node->attr_affinity.group = nextAvailGroupId++;
                    }
                }
            }
        }
    }
    return 0;
}

#if (ENABLE_OPENCL || ENABLE_HIP)
static int agoOptimizeDramaAllocMergeSuperNodes(AgoGraph * graph)
{
    // initialize groupInfo list with SuperNodeInfo
    class SuperNodeInfo {
    public:
        vx_uint32 integ_flags;
        vx_uint32 min_hierarchical_level;
        vx_uint32 max_hierarchical_level;
        std::list<AgoNode *> nodeList;
        std::list<AgoData *> inputList;
        std::list<AgoData *> outputList;
    };
    std::map<vx_uint32, SuperNodeInfo *> groupInfo;
    for (auto node = graph->nodeList.head; node; node = node->next) {
        vx_uint32 group = node->attr_affinity.group;
        if (node->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_GPU && group > 0) {
            auto it = groupInfo.find(group);
            // create/get superNodeInfo for the current group
            SuperNodeInfo * superNodeInfo = nullptr;
            if (it == groupInfo.end()) {
                superNodeInfo = new SuperNodeInfo;
                superNodeInfo->integ_flags = 0;
                superNodeInfo->min_hierarchical_level = INT_MAX;
                superNodeInfo->max_hierarchical_level = 0;
                groupInfo[group] = superNodeInfo;
            }
            else {
                superNodeInfo = groupInfo[group];
            }
            // update superNodeInfo
            superNodeInfo->integ_flags |= node->target_support_flags & AGO_KERNEL_FLAG_GPU_INTEG_MASK;
            if (node->hierarchical_level < superNodeInfo->min_hierarchical_level) superNodeInfo->min_hierarchical_level = node->hierarchical_level;
            if (node->hierarchical_level > superNodeInfo->max_hierarchical_level) superNodeInfo->max_hierarchical_level = node->hierarchical_level;
            superNodeInfo->nodeList.push_back(node);
            for (vx_uint32 i = 0; i < node->paramCount; i++) {
                auto data = node->paramList[i];
                if (data) {
                    auto it = std::find(superNodeInfo->inputList.begin(), superNodeInfo->inputList.end(), data);
                    if (it == superNodeInfo->inputList.end() && (node->parameters[i].direction == VX_INPUT || node->parameters[i].direction == (vx_direction_e)VX_BIDIRECTIONAL))
                        superNodeInfo->inputList.push_back(data);
                    it = std::find(superNodeInfo->outputList.begin(), superNodeInfo->outputList.end(), data);
                    if (it == superNodeInfo->outputList.end() && (node->parameters[i].direction == VX_OUTPUT || node->parameters[i].direction == (vx_direction_e)VX_BIDIRECTIONAL))
                        superNodeInfo->outputList.push_back(data);
                }
            }
        }
    }
    // perform  one hierarchical level at a time
    for (auto enode = graph->nodeList.head; enode;) {
        // get snode..enode with next hierarchical_level
        auto hierarchical_level = enode->hierarchical_level;
        auto snode = enode; enode = enode->next;
        while (enode && enode->hierarchical_level == hierarchical_level)
            enode = enode->next;
        // try to merge with supernodes from previous hierarchical levels
        for (auto cnode = snode; cnode != enode; cnode = cnode->next) {
            if (cnode->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_GPU && cnode->attr_affinity.group > 0) {
                SuperNodeInfo * csuperNodeInfo = groupInfo[cnode->attr_affinity.group];
                for (auto pnode = graph->nodeList.head; pnode != cnode; pnode = pnode->next) {
                    if (pnode->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_GPU && pnode->attr_affinity.group > 0 && pnode->attr_affinity.group != cnode->attr_affinity.group) {
                        SuperNodeInfo * psuperNodeInfo = groupInfo[pnode->attr_affinity.group];
                        // check and merge if csuperNodeInfo can be merged with psuperNodeInfo
                        auto conflictDetected = false;
                        if (cnode->target_support_flags & pnode->target_support_flags & AGO_KERNEL_FLAG_GPU_INTEG_M2R) {
                            // only one M2R allowed per supernode at this time
                            conflictDetected = true;
                        }
                        else if (pnode->paramList[0]->u.img.width != cnode->paramList[0]->u.img.width || pnode->paramList[0]->u.img.height != cnode->paramList[0]->u.img.height) {
                            // all destination images shall have same dimensions
                            conflictDetected = true;
                        }
                        else {
                            for (auto cit = csuperNodeInfo->inputList.begin(); cit != csuperNodeInfo->inputList.end(); cit++) {
                                auto pit = std::find(psuperNodeInfo->outputList.begin(), psuperNodeInfo->outputList.end(), *cit);
                                if (pit == psuperNodeInfo->outputList.end()) {
                                    if ((*cit)->hierarchical_level > psuperNodeInfo->min_hierarchical_level) {
                                        conflictDetected = true;
                                        break;
                                    }
                                }
                                else if (cnode->target_support_flags & AGO_KERNEL_FLAG_GPU_INTEG_M2R) {
                                    // can't gather from output from the same supernode
                                    conflictDetected = true;
                                    break;
                                }
                            }
                        }
                        if (!conflictDetected) {
                            auto cgroup = cnode->attr_affinity.group;
                            psuperNodeInfo->integ_flags |= csuperNodeInfo->integ_flags;
                            psuperNodeInfo->min_hierarchical_level = min(psuperNodeInfo->min_hierarchical_level, csuperNodeInfo->min_hierarchical_level);
                            psuperNodeInfo->max_hierarchical_level = max(psuperNodeInfo->max_hierarchical_level, csuperNodeInfo->max_hierarchical_level);
                            for (auto it = csuperNodeInfo->nodeList.begin(); it != csuperNodeInfo->nodeList.end(); it++) {
                                (*it)->attr_affinity.group = pnode->attr_affinity.group;
                                psuperNodeInfo->nodeList.push_back(*it);
                            }
                            for (auto it = csuperNodeInfo->inputList.begin(); it != csuperNodeInfo->inputList.end(); it++)
                                psuperNodeInfo->inputList.push_back(*it);
                            for (auto it = csuperNodeInfo->outputList.begin(); it != csuperNodeInfo->outputList.end(); it++)
                                psuperNodeInfo->outputList.push_back(*it);
                            groupInfo.erase(cgroup);
                            delete csuperNodeInfo;
                            csuperNodeInfo = nullptr;
                            break;
                        }
                    }
                }
            }
        }
    }
    // release
    for (auto it = groupInfo.begin(); it != groupInfo.end(); it++) {
        delete it->second;
    }
#if _DEBUG
    // count number of CPU & GPU scheduled nodes
    int nodeCpuCount = 0, nodeGpuCount = 0;
    for (auto node = graph->nodeList.head; node; node = node->next) {
        if (node->attr_affinity.device_type == AGO_KERNEL_FLAG_DEVICE_GPU)
            nodeGpuCount++;
        else
            nodeCpuCount++;
    }
    agoAddLogEntry(NULL, VX_SUCCESS, "OK: OpenVX scheduling %d nodes on CPU and %d nodes on GPU\n", nodeCpuCount, nodeGpuCount);
#endif
    return 0;
}
#endif

int agoOptimizeDramaAlloc(AgoGraph * agraph)
{
    // return success if there is nothing to do
    if (!agraph->nodeList.head)
        return 0;

    // make sure all buffers are properly checked and updated
    for (AgoData * adata = agraph->dataList.head; adata; adata = adata->next) {
        if (!adata->buffer && agoDataSanityCheckAndUpdate(adata)) {
            return -1;
        }
    }
    for (AgoData * adata = agraph->ref.context->dataList.head; adata; adata = adata->next) {
        if (!adata->buffer && agoDataSanityCheckAndUpdate(adata)) {
            return -1;
        }
    }

    // compute image valid rectangles
    if (agoPrepareImageValidRectangleBuffers(agraph)) {
        return -1;
    }
    if (agoComputeImageValidRectangleOutputs(agraph)) {
        return -1;
    }

    // set default target assignments
    if (agoOptimizeDramaAllocSetDefaultTargets(agraph) < 0) {
        return -1;
    }

#if (ENABLE_OPENCL || ENABLE_HIP)
    if (!(agraph->optimizer_flags & AGO_GRAPH_OPTIMIZER_FLAG_NO_SUPERNODE_MERGE)) {
        // merge super nodes
        if (agoOptimizeDramaAllocMergeSuperNodes(agraph) < 0) {
            return -1;
        }
    }
    // allocate GPU resources
    if (agoOptimizeDramaAllocGpuResources(agraph) < 0) {
        return -1;
    }
#endif
    // remove unused data
    if (agoOptimizeDramaAllocRemoveUnusedData(agraph)) return -1;

    // make sure all buffers are allocated and initialized
    for (AgoData * adata = agraph->dataList.head; adata; adata = adata->next) {
        if (agoAllocData(adata)) {
            vx_char name[1024]; agoGetDataName(name, adata);
            agoAddLogEntry(&adata->ref, VX_FAILURE, "ERROR: agoOptimizeDramaAlloc: data allocation failed for %s\n", name);
            return -1;
        }
    }
    for (AgoData * adata = agraph->ref.context->dataList.head; adata; adata = adata->next) {
        if (agoAllocData(adata)) {
            vx_char name[1024]; agoGetDataName(name, adata);
            agoAddLogEntry(&adata->ref, VX_FAILURE, "ERROR: agoOptimizeDramaAlloc: data allocation failed for %s\n", name);
            return -1;
        }
    }

    return 0;
}