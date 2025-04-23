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

int agoOptimizeDramaCheckArgs(AgoGraph * agraph)
{
	int astatus = 0;
	for (AgoNode * anode = agraph->nodeList.head; anode; anode = anode->next)
	{
		AgoKernel * akernel = anode->akernel;
		for (vx_uint32 arg = 0; arg < AGO_MAX_PARAMS; arg++) {
			if (!anode->paramList[arg] || (arg >= anode->paramCount))
			{
				if (((akernel->argConfig[arg] & AGO_KERNEL_ARG_OPTIONAL_FLAG) == 0) && ((akernel->argConfig[arg] & (AGO_KERNEL_ARG_INPUT_FLAG | AGO_KERNEL_ARG_OUTPUT_FLAG)) != 0))
				{
					agoAddLogEntry(&akernel->ref, VX_FAILURE, "ERROR: agoOptimizeDramaCheckArgs: kernel %s: missing argument#%d\n", akernel->name, arg);
					astatus = -1;
				}
			}
			else if ((akernel->argConfig[arg] & (AGO_KERNEL_ARG_INPUT_FLAG | AGO_KERNEL_ARG_OUTPUT_FLAG)) == 0)
			{
				agoAddLogEntry(&akernel->ref, VX_FAILURE, "ERROR: agoOptimizeDramaCheckArgs: kernel %s: unexpected argument#%d\n", akernel->name, arg);
				astatus = -1;
			}
		}
	}
	return astatus;
}

static bool DetectRectOverlap(vx_rectangle_t& a, vx_rectangle_t& b)
{
	vx_rectangle_t c;
	c.start_x = max(a.start_x, b.start_x);
	c.start_y = max(a.start_y, b.start_y);
	c.end_x = min(a.end_x, b.end_x);
	c.end_y = min(a.end_y, b.end_y);
	return (c.start_x < c.end_x) && (c.start_y < c.end_y) ? true : false;
}

void agoOptimizeDramaGetDataUsageOfROI(AgoGraph * agraph, AgoData * roiMasterImage, vx_uint32& inputUsageCount, vx_uint32& outputUsageCount, vx_uint32& inoutUsageCount)
{
	std::list<vx_rectangle_t> rectList;
	vx_uint32 outputUsageCount_ = 0;
	for (int isVirtual = 0; isVirtual <= 1; isVirtual++) {
		for (AgoData * data = isVirtual ? agraph->ref.context->dataList.head : agraph->dataList.head; data; data = data->next) {
			if (data->ref.type == VX_TYPE_IMAGE && data->u.img.isROI && data->u.img.roiMasterImage == roiMasterImage) {
				inputUsageCount += data->inputUsageCount;
				inoutUsageCount += data->inoutUsageCount;
				if (data->outputUsageCount > 0) {
					if (outputUsageCount == 0) {
						bool detectedOverlap = false;
						for (auto it = rectList.begin(); it != rectList.end(); it++) {
							if (DetectRectOverlap(*it, data->u.img.rect_roi)) {
								detectedOverlap = true;
								break;
							}
						}
						rectList.push_back(data->u.img.rect_roi);
						if (detectedOverlap) {
							outputUsageCount_ += data->outputUsageCount;
						}
						else {
							outputUsageCount_ = max(outputUsageCount_, data->outputUsageCount);
						}
					}
					else {
						outputUsageCount_ += data->outputUsageCount;
					}
				}
			}
		}
	}
	outputUsageCount += outputUsageCount_;
}

void agoOptimizeDramaMarkDataUsageOfROI(AgoGraph * agraph, AgoData * roiMasterImage, vx_uint32 inputUsageCount, vx_uint32 outputUsageCount, vx_uint32 inoutUsageCount)
{
	for (int isVirtual = 0; isVirtual <= 1; isVirtual++) {
		for (AgoData * data = isVirtual ? agraph->ref.context->dataList.head : agraph->dataList.head; data; data = data->next) {
			if (data->ref.type == VX_TYPE_IMAGE && data->u.img.isROI && data->u.img.roiMasterImage == roiMasterImage) {
				data->inputUsageCount = inputUsageCount;
				data->outputUsageCount = outputUsageCount;
				data->inoutUsageCount = inoutUsageCount;
			}
		}
	}
}

void agoOptimizeDramaMarkDataUsage(AgoGraph * agraph)
{
	// reset the data usage in all data elements
	for (int isVirtual = 0; isVirtual <= 1; isVirtual++) {
		for (AgoData * data = isVirtual ? agraph->ref.context->dataList.head : agraph->dataList.head; data; data = data->next) {
			data->inputUsageCount = 0;
			data->outputUsageCount = 0;
			data->inoutUsageCount = 0;
			for (vx_uint32 i = 0; i < data->numChildren; i++) {
				AgoData * idata = data->children[i];
				if (idata) {
					idata->inputUsageCount = 0;
					idata->outputUsageCount = 0;
					idata->inoutUsageCount = 0;
					for (vx_uint32 j = 0; j < idata->numChildren; j++) {
						AgoData * jdata = idata->children[j];
						if (jdata) {
							jdata->inputUsageCount = 0;
							jdata->outputUsageCount = 0;
							jdata->inoutUsageCount = 0;
							for (vx_uint32 k = 0; k < jdata->numChildren; k++) {
								AgoData * kdata = jdata->children[k];
								if (kdata) {
									kdata->inputUsageCount = 0;
									kdata->outputUsageCount = 0;
									kdata->inoutUsageCount = 0;
								}
							}
						}
					}
				}
			}
		}
	}
	// update the data usage by this graph
	for (AgoNode * anode = agraph->nodeList.head; anode; anode = anode->next) {
		AgoKernel * akernel = anode->akernel;
		for (vx_uint32 arg = 0; arg < anode->paramCount; arg++) {
			AgoData * adata = anode->paramList[arg];
			if (adata) {
				// mark the usage of the data item
				if ((akernel->argConfig[arg] & (AGO_KERNEL_ARG_INPUT_FLAG | AGO_KERNEL_ARG_OUTPUT_FLAG)) == (AGO_KERNEL_ARG_INPUT_FLAG | AGO_KERNEL_ARG_OUTPUT_FLAG))
					adata->inoutUsageCount++;
				else if (akernel->argConfig[arg] & AGO_KERNEL_ARG_OUTPUT_FLAG)
					adata->outputUsageCount++;
				else if (akernel->argConfig[arg] & AGO_KERNEL_ARG_INPUT_FLAG)
					adata->inputUsageCount++;
				// get image plane input/output non-usage count to compensate propagation in the next step
				if (akernel->func && adata->ref.type == VX_TYPE_IMAGE && adata->numChildren > 1) {
					if ((akernel->argConfig[arg] & (AGO_KERNEL_ARG_INPUT_FLAG | AGO_KERNEL_ARG_OUTPUT_FLAG)) != (AGO_KERNEL_ARG_INPUT_FLAG | AGO_KERNEL_ARG_OUTPUT_FLAG)) {
						anode->funcExchange[0] = arg;
						for (vx_uint32 plane = 0; plane < adata->numChildren; plane++)
							anode->funcExchange[1 + plane] = 0;
						if (!akernel->func(anode, ago_kernel_cmd_get_image_plane_nonusage)) {
							for (vx_uint32 plane = 0; plane < adata->numChildren; plane++) {
								if (adata->children[plane] && anode->funcExchange[1 + plane]) {
									if (akernel->argConfig[arg] & AGO_KERNEL_ARG_OUTPUT_FLAG)
										adata->children[plane]->outputUsageCount--;
									else if (akernel->argConfig[arg] & AGO_KERNEL_ARG_INPUT_FLAG)
										adata->children[plane]->inputUsageCount--;
								}
							}
						}
					}
				}
			}
		}
	}
	// propagate usage counts from top-level to children (e.g., PYRAMID to IMAGE)
	for (int isVirtual = 0; isVirtual <= 1; isVirtual++) {
		for (AgoData * data = isVirtual ? agraph->ref.context->dataList.head : agraph->dataList.head; data; data = data->next) {
			if (!data->parent) {
				vx_uint32 min_outputUsageCount = INT_MAX;
				for (vx_uint32 i = 0; i < data->numChildren; i++) {
					AgoData * idata = data->children[i];
					if (idata) {
						idata->outputUsageCount += data->outputUsageCount;
						idata->inoutUsageCount += data->inoutUsageCount;
						idata->inputUsageCount += data->inputUsageCount;
						vx_uint32 imin_outputUsageCount = INT_MAX;
						for (vx_uint32 j = 0; j < idata->numChildren; j++) {
							AgoData * jdata = idata->children[j];
							if (jdata) {
								jdata->outputUsageCount += idata->outputUsageCount;
								jdata->inoutUsageCount += idata->inoutUsageCount;
								jdata->inputUsageCount += idata->inputUsageCount;
								vx_uint32 jmin_outputUsageCount = INT_MAX;
								for (vx_uint32 k = 0; k < jdata->numChildren; k++) {
									AgoData * kdata = jdata->children[k];
									if (kdata) {
										kdata->outputUsageCount += jdata->outputUsageCount;
										kdata->inoutUsageCount += jdata->inoutUsageCount;
										kdata->inputUsageCount += jdata->inputUsageCount;
										// IMPORTANT: parent check is needed to deal with image aliasing inside pyramids (result of agoReplaceDataInGraph)
										if (kdata->parent == jdata && jmin_outputUsageCount > kdata->outputUsageCount) jmin_outputUsageCount = kdata->outputUsageCount;
									}
								}
								if (!jdata->outputUsageCount && jmin_outputUsageCount != INT_MAX) jdata->outputUsageCount = jmin_outputUsageCount;
								// IMPORTANT: parent check is needed to deal with image aliasing inside pyramids (result of agoReplaceDataInGraph)
								if (jdata->parent == idata && imin_outputUsageCount > jdata->outputUsageCount) imin_outputUsageCount = jdata->outputUsageCount;
							}
						}
						if (!idata->outputUsageCount && imin_outputUsageCount != INT_MAX) idata->outputUsageCount = imin_outputUsageCount;
						// IMPORTANT: parent check is needed to deal with image aliasing inside pyramids (result of agoReplaceDataInGraph)
						if (idata->parent == data && min_outputUsageCount > idata->outputUsageCount) min_outputUsageCount = idata->outputUsageCount;
					}
				}
				if (!data->outputUsageCount && min_outputUsageCount != INT_MAX) data->outputUsageCount = min_outputUsageCount;
			}
		}
	}
	// add up ROI data usage
	for (int isVirtual = 0; isVirtual <= 1; isVirtual++) {
		for (AgoData * data = isVirtual ? agraph->ref.context->dataList.head : agraph->dataList.head; data; data = data->next) {
			if (data->ref.type == VX_TYPE_IMAGE && !data->u.img.isROI) {
				agoOptimizeDramaGetDataUsageOfROI(agraph, data, data->inputUsageCount, data->outputUsageCount, data->inoutUsageCount);
				agoOptimizeDramaMarkDataUsageOfROI(agraph, data, data->inputUsageCount, data->outputUsageCount, data->inoutUsageCount);
			}
		}
	}
}

static int agoSetDataHierarchicalLevel(AgoData * data, vx_uint32 hierarchical_level)
{
	data->hierarchical_level = hierarchical_level;
	if(!hierarchical_level) {
		data->hierarchical_life_start = data->hierarchical_life_end = 0;
	}
#if SHOW_DEBUG_HIERARCHICAL_LEVELS
	if (data->hierarchical_level) {
		char name[1024];
		agoGetDataName(name, data);
		printf("DEBUG: HIERARCHICAL DATA %3d %s\n", data->hierarchical_level, name);
	}
#endif
	// propagate hierarchical_level to all of its children (if available)
	for (vx_uint32 child = 0; child < data->numChildren; child++) {
		if (data->children[child]) {
			agoSetDataHierarchicalLevel(data->children[child], hierarchical_level);
		}
	}
	// propagate hierarchical_level to image-ROI master (if available)
	if (data->ref.type == VX_TYPE_IMAGE) {
		if (data->u.img.isROI) {
			if (data->u.img.roiMasterImage && !data->u.img.roiMasterImage->hierarchical_level) {
				agoSetDataHierarchicalLevel(data->u.img.roiMasterImage, hierarchical_level);
			}
		}
		else if (hierarchical_level) {
			for (AgoData * pdata = data->isVirtual ? ((AgoGraph *)data->ref.scope)->dataList.head : data->ref.context->dataList.head; pdata; pdata = pdata->next) {
				if (pdata->ref.type == VX_TYPE_IMAGE && pdata->u.img.isROI && pdata->u.img.roiMasterImage == data && !pdata->hierarchical_level) {
					agoSetDataHierarchicalLevel(pdata, hierarchical_level);
				}
			}
		}
	}
	// propagate hierarchical_level to parent (if possible)
	if (hierarchical_level) {
		if (data->parent) {
			vx_uint32 hierarchical_level_sibling_min = INT_MAX, hierarchical_level_sibling_max = 0;
			for (vx_uint32 child = 0; child < data->parent->numChildren; child++) {
				if (data->parent->children[child]) {
					vx_uint32 hierarchical_level_sibling = data->parent->children[child]->hierarchical_level;
					if (hierarchical_level_sibling_min > hierarchical_level_sibling)
						hierarchical_level_sibling_min = hierarchical_level_sibling;
					if (hierarchical_level_sibling_max < hierarchical_level_sibling)
						hierarchical_level_sibling_max = hierarchical_level_sibling;
				}
			}
			// make sure that all siblings has hierarchical_level the parent hierarchical_level is max of all siblings
			if (hierarchical_level_sibling_min > 0 && hierarchical_level_sibling_max > 0)
				data->parent->hierarchical_level = hierarchical_level_sibling_max;
		}
	}
	return 0;
}

int agoOptimizeDramaComputeGraphHierarchy(AgoGraph * graph)
{
#if SHOW_DEBUG_HIERARCHICAL_LEVELS
	printf("DEBUG: HIERARCHICAL **** *** **************************************\n");
#endif

	agoOptimizeDramaMarkDataUsage(graph);

	////////////////////////////////////////////////
	// make sure that there is only one writer and
	// make sure that virtual buffers always have a writer
	////////////////////////////////////////////////
	for (AgoNode * node = graph->nodeList.head; node; node = node->next)
	{
		node->hierarchical_level = 0;
		for (vx_uint32 arg = 0; arg < node->paramCount; arg++) {
			AgoData * data = node->paramList[arg];
			if (data) {
#if SHOW_DEBUG_HIERARCHICAL_LEVELS
				char name[1024];
				agoGetDataName(name, data);
				printf("DEBUG: DATA USAGE #%d [ %d %d %d ] %s %s\n", arg, data->inputUsageCount, data->outputUsageCount, data->inoutUsageCount, node->akernel->name, name);
#endif
				if (data->outputUsageCount > 1) {
					vx_status status = VX_ERROR_MULTIPLE_WRITERS;
					agoAddLogEntry(&graph->ref, status, "ERROR: vxVerifyGraph: kernel %s: multiple writers for argument#%d (%s)\n", node->akernel->name, arg, data->name.c_str());
					return status;
				}
				else if (data->isVirtual && data->outputUsageCount == 0 && !data->isInitialized) {
					vx_status status = VX_ERROR_MULTIPLE_WRITERS;
					agoAddLogEntry(&graph->ref, status, "ERROR: vxVerifyGraph: kernel %s: no writer/initializer for virtual buffer at argument#%d (%s)\n", node->akernel->name, arg, data->name.c_str());
					return status;
				}
			}
		}
	}

	////////////////////////////////////////////////
	// reset hierarchical_level = 0 for all data
	////////////////////////////////////////////////
	for (int isVirtual = 0; isVirtual <= 1; isVirtual++) {
		for (AgoData * data = isVirtual ? graph->ref.context->dataList.head : graph->dataList.head; data; data = data->next) {
			agoSetDataHierarchicalLevel(data, 0);
		}
	}

	////////////////////////////////////////////////
	// identify object for nodes with hierarchical_level = 1 (head nodes)
	// (i.e., nodes that only take objects not updated by this graph)
	////////////////////////////////////////////////
	for (AgoNode * node = graph->nodeList.head; node; node = node->next)
	{
		for (vx_uint32 arg = 0; arg < node->paramCount; arg++) {
			AgoData * data = node->paramList[arg];
			if (data) {
				if (data->parent && data->parent->ref.type != VX_TYPE_DELAY) 
					data = data->parent;
				vx_uint32 inputUsageCount = data->inputUsageCount;
				vx_uint32 inoutUsageCount = data->inoutUsageCount;
				vx_uint32 outputUsageCount = data->outputUsageCount;
				for (vx_uint32 i = 0; i < data->numChildren; i++) {
					AgoData * idata = data->children[i];
					if (idata) {
						if (outputUsageCount < idata->outputUsageCount) {
							inputUsageCount = idata->inputUsageCount;
							inoutUsageCount = idata->inoutUsageCount;
							outputUsageCount = idata->outputUsageCount;
						}
						for (vx_uint32 j = 0; j < idata->numChildren; j++) {
							AgoData * jdata = idata->children[j];
							if (jdata) {
								if (outputUsageCount < jdata->outputUsageCount) {
									inputUsageCount = jdata->inputUsageCount;
									inoutUsageCount = jdata->inoutUsageCount;
									outputUsageCount = jdata->outputUsageCount;
								}
								for (vx_uint32 k = 0; k < jdata->numChildren; k++) {
									AgoData * kdata = jdata->children[k];
									if (kdata) {
										if (outputUsageCount < kdata->outputUsageCount) {
											inputUsageCount = kdata->inputUsageCount;
											inoutUsageCount = kdata->inoutUsageCount;
											outputUsageCount = kdata->outputUsageCount;
										}
									}
								}
							}
						}
					}
				}
#if 0 // TBD: disabled temporarily as a quick workaround for Birdirectional buffer issue
				if (inoutUsageCount > 0 && inputUsageCount > 0) {
					// can't support a data as input an input parameter as well as bidirectional parameter in a single graph
					printf("ERROR: agoVerifyGraph: detected a buffer used a input parameter as well as bidirectional parameter -- not supported\n");
					return -1;
				}
				else
#endif
				if (outputUsageCount == 0) {
					// mark that this data object can be input to nodes with hierarchical_level = 1
					agoSetDataHierarchicalLevel(data, 1);
				}
			}
		}
	}

	////////////////////////////////////////////////
	// identify nodes for hierarchical_level = 1 (head nodes)
	// (i.e., nodes with hierarchical_level = 1 for all of its inputs)
	////////////////////////////////////////////////
	vx_uint32 num_nodes_marked = 0;
	vx_uint32 num_head_nodes = 0;
	for (AgoNode * node = graph->nodeList.head; node; node = node->next)
	{
		AgoKernel * kernel = node->akernel;
		// a node is a head node if all its inputs have hierarchical_level == 1
		bool is_head_node = true;
		for (vx_uint32 arg = 0; arg < node->paramCount; arg++) {
			AgoData * data = node->paramList[arg];
			if (data && (kernel->argConfig[arg] & AGO_KERNEL_ARG_INPUT_FLAG) && !(data->hierarchical_level == 1))
				is_head_node = false;
		}
		if (is_head_node) {
			// mark that node is a head node
			node->hierarchical_level = 1;
			num_nodes_marked++;
			num_head_nodes++;
#if SHOW_DEBUG_HIERARCHICAL_LEVELS
			printf("DEBUG: HIERARCHICAL NODE %3d %s\n", node->hierarchical_level, node->akernel->name);
#endif
			// set the hierarchical_level of outputs to 2
			for (vx_uint32 arg = 0; arg < node->paramCount; arg++) {
				AgoData * data = node->paramList[arg];
				if (data && (kernel->argConfig[arg] & AGO_KERNEL_ARG_OUTPUT_FLAG))
					agoSetDataHierarchicalLevel(data, node->hierarchical_level + 1);
			}
		}
	}
	if(num_head_nodes == 0){
		vx_status status = VX_ERROR_INVALID_GRAPH;
		vxAddLogEntry(&graph->ref, status, "ERROR: vxVerifyGraph: Cycle: Graph has no head nodes!");
		return status;
	}
	////////////////////////////////////////////////
	// calculate hierarchical_level for rest of the nodes
	////////////////////////////////////////////////
	for (;;)
	{
		bool found_change = false;
		for (AgoNode * node = graph->nodeList.head; node; node = node->next)
		{
			if (node->hierarchical_level == 0) {
				// find min and max hierarchical_level of inputs
				AgoKernel * kernel = node->akernel;
				vx_uint32 hierarchical_level_min = INT_MAX, hierarchical_level_max = 0;
				for (vx_uint32 arg = 0; arg < node->paramCount; arg++) {
					AgoData * data = node->paramList[arg];
					if (data && (kernel->argConfig[arg] & AGO_KERNEL_ARG_INPUT_FLAG)) {
						vx_uint32 hierarchical_level = data->hierarchical_level;
						if (hierarchical_level_min > hierarchical_level)
							hierarchical_level_min = hierarchical_level;
						if (hierarchical_level_max < hierarchical_level)
							hierarchical_level_max = hierarchical_level;
					}
				}
				// check if all inputs have hierarchical_level set
				if (hierarchical_level_min > 0) {
					found_change = true;
					// mark that node is at highest hierarchical_level of all its inputs
					node->hierarchical_level = hierarchical_level_max;
					num_nodes_marked++;
#if SHOW_DEBUG_HIERARCHICAL_LEVELS
					printf("DEBUG: HIERARCHICAL NODE %3d %s\n", node->hierarchical_level, node->akernel->name);
#endif
					// set the hierarchical_level of outputs to (node->hierarchical_level + 1)
					for (vx_uint32 arg = 0; arg < node->paramCount; arg++) {
						AgoData * data = node->paramList[arg];
						if (data && (kernel->argConfig[arg] & AGO_KERNEL_ARG_OUTPUT_FLAG))
							agoSetDataHierarchicalLevel(data, node->hierarchical_level + 1);
					}
				}
			}
		}
		if (!found_change)
			break;
	}
	if (num_nodes_marked != graph->nodeList.count) {
		vx_status status = VX_ERROR_INVALID_GRAPH;
		vxAddLogEntry(&graph->ref, status, "ERROR: vxVerifyGraph: invalid graph: possible cycles? [%d|%d]\n", num_nodes_marked, graph->nodeList.count);
		return status;
	}
	return VX_SUCCESS;
}

void agoOptimizeDramaSortGraphHierarchy(AgoGraph * graph)
{
	if (graph->nodeList.count > 1) {
		for (;;) {
			bool swapped = false;
			AgoNode * prev_node = graph->nodeList.head;
			AgoNode * node = prev_node->next;
			// check for the order of hierarchical_level
			if (node->hierarchical_level < prev_node->hierarchical_level) {
				// swap prev_node and node in the list
				prev_node->next = node->next;
				node->next = prev_node;
				prev_node = node;
				node = prev_node->next;
				graph->nodeList.head = prev_node;
				swapped = true;
			}
			for (; node->next; prev_node = prev_node->next, node = node->next) {
				AgoNode * next_node = node->next;
				// check for the order of hierarchical_level
				if (next_node->hierarchical_level < node->hierarchical_level) {
					// swap node and next_node in the list
					node->next = next_node->next;
					next_node->next = node;
					prev_node->next = next_node;
					node = next_node;
					swapped = true;
				}
			}
			graph->nodeList.tail = node;
			if (!swapped)
				break;
		}
	}
}

int agoOptimizeDrama(AgoGraph * agraph)
{
	// get optimization level requested by user
#if ENABLE_DEBUG_MESSAGES
	agoWriteGraph(agraph, NULL, 0, stdout, "input-to-drama");
#endif
	// perform divide
	if (agoOptimizeDramaCheckArgs(agraph))
		return -1;
	if (!(agraph->optimizer_flags & AGO_GRAPH_OPTIMIZER_FLAG_NO_DIVIDE)) { 
		if(agoOptimizeDramaDivide(agraph)) 
			return -1;
	}
#if ENABLE_DEBUG_MESSAGES
	agoWriteGraph(agraph, NULL, 0, stdout, "after-divide");
#endif
	if (agoOptimizeDramaComputeGraphHierarchy(agraph))
		return -1;
	agoOptimizeDramaSortGraphHierarchy(agraph);

	// perform remove
	if (agoOptimizeDramaCheckArgs(agraph))
		return -1;
	if (agoOptimizeDramaRemove(agraph))
		return -1;
#if ENABLE_DEBUG_MESSAGES
	agoWriteGraph(agraph, NULL, 0, stdout, "after-remove");
#endif
	if (agoOptimizeDramaComputeGraphHierarchy(agraph))
		return -1;
	agoOptimizeDramaSortGraphHierarchy(agraph);

	// perform analyze
	if (agoOptimizeDramaCheckArgs(agraph))
		return -1;
	if (agoOptimizeDramaAnalyze(agraph))
		return -1;
#if ENABLE_DEBUG_MESSAGES
	agoWriteGraph(agraph, NULL, 0, stdout, "after-analyze");
#endif

	// perform merge
	if (agoOptimizeDramaCheckArgs(agraph))
		return -1;
	if (agoOptimizeDramaMerge(agraph))
		return -1;
#if ENABLE_DEBUG_MESSAGES
	agoWriteGraph(agraph, NULL, 0, stdout, "after-merge");
#endif

	// perform alloc
	if (agoOptimizeDramaCheckArgs(agraph))
		return -1;
	if (agoOptimizeDramaAlloc(agraph))
		return -1;
#if ENABLE_DEBUG_MESSAGES
	agoWriteGraph(agraph, NULL, 0, stdout, "after-alloc");
#endif

	return 0;
}