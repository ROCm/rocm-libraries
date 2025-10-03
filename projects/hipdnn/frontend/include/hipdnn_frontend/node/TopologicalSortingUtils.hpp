// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <queue>
#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

#include <hipdnn_frontend/Utilities.hpp>

namespace hipdnn_frontend
{
struct GraphStructure
{
    std::vector<std::vector<size_t>> adjacencyList;
};

struct TopologicalSortResult
{
    std::vector<size_t> order;
    int componentCount;
    bool hasCycle;
};

inline std::vector<int> computeInDegrees(const GraphStructure& structure)
{
    size_t nodeCount = structure.adjacencyList.size();
    std::vector<int> inDegrees(nodeCount, 0);
    for(size_t i = 0; i < nodeCount; ++i)
    {
        for(auto neighbor : structure.adjacencyList[i])
        {
            inDegrees[neighbor]++;
        }
    }
    return inDegrees;
}

inline TopologicalSortResult
    performTopologicalSortWithComponentDetection(const GraphStructure& structure)
{
    size_t nodeCount = structure.adjacencyList.size();
    std::queue<size_t> zeroInDegree;
    std::vector<size_t> topologicalOrder;
    std::vector<int> inDegrees = computeInDegrees(structure);

    std::vector<int> componentId(nodeCount, -1);
    int currentComponent = 0;

    // Find all source nodes (in-degree 0)
    for(size_t i = 0; i < nodeCount; ++i)
    {
        if(inDegrees[i] == 0)
        {
            zeroInDegree.push(i);
            componentId[i] = currentComponent++;
        }
    }

    int componentCount = currentComponent;

    // Process nodes
    while(!zeroInDegree.empty())
    {
        size_t current = zeroInDegree.front();
        zeroInDegree.pop();
        topologicalOrder.push_back(current);

        int currentComponentId = componentId[current];

        for(auto neighbor : structure.adjacencyList[current])
        {
            // Propagate component ID to neighbors
            if(componentId[neighbor] == -1)
            {
                componentId[neighbor] = currentComponentId;
            }

            inDegrees[neighbor]--;
            if(inDegrees[neighbor] == 0)
            {
                zeroInDegree.push(neighbor);
            }
        }
    }

    bool hasCycle = (topologicalOrder.size() != nodeCount);

    return {topologicalOrder, componentCount, hasCycle};
}

inline bool detectCycle(const std::vector<size_t>& topologicalOrder,
                        const GraphStructure& structure)
{
    size_t nodeCount = structure.adjacencyList.size();

    if(topologicalOrder.size() != nodeCount)
    {
        HIPDNN_FE_LOG_ERROR("Graph contains a cycle - not a DAG. Processed {}/{} nodes",
                            topologicalOrder.size(),
                            nodeCount);

        // Log which nodes are part of the cycle
        std::vector<size_t> cycleNodes;
        std::vector<int> inDegrees = computeInDegrees(structure);

        // Recalculate which nodes weren't processed
        for(auto processed : topologicalOrder)
        {
            for(auto neighbor : structure.adjacencyList[processed])
            {
                inDegrees[neighbor]--;
            }
        }

        for(size_t i = 0; i < nodeCount; ++i)
        {
            if(inDegrees[i] > 0)
            {
                cycleNodes.push_back(i);
            }
        }

        if(!cycleNodes.empty())
        {
            std::string nodeList;
            for(auto idx : cycleNodes)
            {
                if(!nodeList.empty())
                {
                    nodeList += ", ";
                }
                nodeList += std::to_string(idx);
            }

            HIPDNN_FE_LOG_ERROR("Nodes involved in cycle: [{}]", nodeList);
        }

        return true; // Cycle detected
    }

    return false; // No cycle
}

}
