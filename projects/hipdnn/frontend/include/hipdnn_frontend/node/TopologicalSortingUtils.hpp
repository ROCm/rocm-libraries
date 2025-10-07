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

inline std::vector<size_t> performTopologicalSort(const GraphStructure& structure)
{
    size_t nodeCount = structure.adjacencyList.size();
    std::queue<size_t> zeroInDegree;
    std::vector<size_t> topologicalOrder;
    std::vector<int> inDegrees = computeInDegrees(structure);

    // Find all source nodes (in-degree 0)
    for(size_t i = 0; i < nodeCount; ++i)
    {
        if(inDegrees[i] == 0)
        {
            zeroInDegree.push(i);
        }
    }

    // Process nodes in topological order
    while(!zeroInDegree.empty())
    {
        size_t current = zeroInDegree.front();
        zeroInDegree.pop();
        topologicalOrder.push_back(current);

        for(auto neighbor : structure.adjacencyList[current])
        {
            inDegrees[neighbor]--;
            if(inDegrees[neighbor] == 0)
            {
                zeroInDegree.push(neighbor);
            }
        }
    }

    return topologicalOrder;
}

inline int countConnectedComponents(const GraphStructure& structure)
{
    size_t nodeCount = structure.adjacencyList.size();

    if(nodeCount == 0)
    {
        return 0;
    }

    // Build undirected version of the graph for connectivity checking
    std::vector<std::vector<size_t>> undirectedGraph(nodeCount);
    for(size_t i = 0; i < nodeCount; ++i)
    {
        for(auto j : structure.adjacencyList[i])
        {
            undirectedGraph[i].push_back(j);
            undirectedGraph[j].push_back(i);
        }
    }

    // DFS to find connected components
    std::unordered_set<size_t> visited;
    int componentCount = 0;

    for(size_t start = 0; start < nodeCount; ++start)
    {
        if(visited.find(start) != visited.end())
        {
            continue;
        }

        componentCount++;
        std::stack<size_t> stack;
        stack.push(start);

        while(!stack.empty())
        {
            size_t current = stack.top();
            stack.pop();

            if(visited.find(current) != visited.end())
            {
                continue;
            }

            visited.insert(current);

            for(auto neighbor : undirectedGraph[current])
            {
                if(visited.find(neighbor) == visited.end())
                {
                    stack.push(neighbor);
                }
            }
        }
    }

    return componentCount;
}

inline bool detectCycle(const std::vector<size_t>& topologicalOrder,
                        const GraphStructure& structure)
{
    size_t nodeCount = structure.adjacencyList.size();

    if(topologicalOrder.size() == nodeCount)
    {
        //No cycle detected
        return false;
    }

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

    return true;
}

inline TopologicalSortResult
    performTopologicalSortWithComponentDetection(const GraphStructure& structure)
{
    auto topologicalOrder = performTopologicalSort(structure);
    int componentCount = countConnectedComponents(structure);
    bool hasCycle = detectCycle(topologicalOrder, structure);

    return {topologicalOrder, componentCount, hasCycle};
}

}
