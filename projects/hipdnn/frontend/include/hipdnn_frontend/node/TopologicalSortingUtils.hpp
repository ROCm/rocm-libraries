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

std::vector<int> computeInDegrees(const GraphStructure& structure)
{
    size_t nodeCount = structure.adjacencyList.size(); //_sub_nodes.size();
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

int checkForDisconnectedComponents(const GraphStructure& structure)
{
    size_t nodeCount = structure.adjacencyList.size(); //_sub_nodes.size();
    std::unordered_set<size_t> visited;
    int componentCount = 0;

    // Build undirected graph for connectivity check
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

std::vector<size_t> performTopologicalSort(const GraphStructure& structure)
{
    size_t nodeCount = structure.adjacencyList.size(); //_sub_nodes.size();
    std::queue<size_t> zeroInDegree;
    std::vector<size_t> topologicalOrder;
    std::vector<int> inDegrees = computeInDegrees(structure);

    // Initialize queue with all nodes that have no incoming edges
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

        // For each neighbor, decrement in-degree
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

bool detectCycle(const std::vector<size_t>& topologicalOrder, const GraphStructure& structure)
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
