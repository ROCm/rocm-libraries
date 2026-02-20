/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <algorithm>
#include <concepts>
#include <queue>
#include <set>

#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    template <typename T, typename Result, typename... Args>
    concept CInvokableTo
        = std::invocable<T, Args...> && std::convertible_to<Result,
                                                            std::invoke_result_t<T, Args...>>;

    template <std::totally_ordered Node, std::totally_ordered Category>
    Generator<Instruction>
        generateNodes(Scheduling::SchedulerPtr                        scheduler,
                      std::set<Node>                                  nodes,
                      std::set<Node>&                                 completedNodes,
                      CInvokableTo<Generator<Instruction>, Node> auto generateNode,
                      CInvokableTo<bool, Node> auto                   nodeIsReady,
                      CInvokableTo<Category, Node> auto               nodeCategory,
                      CInvokableTo<size_t, Category> auto             categoryLimit,
                      CInvokableTo<bool, Node, Node> auto             comparePriorities)
    {
        std::set<Category> allCategories;
        for(auto node : nodes)
            allCategories.insert(nodeCategory(node));

        std::map<Category, std::set<Node>> generating;

        using PQ = std::priority_queue<Node, std::vector<Node>, decltype(comparePriorities)>;
        std::map<Category, PQ> ready;

        for(auto category : allCategories)
            ready.emplace(category, PQ{comparePriorities});

        auto fillReady = [&]() {
            for(auto iter = nodes.begin(); iter != nodes.end();)
            {
                auto idx = *iter;

                if(nodeIsReady(idx))
                {
                    ready.at(nodeCategory(idx)).push(idx);
                    iter = nodes.erase(iter);
                }
                else
                {
                    ++iter;
                }
            }
        };

        auto generate = [&](Node idx) -> Generator<Instruction> {
            auto category = nodeCategory(idx);

            co_yield generateNode(idx);

            completedNodes.insert(idx);
            generating[category].erase(idx);
        };

        auto anyRemainingNodes = [&]() -> bool {
            if(!nodes.empty())
                return true;

            for(auto const& [category, readyQueue] : ready)
            {
                if(!readyQueue.empty())
                    return true;
            }

            for(auto const& [category, generatingSet] : generating)
            {
                AssertFatal(generatingSet.empty());
            }

            return false;
        };

        while(anyRemainingNodes())
        {
            std::vector<Generator<Instruction>> schedulable;

            auto fillSchedulable = [&]() {
                for(auto& [category, readyQueue] : ready)
                {
                    auto limit = categoryLimit(category);
                    AssertFatal(limit > 0);

                    while(!readyQueue.empty() && generating[category].size() < limit)
                    {
                        auto idx = readyQueue.top();
                        readyQueue.pop();

                        generating[category].insert(idx);

                        schedulable.push_back(generate(idx));
                    }
                }
            };

            fillReady();
            fillSchedulable();

            if(!scheduler->supportsAddingStreams())
            {
                co_yield (*scheduler)(schedulable);
            }
            else
            {
                auto generator    = (*scheduler)(schedulable);
                auto prevDoneSize = completedNodes.size();

                for(auto iter = generator.begin(); iter != generator.end(); ++iter)
                {
                    co_yield *iter;

                    if(prevDoneSize != completedNodes.size())
                    {
                        fillReady();

                        prevDoneSize = completedNodes.size();
                    }

                    fillSchedulable();
                }
            }
        }
    }
}
