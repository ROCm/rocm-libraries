/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <array>
#include <functional>
#include <vector>

#include <Tensile/MLFeatures.hpp>
#include <Tensile/ProblemKey.hpp>

namespace Tensile
{
    /**
     * \ingroup Tensile
     * \defgroup DecisionTree Decision Tree
     *
     * @brief Tree based decisions on a list of Feature values
     *
     * Generic decision tress for deciding on an object based on a list of Features
     * derived from the object. Used for DecisionTreeLibrary.
     */

    /**
     * \ingroup DecisionTree
     */
    namespace DecisionTree
    {
        struct Node
        {
            int   featureIdx; // Index into feature array
            float threshold; // Decision threshold value
            int   nextIdxLTE; // Next node index if val <= threshold
            int   nextIdxGT; // Next node index if val > threshold
        };

        enum ReservedIdxs : int
        {
            IDX_RETURN_FALSE = -1,
            IDX_RETURN_TRUE  = -2,
        };

        /**
         * @brief Generic decision tree
         *
         * @tparam Key type used for deciding
         * @tparam Value type managed by tree
         * @tparam ReturnValue type returned by tree
         */
        template <typename Key, typename Value, typename ReturnValue>
        struct Tree
        {
            using Transform = std::function<ReturnValue(Value)>;

            Tree() = default;
            Tree(std::vector<Node> tree)
                : tree(std::move(tree))
            {
            }

            bool predict(Key const& key) const
            {
                int  nodeIdx  = 0;
                int  treeSize = tree.size();
                Node currentNode;

                while(nodeIdx < treeSize)
                {
                    currentNode    = tree[nodeIdx];
                    bool branchLTE = key[currentNode.featureIdx] <= currentNode.threshold;
                    nodeIdx        = branchLTE ? currentNode.nextIdxLTE : currentNode.nextIdxGT;

                    if(nodeIdx == IDX_RETURN_FALSE)
                        return false;
                    if(nodeIdx == IDX_RETURN_TRUE)
                        return true;
                }

                throw std::runtime_error("Decision Tree out of bounds error.");
                return false;
            }

            virtual ReturnValue getSolution(Transform transform) const
            {
                return transform(value);
            }

            bool valid(bool verbose = false) const
            {
                size_t treeSize = tree.size();
                Node   currentNode;
                bool   has_true = false;
                bool   valid    = true;

                if(treeSize == 0)
                {
                    if(verbose)
                    {
                        std::cout << "Tree invalid: no nodes." << std::endl;
                    }
                    return false;
                }

                if(treeSize > ((size_t)std::numeric_limits<signed int>::max() + 1))
                {
                    /* Restrict size to +ve int range, -ve idxs for reserved values */
                    if(verbose)
                    {
                        std::cout << "Tree invalid: too many nodes." << std::endl;
                    }
                    return false;
                }

                // Check for any invalid nodes
                for(int nodeIdx = 0; nodeIdx < treeSize; nodeIdx++)
                {
                    currentNode = tree[nodeIdx];

                    // Avoid OOB on feature array
                    if((currentNode.featureIdx < 0)
                       || (currentNode.featureIdx >= std::tuple_size<Key>::value))
                    {
                        if(verbose)
                        {
                            std::cout << "Node " << std::to_string(nodeIdx)
                                      << " invalid: Unrecognised type '"
                                      << std::to_string(currentNode.featureIdx) << "'" << std::endl;
                        }
                        valid = false;
                    }

                    // Check next indices
                    std::array<int, 2> nextIdxs = {currentNode.nextIdxLTE, currentNode.nextIdxGT};
                    for(auto nextIdx : nextIdxs)
                    {
                        if(nextIdx == IDX_RETURN_TRUE)
                        {
                            has_true = true;
                        }
                        else
                        {
                            if(nextIdx != IDX_RETURN_FALSE)
                            {
                                // Avoid OOB on tree
                                if((nextIdx < 0) || (nextIdx >= treeSize))
                                {
                                    if(verbose)
                                    {
                                        std::cout << "Node " << std::to_string(nodeIdx)
                                                  << " invalid: child index OOB" << std::endl;
                                    }
                                    valid = false;
                                }
                                // Avoid circular trees
                                if(nextIdx <= nodeIdx)
                                {
                                    if(verbose)
                                    {
                                        std::cout << "Node " << std::to_string(nodeIdx)
                                                  << " invalid: potentially circular tree"
                                                  << std::endl;
                                    }
                                    valid = false;
                                }
                            }
                        }
                    }
                }

                if(!has_true)
                {
                    if(verbose)
                    {
                        std::cout << "Tree invalid: no 'true' nodes." << std::endl;
                    }
                    valid = false;
                }

                return valid;
            }

            std::vector<Node> tree;
            Value             value;
        };

        /**
         * @brief Abstract base class for a group of decision trees
         *
         * @tparam Object type used to query trees
         * @tparam Value type managed by trees
         * @tparam ReturnValue type returned by trees
         */
        template <typename Object, typename Value, typename ReturnValue>
        struct Forest
        {
            using Features  = std::vector<std::shared_ptr<MLFeatures::MLFeature<Object>>>;
            using Transform = std::function<ReturnValue(Value)>;

            Forest() = default;
            Forest(Features const& features)
                : features(features)
            {
            }

            virtual ~Forest() = default;

            virtual ReturnValue findBestMatch(Object const& problem, Transform transform) const = 0;

            virtual std::set<ReturnValue> matchesInOrder(Object const& problem,
                                                         Transform     transform) const = 0;

            virtual std::string description() const = 0;

            Features features;
        };

        /**
         * @brief Forest that returns first successful prediction from managed trees
         *
         * @tparam Key type used by trees for deciding
         * @tparam Object type used to query trees
         * @tparam Value type managed by trees
         * @tparam ReturnValue type returned by trees
         */
        template <typename Key, typename Object, typename Value, typename ReturnValue>
        struct BasicForest : public Forest<Object, Value, ReturnValue>
        {
            using Base      = Forest<Object, Value, ReturnValue>;
            using Tree      = Tree<Key, Value, ReturnValue>;
            using Transform = typename Base::Transform;
            using Features  = typename Base::Features;

            BasicForest() {}

            BasicForest(Features const& features)
                : Base(features)
            {
            }

            virtual ReturnValue findBestMatch(Object const& problem,
                                              Transform     transform) const override
            {
                bool debug = Debug::Instance().getSolutionSelectionTrace();

                Key key = ProblemKey::keyForProblem<Key, Object, float>(problem, this->features);

                if(debug)
                {
                    std::cout << "Forest " << this->description() << std::endl;
                    std::cout << "Entering solution selection evaluation loop. Searching forest."
                              << std::endl;
                }

                for(Tree const& tree : trees)
                {
                    ReturnValue rv = tree.getSolution(transform);
                    if(rv != nullptr)
                    {
                        if(debug)
                        {
                            std::cout << "Running predict for kernel: ";
                            std::cout << rv->KernelName();
                            std::cout << " (Library Index: " << rv->libraryLogicIndex;
                            std::cout << ")" << std::endl;
                        }

                        bool result = tree.predict(key);

                        if(debug)
                        {
                            std::cout << "Prediction evaluation result is: ";
                            std::cout << result << std::endl;
                        }

                        if(result)
                        {
                            if(debug)
                                std::cout << "found valid kernel exiting forest evaluation loop"
                                          << std::endl;
                            return rv;
                        }
                    }
                }

                if(debug)
                {
                    std::cout << "Failed to find a valid kernel after searching the full ensamble. "
                                 "will return the fallback kernel."
                              << std::endl;

                    if(transform(nullValue) == nullptr)
                    {
                        std::cout << "No fallback kerenel is defined. Will return a null pointer";
                        std::cout << std::endl;
                    }
                    else
                    {
                        std::cout << "Returning the fallback kernel: "
                                  << transform(nullValue)->KernelName();
                        std::cout << " (Library Index: " << transform(nullValue)->libraryLogicIndex;
                        std::cout << ")" << std::endl;
                    }
                }

                // The nullVallue (fallback) is now a soltuion library that can containe a nullporter
                // as a solution (indecating that no fallback solution is defined in the logic).
                return transform(nullValue);
            }

            virtual std::set<ReturnValue> matchesInOrder(Object const& problem,
                                                         Transform     transform) const override
            {
                std::set<ReturnValue> rv;
                for(Tree const& tree : trees)
                {
                    rv.insert(tree.getSolution(transform));
                }

                return rv;
            }

            virtual std::string description() const override
            {
                return concatenate(
                    "Forest: Features: ", this->features, ", ", trees.size(), " tree(s)");
            }

            std::vector<Tree> trees;
            Value             nullValue;
        };
    } // namespace DecisionTree
} // namespace Tensile
