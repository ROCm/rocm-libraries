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

#include <Tensile/Debug.hpp>
#include <Tensile/DecisionTreeLibrary.hpp>

#include <cstddef>
#include <unordered_set>

namespace Tensile
{
    namespace Serialization
    {
        template <typename Key, typename Value, typename ReturnValue, typename IO>
        struct MappingTraits<DecisionTree::Tree<Key, Value, ReturnValue>, IO>
        {
            using Tree = DecisionTree::Tree<Key, Value, ReturnValue>;
            using iot  = IOTraits<IO>;

            static void mapping(IO& io, Tree& tree)
            {
                iot::mapRequired(io, "tree", tree.tree);
                iot::mapRequired(io, "value", tree.value);
            }
            const static bool flow = false;
        };

        template <typename Key, typename Object, typename Value, typename ReturnValue, typename IO>
        struct MappingTraits<DecisionTree::BasicForest<Key, Object, Value, ReturnValue>, IO>
        {
            using Forest = DecisionTree::BasicForest<Key, Object, Value, ReturnValue>;
            using iot    = IOTraits<IO>;

            static void mapping(IO& io, Forest& lib)
            {
                int32_t index = -1;
                iot::mapRequired(io, "trees", lib.trees);
                iot::mapRequired(io, "nullValue", lib.nullValue);
            }

            const static bool flow = false;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<DecisionTreeLibrary<MyProblem, MySolution>, IO>
        {
            using Library  = DecisionTreeLibrary<MyProblem, MySolution>;
            using Features = typename Library::Forest::Features;
            using Element  = typename Library::Element;

            using iot = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                Features features;
                if(iot::outputting(io))
                {
                    features = lib.forest->features;
                }
                iot::mapRequired(io, "features", features);

                bool success = false;
                if(features.size() == 0)
                    iot::setError(io, "Tree(s) must have at least one feature.");
                else if(features.size() == 1)
                    success = mappingKey<std::array<float, 1>>(io, lib, features);
                else if(features.size() == 2)
                    success = mappingKey<std::array<float, 2>>(io, lib, features);
                else if(features.size() == 3)
                    success = mappingKey<std::array<float, 3>>(io, lib, features);
                else if(features.size() == 4)
                    success = mappingKey<std::array<float, 4>>(io, lib, features);
                else if(features.size() == 5)
                    success = mappingKey<std::array<float, 5>>(io, lib, features);
                else if(features.size() == 6)
                    success = mappingKey<std::array<float, 6>>(io, lib, features);
                else if(features.size() == 7)
                    success = mappingKey<std::array<float, 7>>(io, lib, features);
                else if(features.size() == 8)
                    success = mappingKey<std::array<float, 8>>(io, lib, features);
                else if(features.size() == 9)
                    success = mappingKey<std::array<float, 9>>(io, lib, features);
                else if(features.size() == 10)
                    success = mappingKey<std::array<float, 10>>(io, lib, features);

                if(!success)
                    success = mappingKey<std::vector<float>>(io, lib, features);

                if(!success)
                    iot::setError(io, "Can't write out key: wrong type.");
            }

            template <typename Key>
            static bool mappingKey(IO& io, Library& lib, Features const& features)
            {
                using Forest = DecisionTree::
                    BasicForest<Key, MyProblem, Element, std::shared_ptr<MySolution>>;

                std::shared_ptr<Forest> forest;

                if(iot::outputting(io))
                {
                    forest = std::dynamic_pointer_cast<Forest>(lib.forest);
                    if(!forest)
                        return false;
                }
                else
                {
                    forest           = std::make_shared<Forest>();
                    forest->features = features;
                    lib.forest       = forest;
                }

                MappingTraits<Forest, IO>::mapping(io, *forest);

                return true;
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<DecisionTree::Node, IO>
        {
            using Node = typename DecisionTree::Node;
            using iot  = IOTraits<IO>;

            static void mapping(IO& io, Node& node)
            {
                iot::mapRequired(io, "featureIdx", node.featureIdx);
                iot::mapRequired(io, "threshold", node.threshold);
                iot::mapRequired(io, "nextIdxLTE", node.nextIdxLTE);
                iot::mapRequired(io, "nextIdxGT", node.nextIdxGT);
            }

            const static bool flow = true;
        };
    } // namespace Serialization
} // namespace Tensile
