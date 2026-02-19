/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2026 AMD ROCm(TM) Software
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

#include <rocRoller/CodeGen/LoadStoreTileGenerator.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief TODO
         */
        class ModelAddresses : public GraphTransform
        {
        public:
            ModelAddresses(ContextPtr context);

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override;

        private:
            enum class LDSDirection
            {
                Load,
                Store
            };

            Generator<size_t>
                getLDSAddressesImpl(KernelGraph&                                     graph,
                                    int                                              tag,
                                    LoadStoreTileGenerator::LoadStoreTileInfo const& info,
                                    LDSDirection                                     direction);

            template <typename Op>
            Generator<size_t> getLDSAddresses(KernelGraph& graph, int tag, Op const& op);
            void              setup();
            void              setWorkgroup(uint offset, uint value);
            void              setWorkitem(uint offset, uint value);

            ContextPtr                               m_context;
            KernelArguments                          arguments;
            std::array<uint, 3>                      workgroupOffset, workitemOffset;
            std::array<Expression::ExpressionPtr, 3> kernelWorkgroupIndexes, kernelWorkitemIndexes;
            std::vector<uint8_t>                     rawArguments;
            RuntimeArguments                         runtimeArguments;
        };
    }
}
