// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include <rocRoller/CodeGen/LoadStoreTileGenerator.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Graph transform that models LDS addresses for LoadLDSTile and StoreLDSTile
         * operations.
         *
         * Simulates workgroup/workitem index expressions to compute per-workitem byte offsets into
         * LDS for each tile operation, storing the normalized results in the graph's
         * modelledAddresses map. This is used to analyze LDS access patterns without modifying
         * the real kernel context.
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

            std::vector<size_t>
                getLDSAddressesImpl(KernelGraph&                                     graph,
                                    int                                              tag,
                                    LoadStoreTileGenerator::LoadStoreTileInfo const& info,
                                    LDSDirection                                     direction);

            template <typename Op>
            std::vector<size_t> getLDSAddresses(KernelGraph& graph, int tag, Op const& op);
            void                setup();
            void                setWorkgroup(uint dim, uint value);
            void                setWorkitem(uint dim, uint value);

            ContextPtr                               m_context;
            KernelArguments                          arguments;
            std::array<uint, 3>                      workgroupOffset, workitemOffset;
            std::array<Expression::ExpressionPtr, 3> kernelWorkgroupIndexes, kernelWorkitemIndexes;
            std::vector<uint8_t>                     rawArguments;
            RuntimeArguments                         runtimeArguments;
        };
    }
}
