// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-component.hpp"

namespace hipblaslt_jit
{
    namespace
    {
        class CatalogKnowledge final : public TuningKnowledge
        {
        public:
            std::string_view id() const noexcept override
            {
                return "catalog.v1";
            }
            std::vector<CandidateSeed> seeds(const OperationRequest&,
                                             const DeviceTarget& target) const override
            {
                // Declarative parameter shapes, independent of the installed solution library.
                // Each shape retains its wave topology; MT/MI alone cannot reconstruct it.
                static constexpr std::array<std::array<size_t, 4>, 11> tiles{{
                    {32, 32, 2, 2},
                    {64, 32, 2, 2},
                    {32, 64, 2, 2},
                    {64, 64, 2, 2},
                    {128, 64, 2, 2},
                    {64, 128, 2, 2},
                    {128, 128, 2, 2},
                    {256, 128, 2, 2},
                    {128, 256, 2, 2},
                    {256, 256, 2, 2},
                    {128, 16, 4, 1},
                }};
                // gfx90a has no NT modifier; gfx1250 expresses NT via TemporalHint.
                // Keep both at default hints so the model matches emitted loads.
                std::vector<std::array<int, 2>> hints{{0, 0}};
                if(target.isa != "gfx90a" && target.isa != "gfx1250")
                {
                    hints.push_back({4, 0});
                    hints.push_back({0, 4});
                }
                std::vector<CandidateSeed> seeds;
                for(const auto& tile : tiles)
                    seeds.push_back({tile, {{32, 1}, {64, 2}}, hints});
                return seeds;
            }
            std::vector<TuningParameter>
                defaults(const OperationRequest&, const DeviceTarget&, const Candidate&) const override
            {
                return {};
            }
        };
    }

    std::shared_ptr<const TuningKnowledge> makeCatalogKnowledge()
    {
        return std::make_shared<const CatalogKnowledge>();
    }
}
