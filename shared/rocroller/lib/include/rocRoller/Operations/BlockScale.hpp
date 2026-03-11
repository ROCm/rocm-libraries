// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Block scale MX datatypes command.
 */

#pragma once

#include <optional>
#include <unordered_set>
#include <vector>

#include <rocRoller/Operations/BlockScale_fwd.hpp>
#include <rocRoller/Operations/Operation.hpp>
#include <rocRoller/Serialization/Base_fwd.hpp>

namespace rocRoller
{
    namespace Operations
    {
        /**
         * A block scale operation for MX datatypes
        */
        class BlockScale : public BaseOperation
        {
        public:
            BlockScale() = delete;

            /**
             * @param data Tag of data to-be-scaled
             * @param dimensions Number of dimensions of `data`
             * @param scale Optional tag of scale tensor (if not provided, treated as inline block scale)
             * @param strides Strides of the scale
            */
            explicit BlockScale(OperationTag                data,
                                int                         dimensions,
                                std::optional<OperationTag> scale   = {},
                                std::vector<size_t>         strides = {});

            std::unordered_set<OperationTag> getInputs() const;
            std::string                      toString() const;
            ScaleMode                        scaleMode() const;
            const std::vector<size_t>&       strides() const;

            bool                        operator==(BlockScale const&) const;
            OperationTag                data() const;
            std::optional<OperationTag> scale() const;

        private:
            OperationTag                m_data;
            std::optional<OperationTag> m_scale;
            std::vector<size_t>         m_strides;

            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;
        };

        class SubTileTranspose : public BaseOperation
        {
        public:
            SubTileTranspose() = delete;

            explicit SubTileTranspose(OperationTag        input,
                                      std::vector<size_t> tileDimensions,
                                      bool                transpose = false);

            std::unordered_set<OperationTag> getInputs() const;
            std::string                      toString() const;
            std::vector<size_t> const&       tileDimensions() const;
            bool                             isTranspose() const;

            auto operator<=>(SubTileTranspose const&) const = default;
            bool operator==(SubTileTranspose const& other) const;

            OperationTag input() const;

        private:
            OperationTag        m_input;
            std::vector<size_t> m_tileDimensions;
            bool                m_transpose;

            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;
        };
    }
}
