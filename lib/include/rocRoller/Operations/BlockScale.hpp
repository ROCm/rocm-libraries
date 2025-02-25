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
                                std::vector<size_t> const&  strides = {});

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
    }
}
