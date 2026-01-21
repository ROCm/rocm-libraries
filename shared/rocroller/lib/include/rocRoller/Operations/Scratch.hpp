// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Scratch space command.
 */

#pragma once

#include <rocRoller/Operations/Operation.hpp>
#include <rocRoller/Operations/Scratch_fwd.hpp>
#include <rocRoller/Serialization/Base_fwd.hpp>

namespace rocRoller
{
    namespace Operations
    {
        /**
         * A scratch space operation that is used to set policy guarantees for scratch space content.
        */
        class Scratch : public BaseOperation
        {
        public:
            Scratch() = delete;

            /**
             * @param tag Operation tag for this scratch space
             * @param policy Scratch space policy (guarantees about content)
            */
            explicit Scratch(OperationTag tag, ScratchPolicy policy = ScratchPolicy::None);

            OperationTag  getTag() const;
            ScratchPolicy policy() const;
            std::string   toString() const;

            bool operator==(Scratch const&) const;

        private:
            OperationTag  m_tag;
            ScratchPolicy m_policy;

            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;
        };
    }
}
