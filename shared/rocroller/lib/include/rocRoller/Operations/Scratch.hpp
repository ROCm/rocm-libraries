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
