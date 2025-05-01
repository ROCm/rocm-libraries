/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Update KernelGraph to add LDS padding if F6 datatypes will
         * be transpose loaded.
         *
         * When transpose loading F6 datatypes from LDS addresses must be
         * 128b aligned to 128b. This transform modifies KernelGraph in
         * the following manner:
         *   - MacroTile of F6 datatypes get padding in fast-moving dimension.
         *     Padding information is kept in new MacroTile/WaveTile field --
         *     padBytesOfDim keeps track of byte padding per dimension. Total
         *     padding bytes can be queried by calling paddingBytes. Such padding
         *     causes extra LDS to be allocated;
         *   - LDS coordinates associated transpose load from LDS have their new
         *     holdsTransposedTile field set to indicate to CodeGen that
         *     padding is required when computing strides & offsets;
         *   - User coordinates associate with stores to LDS for tensors that need
         *     transpose loads have their new needsPadding field set to indicate
         *     to CodeGen that padding is required.
         */
        class AddF6LDSPadding : public GraphTransform
        {
        public:
            AddF6LDSPadding(ContextPtr context)
                : m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AddF6LDSPadding";
            }

        private:
            ContextPtr m_context;
        };
    }
}
