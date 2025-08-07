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

            explicit SubTileTranspose(OperationTag input, std::vector<size_t> tileDimensions);

            std::unordered_set<OperationTag> getInputs() const;
            std::string                      toString() const;
            std::vector<size_t> const&       tileDimensions() const;

            auto operator<=>(SubTileTranspose const&) const = default;
            bool operator==(SubTileTranspose const& other) const;

            OperationTag input() const;

        private:
            OperationTag        m_input;
            std::vector<size_t> m_tileDimensions;

            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;
        };
    }
}
