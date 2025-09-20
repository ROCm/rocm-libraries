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

#include <rocRoller/Operations/BlockScale.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operation.hpp>

namespace rocRoller
{
    namespace Operations
    {
        BlockScale::BlockScale(OperationTag                data,
                               int                         dimensions,
                               std::optional<OperationTag> scale,
                               std::vector<size_t>         strides)
            : BaseOperation()
            , m_data(data)
            , m_scale(scale)
        {
            if(!strides.empty())
            {
                m_strides = std::move(strides);
            }

            if(m_strides.empty())
            {
                // Default value for first stride based on hardware arch
                m_strides.push_back(32);
            }

            if(m_strides.size() != dimensions)
            {
                m_strides.resize(dimensions, 1);
            }
        }

        std::unordered_set<OperationTag> BlockScale::getInputs() const
        {
            if(scaleMode() == ScaleMode::Inline)
                return {m_data};
            return {m_data, m_scale.value()};
        }

        OperationTag BlockScale::data() const
        {
            return m_data;
        }

        std::optional<OperationTag> BlockScale::scale() const
        {
            return m_scale;
        }

        std::string BlockScale::toString() const
        {
            std::ostringstream rv;

            rv << "BlockScale(" << scaleMode() << ", {";
            streamJoin(rv, m_strides, ", ");
            rv << "}): Data: " << m_data;

            if(m_scale)
            {
                rv << ", Scale: " << *m_scale;
            }

            return rv.str();
        }

        ScaleMode BlockScale::scaleMode() const
        {
            if(m_strides.empty())
                return ScaleMode::SingleScale;
            return m_scale.has_value() ? ScaleMode::Separate : ScaleMode::Inline;
        }

        const std::vector<size_t>& BlockScale::strides() const
        {
            return m_strides;
        }

        bool BlockScale::operator==(BlockScale const& rhs) const
        {
            return m_tag == rhs.m_tag && m_data == rhs.m_data && m_scale == rhs.m_scale
                   && m_strides == rhs.m_strides;
        }

        std::string toString(ScaleMode const& mode)
        {
            switch(mode)
            {
            case ScaleMode::None:
                return "None";
            case ScaleMode::SingleScale:
                return "SingleScale";
            case ScaleMode::Separate:
                return "Separate";
            case ScaleMode::Inline:
                return "Inline";
            case ScaleMode::Count:;
            }

            return "Invalid";
        }

        std::ostream& operator<<(std::ostream& stream, ScaleMode const& mode)
        {
            return stream << toString(mode);
        }

        SubTileTranspose::SubTileTranspose(OperationTag input, std::vector<size_t> tileDimensions)
            : m_input(input)
            , m_tileDimensions(std::move(tileDimensions))
        {
            if(!m_tileDimensions.empty())
            {
                AssertFatal(m_tileDimensions.size() == 3, ShowValue(m_tileDimensions));
                AssertFatal(m_tileDimensions[0] == 64 && m_tileDimensions[1] == 4
                                && (m_tileDimensions[2] == 2 || m_tileDimensions[2] == 4),
                            ShowValue(m_tileDimensions));
            }
        }

        std::unordered_set<OperationTag> SubTileTranspose::getInputs() const
        {
            return {m_input};
        }
        std::string SubTileTranspose::toString() const
        {
            return fmt::format(
                "SubTileTranspose({}: input {})", concatenate(m_tileDimensions), m_input.value);
        }
        const std::vector<size_t>& SubTileTranspose::tileDimensions() const
        {
            return m_tileDimensions;
        }

        bool SubTileTranspose::operator==(SubTileTranspose const& other) const
        {
            return (*this <=> other) == std::strong_ordering::equal;
        }

        OperationTag SubTileTranspose::input() const
        {
            return m_input;
        }
    }
}
