// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

        SubTileTranspose::SubTileTranspose(OperationTag        input,
                                           std::vector<size_t> tileDimensions,
                                           bool                transpose)
            : m_input(input)
            , m_tileDimensions(std::move(tileDimensions))
            , m_transpose(transpose)
        {
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

        bool SubTileTranspose::isTranspose() const
        {
            return m_transpose;
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
