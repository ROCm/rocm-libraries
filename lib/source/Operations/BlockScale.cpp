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
                               std::vector<size_t> const&  strides)
            : BaseOperation()
            , m_data(data)
            , m_scale(scale)
            , m_strides([&]() {
                AssertFatal(dimensions >= 1);
                std::vector<size_t> rt(dimensions, 1);
                rt[0] = 32; // Default value for first stride based on hardware arch
                std::copy(strides.begin(), strides.end(), rt.begin());
                return rt;
            }())
        {
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
    }
}
