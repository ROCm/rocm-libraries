#include <rocRoller/Operations/BlockScale.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operation.hpp>

namespace rocRoller
{
    namespace Operations
    {
        std::string toString(BlockScale::PointerMode const& p)
        {
            switch(p)
            {
            case BlockScale::PointerMode::Separate:
                return "Separate";
            case BlockScale::PointerMode::Inline:
                return "Inline";
            default:
                break;
            }
            Throw<FatalError>("Invalid PointerMode");
        }

        std::ostream& operator<<(std::ostream& stream, BlockScale::PointerMode p)
        {
            return stream << toString(p);
        }

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
            if(pointerMode() == PointerMode::Inline)
                return {m_data};
            return {m_data, m_scale.value()};
        }

        std::string BlockScale::toString() const
        {
            return "BlockScale";
        }

        BlockScale::PointerMode BlockScale::pointerMode() const
        {
            return m_scale.has_value() ? PointerMode::Separate : PointerMode::Inline;
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
    }
}
