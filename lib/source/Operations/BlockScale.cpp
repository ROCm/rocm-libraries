#include "rocRoller/Operations/BlockScale.hpp"
#include "rocRoller/Operations/Command.hpp"
#include "rocRoller/Operations/Operation.hpp"

namespace rocRoller
{
    namespace Operations
    {
        BlockScale::BlockScale(int                        data,
                               int                        dimensions,
                               std::optional<int>         scale,
                               std::vector<size_t> const& strides)
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

        std::unordered_set<int> BlockScale::getInputs() const
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
    }
}
