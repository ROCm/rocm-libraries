
#include <rocRoller/Scheduling/Costs/WaitCntNopCost.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(WaitCntNopCost);
        static_assert(Component::Component<WaitCntNopCost>);

        inline WaitCntNopCost::WaitCntNopCost(ContextPtr ctx)
            : Cost{ctx}
        {
        }

        inline bool WaitCntNopCost::Match(Argument arg)
        {
            return std::get<0>(arg) == CostFunction::WaitCntNop;
        }

        inline std::shared_ptr<Cost> WaitCntNopCost::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<WaitCntNopCost>(std::get<1>(arg));
        }

        inline std::string WaitCntNopCost::name() const
        {
            return Name;
        }

        inline float WaitCntNopCost::cost(Instruction const&       inst,
                                          InstructionStatus const& status) const
        {
            auto const& architecture = m_ctx.lock()->targetArchitecture();

            int vmCost = 0;
            if(architecture.HasCapability(GPUCapability::MaxVmcnt))
            {
                auto loadcnt  = status.waitCount.loadcnt();
                auto storecnt = status.waitCount.storecnt();
                auto vm       = WaitCount::CombineValues(loadcnt, storecnt);
                vmCost
                    = vm == -1 ? 0 : (architecture.GetCapability(GPUCapability::MaxVmcnt) - vm + 1);
            }

            int lgkmCost = 0;
            if(architecture.HasCapability(GPUCapability::MaxLgkmcnt))
            {
                auto kmcnt = status.waitCount.kmcnt();
                auto dscnt = status.waitCount.dscnt();
                auto lgkm  = WaitCount::CombineValues(kmcnt, dscnt);
                lgkmCost   = lgkm == -1
                                 ? 0
                                 : (architecture.GetCapability(GPUCapability::MaxLgkmcnt) - lgkm + 1);
            }

            int expCost = 0;
            if(architecture.HasCapability(GPUCapability::MaxExpcnt))
            {
                int exp = status.waitCount.expcnt();
                expCost = exp == -1
                              ? 0
                              : (architecture.GetCapability(GPUCapability::MaxExpcnt) - exp + 1);
            }

            return static_cast<float>(status.nops) + static_cast<float>(vmCost)
                   + static_cast<float>(lgkmCost) + static_cast<float>(expCost);
        }
    }
}
