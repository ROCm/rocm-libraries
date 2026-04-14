// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Parameters/Solution/LDSBankSwizzleMode.hpp>
#include <rocRoller/Parameters/Solution/ScaleSkipPermlaneMode.hpp>

#include <rocRoller/AssertOpKinds.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    std::string toString(DSObserverType type)
    {
        switch(type)
        {
        case DSObserverType::DSMEMObserver:
            return "DSMEMObserver";
        case DSObserverType::WeightlessDSMemObserver:
            return "WeightlessDSMemObserver";
        case DSObserverType::Count:
        default:
            return "Unknown";
        }
    }

    static void increaseRegisterLimit(KernelOptionValues& values)
    {
        if(Settings::Get(Settings::NoRegisterLimits))
        {
            values.maxACCVGPRs *= 10;
            values.maxSGPRs *= 10;
            values.maxVGPRs *= 10;
        }
    }

    KernelOptions::KernelOptions()
        : m_values(std::make_unique<KernelOptionValues>())
    {
        increaseRegisterLimit(*m_values);
    }

    KernelOptions::KernelOptions(KernelOptionValues&& other)
        : m_values(std::make_unique<KernelOptionValues>(std::forward<KernelOptionValues>(other)))
    {
        increaseRegisterLimit(*m_values);
    }

    KernelOptions::KernelOptions(KernelOptions const& other)
        : m_values(std::make_unique<KernelOptionValues>(*other.m_values))
    {
        increaseRegisterLimit(*m_values);
    }
    KernelOptions::KernelOptions(KernelOptions&& other)
        : m_values(std::move(other.m_values))
    {
        increaseRegisterLimit(*m_values);
    }

    KernelOptions& KernelOptions::operator=(KernelOptions const& other)
    {
        *m_values = *other;

        return *this;
    }
    KernelOptions& KernelOptions::operator=(KernelOptions&& other)
    {
        m_values = std::move(other.m_values);

        return *this;
    }

    KernelOptions& KernelOptions::operator=(KernelOptionValues const& other)
    {
        *m_values = other;

        return *this;
    }

    KernelOptions& KernelOptions::operator=(KernelOptionValues&& other)
    {
        m_values = std::make_unique<KernelOptionValues>(std::move(other));

        return *this;
    }

    KernelOptions::~KernelOptions() = default;

    KernelOptionValues* KernelOptions::operator->()
    {
        return m_values.get();
    }

    KernelOptionValues& KernelOptions::operator*()
    {
        return *m_values;
    }

    KernelOptionValues const* KernelOptions::operator->() const
    {
        return m_values.get();
    }

    KernelOptionValues const& KernelOptions::operator*() const
    {
        return *m_values;
    }

    std::string KernelOptions::toString() const
    {
        return m_values->toString();
    }

    std::ostream& operator<<(std::ostream& stream, const KernelOptions& options)
    {
        return stream << *options;
    }

    std::ostream& operator<<(std::ostream& os, const KernelOptionValues& input)
    {
        return os << toString(input);
    }

    std::string KernelOptionValues::toString() const
    {
        return rocRoller::toString(*this);
    }

    std::string toString(KernelOptionValues const& values)
    {
        static_assert(sizeof(KernelOptionValues) == 88,
                      "Edit the toString() function when adding a kernel option!");

        std::string rv = "Kernel Options:\n";

#define Show(name, value) rv += fmt::format("  {: <35}{: >10}\n", name ":", value)

#define ShowOption(name) Show(#name, values.name)
#define ShowString(name) Show(#name, toString(values.name))

        ShowString(logLevel);
        ShowOption(alwaysWaitAfterLoad);
        ShowOption(alwaysWaitAfterStore);
        ShowOption(alwaysWaitBeforeBranch);
        ShowOption(alwaysWaitZeroBeforeBarrier);
        ShowOption(systemPreloadedKernelArguments);
        ShowOption(lazyLoadKernelArguments);
        ShowOption(maxACCVGPRs);
        ShowOption(maxSGPRs);
        ShowOption(maxVGPRs);
        ShowOption(loadLocalWidth);
        ShowOption(loadGlobalWidth);
        ShowOption(storeLocalWidth);
        ShowOption(storeGlobalWidth);
        ShowOption(assertWaitCntState);
        ShowOption(setNextFreeVGPRToMax);
        ShowOption(deduplicateArguments);
        ShowOption(minLaunchTimeExpressionComplexity);
        ShowOption(maxConcurrentSubExpressions);
        Show("maxConcurrentControlOps",
             values.maxConcurrentControlOps ? std::to_string(*values.maxConcurrentControlOps)
                                            : "none");
        ShowString(dsObserver);
        ShowOption(enableFullDivision);
        ShowString(scaleSkipPermlane);
        ShowString(assertOpKind);
        ShowOption(removeSetCoordinate);
        ShowString(ldsSwizzleMode);

#undef Show
#undef ShowOption
#undef ShowString

        return rv;
    }
}
