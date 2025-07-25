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

#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/KernelOptions_detail.hpp>

#include <rocRoller/AssertOpKinds.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    KernelOptions::KernelOptions()
        : m_values(std::make_unique<KernelOptionValues>())
    {
        if(Settings::Get(Settings::NoRegisterLimits))
        {
            m_values->maxACCVGPRs *= 10;
            m_values->maxSGPRs *= 10;
            m_values->maxVGPRs *= 10;
        }
    }

    KernelOptions::KernelOptions(KernelOptionValues&& other)
        : m_values(std::make_unique<KernelOptionValues>(std::forward<KernelOptionValues>(other)))
    {
    }

    KernelOptions::KernelOptions(KernelOptions const& other)
        : m_values(std::make_unique<KernelOptionValues>(*other.m_values))
    {
    }
    KernelOptions::KernelOptions(KernelOptions&& other)
        : m_values(std::move(other.m_values))
    {
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
        static_assert(sizeof(KernelOptionValues) == 68,
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
        ShowOption(preloadKernelArguments);
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
        ShowOption(lazyAddArguments);
        ShowOption(minLaunchTimeExpressionComplexity);
        ShowOption(maxConcurrentSubExpressions);
        Show("maxConcurrentControlOps",
             values.maxConcurrentControlOps ? std::to_string(*values.maxConcurrentControlOps)
                                            : "none");
        ShowOption(enableFullDivision);
        ShowOption(scaleSkipPermlane);
        ShowString(assertOpKind);

#undef Show
#undef ShowOption
#undef ShowString

        return rv;
    }
}
