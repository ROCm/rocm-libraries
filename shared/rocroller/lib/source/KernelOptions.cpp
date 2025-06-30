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

    // KernelOptions(KernelOptions const& other);
    // KernelOptions(KernelOptions && other);
    // KernelOptions & operator=(KernelOptions const& other);
    // KernelOptions & operator=(KernelOptions && other);

    std::ostream& operator<<(std::ostream& os, const KernelOptionValues& input)
    {
        os << "Kernel Options:" << std::endl;
        os << "  logLevel:\t\t\t" << input.logLevel << std::endl;
        os << "  alwaysWaitAfterLoad:\t\t" << input.alwaysWaitAfterLoad << std::endl;
        os << "  alwaysWaitAfterStore:\t\t" << input.alwaysWaitAfterStore << std::endl;
        os << "  alwaysWaitBeforeBranch:\t" << input.alwaysWaitBeforeBranch << std::endl;
        os << "  preloadKernelArguments:\t" << input.preloadKernelArguments << std::endl;
        os << "  maxACCVGPRs:\t\t\t" << input.maxACCVGPRs << std::endl;
        os << "  maxSGPRs:\t\t\t" << input.maxSGPRs << std::endl;
        os << "  maxVGPRs:\t\t\t" << input.maxVGPRs << std::endl;
        os << "  loadLocalWidth:\t\t" << input.loadLocalWidth << std::endl;
        os << "  loadGlobalWidth:\t\t" << input.loadGlobalWidth << std::endl;
        os << "  storeLocalWidth:\t\t" << input.storeLocalWidth << std::endl;
        os << "  storeGlobalWidth:\t\t" << input.storeGlobalWidth << std::endl;
        os << "  setNextFreeVGPRToMax:\t" << input.setNextFreeVGPRToMax << std::endl;
        os << "  assertWaitCntState:\t\t" << input.assertWaitCntState << std::endl;
        os << "  deduplicateArguments:\t\t" << input.deduplicateArguments << std::endl;
        os << "  lazyAddArguments:\t\t" << input.lazyAddArguments << std::endl;
        os << "  minLaunchTimeExpressionComplexity:\t\t" << input.minLaunchTimeExpressionComplexity
           << std::endl;

        return os;
    }

    std::string KernelOptionValues::toString() const
    {
        if(logLevel >= LogLevel::Warning)
        {
            std::stringstream ss;
            ss << *this;
            return ss.str();
        }
        else
        {
            return "";
        }
    }
}
