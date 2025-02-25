#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    std::ostream& operator<<(std::ostream& os, const KernelOptions& input)
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

    std::string KernelOptions::toString() const
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
