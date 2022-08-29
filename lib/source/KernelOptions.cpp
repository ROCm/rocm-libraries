#include <rocRoller/KernelOptions.hpp>

#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    KernelOptions::KernelOptions()
        : logLevel(Settings::LogLevel::Verbose)
        , alwaysWaitAfterLoad(false)
        , alwaysWaitAfterStore(false)
        , alwaysWaitBeforeBranch(true)
        , preloadKernelArguments(true)
    {
    }

    std::ostream& operator<<(std::ostream& os, const KernelOptions& input)
    {
        os << "Kernel Options:" << std::endl;
        os << "  logLevel:\t\t" << input.logLevel << std::endl;
        os << "  alwaysWaitAfterLoad:\t" << input.alwaysWaitAfterLoad << std::endl;
        os << "  alwaysWaitAfterStore:\t" << input.alwaysWaitAfterStore << std::endl;
        os << "  alwaysWaitBeforeBranch:\t" << input.alwaysWaitBeforeBranch << std::endl;
        os << "  preloadKernelArguments:\t" << input.preloadKernelArguments << std::endl;
        return os;
    }

    std::string KernelOptions::toString() const
    {
        if(logLevel >= Settings::LogLevel::Warning)
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
