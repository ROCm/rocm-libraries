#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    KernelOptions::KernelOptions()
        : logLevel(LogLevel::Verbose)
        , alwaysWaitAfterLoad(false)
        , alwaysWaitAfterStore(false)
        , alwaysWaitBeforeBranch(true)
        , preloadKernelArguments(true)
        , loadLocalWidth(4)
        , loadGlobalWidth(8)
        , storeLocalWidth(4)
        , storeGlobalWidth(4)
        , fuseLoops(true)
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
        os << "  loadLocalWidth:\t" << input.loadLocalWidth << std::endl;
        os << "  loadGlobalWidth:\t" << input.loadGlobalWidth << std::endl;
        os << "  storeLocalWidth:\t" << input.storeLocalWidth << std::endl;
        os << "  storeGlobalWidth:\t" << input.storeGlobalWidth << std::endl;
        os << "  fuseLoops:\t" << input.fuseLoops << std::endl;
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
