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
        , unrollX(0)
        , unrollY(0)
        , transposeMemoryAccessA(true)
        , transposeMemoryAccessB(true)
        , transposeMemoryAccessOther(false)
    {
    }

    std::ostream& operator<<(std::ostream& os, const KernelOptions& input)
    {
        os << "Kernel Options:" << std::endl;
        os << "  logLevel:\t\t\t" << input.logLevel << std::endl;
        os << "  alwaysWaitAfterLoad:\t\t" << input.alwaysWaitAfterLoad << std::endl;
        os << "  alwaysWaitAfterStore:\t\t" << input.alwaysWaitAfterStore << std::endl;
        os << "  alwaysWaitBeforeBranch:\t" << input.alwaysWaitBeforeBranch << std::endl;
        os << "  preloadKernelArguments:\t" << input.preloadKernelArguments << std::endl;
        os << "  loadLocalWidth:\t\t" << input.loadLocalWidth << std::endl;
        os << "  loadGlobalWidth:\t\t" << input.loadGlobalWidth << std::endl;
        os << "  storeLocalWidth:\t\t" << input.storeLocalWidth << std::endl;
        os << "  storeGlobalWidth:\t\t" << input.storeGlobalWidth << std::endl;
        os << "  fuseLoops:\t\t\t" << input.fuseLoops << std::endl;
        os << "  unrollX:\t\t\t" << input.unrollX << std::endl;
        os << "  unrollY:\t\t\t" << input.unrollY << std::endl;
        os << "  transposeMemoryAccessA:\t" << input.transposeMemoryAccessA << std::endl;
        os << "  transposeMemoryAccessB:\t" << input.transposeMemoryAccessB << std::endl;
        os << "  transposeMemoryAccessOther:\t" << input.transposeMemoryAccessOther << std::endl;
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
