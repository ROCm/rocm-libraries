#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    KernelOptions::KernelOptions()
        : logLevel(LogLevel::Verbose)
        , alwaysWaitAfterLoad(false)
        , alwaysWaitAfterStore(false)
        , alwaysWaitBeforeBranch(false)
        , alwaysWaitZeroBeforeBarrier(false)
        , preloadKernelArguments(true)
        , maxACCVGPRs(256)
        , maxSGPRs(102)
        , maxVGPRs(256)
        , loadLocalWidth(4)
        , loadGlobalWidth(8)
        , storeLocalWidth(4)
        , storeGlobalWidth(4)
        , fuseLoops(true)
        , unrollX(0)
        , unrollY(0)
        , unrollK(0)
        , transposeMemoryAccess({})
        , assertWaitCntState(true)
        , packMultipleElementsInto1VGPR(true)
        , enableLongDwordInstructions(true)
        , setNextFreeVGPRToMax(false)
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
        os << "  maxACCVGPRs:\t\t\t" << input.maxACCVGPRs << std::endl;
        os << "  maxSGPRs:\t\t\t" << input.maxSGPRs << std::endl;
        os << "  maxVGPRs:\t\t\t" << input.maxVGPRs << std::endl;
        os << "  loadLocalWidth:\t\t" << input.loadLocalWidth << std::endl;
        os << "  loadGlobalWidth:\t\t" << input.loadGlobalWidth << std::endl;
        os << "  storeLocalWidth:\t\t" << input.storeLocalWidth << std::endl;
        os << "  storeGlobalWidth:\t\t" << input.storeGlobalWidth << std::endl;
        os << "  fuseLoops:\t\t\t" << input.fuseLoops << std::endl;
        os << "  unrollX:\t\t\t" << input.unrollX << std::endl;
        os << "  unrollY:\t\t\t" << input.unrollY << std::endl;
        os << "  unrollK:\t\t\t" << input.unrollK << std::endl;
        os << "  setNextFreeVGPRToMax:\t" << input.setNextFreeVGPRToMax << std::endl;

        for(int i = 0; i < static_cast<int>(LayoutType::Count); i++)
            os << "transposeMemoryAccess[" << static_cast<LayoutType>(i)
               << "]: " << input.transposeMemoryAccess[i] << std::endl;

        os << "  assertWaitCntState:\t\t" << input.assertWaitCntState << std::endl;
        os << "  packMultipleElementsInto1VGPR:\t\t" << input.packMultipleElementsInto1VGPR
           << std::endl;
        os << "  enableLongDwordInstructions:\t\t" << input.enableLongDwordInstructions
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
