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
        , unrollK(0)
        , assertWaitCntState(true)
        , packMultipleElementsInto1VGPR(false)
        , enableLongDwordInstructions(false)
    {
        transposeMemoryAccess.set(0);
        transposeMemoryAccess[LayoutType::MATRIX_A]           = false;
        transposeMemoryAccess[LayoutType::MATRIX_B]           = true;
        transposeMemoryAccess[LayoutType::MATRIX_ACCUMULATOR] = true;
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
        os << "  unrollK:\t\t\t" << input.unrollK << std::endl;

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
