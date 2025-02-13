
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

#include <rocRoller/KernelGraph/Transforms/AddComputeIndex.hpp>
#include <rocRoller/KernelGraph/Transforms/AddConvert.hpp>
#include <rocRoller/KernelGraph/Transforms/AddDeallocate.hpp>
#include <rocRoller/KernelGraph/Transforms/AddDirect2LDS.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDS.hpp>
#include <rocRoller/KernelGraph/Transforms/AddPRNG.hpp>
#include <rocRoller/KernelGraph/Transforms/AddStreamK.hpp>
#include <rocRoller/KernelGraph/Transforms/CleanArguments.hpp>
#include <rocRoller/KernelGraph/Transforms/CleanLoops.hpp>
#include <rocRoller/KernelGraph/Transforms/ConnectWorkgroups.hpp>
#include <rocRoller/KernelGraph/Transforms/ConstantPropagation.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseExpressions.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseLoops.hpp>
#include <rocRoller/KernelGraph/Transforms/IdentifyParallelDimensions.hpp>
#include <rocRoller/KernelGraph/Transforms/InlineIncrements.hpp>
#include <rocRoller/KernelGraph/Transforms/LoopOverTileNumbers.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerLinear.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTensorContraction.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTile.hpp>
#include <rocRoller/KernelGraph/Transforms/OrderEpilogueBlocks.hpp>
#include <rocRoller/KernelGraph/Transforms/OrderMemory.hpp>
#include <rocRoller/KernelGraph/Transforms/RemoveDuplicates.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Transforms/UnrollLoops.hpp>
#include <rocRoller/KernelGraph/Transforms/UpdateParameters.hpp>
