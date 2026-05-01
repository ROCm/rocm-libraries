#pragma once

#include <unordered_map>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/bindings/python/LogicalModule.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"

namespace stinkytofu {
/**
 * @brief Converts PyLogicalModule (Python-side IR) to Function (C++-side IR)
 *
 * This converter decouples the Python module from StinkyTofu's internal
 * optimization pipeline. It performs the following tasks:
 *
 * 1. Extracts raw pointers from shared_ptr<LogicalInstruction>
 * 2. Adds LogicalInstruction* directly to IRList (both inherit from IRBase)
 * 3. Splits instructions into BasicBlocks based on labels
 * 4. Populates Function with BasicBlocks and IRLists
 *
 * Note: NO LOWERING happens here. LogicalInstruction* are added directly to IRList.
 * Lowering from LogicalInstruction to StinkyInstruction happens later as a pass.
 *
 * Architecture:
 *   Python: PyLogicalModule + shared_ptr<LogicalInstruction>
 *       ? [Converter - NO lowering]
 *   C++: Function + BasicBlock + IRList (raw LogicalInstruction*)
 *       ? [OptimizationPipeline with lowering pass]
 *   Optimized assembly
 *
 * Example usage:
 * @code
 *   // Python creates PyLogicalModule
 *   auto module = std::make_shared<PyLogicalModule>("kernel");
 *   module->add(makeLogicalInstructionShared(VAddF32(...)));
 *
 *   // C++ converts to Function; PyLogicalFunction wraps it and detaches Python-owned IRs on
 * destroy Function func("kernel"); PyLogicalFunction pyFunc(&func); LogicalToFunctionConverter
 * converter(GfxArchID::Gfx1250); converter.convert(module.get(), pyFunc);
 *
 *   // Now run optimization pipeline on func
 *   OptimizationPipeline::run(func, config);
 * @endcode
 */
class STINKYTOFU_EXPORT LogicalToFunctionConverter {
   public:
    /**
     * @brief Construct a converter for a specific target architecture
     * @param arch Target GPU architecture (needed for lowering)
     */
    explicit LogicalToFunctionConverter(GfxArchID arch);

    /**
     * @brief Convert PyLogicalModule to PyLogicalFunction
     *
     * This performs the following steps:
     * 1. Extract raw pointers from shared_ptr<LogicalInstruction>
     * 2. Add LogicalInstruction* directly to IRList (NO lowering)
     * 3. On destruction, PyLogicalFunction detaches
     *    ownedExternally IRs so the list does not delete them.
     *
     * @param module Source PyLogicalModule (Python-side IR)
     * @param pyFunc Destination PyLogicalFunction
     */
    void convert(PyLogicalModule* module, PyLogicalFunction& pyFunc);

    /**
     * @brief Convert with automatic label detection and block splitting
     *
     * This version automatically:
     * - Detects label instructions
     * - Creates BasicBlocks for each label
     * - Adds LogicalInstruction* directly to IRList (NO lowering)
     *
     * @param module Source PyLogicalModule
     * @param pyFunc Destination PyLogicalFunction
     * @param autoSplitBlocks If true, split on labels (default: true)
     */
    void convertWithAutoBlocks(PyLogicalModule* module, PyLogicalFunction& pyFunc,
                               bool autoSplitBlocks = true);

   private:
    GfxArchID arch;

    /**
     * @brief Identify label instructions in the instruction stream
     *
     * @param instructions Vector of LogicalInstructions
     * @return Map of label name -> instruction index
     */
    std::unordered_map<std::string, size_t> identifyLabels(
        const std::vector<std::shared_ptr<LogicalInstruction>>& instructions);

    /**
     * @brief Split instructions into BasicBlocks based on labels
     *
     * @param func Destination Function
     * @param instructions Raw LogicalInstruction pointers
     * @param labelMap Map of label name -> instruction index
     */
    void splitIntoBasicBlocks(Function& func, const std::vector<LogicalInstruction*>& instructions,
                              const std::unordered_map<std::string, size_t>& labelMap);
};

}  // namespace stinkytofu
