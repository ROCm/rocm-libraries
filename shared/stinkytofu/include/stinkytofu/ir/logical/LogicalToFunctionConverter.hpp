#pragma once

#include "stinkytofu/core/PyLogicalModule.hpp"
#include "stinkytofu/core/stinkytofu.hpp"
#include <unordered_map>

namespace stinkytofu
{
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
     *   module->add(std::make_shared<VAddF32>(...));
     *
     *   // C++ converts to Function for optimization
     *   Function func("kernel");
     *   PassContext ctx;
     *   LogicalToFunctionConverter converter(GfxArchID::Gfx942);
     *   converter.convert(module.get(), func, ctx);
     *
     *   // Now run optimization pipeline
     *   OptimizationPipeline::run(func, config);
     * @endcode
     */
    class LogicalToFunctionConverter
    {
    public:
        /**
         * @brief Construct a converter for a specific target architecture
         * @param arch Target GPU architecture (needed for lowering)
         */
        explicit LogicalToFunctionConverter(GfxArchID arch);

        /**
         * @brief Convert PyLogicalModule to Function
         *
         * This performs the following steps:
         * 1. Extract raw pointers from shared_ptr<LogicalInstruction>
         * 2. Add LogicalInstruction* directly to IRList (NO lowering)
         * 3. Transfer ownership: shared_ptr remains in Python, IRList holds raw pointers
         *
         * Note: LogicalInstruction* and StinkyInstruction* both inherit from IRBase*,
         * so IRList can hold both. Lowering happens later as an optimization pass.
         *
         * @param module Source PyLogicalModule (Python-side IR)
         * @param func Destination Function (C++-side IR)
         */
        void convert(PyLogicalModule* module, Function& func);

        /**
         * @brief Convert with automatic label detection and block splitting
         *
         * This version automatically:
         * - Detects label instructions
         * - Creates BasicBlocks for each label
         * - Adds LogicalInstruction* directly to IRList (NO lowering)
         *
         * @param module Source PyLogicalModule
         * @param func Destination Function
         * @param autoSplitBlocks If true, split on labels (default: true)
         */
        void convertWithAutoBlocks(PyLogicalModule* module,
                                   Function&        func,
                                   bool             autoSplitBlocks = true);

    private:
        GfxArchID arch;

        /**
         * @brief Identify label instructions in the instruction stream
         *
         * @param instructions Vector of LogicalInstructions
         * @return Map of label name -> instruction index
         */
        std::unordered_map<std::string, size_t>
            identifyLabels(const std::vector<std::shared_ptr<LogicalInstruction>>& instructions);

        /**
         * @brief Split instructions into BasicBlocks based on labels
         *
         * @param func Destination Function
         * @param instructions Raw LogicalInstruction pointers
         * @param labelMap Map of label name -> instruction index
         */
        void splitIntoBasicBlocks(Function&                                      func,
                                  const std::vector<LogicalInstruction*>&        instructions,
                                  const std::unordered_map<std::string, size_t>& labelMap);
    };

} // namespace stinkytofu
