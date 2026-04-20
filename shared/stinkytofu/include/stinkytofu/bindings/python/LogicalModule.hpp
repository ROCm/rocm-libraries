/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "stinkytofu/core/Function.hpp"

namespace stinkytofu {
// Forward declarations
class LogicalInstruction;

// ========================================================================
// PYTHON-SPECIFIC MODULE - Can be removed when Python bindings are deprecated
// ========================================================================
//
// PyLogicalModule is ONLY for Python bindings. It provides a simple container
// for shared_ptr<LogicalInstruction> that Python can easily manage.
//
// C++ code should use the unified Pass infrastructure directly:
//   - Create raw LogicalInstruction* (via factory functions)
//   - Add directly to Function's IRList
//   - Run passes via PassManager
//
// Python workflow:
//   1. Python creates PyLogicalModule and adds shared_ptr<LogicalInstruction>
//   2. LogicalToFunctionConverter extracts raw pointers for Function/IRList
//   3. Caller keeps PyLogicalModule alive for the lifetime of the Function
//   4. PassManager runs optimization and lowering passes
//
// This design allows Python to be easily removed/deprecated in the future
// without impacting the core C++ optimization infrastructure.
// ========================================================================

/**
 * @brief Python-specific IR Module container
 *
 * **IMPORTANT**: This class is ONLY for Python bindings. C++ code should
 * use Function + IRList directly.
 *
 * PyLogicalModule holds high-level, architecture-independent IR instructions
 * (LogicalInstruction) with shared_ptr for Python's memory management.
 *
 * The module itself is architecture-independent. Target architecture is
 * specified when converting to Function via LogicalToFunctionConverter.
 *
 * Example Python usage:
 * @code{.py}
 *   import stinkytofu as st
 *
 *   module = st.PyLogicalModule("myKernel")
 *   module.add(st.VAddF32(st.vgpr(0), st.vgpr(1), st.vgpr(2)))
 *   module.add(st.VAddPKF32(st.vgpr(3), st.vgpr(4), st.vgpr(5)))
 *
 *   # Convert to Function and run passes (C++ side)
 *   converter = st.LogicalToFunctionConverter(st.GfxArchID.Gfx942)
 *   func = converter.convertWithAutoBlocks(module)
 *   # ... run passes ...
 * @endcode
 */
class PyLogicalModule {
   public:
    /**
     * @brief Construct a new PyLogicalModule
     * @param name Module/kernel name
     */
    PyLogicalModule(const std::string& name = "");

    /**
     * @brief Destructor - shared_ptrs handle cleanup
     */
    ~PyLogicalModule();

    // Disable copy (we own shared_ptrs)
    PyLogicalModule(const PyLogicalModule&) = delete;
    PyLogicalModule& operator=(const PyLogicalModule&) = delete;

    // Enable move
    PyLogicalModule(PyLogicalModule&&) noexcept;
    PyLogicalModule& operator=(PyLogicalModule&&) noexcept;

    /**
     * @brief Get the module name
     * @return Module name string
     */
    std::string getName() const;

    /**
     * @brief Add a single IR instruction to this module
     *
     * The module shares ownership of the instruction via shared_ptr.
     *
     * @param inst IR instruction to add
     * @return The same instruction (for chaining)
     */
    std::shared_ptr<LogicalInstruction> add(std::shared_ptr<LogicalInstruction> inst);

    /**
     * @brief Get all IR instructions in this module (const version)
     * @return Vector of IR instructions (shared ownership)
     */
    const std::vector<std::shared_ptr<LogicalInstruction>>& getInstructions() const;

    /**
     * @brief Get all IR instructions in this module (non-const version)
     * @return Vector of IR instructions (shared ownership)
     */
    std::vector<std::shared_ptr<LogicalInstruction>>& getMutableInstructions();

    /**
     * @brief Remove an instruction from the module
     * @param inst Pointer to the instruction to remove
     * @return true if instruction was found and removed, false otherwise
     */
    bool removeInstruction(LogicalInstruction* inst);

    /**
     * @brief Get number of instructions in this module
     * @return Instruction count
     */
    size_t size() const;

    /**
     * @brief Dump the IR instructions for debugging
     * @param out Output stream
     */
    void dump(std::ostream& out) const;

   private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * @brief Python-specific Function wrapper that holds an external Function*
 *        and detaches externally owned IRs when destroyed.
 *
 * Does not own the Function. When the destructor runs, it traverses all IRs
 * in all BasicBlocks and removes (detaches) any LogicalInstruction with
 * ownedExternally == true from the IRList, so that the list does not delete
 * them. The caller owns the Function and must keep it alive while this
 * wrapper is in use.
 */
class PyLogicalFunction {
   public:
    /** @brief Take an external Function* (caller owns it; this wrapper does not). */
    explicit PyLogicalFunction(Function* func);

    ~PyLogicalFunction();

    PyLogicalFunction(const PyLogicalFunction&) = delete;
    PyLogicalFunction& operator=(const PyLogicalFunction&) = delete;

    /** @brief Get the wrapped Function pointer. */
    Function* getFunction() {
        return func;
    }
    const Function* getFunction() const {
        return func;
    }

   private:
    void detachExternallyOwnedIRs();
    Function* func;
};

}  // namespace stinkytofu
