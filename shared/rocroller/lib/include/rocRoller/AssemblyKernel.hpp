/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

/**
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include <rocRoller/AssemblyKernelArgument.hpp>
#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    class AssemblyKernel
    {
    public:
        AssemblyKernel(ContextPtr context, std::string const& kernelName);

        AssemblyKernel() noexcept;
        AssemblyKernel(AssemblyKernel const& rhs) noexcept;
        AssemblyKernel(AssemblyKernel&& rhs) noexcept;
        AssemblyKernel& operator=(AssemblyKernel const& rhs);
        AssemblyKernel& operator=(AssemblyKernel&& rhs);

        ~AssemblyKernel();

        ContextPtr context() const;
        CommandPtr command() const;

        std::string kernelName() const;
        void        setKernelName(std::string const& name);

        int  kernelDimensions() const;
        void setKernelDimensions(int dims);

        Register::ValuePtr argumentPointer() const;

        /**
         * Called by `preamble()`.  Allocates all the registers set by the initial kernel execution state.
         *
         * @return Generator<Instruction>
         */
        Generator<Instruction> allocateInitialRegisters();

        /**
         * Everything up to and including the label that marks the beginning of the kernel.
         *
         * @return Generator<Instruction>
         */
        Generator<Instruction> preamble();

        /**
         * Instructions that set up the initial kernel state.
         */
        Generator<Instruction> prolog();

        /**
         * `s_endpgm`, plus assembler directives that must follow the kernel, not
         * including the `.amdgpu_metadata` section.
         *
         * @return Generator<Instruction>
         */
        Generator<Instruction> postamble() const;

        /**
         * Metadata YAML document, surrounded by
         *  `.amdgpu_metadata`/`.end_amdgpu_metadata` directives
         *
         *
         * @return Generator<Instruction>
         */
        Generator<Instruction> amdgpu_metadata();

        /**
         * Just the YAML document.
         *
         * @return std::string
         */
        std::string amdgpu_metadata_yaml();

        int kernarg_segment_size() const;
        int group_segment_fixed_size() const;
        int private_segment_fixed_size() const;
        int kernarg_segment_align() const;
        int wavefront_size() const;
        int sgpr_count() const;
        int vgpr_count() const;
        int agpr_count() const;

        /**
         * Returns the value which should be supplied to the `accum_offset` meta value in the YAML metadata.
         * Equal to the number of VGPRs used rounded up to a multiple of 4.
         */
        int accum_offset() const;
        /**
         * Returns the total number of VGPRs used, including AGPRs.
         */
        int total_vgprs() const;

        int max_flat_workgroup_size() const;

        std::shared_ptr<KernelGraph::KernelGraph> kernel_graph() const;

        AssemblyKernelArgument const&       findArgument(std::string const& name) const;
        bool                                hasArgument(std::string const& name) const;
        std::vector<AssemblyKernelArgument> resetArguments();

        /**
         * If a kernel argument exists with an expression equivalent to `exp`, return an
         * expression referencing that argument, otherwise `nullptr`.
         */
        Expression::ExpressionPtr findArgumentForExpression(Expression::ExpressionPtr exp) const;

        std::vector<AssemblyKernelArgument> const& arguments() const;

        Expression::ExpressionPtr addArgument(AssemblyKernelArgument arg);

        std::string uniqueArgName(std::string const& base) const;

        /**
         * Adds a vector of CommandArguments as arguments to the AssemblyKernel.
         *
         * @param args Vector of CommandArgument pointers that should be added as arguments.
         */
        void                      addCommandArguments(std::vector<CommandArgumentPtr> args);
        Expression::ExpressionPtr addCommandArgument(CommandArgumentPtr arg);

        std::string args_string();

        /** The size in bytes of all the arguments. */
        size_t argumentSize() const;

        std::array<unsigned int, 3> const&              workgroupSize() const;
        Expression::ExpressionPtr                       workgroupCount(size_t index);
        std::array<Expression::ExpressionPtr, 3> const& workitemCount() const;
        Expression::ExpressionPtr const&                dynamicSharedMemBytes() const;

        void setWorkgroupSize(std::array<unsigned int, 3> const& val);
        void setWorkitemCount(std::array<Expression::ExpressionPtr, 3> const& val);
        void setDynamicSharedMemBytes(Expression::ExpressionPtr const& val);
        void setKernelGraphMeta(KernelGraph::KernelGraphPtr graph);
        void setCommandMeta(CommandPtr graph);
        void setWavefrontSize(int);

        std::array<Register::ValuePtr, 3> const& workgroupIndex() const;
        std::array<Register::ValuePtr, 3> const& workitemIndex() const;

        Register::ValuePtr kernelStartLabel() const;
        Register::ValuePtr kernelEndLabel() const;

        bool startedCodeGeneration() const;
        void startCodeGeneration();

        /**
         * Clears the index register pointers, allowing the registers to be freed
         * if they are not referenced elsewhere.
         */
        void clearIndexRegisters();

    private:
        template <typename T1, typename T2, typename T3>
        friend struct rocRoller::Serialization::MappingTraits;

        /**
         * If a kernel argument exists with an expression equivalent to `exp`, return an
         * expression referencing that argument, and return its index in `idx`, otherwise return
         * `nullptr` and set `idx` to -1.
         */
        Expression::ExpressionPtr findArgumentForExpression(Expression::ExpressionPtr exp,
                                                            ptrdiff_t&                idx) const;

        std::weak_ptr<Context> m_context;

        std::string        m_kernelName;
        Register::ValuePtr m_kernelStartLabel;
        Register::ValuePtr m_kernelEndLabel;

        int m_kernelDimensions = 3;

        bool m_startedCodeGeneration = false;

        std::array<unsigned int, 3>              m_workgroupSize = {1, 1, 1};
        std::array<Expression::ExpressionPtr, 3> m_workitemCount;
        std::array<Expression::ExpressionPtr, 3> m_workgroupCount;
        Expression::ExpressionPtr                m_dynamicSharedMemBytes;

        Register::ValuePtr                m_argumentPointer;
        std::array<Register::ValuePtr, 3> m_workgroupIndex;

        Register::ValuePtr                m_packedWorkitemIndex;
        std::array<Register::ValuePtr, 3> m_workitemIndex;

        std::vector<AssemblyKernelArgument>     m_arguments;
        std::unordered_map<std::string, size_t> m_argumentNames;
        int                                     m_argumentSize = 0;

        int m_wavefrontSize = 64;

        KernelGraph::KernelGraphPtr m_kernelGraph;
        CommandPtr                  m_command;
    };

    struct AssemblyKernels
    {
        constexpr std::array<int, 2> hsa_version()
        {
            return {1, 2}; // Assuming -mcode-object-version=5
        }
        std::vector<AssemblyKernel> kernels;

        static AssemblyKernels fromYAML(std::string const& str);
    };
}

#include <rocRoller/AssemblyKernel_impl.hpp>
