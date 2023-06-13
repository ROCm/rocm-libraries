/**
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "AssemblyKernel_fwd.hpp"
#include "CodeGen/Instruction.hpp"
#include "Expression_fwd.hpp"
#include "KernelGraph/KernelGraph.hpp"
#include "Scheduling/Scheduling.hpp"
#include "Utilities/Generator.hpp"

namespace rocRoller
{

    struct AssemblyKernelArgument
    {
        std::string   name;
        VariableType  variableType;
        DataDirection dataDirection = DataDirection::ReadOnly;

        Expression::ExpressionPtr expression = nullptr;

        int offset = -1;
        int size   = -1;

        bool operator==(AssemblyKernelArgument const&) const = default;
    };

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

        std::optional<std::string> kernel_graph() const;

        AssemblyKernelArgument const& findArgument(std::string const& name) const;

        std::vector<AssemblyKernelArgument> const& arguments() const;
        void                                       addArgument(AssemblyKernelArgument arg);

        /**
         * Adds a vector of CommandArguments as arguments to the AssemblyKernel.
         *
         * @param args Vector of CommandArgument pointers that should be added as arguments.
         */
        void addCommandArguments(std::vector<CommandArgumentPtr> args);

        /** The size in bytes of all the arguments. */
        size_t argumentSize() const;

        std::array<unsigned int, 3> const&              workgroupSize() const;
        Expression::ExpressionPtr                       workgroupCount(size_t index);
        std::array<Expression::ExpressionPtr, 3> const& workitemCount() const;
        Expression::ExpressionPtr const&                dynamicSharedMemBytes() const;

        void setWorkgroupSize(std::array<unsigned int, 3> const& val);
        void setWorkitemCount(std::array<Expression::ExpressionPtr, 3> const& val);
        void setDynamicSharedMemBytes(Expression::ExpressionPtr const& val);
        void setKernelGraphMeta(std::shared_ptr<KernelGraph::KernelGraph> graph);
        void setWavefrontSize(int);

        std::array<Register::ValuePtr, 3> const& workgroupIndex() const;
        std::array<Register::ValuePtr, 3> const& workitemIndex() const;

        Register::ValuePtr kernelStartLabel() const;
        Register::ValuePtr kernelEndLabel() const;

        /**
         * Clears the index register pointers, allowing the registers to be freed
         * if they are not referenced elsewhere.
         */
        void clearIndexRegisters();

    private:
        std::weak_ptr<Context> m_context;

        std::string        m_kernelName;
        Register::ValuePtr m_kernelStartLabel;
        Register::ValuePtr m_kernelEndLabel;

        int m_kernelDimensions = 3;

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

        std::shared_ptr<KernelGraph::KernelGraph> m_kernelGraph;

        /**
         *
         *   1 .amdgcn_target "amdgcn-amd-amdhsa--gfx900+xnack" // optional
         *   2
         *   3 .text
         *   4 .globl hello_world
         *   5 .p2align 8
         *   6 .type hello_world,@function
         *   7 hello_world:
         *   8   s_load_dwordx2 s[0:1], s[0:1] 0x0
         *   9   v_mov_b32 v0, 3.14159
         *  10   s_waitcnt lgkmcnt(0)
         *  11   v_mov_b32 v1, s0
         *  12   v_mov_b32 v2, s1
         *  13   flat_store_dword v[1:2], v0
         *  14   s_endpgm
         *  15 .Lfunc_end0:
         *  16   .size   hello_world, .Lfunc_end0-hello_world
         *  17
         *  18 .rodata
         *  19 .p2align 6
         *  20 .amdhsa_kernel hello_world
         *  21   .amdhsa_user_sgpr_kernarg_segment_ptr 1
         *  22   .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
         *  23   .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
         *  24 .end_amdhsa_kernel
         *  25
         *  26 .amdgpu_metadata
         *  27 ---
         *  28 amdhsa.version:
         *  29   - 1
         *  30   - 0
         *  31 amdhsa.kernels:
         *  32   - .name: hello_world
         *  33     .symbol: hello_world.kd
         *  34     .kernarg_segment_size: 48
         *  35     .group_segment_fixed_size: 0
         *  36     .private_segment_fixed_size: 0
         *  37     .kernarg_segment_align: 4
         *  38     .wavefront_size: 64
         *  39     .sgpr_count: 2
         *  40     .vgpr_count: 3
         *  41     .max_flat_workgroup_size: 256
         *  42     .args:
         *  43       - .size: 8
         *  44         .offset: 0
         *  45         .value_kind: global_buffer
         *  46         .address_space: global
         *  47         .actual_access: write_only
         *  48 //...
         *  49 .end_amdgpu_metadata
         */
    };

    struct AssemblyKernels
    {
        constexpr std::array<int, 2> hsa_version()
        {
            return {1, 0};
        }
        std::vector<AssemblyKernel> kernels;
    };
}

#include "AssemblyKernel_impl.hpp"
