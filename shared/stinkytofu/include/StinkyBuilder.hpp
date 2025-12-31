#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "ErrorHandling.hpp"

// Forward declare stinkytofu namespace types (will be fully defined in .cpp)
namespace stinkytofu
{
    struct StinkyRegister;
    struct StinkyInstruction;
    struct AsmDirective;
    struct MacroInstruction;
    enum class GfxArchID : unsigned int;

    // Forward declare IRListModule
    class IRListModule;

    /**
     * @brief Standalone StinkyTofu builder for Python bindings.
     *
     * This class provides a clean, instance-based API for building GPU assembly.
     * Each instance maintains its own architecture context.
     *
     * Instructions are created via methods on this class, then explicitly added to
     * IRListModule instances.
     *
     * Example usage:
     *   StinkyTofu st([9, 4, 2]);
     *   auto module = st.createIRList("my_kernel");
     *   auto inst = st.VAddU32(vgpr(0), vgpr(1), vgpr(2), "add");
     *   module->add(inst);
     *   std::cout << module->emitAssembly() << std::endl;
     */
    class StinkyTofu
    {
    public:
        /**
         * @brief Construct a new StinkyTofu builder for the specified architecture.
         * @param arch Target architecture as [major, minor, stepping] (e.g., [9, 4, 2]).
         */
        explicit StinkyTofu(std::array<int, 3> arch);

        /**
         * @brief Destructor.
         */
        ~StinkyTofu();

        /**
         * @brief Create a new IRList module with the given name.
         * @param name Module/kernel name.
         * @return Shared pointer to the new IRListModule.
         */
        std::shared_ptr<IRListModule> createIRList(const std::string& name = "");

        // ========================================================================
        // Vector ALU Instructions
        // ========================================================================

        /**
         * @brief Create a V_ADD_U32 instruction (32-bit unsigned integer add).
         * @param dst Destination register.
         * @param src0 First source operand.
         * @param src1 Second source operand.
         * @param comment Optional user comment.
         * @return Vector of created instructions (typically 1).
         */
        std::vector<StinkyInstruction*> VAddU32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        /**
         * @brief Create a V_MUL_F32 instruction (32-bit float multiply).
         * @param dst Destination register.
         * @param src0 First source operand.
         * @param src1 Second source operand.
         * @param comment Optional user comment.
         * @return Vector of created instructions (typically 1).
         */
        std::vector<StinkyInstruction*> VMulF32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        // ========================================================================
        // Vector Arithmetic Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> VAddF16(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VAddF32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VAddF64(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VAddI32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VAddCOU32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VAddCCOU32(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> VAddPKF16(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VAdd3U32(const StinkyRegister& dst,
                                                 const StinkyRegister& src0,
                                                 const StinkyRegister& src1,
                                                 const std::string&    comment = "");

        std::vector<StinkyInstruction*> VSubF32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VSubI32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VSubU32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VSubCoU32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        // ========================================================================
        // Vector Multiply Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> VMulF16(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMulF64(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMulPKF16(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMulPKF32S(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMulLOU32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMulHII32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMulHIU32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMulI32I24(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMulU32U24(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const std::string&    comment = "");

        // ========================================================================
        // Vector MAC/FMA Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> VMacF32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VFmaF16(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const StinkyRegister& src2,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VFmaF32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const StinkyRegister& src2,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VFmaF64(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const StinkyRegister& src2,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VFmaPKF16(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const StinkyRegister& src2,
                                                  const std::string&    comment = "");

        Expected<std::vector<StinkyInstruction*>> VFmaMixF32(const StinkyRegister& dst,
                                                             const StinkyRegister& src0,
                                                             const StinkyRegister& src1,
                                                             const StinkyRegister& src2,
                                                             const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMadI32I24(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const StinkyRegister& src2,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMadU32U24(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const StinkyRegister& src2,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMadMixF32(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const StinkyRegister& src2,
                                                   const std::string&    comment = "");

        // ========================================================================
        // Vector Dot Product Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> VDot2CF32F16(const StinkyRegister& dst,
                                                     const StinkyRegister& src0,
                                                     const StinkyRegister& src1,
                                                     const StinkyRegister& src2,
                                                     const std::string&    comment = "");

        std::vector<StinkyInstruction*> VDot2F32F16(const StinkyRegister& dst,
                                                    const StinkyRegister& src0,
                                                    const StinkyRegister& src1,
                                                    const StinkyRegister& src2,
                                                    const std::string&    comment = "");

        std::vector<StinkyInstruction*> VDot2F32BF16(const StinkyRegister& dst,
                                                     const StinkyRegister& src0,
                                                     const StinkyRegister& src1,
                                                     const StinkyRegister& src2,
                                                     const std::string&    comment = "");

        std::vector<StinkyInstruction*> VDot2CF32BF16(const StinkyRegister& dst,
                                                      const StinkyRegister& src0,
                                                      const StinkyRegister& src1,
                                                      const StinkyRegister& src2,
                                                      const std::string&    comment = "");

        // ========================================================================
        // Vector Transcendental Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> VExpF16(const StinkyRegister& dst,
                                                const StinkyRegister& src,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VExpF32(const StinkyRegister& dst,
                                                const StinkyRegister& src,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VRcpF16(const StinkyRegister& dst,
                                                const StinkyRegister& src,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VRcpF32(const StinkyRegister& dst,
                                                const StinkyRegister& src,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VRcpIFlagF32(const StinkyRegister& dst,
                                                     const StinkyRegister& src,
                                                     const std::string&    comment = "");

        std::vector<StinkyInstruction*> VRsqF16(const StinkyRegister& dst,
                                                const StinkyRegister& src,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VRsqF32(const StinkyRegister& dst,
                                                const StinkyRegister& src,
                                                const std::string&    comment = "");

        Expected<std::vector<StinkyInstruction*>> VRsqIFlagF32(const StinkyRegister& dst,
                                                               const StinkyRegister& src,
                                                               const std::string&    comment = "");

        std::vector<StinkyInstruction*> VRndneF32(const StinkyRegister& dst,
                                                  const StinkyRegister& src,
                                                  const std::string&    comment = "");

        // ========================================================================
        // Vector Min/Max/Med Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> VMaxF16(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMaxF32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMaxF64(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMaxI32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMaxPKF16(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMinF16(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMinF32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMinF64(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMinI32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMed3I32(const StinkyRegister& dst,
                                                 const StinkyRegister& src0,
                                                 const StinkyRegister& src1,
                                                 const StinkyRegister& src2,
                                                 const std::string&    comment = "");

        std::vector<StinkyInstruction*> VMed3F32(const StinkyRegister& dst,
                                                 const StinkyRegister& src0,
                                                 const StinkyRegister& src1,
                                                 const StinkyRegister& src2,
                                                 const std::string&    comment = "");

        // ========================================================================
        // Vector Bitwise Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> VAndB32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VAndOrB32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const StinkyRegister& src2,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VOrB32(const StinkyRegister& dst,
                                               const StinkyRegister& src0,
                                               const StinkyRegister& src1,
                                               const std::string&    comment = "");

        std::vector<StinkyInstruction*> VXorB32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VNotB32(const StinkyRegister& dst,
                                                const StinkyRegister& src,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VPrngB32(const StinkyRegister& dst,
                                                 const StinkyRegister& src,
                                                 const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCndMaskB32(const StinkyRegister& dst,
                                                    const StinkyRegister& src0,
                                                    const StinkyRegister& src1,
                                                    const std::string&    comment = "");

        // ========================================================================
        // Vector Shift Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> VLShiftLeftB16(const StinkyRegister& dst,
                                                       const StinkyRegister& src0,
                                                       const StinkyRegister& src1,
                                                       const std::string&    comment = "");

        std::vector<StinkyInstruction*> VLShiftLeftB32(const StinkyRegister& dst,
                                                       const StinkyRegister& src0,
                                                       const StinkyRegister& src1,
                                                       const std::string&    comment = "");

        std::vector<StinkyInstruction*> VLShiftRightB32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment = "");

        std::vector<StinkyInstruction*> VLShiftLeftB64(const StinkyRegister& dst,
                                                       const StinkyRegister& src0,
                                                       const StinkyRegister& src1,
                                                       const std::string&    comment = "");

        std::vector<StinkyInstruction*> VLShiftRightB64(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment = "");

        std::vector<StinkyInstruction*> VAShiftRightI32(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment = "");

        // ========================================================================
        // Vector Move/Utility Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> VMovB32(const StinkyRegister& dst,
                                                const StinkyRegister& src,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VSwapB32(const StinkyRegister& dst,
                                                 const StinkyRegister& src,
                                                 const std::string&    comment = "");

        std::vector<StinkyInstruction*> VPackF16toB32(const StinkyRegister& dst,
                                                      const StinkyRegister& src0,
                                                      const StinkyRegister& src1,
                                                      const std::string&    comment = "");

        std::vector<StinkyInstruction*> VPermB32(const StinkyRegister& dst,
                                                 const StinkyRegister& src0,
                                                 const StinkyRegister& src1,
                                                 const StinkyRegister& src2,
                                                 const std::string&    comment = "");

        // ========================================================================
        // Vector Bit Field Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> VBfeI32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const StinkyRegister& src2,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VBfeU32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const StinkyRegister& src2,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> VBfiB32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const StinkyRegister& src2,
                                                const std::string&    comment = "");

        // ========================================================================
        // Vector Accumulator Access Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> VAccvgprReadB32(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment = "");

        std::vector<StinkyInstruction*> VAccvgprWrite(const StinkyRegister& dst,
                                                      const StinkyRegister& src,
                                                      const std::string&    comment = "");

        std::vector<StinkyInstruction*> VAccvgprWriteB32(const StinkyRegister& dst,
                                                         const StinkyRegister& src,
                                                         const std::string&    comment = "");

        std::vector<StinkyInstruction*> VReadfirstlaneB32(const StinkyRegister& dst,
                                                          const StinkyRegister& src,
                                                          const std::string&    comment = "");

        // ========================================================================
        // Branch Instructions
        // ========================================================================

        /**
         * @brief Unconditional branch to label.
         * Instruction: s_branch <label>
         */
        std::vector<StinkyInstruction*> SBranch(const std::string& labelName,
                                                const std::string& comment = "");

        /**
         * @brief Conditional branch if SCC == 0.
         * Instruction: s_cbranch_scc0 <label>
         */
        std::vector<StinkyInstruction*> SCBranchSCC0(const std::string& labelName,
                                                     const std::string& comment = "");

        /**
         * @brief Conditional branch if SCC == 1.
         * Instruction: s_cbranch_scc1 <label>
         */
        std::vector<StinkyInstruction*> SCBranchSCC1(const std::string& labelName,
                                                     const std::string& comment = "");

        /**
         * @brief Conditional branch if VCC != 0.
         * Instruction: s_cbranch_vccnz <label>
         */
        std::vector<StinkyInstruction*> SCBranchVCCNZ(const std::string& labelName,
                                                      const std::string& comment = "");

        /**
         * @brief Conditional branch if VCC == 0.
         * Instruction: s_cbranch_vccz <label>
         */
        std::vector<StinkyInstruction*> SCBranchVCCZ(const std::string& labelName,
                                                     const std::string& comment = "");

        /**
         * @brief Set program counter to value in sgpr.
         * Instruction: s_setpc_b64 <sgpr>
         */
        std::vector<StinkyInstruction*> SSetPCB64(const StinkyRegister& src,
                                                  const std::string&    comment = "");

        /**
         * @brief Swap program counter with value in sgpr.
         * Instruction: s_swappc_b64 <dst_sgpr>, <src_sgpr>
         */
        std::vector<StinkyInstruction*> SSwapPCB64(const StinkyRegister& dst,
                                                   const StinkyRegister& src,
                                                   const std::string&    comment = "");

        /**
         * @brief Conditional branch if EXEC == 0.
         * Instruction: s_cbranch_execz <label>
         */
        std::vector<StinkyInstruction*> SCBranchExecZ(const std::string& labelName,
                                                      const std::string& comment = "");

        /**
         * @brief Conditional branch if EXEC != 0.
         * Instruction: s_cbranch_execnz <label>
         */
        std::vector<StinkyInstruction*> SCBranchExecNZ(const std::string& labelName,
                                                       const std::string& comment = "");

        // ========================================================================
        // Compare Instructions (from cmp.hpp)
        // ========================================================================

        // Scalar Compare Instructions (set SCC, no destination)
        std::vector<StinkyInstruction*> SCmpEQI32(const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SCmpEQU32(const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SCmpEQU64(const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SCmpGeI32(const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SCmpGeU32(const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SCmpGtI32(const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SCmpGtU32(const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SCmpLeI32(const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SCmpLeU32(const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SCmpLgU32(const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SCmpLgI32(const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SCmpLgU64(const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SCmpLtI32(const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SCmpLtU32(const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SBitcmp1B32(const StinkyRegister& src0,
                                                    const StinkyRegister& src1,
                                                    const std::string&    comment = "");

        // Scalar Compare with Immediate (SCMPK)
        std::vector<StinkyInstruction*>
            SCmpKEQU32(const StinkyRegister& src, int simm16, const std::string& comment = "");

        std::vector<StinkyInstruction*>
            SCmpKGeU32(const StinkyRegister& src, int simm16, const std::string& comment = "");

        std::vector<StinkyInstruction*>
            SCmpKGtU32(const StinkyRegister& src, int simm16, const std::string& comment = "");

        std::vector<StinkyInstruction*>
            SCmpKLGU32(const StinkyRegister& src, int simm16, const std::string& comment = "");

        // Vector Compare Instructions (write to mask register)
        std::vector<StinkyInstruction*> VCmpEQF32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpEQF64(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpEQU32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpEQI32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpGEF16(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpGTF16(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpGEF32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpGTF32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpGEF64(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpGTF64(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpGEI32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpGTI32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpGEU32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpGtU32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpLeU32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpLeI32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpLtI32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpLtU32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpUF32(const StinkyRegister& dst,
                                                 const StinkyRegister& src0,
                                                 const StinkyRegister& src1,
                                                 const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpNeI32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpNeU32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpNeU64(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpClassF32(const StinkyRegister& dst,
                                                     const StinkyRegister& src0,
                                                     const StinkyRegister& src1,
                                                     const std::string&    comment = "");

        // Vector Compare with EXEC modification (VCmpX)
        std::vector<StinkyInstruction*> VCmpXClassF32(const StinkyRegister& dst,
                                                      const StinkyRegister& src0,
                                                      const StinkyRegister& src1,
                                                      const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpXEqU32(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpXGeU32(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpXGtU32(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpXLeU32(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpXLeI32(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpXLtF32(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpXLtI32(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpXLtU32(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpXLtU64(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpXNeU16(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCmpXNeU32(const StinkyRegister& dst,
                                                   const StinkyRegister& src0,
                                                   const StinkyRegister& src1,
                                                   const std::string&    comment = "");

        // ========================================================================
        // Conversion Instructions (from cvt.hpp)
        // ========================================================================

        std::vector<StinkyInstruction*> VCvtF16toF32(const StinkyRegister& dst,
                                                     const StinkyRegister& src,
                                                     const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtF32toF16(const StinkyRegister& dst,
                                                     const StinkyRegister& src,
                                                     const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtF32toU32(const StinkyRegister& dst,
                                                     const StinkyRegister& src,
                                                     const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtU32toF32(const StinkyRegister& dst,
                                                     const StinkyRegister& src,
                                                     const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtI32toF32(const StinkyRegister& dst,
                                                     const StinkyRegister& src,
                                                     const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtF32toI32(const StinkyRegister& dst,
                                                     const StinkyRegister& src,
                                                     const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtFP8toF32(const StinkyRegister& dst,
                                                     const StinkyRegister& src,
                                                     const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtBF8toF32(const StinkyRegister& dst,
                                                     const StinkyRegister& src,
                                                     const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtPkFP8toF32(const StinkyRegister& dst,
                                                       const StinkyRegister& src,
                                                       const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtPkBF8toF32(const StinkyRegister& dst,
                                                       const StinkyRegister& src,
                                                       const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtPkF32toFP8(const StinkyRegister& dst,
                                                       const StinkyRegister& src0,
                                                       const StinkyRegister& src1,
                                                       const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtPkF32toBF8(const StinkyRegister& dst,
                                                       const StinkyRegister& src0,
                                                       const StinkyRegister& src1,
                                                       const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtSRF32toFP8(const StinkyRegister& dst,
                                                       const StinkyRegister& src0,
                                                       const StinkyRegister& src1,
                                                       const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtSRF32toBF8(const StinkyRegister& dst,
                                                       const StinkyRegister& src0,
                                                       const StinkyRegister& src1,
                                                       const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtScalePkFP8toF16(const StinkyRegister& dst,
                                                            const StinkyRegister& src0,
                                                            const StinkyRegister& src1,
                                                            const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtScalePkBF8toF16(const StinkyRegister& dst,
                                                            const StinkyRegister& src0,
                                                            const StinkyRegister& src1,
                                                            const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtScaleFP8toF16(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtScalePkF16toFP8(const StinkyRegister& dst,
                                                            const StinkyRegister& src0,
                                                            const StinkyRegister& src1,
                                                            const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtScalePkF16toBF8(const StinkyRegister& dst,
                                                            const StinkyRegister& src0,
                                                            const StinkyRegister& src1,
                                                            const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtScaleSRF16toFP8(const StinkyRegister& dst,
                                                            const StinkyRegister& src0,
                                                            const StinkyRegister& src1,
                                                            const std::string&    comment = "");

        std::vector<StinkyInstruction*> VCvtScaleSRF16toBF8(const StinkyRegister& dst,
                                                            const StinkyRegister& src0,
                                                            const StinkyRegister& src1,
                                                            const std::string&    comment = "");

        Expected<std::vector<StinkyInstruction*>> VCvtBF16toFP32(const StinkyRegister& dst,
                                                                 const StinkyRegister& src,
                                                                 const StinkyRegister* vgprMask
                                                                 = nullptr,
                                                                 int                vi      = 0,
                                                                 const std::string& comment = "");

        std::vector<StinkyInstruction*> VCvtPkF32toBF16(const StinkyRegister& dst,
                                                        const StinkyRegister& src0,
                                                        const StinkyRegister& src1,
                                                        const std::string&    comment = "");

        // ========================================================================
        // Memory Instructions (from mem.hpp)
        // ========================================================================

        // DS (LDS) Instructions
        std::vector<StinkyInstruction*> DSLoadU8(const StinkyRegister& dst,
                                                 const StinkyRegister& addr,
                                                 const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSLoadI8(const StinkyRegister& dst,
                                                 const StinkyRegister& addr,
                                                 const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSLoadU16(const StinkyRegister& dst,
                                                  const StinkyRegister& addr,
                                                  const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSLoadI16(const StinkyRegister& dst,
                                                  const StinkyRegister& addr,
                                                  const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSLoadB32(const StinkyRegister& dst,
                                                  const StinkyRegister& addr,
                                                  const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSLoadB64(const StinkyRegister& dst,
                                                  const StinkyRegister& addr,
                                                  const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSLoadB96(const StinkyRegister& dst,
                                                  const StinkyRegister& addr,
                                                  const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSLoadB128(const StinkyRegister& dst,
                                                   const StinkyRegister& addr,
                                                   const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSLoadD16HIU8(const StinkyRegister& dst,
                                                      const StinkyRegister& addr,
                                                      const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSLoadD16HIU16(const StinkyRegister& dst,
                                                       const StinkyRegister& addr,
                                                       const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSLoadB64TrB4(const StinkyRegister& dst,
                                                      const StinkyRegister& addr,
                                                      const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSLoadB96TrB6(const StinkyRegister& dst,
                                                      const StinkyRegister& addr,
                                                      const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSLoadB64TrB8(const StinkyRegister& dst,
                                                      const StinkyRegister& addr,
                                                      const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSLoadB64TrB16(const StinkyRegister& dst,
                                                       const StinkyRegister& addr,
                                                       const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSLoad2B32(const StinkyRegister& dst,
                                                   const StinkyRegister& addr0,
                                                   const StinkyRegister& addr1,
                                                   const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSLoad2B64(const StinkyRegister& dst,
                                                   const StinkyRegister& addr0,
                                                   const StinkyRegister& addr1,
                                                   const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSStoreB8(const StinkyRegister& addr,
                                                  const StinkyRegister& src,
                                                  const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSStoreB16(const StinkyRegister& addr,
                                                   const StinkyRegister& src,
                                                   const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSStoreB32(const StinkyRegister& addr,
                                                   const StinkyRegister& src,
                                                   const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSStoreB64(const StinkyRegister& addr,
                                                   const StinkyRegister& src,
                                                   const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSStoreB96(const StinkyRegister& addr,
                                                   const StinkyRegister& src,
                                                   const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSStoreB128(const StinkyRegister& addr,
                                                    const StinkyRegister& src,
                                                    const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSStoreD16HIB8(const StinkyRegister& addr,
                                                       const StinkyRegister& src,
                                                       const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSStoreD16HIB16(const StinkyRegister& addr,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSStore2B32(const StinkyRegister& addr0,
                                                    const StinkyRegister& addr1,
                                                    const StinkyRegister& src,
                                                    const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSStore2B64(const StinkyRegister& addr0,
                                                    const StinkyRegister& addr1,
                                                    const StinkyRegister& src,
                                                    const std::string&    comment = "");
        std::vector<StinkyInstruction*> DSBPermuteB32(const StinkyRegister& dst,
                                                      const StinkyRegister& src,
                                                      const std::string&    comment = "");

        // Buffer (MUBUF) Instructions
        std::vector<StinkyInstruction*> BufferLoadU8(const StinkyRegister& dst,
                                                     const StinkyRegister& addr,
                                                     const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferLoadI8(const StinkyRegister& dst,
                                                     const StinkyRegister& addr,
                                                     const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferLoadU16(const StinkyRegister& dst,
                                                      const StinkyRegister& addr,
                                                      const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferLoadI16(const StinkyRegister& dst,
                                                      const StinkyRegister& addr,
                                                      const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferLoadB32(const StinkyRegister& dst,
                                                      const StinkyRegister& addr,
                                                      const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferLoadB64(const StinkyRegister& dst,
                                                      const StinkyRegister& addr,
                                                      const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferLoadB96(const StinkyRegister& dst,
                                                      const StinkyRegister& addr,
                                                      const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferLoadB128(const StinkyRegister& dst,
                                                       const StinkyRegister& addr,
                                                       const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferLoadD16U8(const StinkyRegister& dst,
                                                        const StinkyRegister& addr,
                                                        const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferLoadD16HIU8(const StinkyRegister& dst,
                                                          const StinkyRegister& addr,
                                                          const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferLoadD16I8(const StinkyRegister& dst,
                                                        const StinkyRegister& addr,
                                                        const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferLoadD16HII8(const StinkyRegister& dst,
                                                          const StinkyRegister& addr,
                                                          const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferLoadD16B16(const StinkyRegister& dst,
                                                         const StinkyRegister& addr,
                                                         const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferLoadD16HIB16(const StinkyRegister& dst,
                                                           const StinkyRegister& addr,
                                                           const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferStoreB8(const StinkyRegister& addr,
                                                      const StinkyRegister& src,
                                                      const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferStoreD16HIU8(const StinkyRegister& addr,
                                                           const StinkyRegister& src,
                                                           const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferStoreB16(const StinkyRegister& addr,
                                                       const StinkyRegister& src,
                                                       const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferStoreD16HIB16(const StinkyRegister& addr,
                                                            const StinkyRegister& src,
                                                            const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferStoreB32(const StinkyRegister& addr,
                                                       const StinkyRegister& src,
                                                       const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferStoreB64(const StinkyRegister& addr,
                                                       const StinkyRegister& src,
                                                       const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferStoreB96(const StinkyRegister& addr,
                                                       const StinkyRegister& src,
                                                       const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferStoreB128(const StinkyRegister& addr,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferAtomicAddF32(const StinkyRegister& dst,
                                                           const StinkyRegister& src,
                                                           const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferAtomicCmpswapB32(const StinkyRegister& dst,
                                                               const StinkyRegister& addr0,
                                                               const StinkyRegister& addr1,
                                                               const std::string&    comment = "");
        std::vector<StinkyInstruction*> BufferAtomicCmpswapB64(const StinkyRegister& dst,
                                                               const StinkyRegister& addr0,
                                                               const StinkyRegister& addr1,
                                                               const std::string&    comment = "");

        // Scalar Memory (SMEM) Instructions
        // Scalar Memory Load Instructions (size-based naming, architecture-agnostic)
        // Backend will emit correct ISA: s_load_dword (GFX9) or s_load_b32 (GFX12)
        std::vector<StinkyInstruction*> SLoadB32(const StinkyRegister& dst,
                                                 const StinkyRegister& base,
                                                 int                   offset  = 0,
                                                 const std::string&    comment = "");
        std::vector<StinkyInstruction*> SLoadB64(const StinkyRegister& dst,
                                                 const StinkyRegister& base,
                                                 int                   offset  = 0,
                                                 const std::string&    comment = "");
        std::vector<StinkyInstruction*> SLoadB128(const StinkyRegister& dst,
                                                  const StinkyRegister& base,
                                                  int                   offset  = 0,
                                                  const std::string&    comment = "");
        std::vector<StinkyInstruction*> SLoadB256(const StinkyRegister& dst,
                                                  const StinkyRegister& base,
                                                  int                   offset  = 0,
                                                  const std::string&    comment = "");
        std::vector<StinkyInstruction*> SLoadB512(const StinkyRegister& dst,
                                                  const StinkyRegister& base,
                                                  int                   offset  = 0,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SStoreB32(const StinkyRegister& addr,
                                                  const StinkyRegister& src,
                                                  const std::string&    comment = "");
        std::vector<StinkyInstruction*> SStoreB64(const StinkyRegister& addr,
                                                  const StinkyRegister& src,
                                                  const std::string&    comment = "");
        std::vector<StinkyInstruction*> SStoreB128(const StinkyRegister& addr,
                                                   const StinkyRegister& src,
                                                   const std::string&    comment = "");
        std::vector<StinkyInstruction*> SAtomicDec(const StinkyRegister& dst,
                                                   const StinkyRegister& src,
                                                   const std::string&    comment = "");

        // Flat Memory Instructions
        std::vector<StinkyInstruction*> FlatLoadU8(const StinkyRegister& dst,
                                                   const StinkyRegister& addr,
                                                   const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatLoadI8(const StinkyRegister& dst,
                                                   const StinkyRegister& addr,
                                                   const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatLoadU16(const StinkyRegister& dst,
                                                    const StinkyRegister& addr,
                                                    const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatLoadI16(const StinkyRegister& dst,
                                                    const StinkyRegister& addr,
                                                    const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatLoadD16U8(const StinkyRegister& dst,
                                                      const StinkyRegister& addr,
                                                      const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatLoadD16HIU8(const StinkyRegister& dst,
                                                        const StinkyRegister& addr,
                                                        const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatLoadD16I8(const StinkyRegister& dst,
                                                      const StinkyRegister& addr,
                                                      const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatLoadD16HII8(const StinkyRegister& dst,
                                                        const StinkyRegister& addr,
                                                        const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatLoadD16B16(const StinkyRegister& dst,
                                                       const StinkyRegister& addr,
                                                       const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatLoadD16HIB16(const StinkyRegister& dst,
                                                         const StinkyRegister& addr,
                                                         const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatLoadB32(const StinkyRegister& dst,
                                                    const StinkyRegister& addr,
                                                    const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatLoadB64(const StinkyRegister& dst,
                                                    const StinkyRegister& addr,
                                                    const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatLoadB96(const StinkyRegister& dst,
                                                    const StinkyRegister& addr,
                                                    const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatLoadB128(const StinkyRegister& dst,
                                                     const StinkyRegister& addr,
                                                     const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatStoreB8(const StinkyRegister& addr,
                                                    const StinkyRegister& src,
                                                    const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatStoreD16HIU8(const StinkyRegister& addr,
                                                         const StinkyRegister& src,
                                                         const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatStoreB16(const StinkyRegister& addr,
                                                     const StinkyRegister& src,
                                                     const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatStoreD16HIB16(const StinkyRegister& addr,
                                                          const StinkyRegister& src,
                                                          const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatStoreB32(const StinkyRegister& addr,
                                                     const StinkyRegister& src,
                                                     const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatStoreB64(const StinkyRegister& addr,
                                                     const StinkyRegister& src,
                                                     const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatStoreB96(const StinkyRegister& addr,
                                                     const StinkyRegister& src,
                                                     const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatStoreB128(const StinkyRegister& addr,
                                                      const StinkyRegister& src,
                                                      const std::string&    comment = "");
        std::vector<StinkyInstruction*> FlatAtomicCmpswapB32(const StinkyRegister& dst,
                                                             const StinkyRegister& addr0,
                                                             const StinkyRegister& addr1,
                                                             const std::string&    comment = "");

        // ========================================================================
        // Label Creation
        // ========================================================================

        /**
         * @brief Create a label instruction.
         *
         * Labels are treated as instruction modifiers in StinkyTofu, similar to comments.
         * They are emitted as `label_name:` in the assembly output.
         *
         * @param label_name Name of the label
         * @return Vector containing the label instruction
         */
        std::vector<StinkyInstruction*> createLabel(const std::string& label_name);

        // ========================================================================
        // Composite Instructions (Architecture-Aware Lowering)
        // ========================================================================

        /**
         * @brief Create V_PK_MUL_F32 instruction (@composite).
         *
         * On architectures with packed support, generates 1 v_pk_mul_f32 instruction.
         * On older architectures, generates 2 v_mul_f32 instructions.
         */
        std::vector<StinkyInstruction*> VMulPKF32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        /**
         * @brief Create V_MOV_B64 instruction (@composite).
         *
         * Falls back to 2x V_MOV_B32 if native V_MOV_B64 is not available.
         */
        std::vector<StinkyInstruction*> VMovB64(const StinkyRegister& dst,
                                                const StinkyRegister& src,
                                                const std::string&    comment = "");

        /**
         * @brief Create V_LSHLREV_B32 | V_OR_B32 (@composite).
         *
         * Performs: dst = (src0 << shiftHex) | src1
         */
        std::vector<StinkyInstruction*> VLShiftLeftOrB32(const StinkyRegister& dst,
                                                         const StinkyRegister& shiftHex,
                                                         const StinkyRegister& src0,
                                                         const StinkyRegister& src1,
                                                         const std::string&    comment = "");

        /**
         * @brief Create V_ADD_LSHL_U32 instruction (@composite).
         *
         * Performs: dst = (src0 + src1) << shiftHex
         */
        std::vector<StinkyInstruction*> VAddLShiftLeftU32(const StinkyRegister& dst,
                                                          const StinkyRegister& shiftHex,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment = "");

        /**
         * @brief Create V_LSHL_ADD_U32 instruction (@composite).
         *
         * Performs: dst = (src0 << shiftHex) + src1
         */
        std::vector<StinkyInstruction*> VLShiftLeftAddU32(const StinkyRegister& dst,
                                                          const StinkyRegister& shiftHex,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment = "");

        /**
         * @brief Create S_WAITCNT instruction (@composite).
         *
         * Generates appropriate wait instructions based on the target architecture's
         * counter capabilities. Parameters correspond to:
         * - vlcnt: VMEM load counter
         * - vscnt: VMEM store counter
         * - dscnt: LDS counter
         * - kmcnt: Scalar memory/message counter
         *
         * Use -1 to skip a particular counter.
         */
        std::vector<StinkyInstruction*> SWaitCnt(int                vlcnt   = -1,
                                                 int                vscnt   = -1,
                                                 int                dscnt   = -1,
                                                 int                kmcnt   = -1,
                                                 const std::string& comment = "");

        std::vector<StinkyInstruction*> SWaitAlu(int                va_vdst  = -1,
                                                 int                va_sdst  = -1,
                                                 int                va_ssrc  = -1,
                                                 int                hold_cnt = -1,
                                                 int                vm_vsrc  = -1,
                                                 int                va_vcc   = -1,
                                                 int                sa_sdst  = -1,
                                                 const std::string& comment  = "");

        std::vector<StinkyInstruction*> SWaitTensorcnt(int                tensorcnt = 0,
                                                       const std::string& comment   = "");

        std::vector<StinkyInstruction*> SNop(int waitState, const std::string& comment = "");

        std::vector<StinkyInstruction*> VNop(int count, const std::string& comment = "");

        std::vector<StinkyInstruction*> SEndpgm(const std::string& comment = "");

        std::vector<StinkyInstruction*> SSleep(int simm16, const std::string& comment = "");

        std::vector<StinkyInstruction*> SDcacheWb(const std::string& comment = "");

        std::vector<StinkyInstruction*> SDelayAlu(int                instid0,
                                                  int                instid0cnt,
                                                  int                instskip   = -1,
                                                  int                instid1    = -1,
                                                  int                instid1cnt = -1,
                                                  const std::string& comment    = "");

        std::vector<StinkyInstruction*> SSetPrior(int prior, const std::string& comment = "");

        std::vector<StinkyInstruction*> SSetVgprMsb(int simm16, const std::string& comment = "");

        std::vector<StinkyInstruction*> SSetVgprMsb(
            int msbSrc0, int msbSrc1, int msbSrc2, int msbDst, const std::string& comment = "");

        // ========================================================================
        // Scalar Instructions
        // ========================================================================

        /**
         * @brief Create an S_ABS_I32 instruction (scalar absolute value).
         * @param dst Destination register.
         * @param src Source operand.
         * @param comment Optional user comment.
         * @return Vector of created instructions (typically 1).
         */
        std::vector<StinkyInstruction*> SAbsI32(const StinkyRegister& dst,
                                                const StinkyRegister& src,
                                                const std::string&    comment = "");

        /**
         * @brief Create an S_BARRIER instruction (synchronization barrier).
         * @param comment Optional user comment.
         * @return Vector of created instructions (typically 1).
         */
        std::vector<StinkyInstruction*> SBarrier(const std::string& comment = "");

        // ========================================================================
        // Scalar Arithmetic Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> SMaxI32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> SMaxU32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> SMinI32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> SMinU32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> SAddI32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> SAddU32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> SAddCU32(const StinkyRegister& dst,
                                                 const StinkyRegister& src0,
                                                 const StinkyRegister& src1,
                                                 const std::string&    comment = "");

        std::vector<StinkyInstruction*> SMulI32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> SMulHII32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SMulHIU32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        Expected<std::vector<StinkyInstruction*>> SMulLOU32(const StinkyRegister& dst,
                                                            const StinkyRegister& src0,
                                                            const StinkyRegister& src1,
                                                            const std::string&    comment = "");

        std::vector<StinkyInstruction*> SSubI32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> SSubU32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> SSubBU32(const StinkyRegister& dst,
                                                 const StinkyRegister& src0,
                                                 const StinkyRegister& src1,
                                                 const std::string&    comment = "");

        Expected<std::vector<StinkyInstruction*>> SSubU64(const StinkyRegister& dst,
                                                          const StinkyRegister& src0,
                                                          const StinkyRegister& src1,
                                                          const std::string&    comment = "");

        // ========================================================================
        // Scalar Bitwise Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> SAndB32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> SAndB64(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> SAndN2B32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SOrB32(const StinkyRegister& dst,
                                               const StinkyRegister& src0,
                                               const StinkyRegister& src1,
                                               const std::string&    comment = "");

        std::vector<StinkyInstruction*> SOrB64(const StinkyRegister& dst,
                                               const StinkyRegister& src0,
                                               const StinkyRegister& src1,
                                               const std::string&    comment = "");

        std::vector<StinkyInstruction*> SXorB32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        // ========================================================================
        // Scalar Shift Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> SLShiftLeftB32(const StinkyRegister& dst,
                                                       const StinkyRegister& src,
                                                       const StinkyRegister& shift,
                                                       const std::string&    comment = "");

        std::vector<StinkyInstruction*> SLShiftRightB32(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const StinkyRegister& shift,
                                                        const std::string&    comment = "");

        std::vector<StinkyInstruction*> SLShiftLeftB64(const StinkyRegister& dst,
                                                       const StinkyRegister& src,
                                                       const StinkyRegister& shift,
                                                       const std::string&    comment = "");

        std::vector<StinkyInstruction*> SLShiftRightB64(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const StinkyRegister& shift,
                                                        const std::string&    comment = "");

        std::vector<StinkyInstruction*> SAShiftRightI32(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const StinkyRegister& shift,
                                                        const std::string&    comment = "");

        std::vector<StinkyInstruction*> SLShiftLeft1AddU32(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment = "");

        std::vector<StinkyInstruction*> SLShiftLeft2AddU32(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment = "");

        std::vector<StinkyInstruction*> SLShiftLeft3AddU32(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment = "");

        std::vector<StinkyInstruction*> SLShiftLeft4AddU32(const StinkyRegister& dst,
                                                           const StinkyRegister& src0,
                                                           const StinkyRegister& src1,
                                                           const std::string&    comment = "");

        // ========================================================================
        // Scalar Move/Control Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> SMovB32(const StinkyRegister& dst,
                                                const StinkyRegister& src,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> SMovB64(const StinkyRegister& dst,
                                                const StinkyRegister& src,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> SCMovB32(const StinkyRegister& dst,
                                                 const StinkyRegister& src,
                                                 const std::string&    comment = "");

        std::vector<StinkyInstruction*> SCMovB64(const StinkyRegister& dst,
                                                 const StinkyRegister& src,
                                                 const std::string&    comment = "");

        std::vector<StinkyInstruction*> SCSelectB32(const StinkyRegister& dst,
                                                    const StinkyRegister& src0,
                                                    const StinkyRegister& src1,
                                                    const std::string&    comment = "");

        std::vector<StinkyInstruction*> SGetPCB64(const StinkyRegister& dst,
                                                  const std::string&    comment = "");

        std::vector<StinkyInstruction*> SSetMask(const StinkyRegister& dst,
                                                 const StinkyRegister& src,
                                                 const std::string&    comment = "");

        // ========================================================================
        // Scalar Bit Manipulation Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> SFf1B32(const StinkyRegister& dst,
                                                const StinkyRegister& src,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> SBfmB32(const StinkyRegister& dst,
                                                const StinkyRegister& src0,
                                                const StinkyRegister& src1,
                                                const std::string&    comment = "");

        std::vector<StinkyInstruction*> SMovkI32(const StinkyRegister& dst,
                                                 const StinkyRegister& src,
                                                 const std::string&    comment = "");

        std::vector<StinkyInstruction*> SSExtI16toI32(const StinkyRegister& dst,
                                                      const StinkyRegister& src,
                                                      const std::string&    comment = "");

        // ========================================================================
        // Scalar Exec Mask Instructions
        // ========================================================================

        Expected<std::vector<StinkyInstruction*>> SAndSaveExecB32(const StinkyRegister& dst,
                                                                  const StinkyRegister& src,
                                                                  const std::string& comment = "");

        std::vector<StinkyInstruction*> SAndSaveExecB64(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment = "");

        Expected<std::vector<StinkyInstruction*>> SOrSaveExecB32(const StinkyRegister& dst,
                                                                 const StinkyRegister& src,
                                                                 const std::string& comment = "");

        std::vector<StinkyInstruction*> SOrSaveExecB64(const StinkyRegister& dst,
                                                       const StinkyRegister& src,
                                                       const std::string&    comment = "");

        // ========================================================================
        // Scalar Register Access Instructions
        // ========================================================================

        std::vector<StinkyInstruction*> SGetRegB32(const StinkyRegister& dst,
                                                   const StinkyRegister& src,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> SSetRegB32(const StinkyRegister& dst,
                                                   const StinkyRegister& src,
                                                   const std::string&    comment = "");

        std::vector<StinkyInstruction*> SSetRegIMM32B32(const StinkyRegister& dst,
                                                        const StinkyRegister& src,
                                                        const std::string&    comment = "");

        /**
         * @brief Create a V_PK_ADD_F32 instruction (@composite).
         *
         * Architecture-aware implementation:
         * - On architectures that support it, returns 1 v_pk_add_f32 instruction
         * - On older architectures, returns 2 v_add_f32 instructions (lowered)
         *
         * @param dst Destination register (should be 2 consecutive registers).
         * @param src0 First source operand.
         * @param src1 Second source operand.
         * @param comment Optional user comment.
         * @return Vector of created instructions (1 or 2 depending on architecture).
         */
        std::vector<StinkyInstruction*> VAddPKF32(const StinkyRegister& dst,
                                                  const StinkyRegister& src0,
                                                  const StinkyRegister& src1,
                                                  const std::string&    comment = "");

        /**
         * @brief Create a generic MFMA instruction based on data types and dimensions.
         *
         * Constructs instruction string like "v_mfma_f32_32x32x8_bf16" or "v_wmma_f32_16x16x16_bf16"
         * based on the input type, accumulator type, and matrix dimensions.
         *
         * @param instType Input data type (e.g., bf16, f16, i8, f8, etc.).
         * @param accType Accumulator type (e.g., f32, i32).
         * @param m Matrix M dimension (e.g., 16, 32).
         * @param n Matrix N dimension (e.g., 16, 32).
         * @param k Matrix K dimension (e.g., 4, 8, 16).
         * @param blocks Number of blocks (default 1, can be higher for some variants).
         * @param mfma1k Whether this is a _1k variant (default false).
         * @param acc Accumulator destination register.
         * @param a Matrix A source register.
         * @param b Matrix B source register.
         * @param acc2 Accumulator source register (default nullptr = use acc).
         * @param neg Negate operands (default false).
         * @param comment Optional user comment.
         * @return Vector of created instructions (typically 1).
         */
        Expected<std::vector<StinkyInstruction*>> createMFMA(const std::string&    instType,
                                                             const std::string&    accType,
                                                             int                   m,
                                                             int                   n,
                                                             int                   k,
                                                             int                   blocks,
                                                             bool                  mfma1k,
                                                             const StinkyRegister& acc,
                                                             const StinkyRegister& a,
                                                             const StinkyRegister& b,
                                                             const StinkyRegister* acc2 = nullptr,
                                                             bool                  neg  = false,
                                                             const std::string&    comment = "");

        /**
         * @brief Create an MX format MFMA instruction (with scale factors).
         *
         * Constructs instruction string like "v_wmma_scale_f32_16x16x128_f8f6f4" or "v_wmma_scale16_f32_16x16x128_f4"
         * for MX format matrix operations with scale factors.
         *
         * @param instType Input data type (e.g., f8, f4, f6, bf8, etc.).
         * @param accType Accumulator type (typically f32).
         * @param mxScaleATypeStr Scale format for matrix A (e.g., "e5m3", "fp8").
         * @param mxScaleBTypeStr Scale format for matrix B (e.g., "e5m3", "fp8").
         * @param m Matrix M dimension.
         * @param n Matrix N dimension.
         * @param k Matrix K dimension.
         * @param block Block size (16 or other).
         * @param acc Accumulator destination register.
         * @param a Matrix A source register.
         * @param b Matrix B source register.
         * @param acc2 Accumulator source register.
         * @param mxsa Scale factor A register.
         * @param mxsb Scale factor B register.
         * @param reuseA Matrix A reuse flag (default false).
         * @param reuseB Matrix B reuse flag (default false).
         * @param comment Optional user comment.
         * @return Vector of created instructions (typically 1).
         */
        Expected<std::vector<StinkyInstruction*>> createMXMFMA(const std::string& instType,
                                                               const std::string& accType,
                                                               const std::string& mxScaleATypeStr,
                                                               const std::string& mxScaleBTypeStr,
                                                               int                m,
                                                               int                n,
                                                               int                k,
                                                               int                block,
                                                               const StinkyRegister& acc,
                                                               const StinkyRegister& a,
                                                               const StinkyRegister& b,
                                                               const StinkyRegister& acc2,
                                                               const StinkyRegister& mxsa,
                                                               const StinkyRegister& mxsb,
                                                               bool                  reuseA = false,
                                                               bool                  reuseB = false,
                                                               const std::string&    comment = "");

        /**
         * @brief Create a sparse MFMA instruction.
         *
         * Constructs instruction string like "v_smfmac_f32_16x16x32_bf16" or "v_swmmac_f32_16x16x32_bf16"
         * for sparse matrix operations.
         *
         * @param instType Input data type (e.g., bf16, f16, i8, f8, bf8).
         * @param accType Accumulator type (e.g., f32, i32).
         * @param m Matrix M dimension.
         * @param n Matrix N dimension.
         * @param k Matrix K dimension.
         * @param blocks Number of micro-blocks (default 1).
         * @param mfma1k Whether this is a _1k variant (default false).
         * @param acc Accumulator destination register.
         * @param a Matrix A source register.
         * @param b Matrix B source register.
         * @param metadata Sparsity metadata register.
         * @param neg Negate operands (default false).
         * @param comment Optional user comment.
         * @return Vector of created instructions (typically 1).
         */
        Expected<std::vector<StinkyInstruction*>> createSMFMA(const std::string&    instType,
                                                              const std::string&    accType,
                                                              int                   m,
                                                              int                   n,
                                                              int                   k,
                                                              int                   blocks,
                                                              bool                  mfma1k,
                                                              const StinkyRegister& acc,
                                                              const StinkyRegister& a,
                                                              const StinkyRegister& b,
                                                              const StinkyRegister& metadata,
                                                              bool                  neg     = false,
                                                              const std::string&    comment = "");

        // ========================================================================
        // Assembly Directive Factory Functions (similar to VAddU32, etc.)
        // ========================================================================

        /**
     * @brief Create .set directive
     * Example: .set vgprBase, 12
     */
        AsmDirective* createSetDirective(const std::string& symbol,
                                         const std::string& value,
                                         const std::string& comment = "");

        /**
     * @brief Create .if directive
     * Example: .if WAVE_SIZE == 64
     */
        AsmDirective* createIfDirective(const std::string& condition,
                                        const std::string& comment = "");

        /**
     * @brief Create .else directive
     */
        AsmDirective* createElseDirective(const std::string& comment = "");

        /**
     * @brief Create .elseif directive
     */
        AsmDirective* createElseIfDirective(const std::string& condition,
                                            const std::string& comment = "");

        /**
     * @brief Create .endif directive
     */
        AsmDirective* createEndIfDirective(const std::string& comment = "");

        // ========================================================================
        // Macro Factory Functions
        // ========================================================================

        /**
     * @brief Create VMagicDiv macro
     * Optimized division by compile-time constant
     */
        MacroInstruction* createVMagicDivMacro(uint32_t           divisor,
                                               StinkyRegister     quotient,
                                               StinkyRegister     dividend,
                                               StinkyRegister     tmpVgpr,
                                               StinkyRegister     tmpSgpr,
                                               const std::string& comment = "");

        /**
     * @brief Create PseudoRandomGenerator macro
     * Linear congruential generator
     */
        MacroInstruction* createPseudoRandomGeneratorMacro(StinkyRegister     dst,
                                                           StinkyRegister     seed,
                                                           StinkyRegister     tmpSgpr,
                                                           const std::string& comment = "");

        /**
     * @brief Create VectorStaticDivide macro
     */
        MacroInstruction* createVectorStaticDivideMacro(StinkyRegister     dst,
                                                        StinkyRegister     dividend,
                                                        uint32_t           divisor,
                                                        StinkyRegister     tmpVgpr,
                                                        StinkyRegister     tmpSgpr,
                                                        const std::string& comment = "");

        /**
     * @brief Create VectorStaticRemainder macro
     */
        MacroInstruction* createVectorStaticRemainderMacro(StinkyRegister     dst,
                                                           StinkyRegister     dividend,
                                                           uint32_t           divisor,
                                                           StinkyRegister     tmpVgpr,
                                                           StinkyRegister     tmpSgpr,
                                                           const std::string& comment = "");

        /**
     * @brief Create ScalarStaticDivide macro
     */
        MacroInstruction* createScalarStaticDivideMacro(StinkyRegister     dst,
                                                        StinkyRegister     dividend,
                                                        uint32_t           divisor,
                                                        StinkyRegister     tmpSgpr,
                                                        const std::string& comment = "");

        /**
     * @brief Create ScalarStaticRemainder macro
     */
        MacroInstruction* createScalarStaticRemainderMacro(StinkyRegister     dst,
                                                           StinkyRegister     dividend,
                                                           uint32_t           divisor,
                                                           StinkyRegister     tmpSgpr,
                                                           const std::string& comment = "");

        /**
     * @brief Create BranchIfZero macro
     */
        MacroInstruction* createBranchIfZeroMacro(StinkyRegister     src,
                                                  const std::string& label,
                                                  const std::string& comment = "");

        /**
     * @brief Create BranchIfNotZero macro
     */
        MacroInstruction* createBranchIfNotZeroMacro(StinkyRegister     src,
                                                     const std::string& label,
                                                     const std::string& comment = "");

        /**
     * @brief Create DSInit macro
     * Initialize LDS memory
     */
        MacroInstruction* createDSInitMacro(uint32_t           sizeBytes,
                                            uint32_t           value,
                                            StinkyRegister     tmpVgpr,
                                            StinkyRegister     tmpSgpr,
                                            const std::string& comment = "");

        /**
     * @brief Create ArgumentLoader macro
     * Load kernel arguments
     */
        MacroInstruction* createArgumentLoaderMacro(StinkyRegister     dst,
                                                    StinkyRegister     kernargPtr,
                                                    uint32_t           offsetBytes,
                                                    uint32_t           sizeBytes,
                                                    const std::string& comment = "");

        struct Impl;

    private:
        // Private implementation (PIMPL) to hide internal details
        std::unique_ptr<Impl> pImpl;

        // Allow IRListModule to access implementation
        friend class IRListModule;
    };

    /**
     * @brief Module/container for a list of StinkyInstructions.
     *
     * This class represents a named IR list that can hold instructions.
     * Instructions are explicitly added to the module using the add() method.
     *
     * Example usage:
     *   auto module = st.createIRList("my_kernel");
     *   auto inst = st.VAddU32(vgpr(0), vgpr(1), vgpr(2));
     *   module->add(inst);  // Returns inst for chaining
     *   std::cout << module->emitAssembly() << std::endl;
     */
    class IRListModule
    {
    public:
        /**
         * @brief Construct a new IRListModule.
         * @param arch Architecture ID for this module.
         * @param name Module/kernel name.
         */
        IRListModule(GfxArchID arch, const std::string& name = "");

        /**
         * @brief Destructor.
         */
        ~IRListModule();

        /**
         * @brief Get the module name.
         * @return Module name string.
         */
        std::string getName() const;

        /**
         * @brief Add instruction(s) to this module.
         * @param insts Vector of instructions to add.
         * @return The same vector of instructions (for chaining).
         */
        std::vector<StinkyInstruction*> add(const std::vector<StinkyInstruction*>& insts);

        /**
         * @brief Add all instructions from another module to this module.
         * @param module The module whose instructions to append.
         */
        void addModule(const IRListModule& module);

        /**
         * @brief Emit the assembly code for all instructions in this module.
         * @param emitCycleInfo If true, emit cycle information.
         * @param emitComments If true, emit comments.
         * @return Assembly code as string.
         */
        std::string emitAssembly(bool emitCycleInfo = false, bool emitComments = true) const;

        /**
         * @brief Convert the object to its string representation.
         *
         * @return std::string A string representation of the object.
         */
        std::string toString() const;

        /**
         * @brief Remap all virtual registers in this module (in-place)
         * 
         * Walks through all instructions in the module and applies register offset
         * remapping to virtual registers. Modifies the instructions in-place.
         * 
         * Use this when you want to modify the original module (single instantiation).
         * For multiple instantiations, use cloneAndRemap() instead.
         * 
         * @param vgprOffset Offset to add to virtual VGPRs
         * @param sgprOffset Offset to add to virtual SGPRs
         * 
         * Example:
         *   auto module = st.createIRList("kernel");
         *   module->add(generateTemplateWithVirtuals(st));
         *   module->remapVirtualRegisters(10, 5);  // Virtual v0→v10, s0→s5
         */
        void remapVirtualRegisters(int vgprOffset, int sgprOffset);

        /**
         * @brief Clone this module and remap virtual registers in the clone
         * 
         * Creates a deep copy of all instructions in this module, then applies
         * register offset remapping to the cloned instructions. The original
         * module remains unchanged.
         * 
         * Use this when you want to instantiate the same template multiple times
         * with different register allocations (e.g., activation functions).
         * 
         * @param vgprOffset Offset to add to virtual VGPRs in the clone
         * @param sgprOffset Offset to add to virtual SGPRs in the clone
         * @return New module with remapped registers
         * 
         * Example:
         *   // Generate template once
         *   auto absTemplate = st.createIRList("abs_template");
         *   absTemplate->add(generateAbsWithVirtuals(st));
         *   
         *   // Instantiate multiple times
         *   auto inst1 = absTemplate->cloneAndRemap(10, 0);  // v[10:12]
         *   auto inst2 = absTemplate->cloneAndRemap(20, 0);  // v[20:22]
         *   auto inst3 = absTemplate->cloneAndRemap(30, 0);  // v[30:32]
         *   
         *   // Original template unchanged, can reuse
         */
        Expected<std::shared_ptr<IRListModule>> cloneAndRemap(int vgprOffset, int sgprOffset) const;

        struct Impl;

    private:
        std::unique_ptr<Impl> pImpl;
    };

    // ========================================================================
    // Register Helper Functions
    // ========================================================================

    /**
     * @brief Create a VGPR (Vector General Purpose Register).
     * @param idx Register index.
     * @param count Number of consecutive registers (default: 1).
     * @return StinkyRegister object representing the VGPR.
     *
     * Examples:
     *   vgpr(0)      -> v0 (single register)
     *   vgpr(0, 4)   -> v[0:3] (4 consecutive registers)
     */
    StinkyRegister vgpr(uint32_t idx, uint32_t count = 1);

    /**
     * @brief Create an SGPR (Scalar General Purpose Register).
     * @param idx Register index.
     * @param count Number of consecutive registers (default: 1).
     * @return StinkyRegister object representing the SGPR.
     *
     * Examples:
     *   sgpr(10)     -> s10 (single register)
     *   sgpr(10, 8)  -> s[10:17] (8 consecutive registers)
     */
    StinkyRegister sgpr(uint32_t idx, uint32_t count = 1);

    /**
     * @brief Create an ACC (Accumulator Register).
     * @param idx Register index.
     * @param count Number of consecutive registers (default: 1).
     * @return StinkyRegister object representing the ACC.
     *
     * Examples:
     *   acc(0)       -> a0 (single register)
     *   acc(0, 4)    -> a[0:3] (4 consecutive registers)
     */
    StinkyRegister acc(uint32_t idx, uint32_t count = 1);

    // ========================================================================
    // Assembly Directive Factory Functions (similar to VAddU32, etc.)
    // ========================================================================

    /**
     * @brief Create .set directive
     * Example: .set vgprBase, 12
     */
    AsmDirective* createSetDirective(const std::string& symbol,
                                     const std::string& value,
                                     const std::string& comment = "");

    /**
     * @brief Create .if directive
     * Example: .if WAVE_SIZE == 64
     */
    AsmDirective* createIfDirective(const std::string& condition, const std::string& comment = "");

    /**
     * @brief Create .else directive
     */
    AsmDirective* createElseDirective(const std::string& comment = "");

    /**
     * @brief Create .elseif directive
     */
    AsmDirective* createElseIfDirective(const std::string& condition,
                                        const std::string& comment = "");

    /**
     * @brief Create .endif directive
     */
    AsmDirective* createEndIfDirective(const std::string& comment = "");

    // ========================================================================
    // Macro Factory Functions
    // ========================================================================

    /**
     * @brief Create VMagicDiv macro
     * Optimized division by compile-time constant
     */
    MacroInstruction* createVMagicDivMacro(uint32_t           divisor,
                                           StinkyRegister     quotient,
                                           StinkyRegister     dividend,
                                           StinkyRegister     tmpVgpr,
                                           StinkyRegister     tmpSgpr,
                                           const std::string& comment = "");

    /**
     * @brief Create PseudoRandomGenerator macro
     * Linear congruential generator
     */
    MacroInstruction* createPseudoRandomGeneratorMacro(StinkyRegister     dst,
                                                       StinkyRegister     seed,
                                                       StinkyRegister     tmpSgpr,
                                                       const std::string& comment = "");

    /**
     * @brief Create VectorStaticDivide macro
     */
    MacroInstruction* createVectorStaticDivideMacro(StinkyRegister     dst,
                                                    StinkyRegister     dividend,
                                                    uint32_t           divisor,
                                                    StinkyRegister     tmpVgpr,
                                                    StinkyRegister     tmpSgpr,
                                                    const std::string& comment = "");

    /**
     * @brief Create VectorStaticRemainder macro
     */
    MacroInstruction* createVectorStaticRemainderMacro(StinkyRegister     dst,
                                                       StinkyRegister     dividend,
                                                       uint32_t           divisor,
                                                       StinkyRegister     tmpVgpr,
                                                       StinkyRegister     tmpSgpr,
                                                       const std::string& comment = "");

    /**
     * @brief Create ScalarStaticDivide macro
     */
    MacroInstruction* createScalarStaticDivideMacro(StinkyRegister     dst,
                                                    StinkyRegister     dividend,
                                                    uint32_t           divisor,
                                                    StinkyRegister     tmpSgpr,
                                                    const std::string& comment = "");

    /**
     * @brief Create ScalarStaticRemainder macro
     */
    MacroInstruction* createScalarStaticRemainderMacro(StinkyRegister     dst,
                                                       StinkyRegister     dividend,
                                                       uint32_t           divisor,
                                                       StinkyRegister     tmpSgpr,
                                                       const std::string& comment = "");

    /**
     * @brief Create BranchIfZero macro
     */
    MacroInstruction* createBranchIfZeroMacro(StinkyRegister     src,
                                              const std::string& label,
                                              const std::string& comment = "");

    /**
     * @brief Create BranchIfNotZero macro
     */
    MacroInstruction* createBranchIfNotZeroMacro(StinkyRegister     src,
                                                 const std::string& label,
                                                 const std::string& comment = "");

    /**
     * @brief Create DSInit macro
     * Initialize LDS memory
     */
    MacroInstruction* createDSInitMacro(uint32_t           sizeBytes,
                                        uint32_t           value,
                                        StinkyRegister     tmpVgpr,
                                        StinkyRegister     tmpSgpr,
                                        const std::string& comment = "");

    /**
     * @brief Create ArgumentLoader macro
     * Load kernel arguments
     */
    MacroInstruction* createArgumentLoaderMacro(StinkyRegister     dst,
                                                StinkyRegister     kernargPtr,
                                                uint32_t           offsetBytes,
                                                uint32_t           sizeBytes,
                                                const std::string& comment = "");

} // namespace stinkytofu
