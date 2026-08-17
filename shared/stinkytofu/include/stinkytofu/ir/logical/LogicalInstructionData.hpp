#pragma once

#include <optional>
#include <string>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
/**
 * @file LogicalInstructionData.hpp
 * @brief Data structures for special LogicalInstructions
 *
 * These structs hold the additional data for instructions that don't fit
 * the standard dest/src pattern (MFMA, Label, IntrinsicCall, etc.).
 *
 * Used via void* in LogicalInstruction::specialData_ for pure enum design.
 */

// ========================================================================
// Matrix Operations
// ========================================================================

/**
 * @brief Data for MFMA (Matrix Fused Multiply-Add) instructions
 *
 * Registers are stored in LogicalInstruction::dests and ::srcs:
 * - dests[0] = acc (accumulator destination)
 * - srcs[0] = a (matrix A)
 * - srcs[1] = b (matrix B)
 * - srcs[2] = acc2 (optional accumulator source, if present)
 */
struct MFMAData {
    std::string instType;  ///< Input data type (bf16, f16, i8, etc.)
    std::string accType;   ///< Accumulator type (f32, i32)
    int m;                 ///< Matrix M dimension
    int n;                 ///< Matrix N dimension
    int k;                 ///< Matrix K dimension
    int blocks;            ///< Number of blocks
    bool mfma1k;           ///< Whether this is a _1k variant
    bool neg;              ///< Negate operands

    /// Per-matrix input format tokens for gfx1250 f8f6f4-family WMMA, e.g.
    /// "MATRIX_FMT_FP6" (empty when the instruction needs no matrix_*_fmt).
    std::string matrixAFmt;
    std::string matrixBFmt;

    /// gfx1250 forceScaledWMMA workaround: emit the scale-instruction mnemonic
    /// (v_wmma_scale_*). rocisa gates this only on isaVersion (mfma.hpp
    /// forceScaledWMMA()), independent of the assembler f8f6f4 capability.
    bool scaled;

    /// Append the two trailing zero scale source operands (", 0, 0"). rocisa
    /// only emits these under the HasWMMA_f8f6f4 branch of getArgStr, so this
    /// is tracked separately from `scaled` (which controls only the mnemonic).
    bool scaleOperands;

    MFMAData(const std::string& instType_, const std::string& accType_, int m_, int n_, int k_,
             int blocks_, bool mfma1k_, bool neg_ = false, const std::string& matrixAFmt_ = "",
             const std::string& matrixBFmt_ = "", bool scaled_ = false, bool scaleOperands_ = false)
        : instType(instType_),
          accType(accType_),
          m(m_),
          n(n_),
          k(k_),
          blocks(blocks_),
          mfma1k(mfma1k_),
          neg(neg_),
          matrixAFmt(matrixAFmt_),
          matrixBFmt(matrixBFmt_),
          scaled(scaled_),
          scaleOperands(scaleOperands_) {}
};

/**
 * @brief Data for MXMFMA (Mixed-precision Matrix FMA) instructions
 *
 * Registers are stored in LogicalInstruction::dests and ::srcs:
 * - dests[0] = acc (accumulator destination)
 * - srcs[0] = a (matrix A)
 * - srcs[1] = b (matrix B)
 * - srcs[2] = acc2 (accumulator source)
 * - srcs[3] = mxsa (scale factor A)
 * - srcs[4] = mxsb (scale factor B)
 */
struct MXMFMAData {
    std::string instType;         ///< Input data type (f8, f4, bf8, etc.)
    std::string accType;          ///< Accumulator type (f32)
    std::string mxScaleATypeStr;  ///< Scale format for matrix A
    std::string mxScaleBTypeStr;  ///< Scale format for matrix B
    int m;                        ///< Matrix M dimension
    int n;                        ///< Matrix N dimension
    int k;                        ///< Matrix K dimension
    int block;                    ///< Block size
    bool reuseA;                  ///< Matrix A reuse flag
    bool reuseB;                  ///< Matrix B reuse flag
    std::string matrixAFmt;       ///< Per-matrix input format for A (e.g. MATRIX_FMT_FP4)
    std::string matrixBFmt;       ///< Per-matrix input format for B (e.g. MATRIX_FMT_FP4)

    MXMFMAData(const std::string& instType_, const std::string& accType_,
               const std::string& mxScaleATypeStr_, const std::string& mxScaleBTypeStr_, int m_,
               int n_, int k_, int block_, bool reuseA_, bool reuseB_,
               const std::string& matrixAFmt_ = "", const std::string& matrixBFmt_ = "")
        : instType(instType_),
          accType(accType_),
          mxScaleATypeStr(mxScaleATypeStr_),
          mxScaleBTypeStr(mxScaleBTypeStr_),
          m(m_),
          n(n_),
          k(k_),
          block(block_),
          reuseA(reuseA_),
          reuseB(reuseB_),
          matrixAFmt(matrixAFmt_),
          matrixBFmt(matrixBFmt_) {}
};

/**
 * @brief Data for SMFMA (Sparse Matrix FMA) instructions
 *
 * Registers are stored in LogicalInstruction::dests and ::srcs:
 * - dests[0] = acc (accumulator destination)
 * - srcs[0] = a (matrix A)
 * - srcs[1] = b (matrix B)
 * - srcs[2] = metadata (sparsity metadata)
 */
struct SMFMAData {
    std::string instType;  ///< Input data type (bf16, f16, i8, etc.)
    std::string accType;   ///< Accumulator type (f32, i32)
    int m;                 ///< Matrix M dimension
    int n;                 ///< Matrix N dimension
    int k;                 ///< Matrix K dimension
    int blocks;            ///< Number of blocks
    bool mfma1k;           ///< Whether this is a _1k variant
    bool neg;              ///< Negate operands

    SMFMAData(const std::string& instType_, const std::string& accType_, int m_, int n_, int k_,
              int blocks_, bool mfma1k_, bool neg_ = false)
        : instType(instType_),
          accType(accType_),
          m(m_),
          n(n_),
          k(k_),
          blocks(blocks_),
          mfma1k(mfma1k_),
          neg(neg_) {}
};

// ========================================================================
// Control Flow
// ========================================================================

/**
 * @brief Data for SWaitAlu instructions (logical IR)
 *
 * Carries the 7 optional dependency counter fields.
 * A field value of -1 means "not specified" (omit from emitted instruction).
 */
struct SWaitAluLogicalData {
    int va_vdst;
    int va_sdst;
    int va_ssrc;
    int hold_cnt;
    int vm_vsrc;
    int va_vcc;
    int sa_sdst;

    SWaitAluLogicalData(int va_vdst_ = -1, int va_sdst_ = -1, int va_ssrc_ = -1, int hold_cnt_ = -1,
                        int vm_vsrc_ = -1, int va_vcc_ = -1, int sa_sdst_ = -1)
        : va_vdst(va_vdst_),
          va_sdst(va_sdst_),
          va_ssrc(va_ssrc_),
          hold_cnt(hold_cnt_),
          vm_vsrc(vm_vsrc_),
          va_vcc(va_vcc_),
          sa_sdst(sa_sdst_) {}
};

/**
 * @brief Data for Label instructions (logical IR)
 * Note: Different from asm::LabelData which is a modifier
 */
struct LogicalLabelData {
    std::string labelName;

    explicit LogicalLabelData(const std::string& name) : labelName(name) {}
};

// ========================================================================
// High-Level Constructs
// ========================================================================

/**
 * @brief Data for IntrinsicCall instructions
 *
 * Stores the function name; registers are stored in LogicalInstruction::dests
 */
struct IntrinsicCallData {
    std::string functionName;

    explicit IntrinsicCallData(const std::string& name) : functionName(name) {}
};

}  // namespace stinkytofu
