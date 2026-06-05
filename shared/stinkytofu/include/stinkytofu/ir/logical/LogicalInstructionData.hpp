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

    MFMAData(const std::string& instType_, const std::string& accType_, int m_, int n_, int k_,
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

    MXMFMAData(const std::string& instType_, const std::string& accType_,
               const std::string& mxScaleATypeStr_, const std::string& mxScaleBTypeStr_, int m_,
               int n_, int k_, int block_, bool reuseA_, bool reuseB_)
        : instType(instType_),
          accType(accType_),
          mxScaleATypeStr(mxScaleATypeStr_),
          mxScaleBTypeStr(mxScaleBTypeStr_),
          m(m_),
          n(n_),
          k(k_),
          block(block_),
          reuseA(reuseA_),
          reuseB(reuseB_) {}
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
