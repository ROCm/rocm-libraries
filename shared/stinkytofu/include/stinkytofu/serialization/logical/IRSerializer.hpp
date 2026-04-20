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

#include <iostream>
#include <string>
#include <vector>

#include "stinkytofu/serialization/asm/PatternParser.hpp"

namespace stinkytofu {
/**
 * @brief Serializer for intrinsics to/from binary bitcode format (.st.bc)
 *
 * This serializer converts intrinsic definitions (Pattern objects)
 * to a binary format for storage and later deserialization.
 *
 * Binary Format (.st.bc):
 *   Magic Number: "STBC" (4 bytes)
 *   Version: uint32_t
 *   Number of Intrinsics: uint32_t
 *   For each intrinsic:
 *     - Name length: uint32_t
 *     - Name: string
 *     - Num arguments: uint32_t
 *     - Arguments: [name_len, name, reg_type_len, reg_type]*
 *     - Num instructions: uint32_t
 *     - Instructions: [dest_len, dest, op_len, op, num_operands, [operand]*]*
 *     - Comment length: uint32_t
 *     - Comment: string
 *     - Python binding: uint8_t (0/1)
 */
class IRSerializer {
   public:
    /**
     * @brief Serialize an intrinsic definition to binary stream
     *
     * @param name Intrinsic name
     * @param arguments List of intrinsic arguments
     * @param instructions List of high-level IR instructions
     * @param comment Documentation comment
     * @param pythonBinding Whether to generate Python bindings
     * @param out Output stream
     */
    static void serializeIntrinsic(const std::string& name,
                                   const std::vector<IntrinsicArgument>& arguments,
                                   const std::vector<IntrinsicInstruction>& instructions,
                                   const std::string& comment, bool pythonBinding,
                                   std::ostream& out);

    /**
     * @brief Serialize multiple intrinsics to .st.bc file
     *
     * @param intrinsics Vector of Pattern objects (intrinsic type)
     * @param filename Output filename (.st.bc)
     * @return true if successful, false otherwise
     */
    static bool serializeToFile(const std::vector<Pattern>& intrinsics,
                                const std::string& filename);

    /**
     * @brief Deserialize intrinsics from binary stream
     *
     * @param in Input stream
     * @return Vector of Pattern objects (intrinsic type)
     */
    static std::vector<Pattern> deserializeIntrinsics(std::istream& in);

    /**
     * @brief Deserialize intrinsics from .st.bc file
     *
     * @param filename Input filename (.st.bc)
     * @return Vector of Pattern objects (intrinsic type)
     */
    static std::vector<Pattern> deserializeFromFile(const std::string& filename);

   private:
    // Magic number for .st.bc files: "STBC"
    static constexpr uint32_t MAGIC_NUMBER = 0x43425453;  // 'STBC' in little-endian
    static constexpr uint32_t VERSION = 1;

    // Helper methods for binary I/O
    static void writeUInt32(std::ostream& out, uint32_t value);
    static void writeUInt8(std::ostream& out, uint8_t value);
    static void writeString(std::ostream& out, const std::string& str);

    static uint32_t readUInt32(std::istream& in);
    static uint8_t readUInt8(std::istream& in);
    static std::string readString(std::istream& in, uint32_t length);
};

}  // namespace stinkytofu
