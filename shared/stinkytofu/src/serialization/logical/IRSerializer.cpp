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

#include "stinkytofu/serialization/logical/IRSerializer.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace stinkytofu {

// Helper to write double
static void writeDouble(std::ostream& out, double value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

// Helper to read double
static double readDouble(std::istream& in) {
    double value;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

// Helper to write int64_t
static void writeInt64(std::ostream& out, int64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

// Helper to read int64_t
static int64_t readInt64(std::istream& in) {
    int64_t value;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}
//===----------------------------------------------------------------------===//
// Binary I/O Helper Methods
//===----------------------------------------------------------------------===//

void IRSerializer::writeUInt32(std::ostream& out, uint32_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void IRSerializer::writeUInt8(std::ostream& out, uint8_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void IRSerializer::writeString(std::ostream& out, const std::string& str) {
    writeUInt32(out, static_cast<uint32_t>(str.size()));
    out.write(str.data(), str.size());
}

uint32_t IRSerializer::readUInt32(std::istream& in) {
    uint32_t value;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

uint8_t IRSerializer::readUInt8(std::istream& in) {
    uint8_t value;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

std::string IRSerializer::readString(std::istream& in, uint32_t length) {
    std::string str(length, '\0');
    in.read(&str[0], length);
    return str;
}

//===----------------------------------------------------------------------===//
// Serialization
//===----------------------------------------------------------------------===//

void IRSerializer::serializeIntrinsic(const std::string& name,
                                      const std::vector<IntrinsicArgument>& arguments,
                                      const std::vector<IntrinsicInstruction>& instructions,
                                      const std::string& comment, bool pythonBinding,
                                      std::ostream& out) {
    // Write intrinsic name
    writeString(out, name);

    // Write arguments
    writeUInt32(out, static_cast<uint32_t>(arguments.size()));
    for (const auto& arg : arguments) {
        writeString(out, arg.name);
        writeString(out, arg.regType);
    }

    // Write instructions
    writeUInt32(out, static_cast<uint32_t>(instructions.size()));
    for (const auto& inst : instructions) {
        writeString(out, inst.destReg);
        writeString(out, inst.operation);
        writeUInt32(out, static_cast<uint32_t>(inst.operands.size()));
        for (const auto& operand : inst.operands) {
            // Write operand type
            writeUInt8(out, static_cast<uint8_t>(operand.type));

            // Write operand value based on type
            switch (operand.type) {
                case IntrinsicOperand::Register:
                    writeString(out, operand.registerName);
                    break;
                case IntrinsicOperand::IntLiteral:
                    writeInt64(out, operand.intValue);
                    break;
                case IntrinsicOperand::FloatLiteral:
                case IntrinsicOperand::HexLiteral:
                    writeDouble(out, operand.floatValue);
                    break;
            }
        }
    }

    // Write comment and python_binding flag
    writeString(out, comment);
    writeUInt8(out, pythonBinding ? 1 : 0);
}

bool IRSerializer::serializeToFile(const std::vector<Pattern>& intrinsics,
                                   const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        return false;
    }

    // Write header
    writeUInt32(out, MAGIC_NUMBER);
    writeUInt32(out, VERSION);

    // Count intrinsic patterns only
    uint32_t intrinsicCount = 0;
    for (const auto& pattern : intrinsics) {
        if (pattern.type == PatternType::Intrinsic) {
            intrinsicCount++;
        }
    }
    writeUInt32(out, intrinsicCount);

    // Write each intrinsic
    for (const auto& pattern : intrinsics) {
        if (pattern.type == PatternType::Intrinsic) {
            serializeIntrinsic(pattern.name, pattern.arguments, pattern.body, pattern.comment,
                               pattern.pythonBinding, out);
        }
    }

    return out.good();
}

//===----------------------------------------------------------------------===//
// Deserialization
//===----------------------------------------------------------------------===//

std::vector<Pattern> IRSerializer::deserializeIntrinsics(std::istream& in) {
    std::vector<Pattern> intrinsics;

    // Read and verify header
    uint32_t magic = readUInt32(in);
    if (magic != MAGIC_NUMBER) {
        return {};  // Invalid file format
    }

    uint32_t version = readUInt32(in);
    if (version != VERSION) {
        return {};  // Unsupported version
    }

    uint32_t count = readUInt32(in);

    // Read each intrinsic
    for (uint32_t i = 0; i < count && in.good(); ++i) {
        Pattern pattern;
        pattern.type = PatternType::Intrinsic;

        // Read name
        uint32_t nameLen = readUInt32(in);
        pattern.name = readString(in, nameLen);

        // Read arguments
        uint32_t argCount = readUInt32(in);
        for (uint32_t j = 0; j < argCount && in.good(); ++j) {
            IntrinsicArgument arg;
            uint32_t argNameLen = readUInt32(in);
            arg.name = readString(in, argNameLen);
            uint32_t argTypeLen = readUInt32(in);
            arg.regType = readString(in, argTypeLen);
            pattern.arguments.push_back(arg);
        }

        // Read instructions
        uint32_t instCount = readUInt32(in);
        for (uint32_t j = 0; j < instCount && in.good(); ++j) {
            IntrinsicInstruction inst;

            uint32_t destLen = readUInt32(in);
            inst.destReg = readString(in, destLen);

            uint32_t opLen = readUInt32(in);
            inst.operation = readString(in, opLen);

            uint32_t operandCount = readUInt32(in);
            for (uint32_t k = 0; k < operandCount && in.good(); ++k) {
                // Read operand type
                uint8_t opType = readUInt8(in);

                IntrinsicOperand operand;
                operand.type = static_cast<IntrinsicOperand::Type>(opType);

                // Read operand value based on type
                switch (operand.type) {
                    case IntrinsicOperand::Register: {
                        uint32_t len = readUInt32(in);
                        operand.registerName = readString(in, len);
                        break;
                    }
                    case IntrinsicOperand::IntLiteral:
                        operand.intValue = readInt64(in);
                        break;
                    case IntrinsicOperand::FloatLiteral:
                    case IntrinsicOperand::HexLiteral:
                        operand.floatValue = readDouble(in);
                        break;
                }

                inst.operands.push_back(operand);
            }

            pattern.body.push_back(inst);
        }

        // Read comment
        uint32_t commentLen = readUInt32(in);
        pattern.comment = readString(in, commentLen);

        // Read python_binding flag
        pattern.pythonBinding = (readUInt8(in) != 0);

        intrinsics.push_back(pattern);
    }

    return intrinsics;
}

std::vector<Pattern> IRSerializer::deserializeFromFile(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        return {};
    }

    return deserializeIntrinsics(in);
}

}  // namespace stinkytofu
