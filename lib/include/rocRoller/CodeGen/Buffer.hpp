/**
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include <functional>
#include <memory>

#include "Instruction.hpp"

#include "../Context.hpp"
#include "../Utilities/Comparison.hpp"
#include "../Utilities/Component.hpp"
#include "../Utilities/Generator.hpp"

namespace rocRoller
{
    class BufferDescriptor
    {
    public:
        BufferDescriptor(std::shared_ptr<Context> context);
        Generator<Instruction> setup();
        Generator<Instruction> incrementBasePointer(std::shared_ptr<Register::Value> value);
        Generator<Instruction> setBasePointer(std::shared_ptr<Register::Value> value);
        Generator<Instruction> setSize(std::shared_ptr<Register::Value> value);
        Generator<Instruction> setOptions(std::shared_ptr<Register::Value> value);

        std::shared_ptr<Register::Value> allRegisters() const;
        std::shared_ptr<Register::Value> basePointerAndStride() const;
        std::shared_ptr<Register::Value> size() const;
        std::shared_ptr<Register::Value> descriptorOptions() const;

    private:
        std::shared_ptr<Register::Value> m_bufferResourceDescriptor;
        std::shared_ptr<Context>         m_context;
    };
}
