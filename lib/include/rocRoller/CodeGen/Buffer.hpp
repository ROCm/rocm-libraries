/**
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include <functional>
#include <memory>

#include <rocRoller/CodeGen/Instruction.hpp>

#include <rocRoller/Context.hpp>
#include <rocRoller/Utilities/Comparison.hpp>
#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    struct GFX9BufferDescriptorOptions
    {
        enum DataFormatValue
        {
            DFInvalid = 0,
            DF8,
            DF16,
            DF8_8,
            DF32,
            DF16_16,
            DF10_11_11,
            DF11_11_10,
            DF10_10_10_2,
            DF2_10_10_10,
            DF8_8_8_8,
            DF32_32,
            DF16_16_16_16,
            DF32_32_32,
            DF32_32_32_32,
            DFReserved
        };

        unsigned int    dst_sel_x : 3;
        unsigned int    dst_sel_y : 3;
        unsigned int    dst_sel_z : 3;
        unsigned int    dst_sel_w : 3;
        unsigned int    num_format : 3;
        DataFormatValue data_format : 4;
        unsigned int    user_vm_enable : 1;
        unsigned int    user_vm_mode : 1;
        unsigned int    index_stride : 2;
        unsigned int    add_tid_enable : 1;
        unsigned int    _unusedA : 3;
        unsigned int    nv : 1;
        unsigned int    _unusedB : 2;
        unsigned int    type : 2;

        GFX9BufferDescriptorOptions();
        GFX9BufferDescriptorOptions(uint32_t raw);

        void               setRawValue(uint32_t val);
        uint32_t           rawValue() const;
        Register::ValuePtr literal() const;

        std::string toString() const;

        void validate() const;
    };

    std::string   toString(GFX9BufferDescriptorOptions::DataFormatValue val);
    std::ostream& operator<<(std::ostream&                                stream,
                             GFX9BufferDescriptorOptions::DataFormatValue val);

    static_assert(sizeof(GFX9BufferDescriptorOptions) == 4);
    static_assert(GFX9BufferDescriptorOptions::DFReserved == 15);

    class BufferDescriptor
    {
    public:
        BufferDescriptor(Register::ValuePtr srd, ContextPtr context);
        BufferDescriptor(ContextPtr context);
        Generator<Instruction> setup();
        Generator<Instruction> setDefaultOpts();
        Generator<Instruction> incrementBasePointer(Register::ValuePtr value);
        Generator<Instruction> setBasePointer(Register::ValuePtr value);
        Generator<Instruction> setSize(Register::ValuePtr value);
        Generator<Instruction> setOptions(Register::ValuePtr value);

        Register::ValuePtr allRegisters() const;
        Register::ValuePtr basePointerAndStride() const;
        Register::ValuePtr size() const;
        Register::ValuePtr descriptorOptions() const;

    private:
        Register::ValuePtr m_bufferResourceDescriptor;
        ContextPtr         m_context;
    };
}
