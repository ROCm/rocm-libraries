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

#pragma once

#include <bit>
#include <cassert>
#include <concepts>

#include <rocRoller/CodeGen/Instruction_fwd.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/InstructionValues/LDSAllocator.hpp>
#include <rocRoller/InstructionValues/RegisterAllocator_fwd.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    /**
     * @brief
     *
     * TODO: Figure out how to handle multi-dimensional arrays of registers
     */
    namespace Register
    {
        struct RegisterId
        {
            Type regType;
            int  regIndex;
            int  ldsSize = 0;

            auto        operator<=>(RegisterId const&) const = default;
            std::string toString() const;
        };
        struct RegisterIdHash
        {
            size_t operator()(RegisterId const& regId) const noexcept
            {
                size_t h1 = static_cast<size_t>(regId.regType);
                size_t h2 = static_cast<size_t>(regId.regIndex);
                size_t h3 = static_cast<size_t>(regId.ldsSize);
                return h1 | (h2 << std::bit_width(static_cast<unsigned int>(Type::Count)))
                       | (h3 << 32);
            }
        };

        std::string TypePrefix(Type t);

        /**
         * Returns a register type suitable for holding the result of an arithmetic
         * operation between two types.  Generally Literal -> Scalar -> Vector.
         */
        constexpr Type PromoteType(Type lhs, Type rhs);

        /**
         * @brief Says whether a Register::Type represents an actual register, as opposed
         * to a literal, or other non-register value.
         *
         * @param t Register::Type to test
         * @return true If the type represents an actual register
         * @return false If the type does not represent an actual register
         */
        constexpr bool IsRegister(Type t);

        /**
         * @brief Says whether a Register::Type represents a special register.
         *
         * @param t Register::Type to test
         * @return true If the type represents a special
         * @return false If the type does not represent a special register.
         */
        constexpr bool IsSpecial(Type t);

        /**
         * Represents a single value (or single value per lane) stored in one or more registers,
         * or a literal value, or a one-dimensional array of registers suitable for use in a
         * MFMA or similar instruction.
         *
         * Maintains a `shared_ptr` reference to the `Allocation` object.
         *
         */
        struct Value : public std::enable_shared_from_this<Value>
        {
        public:
            Value();

            ~Value();

            template <CCommandArgumentValue T>
            static ValuePtr Literal(T const& value);

            static ValuePtr Literal(CommandArgumentValue const& value);

            static ValuePtr Label(const std::string& label);

            static ValuePtr NullLiteral();

            /**
             * Placeholder value to be filled in later.
             */
            static ValuePtr Placeholder(ContextPtr        ctx,
                                        Type              regType,
                                        VariableType      varType,
                                        int               count,
                                        AllocationOptions allocOptions = {});

            static ValuePtr WavefrontPlaceholder(ContextPtr context, int count = 1);

            static ValuePtr AllocateLDS(ContextPtr   ctx,
                                        VariableType varType,
                                        int          count,
                                        unsigned int alignment    = 4,
                                        uint         paddingBytes = 0);

            constexpr AllocationState allocationState() const;

            AllocationPtr allocation() const;

            std::vector<int> allocationCoord() const;

            //> Returns a new instruction that only allocates registers for this value.
            Instruction allocate();
            void        allocate(Instruction& inst);

            bool canAllocateNow() const;
            void allocateNow();

            constexpr bool isPlaceholder() const;
            bool           isZeroLiteral() const;

            constexpr bool isSpecial() const;
            constexpr bool isTTMP() const;
            constexpr bool isSCC() const;
            bool           isVCC() const;
            constexpr bool isExec() const;

            /**
             * Asserts that `this` is in a valid state to be used as an operand to an instruction.
             */
            void assertCanUseAsOperand() const;
            bool canUseAsOperand() const;

            /**
             * Returns a new unallocated RegisterValue with the same characteristics (register type,
             * data type, count, etc.)
             */
            ValuePtr placeholder() const;

            /**
             * Returns a new unallocated Value with the specified register type but
             * the same other properties.
             */
            ValuePtr placeholder(Type regType, AllocationOptions allocOptions) const;

            Type         regType() const;
            VariableType variableType() const;

            void setVariableType(VariableType value);

            void        toStream(std::ostream& os) const;
            std::string toString() const;
            std::string description() const;

            Value(ContextPtr        ctx,
                  Type              regType,
                  VariableType      variableType,
                  int               count,
                  AllocationOptions options = {});

            Value(ContextPtr         ctx,
                  Type               regType,
                  VariableType       variableType,
                  std::vector<int>&& coord);

            template <std::ranges::input_range T>
            Value(ContextPtr ctx, Type regType, VariableType variableType, T const& coord);

            Value(AllocationPtr alloc, Type regType, VariableType variableType, int count);

            Value(AllocationPtr      alloc,
                  Type               regType,
                  VariableType       variableType,
                  std::vector<int>&& coord);

            template <std::ranges::input_range T>
            Value(AllocationPtr alloc, Type regType, VariableType variableType, T& coord);

            std::string name() const;
            void        setName(std::string name);

            void setReadOnly();
            bool readOnly() const;

            /**
             * Return negated copy.
             */
            ValuePtr negate() const;

            /**
             * Return subset of 32bit registers from multi-register values; always DataType::Raw32.
             *
             * For example,
             *
             *   auto v = Value(Register::Type::Vector, DataType::Double, count=4);
             *
             * represents 4 64-bit floating point numbers and spans 8
             * 32-bit registers.  Then
             *
             *   v->subset({1})
             *
             * would give v1, a single 32-bit register.
             */
            template <std::ranges::forward_range T>
            ValuePtr subset(T const& indices) const;

            template <std::integral T>
            ValuePtr subset(std::initializer_list<T> indices) const;

            /**
             * Splits the registers allocated into individual values.
             *
             * For each entry in `indices`, will return a `Value` that now has ownership
             * over those individual registers.
             *
             * The indices may not have any overlap, and any registers assigned to `this`
             * that are not used will be freed.

             * `this` will be in an unallocated state.
             *
             */
            std::vector<ValuePtr> split(std::vector<std::vector<int>> const& indices);

            /**
             * Returns true if `this` and `other` share any registers or LDS addresses.
             */
            bool intersects(Value const& other) const;

            /**
             * Returns true if `this` and `other` share any registers or LDS addresses.
             */
            bool intersects(ValuePtr const& other) const;

            /**
             * Return sub-elements of multi-value values.
             *
             * For example,
             *
             *   auto v = Value(Register::Type::Vector, DataType::Double, count=4);
             *
             * represents 4 64-bit floating point numbers and spans 8
             * 32-bit registers.  Then
             *
             *   v->element({1})
             *
             * would give v[2:3], a single 64-bit floating point value
             * (that spans two 32-bit registers).
             */
            template <std::ranges::forward_range T>
            ValuePtr element(T const& indices) const;
            template <typename T>
            std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, ValuePtr>
                element(std::initializer_list<T> indices) const;

            size_t registerCount() const;
            size_t valueCount() const;

            bool           hasContiguousIndices() const;
            Generator<int> registerIndices() const;

            Generator<RegisterId> getRegisterIds() const;

            std::string getLiteral() const;
            std::string getConstant() const;

            /**
             * Return a literal's actual value.
             */
            CommandArgumentValue getLiteralValue() const;

            std::string getLabel() const;

            std::shared_ptr<LDSAllocation> getLDSAllocation() const;

            rocRoller::Expression::ExpressionPtr expression();

            ContextPtr context() const;

            /**
             * @brief Determine if the other registers are the same as this one, including datatype
             *
             */
            bool sameAs(ValuePtr) const;

            /**
             * @brief Determine if the registers are the same as this one (ignoring datatype)
             *
             */
            bool sameRegistersAs(ValuePtr) const;

            /**
             * Return the bitfield located at bitOffset with bitWidth bits
             * of the 32-bit register represented by this value.
             *
             * For example, a register that contains 0xDEADBEAF is represented by value r then
             * r.bitfield(8, 8) is a ValuePtr that refers to the bitfield with value 0xBE.
             *
             * Throws FatalError if:
             *  - this value is not backed by a register;
             *  - bitOffset is out-of-range of bits in this value;
             *  - bitWidth is zero or greater than number of bits in a register; and/or
             *  - the bitfield would cross register boundaries.
             */
            ValuePtr bitfield(uint8_t bitOffset, uint8_t bitWidth) const;

            /**
             * Return contiguous segments of a packed datatype denotated by indices.
             * Throws FatalError if this value data type is not packed.
             */
            template <std::ranges::forward_range T>
            ValuePtr segment(T const& indices) const;
            template <typename T>
            std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, ValuePtr>
                segment(std::initializer_list<T> indices) const;

            /**
             * Returns true if Value is a bitfield, i.e. has bitOffset &
             * bitWitdh, and false otherwise.
             */
            bool isBitfield() const;

            /**
             * Returns the bit offset of this bitfield Value.
             * Throws FatalError if this is not a bitfield Value.
             */
            uint8_t getBitOffset() const;

            /**
             * Returns the bit width (length in bits) of this bitfield.
             * Throws FatalError if this is not a bitfield Value.
             */
            uint8_t getBitWidth() const;

        private:
            /**
             * Implementation of toString() for general-purpose registers.
             */
            void gprString(std::ostream& os) const;

            /**
             * Implementation of toString() for special registers.
             */
            void specialString(std::ostream& os) const;

            /**
             * Must only be called during `split()`.
             * Creates a new Allocation which consists of only the registers assigned to `this`.
             * The parent `Value` and `Allocation` will be left in an invalid state.
             */
            void takeAllocation();

            friend class Allocation;

            std::weak_ptr<Context> m_context;

            std::string m_name;

            std::string m_label;

            CommandArgumentValue m_literalValue;

            AllocationPtr                  m_allocation;
            std::shared_ptr<LDSAllocation> m_ldsAllocation;

            Type         m_regType = Type::Count;
            VariableType m_varType;
            bool         m_negate = false;

            /**
             * Offset and bit-width of the bitfielf in the 32-bit register represented by this value.
             * Multi-register values must have empty bitOffset & bitWidth, otherwise they are invalid.
             */
            std::optional<uint8_t> m_bitOffset;
            std::optional<uint8_t> m_bitWidth;

            /**
             * Pulls values from the allocation
             */
            std::vector<int> m_allocationCoord;
            /**
             * If true, m_indices contains a contiguous set of
             * numbers, so we can represent as a range, e.g. v[0:3]
             * If false, we must be represented as a list, e.g. [v0, v2, v5]
             * If no value, state is unknown and we will check when converting to string.
             */
            mutable std::optional<bool> m_contiguousIndices;

            void updateContiguousIndices() const;
        };

        ValuePtr Representative(std::initializer_list<ValuePtr> values);

        /**
         * Represents one (possible) allocation of register(s) that are thought of collectively.
         *
         * TODO: Make not copyable, enforce construction through shared_ptr
         */
        struct Allocation : public std::enable_shared_from_this<Allocation>
        {
            Allocation(ContextPtr        context,
                       Type              regType,
                       VariableType      variableType,
                       int               count   = 1,
                       AllocationOptions options = {});

            ~Allocation();

            static AllocationPtr
                SameAs(Value const& val, std::string name, AllocationOptions const& options);

            Instruction allocate();

            bool canAllocateNow() const;
            void allocateNow();

            Type regType() const;

            AllocationState allocationState() const;

            ValuePtr operator*();

            std::string descriptiveComment(std::string const& prefix) const;

            int               registerCount() const;
            AllocationOptions options() const;

            std::vector<int> const& registerIndices() const;

            void setAllocation(std::shared_ptr<Allocator> allocator,
                               std::vector<int> const&    registers);
            void setAllocation(std::shared_ptr<Allocator> allocator, std::vector<int>&& registers);

            void free();

            std::string name() const;
            void        setName(std::string name);

            std::optional<int> controlOp() const;
            void               setControlOp(int op);

            void setReadOnly();
            bool readOnly() const;

        private:
            friend class Value;
            friend class Allocator;

            bool m_readOnly = false;

            std::weak_ptr<Context> m_context;

            Type         m_regType;
            VariableType m_variableType;

            std::optional<int> m_controlOp;

            AllocationOptions m_options;

            int m_valueCount;
            int m_registerCount;

            AllocationState m_allocationState = AllocationState::Unallocated;

            std::shared_ptr<Allocator> m_allocator;
            std::vector<int>           m_registerIndices;

            std::string m_name;

            void setRegisterCount();
        };

        std::string   toString(AllocationOptions const& opts);
        std::ostream& operator<<(std::ostream& stream, AllocationOptions const& opts);
    }
}

#include <rocRoller/InstructionValues/Register_impl.hpp>
