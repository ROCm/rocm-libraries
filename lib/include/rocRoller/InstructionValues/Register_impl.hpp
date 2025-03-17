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

#include <algorithm>
#include <concepts>
#include <numeric>
#include <ranges>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/RegisterAllocator.hpp>
#include <rocRoller/Operations/CommandArgument.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/Settings_fwd.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{

    namespace Register
    {
        uint constexpr bitsPerRegister = 32;

        inline std::string RegisterId::toString() const
        {
            if(IsSpecial(regType))
            {
                return concatenate(regType, regIndex);
            }
            else
            {
                return concatenate(TypePrefix(regType), regIndex);
            }
        }

        inline std::string toString(RegisterId const& regId)
        {
            return regId.toString();
        }

        inline std::ostream& operator<<(std::ostream& stream, RegisterId const& regId)
        {
            return stream << toString(regId);
        }

        inline std::string TypePrefix(Type t)
        {
            switch(t)
            {
            case Type::Scalar:
                return "s";
            case Type::Vector:
                return "v";
            case Type::Accumulator:
                return "a";

            default:
                throw std::runtime_error("No prefix available for literal values");
            }
        }

        constexpr inline bool IsRegister(Type t)
        {
            switch(t)
            {
            case Type::Scalar:
            case Type::Vector:
            case Type::Accumulator:
                return true;

            default:
                return false;
            }
        }

        constexpr inline bool IsSpecial(Type t)
        {
            switch(t)
            {
            case Type::M0:
            case Type::SCC:
            case Type::VCC:
            case Type::VCC_LO:
            case Type::VCC_HI:
            case Type::EXEC:
            case Type::EXEC_LO:
            case Type::EXEC_HI:
            case Type::TTMP7:
            case Type::TTMP9:
                return true;

            default:
                return false;
            }
        }

        constexpr inline bool IsTTMP(Type t)
        {
            switch(t)
            {
            case Type::TTMP7:
            case Type::TTMP9:
                return true;

            default:
                return false;
            }
        }

        constexpr inline Type PromoteType(Type lhs, Type rhs)
        {
            if(lhs == rhs)
                return lhs;

            if(IsRegister(lhs) && !IsRegister(rhs))
                return lhs;
            if(!IsRegister(lhs) && IsRegister(rhs))
                return rhs;
            if((lhs == Type::M0 && rhs == Type::Literal)
               || (lhs == Type::Literal && rhs == Type::M0))
                return Type::M0;

            if(lhs == Type::Count || rhs == Type::Count)
            {
                return Type::Count;
            }

            if((IsTTMP(lhs) && rhs == Type::Literal) || (lhs == Type::Literal && IsTTMP(rhs)))
                return Type::Scalar;

            if(!IsRegister(lhs) && !IsRegister(rhs))
            {
                AssertFatal(lhs == rhs,
                            "Can't promote between two non-register types.",
                            ShowValue(lhs),
                            ShowValue(rhs));
                return rhs;
            }

            if(lhs == Type::Accumulator || rhs == Type::Accumulator)
                return Type::Accumulator;

            if(lhs == Type::Vector || rhs == Type::Vector)
                return Type::Vector;

            Throw<FatalError>("Invalid Register::Type combo: ", ShowValue(lhs), ShowValue(rhs));
        }

        inline std::string toString(Type t)
        {
            switch(t)
            {
            case Type::Literal:
                return "Literal";
            case Type::Label:
                return "Label";
            case Type::NullLiteral:
                return "NullLiteral";
            case Type::Scalar:
                return "SGPR";
            case Type::Vector:
                return "VGPR";
            case Type::Accumulator:
                return "ACCVGPR";
            case Type::LocalData:
                return "LDS";
            case Type::Count:
                return "Count";
            case Type::M0:
                return "M0";
            case Type::SCC:
                return "SCC";
            case Type::VCC:
                return "VCC";
            case Type::VCC_LO:
                return "VCC_LO";
            case Type::VCC_HI:
                return "VCC_HI";
            case Type::EXEC:
                return "EXEC";
            case Type::EXEC_LO:
                return "EXEC_LO";
            case Type::EXEC_HI:
                return "EXEC_HI";
            case Type::TTMP7:
                return "TTMP7";
            case Type::TTMP9:
                return "TTMP9";
            }
            Throw<FatalError>("Invalid register type!");
        }

        inline std::ostream& operator<<(std::ostream& stream, Type t)
        {
            return stream << toString(t);
        }

        inline std::string toString(AllocationState state)
        {
            switch(state)
            {
            case AllocationState::Unallocated:
                return "Unallocated";
            case AllocationState::Allocated:
                return "Allocated";
            case AllocationState::Freed:
                return "Freed";
            case AllocationState::NoAllocation:
                return "NoAllocation";
            case AllocationState::Error:
                return "Error";
            case AllocationState::Count:
                return "Count";

            default:
                throw std::runtime_error("Invalid allocation state!");
            }
        }

        inline std::ostream& operator<<(std::ostream& stream, AllocationState state)
        {
            return stream << toString(state);
        }

        inline Value::Value() = default;

        inline Value::~Value() = default;

        inline Value::Value(ContextPtr        ctx,
                            Type              regType,
                            VariableType      variableType,
                            int               count,
                            AllocationOptions options)
            : m_context(ctx)
            , m_regType(regType)
            , m_varType(variableType)
        {
            AssertFatal(ctx != nullptr);
            AssertFatal(count > 0, "Invalid register count ", ShowValue(count));

            auto const info = DataTypeInfo::Get(variableType);

            uint numRegisters = count;
            if(info.elementBits > bitsPerRegister)
                numRegisters = CeilDivide(count * info.elementBits, bitsPerRegister);
            m_allocationCoord = std::vector<int>(numRegisters);
            std::iota(m_allocationCoord.begin(), m_allocationCoord.end(), 0);

            m_allocation = Allocation::SameAs(*this, m_name, options);
        }

        template <std::ranges::input_range T>
        inline Value::Value(ContextPtr ctx, Type regType, VariableType variableType, T const& coord)
            : m_context(ctx)
            , m_regType(regType)
            , m_varType(variableType)
            , m_allocationCoord(coord.begin(), coord.end())
        {
            AssertFatal(ctx != nullptr);
            m_allocation = Allocation::SameAs(*this, m_name, {});
        }

        inline Value::Value(ContextPtr         ctx,
                            Type               regType,
                            VariableType       variableType,
                            std::vector<int>&& coord)
            : m_context(ctx)
            , m_regType(regType)
            , m_varType(variableType)
            , m_allocationCoord(coord)
        {
            AssertFatal(ctx != nullptr);
            m_allocation = Allocation::SameAs(*this, m_name, {});
        }

        inline Value::Value(AllocationPtr alloc, Type regType, VariableType variableType, int count)
            : m_context(alloc->m_context)
            , m_allocation(alloc)
            , m_regType(regType)
            , m_varType(variableType)
        {
            // auto range = std::ranges::iota_view{0, count};
            // m_allocationCoord = std::vector(range.begin(), range.end());
            m_allocationCoord = std::vector<int>(count);
            std::iota(m_allocationCoord.begin(), m_allocationCoord.end(), 0);

            AssertFatal(m_context.lock() != nullptr);
        }

        template <std::ranges::input_range T>
        inline Value::Value(AllocationPtr alloc, Type regType, VariableType variableType, T& coord)
            : m_context(alloc->m_context)
            , m_allocation(alloc)
            , m_regType(regType)
            , m_varType(variableType)
            , m_allocationCoord(coord.begin(), coord.end())
        {
            AssertFatal(m_context.lock() != nullptr);
        }

        inline Value::Value(AllocationPtr      alloc,
                            Type               regType,
                            VariableType       variableType,
                            std::vector<int>&& coord)
            : m_context(alloc->m_context)
            , m_allocation(alloc)
            , m_regType(regType)
            , m_varType(variableType)
            , m_allocationCoord(coord)
        {
            AssertFatal(m_context.lock() != nullptr);
        }

        template <CCommandArgumentValue T>
        inline ValuePtr Value::Literal(T const& value)
        {
            return Literal(CommandArgumentValue(value));
        }

        inline ValuePtr Value::Literal(CommandArgumentValue const& value)
        {
            auto v = std::make_shared<Value>();

            v->m_literalValue = value;
            v->m_varType      = rocRoller::variableType(value);
            v->m_regType      = Type::Literal;

            return v;
        }

        inline ValuePtr Value::Label(const std::string& label)
        {
            auto v       = std::make_shared<Value>();
            v->m_regType = Type::Label;
            v->m_label   = label;

            return v;
        }

        inline ValuePtr Value::NullLiteral()
        {
            auto v       = std::make_shared<Value>();
            v->m_regType = Type::NullLiteral;
            v->m_label   = "null";

            return v;
        }

        inline ValuePtr Value::AllocateLDS(ContextPtr   ctx,
                                           VariableType variableType,
                                           int          count,
                                           unsigned int alignment,
                                           uint         paddingBytes)
        {
            auto v                  = std::make_shared<Value>();
            v->m_regType            = Type::LocalData;
            v->m_varType            = variableType;
            auto       allocator    = ctx->ldsAllocator();
            auto const info         = DataTypeInfo::Get(variableType);
            auto const elementBytes = CeilDivide(info.elementBits * count, 8u);
            v->m_ldsAllocation      = allocator->allocate(elementBytes + paddingBytes, alignment);

            return v;
        }

        inline ValuePtr Value::Placeholder(ContextPtr        ctx,
                                           Type              regType,
                                           VariableType      variableType,
                                           int               count,
                                           AllocationOptions allocOptions)
        {
            AssertFatal(ctx != nullptr);
            return std::make_shared<Value>(ctx, regType, variableType, count, allocOptions);
        }

        inline constexpr AllocationState Value::allocationState() const
        {
            if(!IsRegister(m_regType))
                return AllocationState::NoAllocation;

            AssertFatal(m_allocation);

            return m_allocation->allocationState();
        }

        inline AllocationPtr Value::allocation() const
        {
            return m_allocation;
        }

        inline std::vector<int> Value::allocationCoord() const
        {
            return m_allocationCoord;
        }

        inline Instruction Value::allocate()
        {
            AssertFatal(allocationState() == AllocationState::Unallocated,
                        "Unneccessary allocation",
                        ShowValue(allocationState()));

            AssertFatal(m_allocation);

            return Instruction::Allocate(shared_from_this());
        }

        inline void Value::allocate(Instruction& inst)
        {
            if(allocationState() != AllocationState::Unallocated)
                return;

            AssertFatal(m_allocation);

            inst.addAllocation(m_allocation);
        }

        inline bool Value::canAllocateNow() const
        {
            AssertFatal(m_allocation);

            return m_allocation->canAllocateNow();
        }

        inline void Value::allocateNow()
        {
            AssertFatal(m_allocation);

            m_allocation->allocateNow();
            m_contiguousIndices.reset();
        }

        inline constexpr bool Value::isPlaceholder() const
        {
            return allocationState() == Register::AllocationState::Unallocated;
        }

        inline bool Value::isZeroLiteral() const
        {
            if(m_regType != Type::Literal)
                return false;

            return std::visit([](auto const& val) { return val == 0; }, m_literalValue);
        }

        inline constexpr bool Value::isSpecial() const
        {
            return IsSpecial(m_regType);
        }

        inline constexpr bool Value::isTTMP() const
        {
            return IsTTMP(m_regType);
        }

        inline constexpr bool Value::isSCC() const
        {
            return m_regType == Type::SCC;
        }

        inline constexpr bool Value::isExec() const
        {
            return m_regType == Type::EXEC;
        }

        inline ValuePtr Value::placeholder() const
        {
            return Placeholder(
                m_context.lock(), m_regType, m_varType, valueCount(), m_allocation->options());
        }

        inline ValuePtr Value::placeholder(Type regType, AllocationOptions allocOptions) const
        {
            return Placeholder(m_context.lock(), regType, m_varType, valueCount(), allocOptions);
        }

        inline Type Value::regType() const
        {
            return m_regType;
        }

        inline VariableType Value::variableType() const
        {
            return m_varType;
        }

        inline void Value::setVariableType(VariableType value)
        {
            // TODO: Assert we have enough registers.
            m_varType = value;
        }

        inline void Value::updateContiguousIndices() const
        {
            AssertFatal(m_allocation);

            if(m_allocationCoord.empty() || allocationState() == AllocationState::Unallocated)
            {
                m_contiguousIndices.reset();
                return;
            }

            auto iter      = m_allocationCoord.begin();
            auto prevValue = m_allocation->m_registerIndices[*iter];

            iter++;
            for(; iter != m_allocationCoord.end(); iter++)
            {
                auto myValue = m_allocation->m_registerIndices[*iter];
                if(myValue != prevValue + 1)
                {
                    m_contiguousIndices = false;
                    return;
                }
                prevValue = myValue;
            }

            m_contiguousIndices = true;
        }

        inline bool Value::canUseAsOperand() const
        {
            if(m_regType == Type::Accumulator || m_regType == Type::Scalar
               || m_regType == Type::Vector)
            {
                return allocationState() == AllocationState::Allocated;
            }

            return true;
        }

        inline void Value::assertCanUseAsOperand() const
        {
            AssertFatal(canUseAsOperand(),
                        "Tried to use unallocated register value!",
                        ShowValue(this->toString()));
        }

        inline void Value::specialString(std::ostream& os) const
        {
            switch(m_regType)
            {
            case Type::M0:
                os << "m0";
                return;
            case Type::SCC:
                os << "scc";
                return;
            case Type::VCC:
                os << "vcc";
                return;
            case Type::VCC_LO:
                os << "vcc_lo";
                return;
            case Type::VCC_HI:
                os << "vcc_hi";
                return;
            case Type::EXEC:
                os << "exec";
                return;
            case Type::EXEC_LO:
                os << "exec_lo";
                return;
            case Type::EXEC_HI:
                os << "exec_hi";
                return;
            case Type::TTMP7:
                os << "ttmp7";
                return;
            case Type::TTMP9:
                os << "ttmp9";
                return;
            default:
                break;
            }
            Throw<FatalError>("Invalid special register: ", (int)m_regType);
        }

        inline void Value::gprString(std::ostream& os) const
        {
            AssertFatal(m_regType == Type::Accumulator || m_regType == Type::Scalar
                            || m_regType == Type::Vector,
                        "gprString is only applicable for actual GPRs.");

            if(!canUseAsOperand())
            {
                os << TypePrefix(m_regType);
                os << "**UNALLOCATED**";
                return;
            }

            auto prefix = TypePrefix(m_regType);

            auto const& regIndices = m_allocation->m_registerIndices;

            if(m_negate)
            {
                AssertFatal(hasContiguousIndices());
                os << "neg(";
            }

            if(m_allocationCoord.size() == 1)
            {
                // one register, use simple syntax, e.g. 'v0'
                os << concatenate(prefix, regIndices[m_allocationCoord[0]]);
            }
            else if(hasContiguousIndices())
            {
                // contiguous range of registers, e.g. v[0:3].
                os << concatenate(prefix,
                                  "[",
                                  regIndices[m_allocationCoord.front()],
                                  ":",
                                  regIndices[m_allocationCoord.back()],
                                  "]");
            }
            else
            {
                // non-contiguous registers, e.g. [v0, v2, v6, v7]
                os << "[";
                auto coordIter = m_allocationCoord.begin();
                os << prefix << regIndices[*coordIter];
                coordIter++;

                for(; coordIter != m_allocationCoord.end(); coordIter++)
                {
                    os << ", " << prefix << regIndices.at(*coordIter);
                }
                os << "]";
            }

            if(m_negate)
            {
                os << ")";
            }
        }

        inline std::string Value::toString() const
        {
            std::stringstream oss;
            toStream(oss);
            return oss.str();
        }

        inline void Value::toStream(std::ostream& os) const
        {
            switch(m_regType)
            {
            case Type::Accumulator:
            case Type::Scalar:
            case Type::Vector:
                gprString(os);
                return;
            case Type::Label:
            case Type::NullLiteral:
                os << m_label;
                return;
            case Type::M0:
            case Type::SCC:
            case Type::VCC:
            case Type::VCC_LO:
            case Type::VCC_HI:
            case Type::EXEC:
            case Type::EXEC_LO:
            case Type::EXEC_HI:
            case Type::TTMP7:
            case Type::TTMP9:
                specialString(os);
                return;
            case Type::Literal:
                os << getLiteral();
                return;
            case Type::LocalData:
                os << "LDS:" << m_ldsAllocation->toString();
                return;
            case Type::Count:
                break;
            }

            throw std::runtime_error("Bad Register::Type");
        }

        inline std::string Value::description() const
        {
            auto rv = concatenate(regType(), " ", variableType(), " x ", valueCount(), ": ");

            if(allocationState() == AllocationState::Unallocated)
                rv += "(unallocated)";
            else
                rv += toString();

            if(!m_name.empty())
                rv = m_name + ": " + rv;
            return rv;
        }

        inline bool Value::hasContiguousIndices() const
        {
            if(!m_contiguousIndices.has_value())
                updateContiguousIndices();

            return *m_contiguousIndices;
        }

        inline Generator<int> Value::registerIndices() const
        {
            AssertFatal(m_regType != Type::Literal, "Literal values have no register indices.");
            AssertFatal(m_regType != Type::Label, "Label values have no register indices.");

            AssertFatal(allocationState() == AllocationState::Allocated,
                        "Can't get indices for unallocated registers.",
                        ShowValue(allocationState()));

            for(int coord : m_allocationCoord)
                co_yield m_allocation->m_registerIndices.at(coord);
        }

        inline std::string Value::getLiteral() const
        {
            if(m_regType != Type::Literal)
            {
                return "";
            }
            return rocRoller::toString(m_literalValue);
        }

        inline CommandArgumentValue Value::getLiteralValue() const
        {
            assert(m_regType == Type::Literal);

            return m_literalValue;
        }

        inline std::string Value::getLabel() const
        {
            return m_label;
        }

        inline std::shared_ptr<LDSAllocation> Value::getLDSAllocation() const
        {
            return m_ldsAllocation;
        }

        inline ContextPtr Value::context() const
        {
            return m_context.lock();
        }

        inline size_t Value::registerCount() const
        {
            return m_allocationCoord.size();
        }

        inline size_t Value::valueCount() const
        {
            if(DataTypeInfo::Get(m_varType).elementBits < bitsPerRegister)
                return m_allocationCoord.size();
            return m_allocationCoord.size() * bitsPerRegister
                   / DataTypeInfo::Get(m_varType).elementBits;
        }

        inline std::string Value::name() const
        {
            return m_name;
        }

        inline void Value::setName(std::string name)
        {
            if(allocationState() != AllocationState::NoAllocation)
            {
                AssertFatal(m_allocation);

                if(m_allocation->name() == "")
                    m_allocation->setName(name);
            }

            m_name = std::move(name);
        }

        inline void Value::setReadOnly()
        {
            AssertFatal(m_allocation);

            m_allocation->setReadOnly();
        }

        inline bool Value::readOnly() const
        {
            if(regType() == Type::Literal)
                return true;

            if(!m_allocation)
                return false;

            return m_allocation->readOnly();
        }

        inline ValuePtr Value::negate() const
        {
            AssertFatal(IsRegister(m_regType) && hasContiguousIndices());
            auto r = std::make_shared<Value>(
                allocation(), regType(), variableType(), m_allocationCoord);
            r->m_negate = true;
            return r;
        }

        template <std::ranges::forward_range T>
        inline ValuePtr Value::subset(T const& indices) const
        {
            AssertFatal(allocationState() != AllocationState::NoAllocation,
                        ShowValue(allocationState()));
            AssertFatal(!m_allocationCoord.empty(), ShowValue(m_allocationCoord.size()));
            std::vector<int> coords;
            for(auto i : indices)
            {
                AssertFatal(i >= 0 && (size_t)i < m_allocationCoord.size(),
                            "Register subset out of bounds.",
                            ShowValue(m_allocationCoord.size()),
                            ShowValue(i));
                coords.push_back(m_allocationCoord.at(i));
            }
            return std::make_shared<Value>(allocation(), regType(), DataType::Raw32, coords);
        }

        template <std::integral T>
        inline ValuePtr Value::subset(std::initializer_list<T> indices) const
        {
            return subset<std::initializer_list<T>>(indices);
        }

        template <std::ranges::forward_range T>
        inline ValuePtr Value::element(T const& indices) const
        {
            AssertFatal(!m_allocationCoord.empty(), ShowValue(m_allocationCoord.size()));
            auto const info = DataTypeInfo::Get(m_varType);

            auto const registersPerElement = CeilDivide(info.elementBits, bitsPerRegister);

            std::vector<int> coords;
            for(auto i : indices)
            {
                for(auto j = 0; j < registersPerElement; ++j)
                {
                    auto idx = i * registersPerElement + j;
                    AssertFatal(idx < m_allocationCoord.size(),
                                ShowValue(i),
                                ShowValue(j),
                                ShowValue(registersPerElement),
                                ShowValue(idx),
                                ShowValue(m_allocationCoord.size()));
                    coords.push_back(m_allocationCoord.at(idx));
                }
            }
            return std::make_shared<Value>(m_allocation, m_regType, m_varType, coords);
        }

        template <typename T>
        std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>,
                         ValuePtr> inline Value::element(std::initializer_list<T> indices) const
        {
            return element<std::initializer_list<T>>(indices);
        }

        inline ValuePtr Value::bitfield(uint8_t bitOffset, uint8_t bitWidth) const
        {
            AssertFatal(allocationState() != AllocationState::NoAllocation,
                        ShowValue(allocationState()));
            AssertFatal(!m_allocationCoord.empty(), ShowValue(m_allocationCoord.size()));

            // Does it make sense to allow taking a bitfield of another bitfield?
            AssertFatal(!this->isBitfield());

            AssertFatal(bitWidth != 0);
            AssertFatal(bitWidth < bitsPerRegister);

            AssertFatal(bitOffset < registerCount() * bitsPerRegister,
                        "bitOffset is greater than number of bits in this value.");

            uint registerOfBitOffset = bitOffset / bitsPerRegister;
            uint registerOfLastBit   = (bitOffset + bitWidth - 1) / bitsPerRegister;

            AssertFatal(registerOfBitOffset == registerOfLastBit,
                        "cannot get bitfield across registers.");

            std::vector<int> allocationCoord = {m_allocationCoord.at(registerOfBitOffset)};

            auto bitfieldValue
                = std::make_shared<Value>(m_allocation, m_regType, m_varType, allocationCoord);

            bitfieldValue->m_bitOffset = bitOffset - registerOfBitOffset * bitsPerRegister;
            bitfieldValue->m_bitWidth  = bitWidth;

            return bitfieldValue;
        }

        template <std::ranges::forward_range T>
        inline ValuePtr Value::segment(T const& indices) const
        {
            auto const info = DataTypeInfo::Get(m_varType);

            AssertFatal(info.packing > 1,
                        "bitfield access by index is only supported for packed types.");

            auto isContiguousRange = [](T v) -> bool {
                return std::adjacent_find(
                           v.begin(), v.end(), [](auto a, auto b) { return (b - a) != 1; })
                       == v.end();
            };

            AssertFatal(isContiguousRange(indices), "indices must refer to adjacent elements.");

            auto first = *std::ranges::begin(indices);
            auto last  = *std::ranges::prev(std::ranges::end(indices));

            uint startBitOffset = first * (info.elementBits / info.packing);
            uint endBitOffset   = (last + 1) * (info.elementBits / info.packing);
            uint bitWidth       = endBitOffset - startBitOffset;

            return bitfield(startBitOffset, bitWidth);
        }

        template <typename T>
        std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>,
                         ValuePtr> inline Value::segment(std::initializer_list<T> indices) const
        {
            return segment<std::initializer_list<T>>(indices);
        }

        inline bool Value::isBitfield() const
        {
            return this->m_bitOffset.has_value();
        }

        inline uint8_t Value::getBitOffset() const
        {
            AssertFatal(this->m_bitOffset.has_value());
            return this->m_bitOffset.value();
        }

        inline uint8_t Value::getBitWidth() const
        {
            AssertFatal(this->m_bitWidth.has_value());
            return this->m_bitWidth.value();
        }

        inline Allocation::Allocation(ContextPtr        context,
                                      Type              regType,
                                      VariableType      variableType,
                                      int               count,
                                      AllocationOptions options)
            : m_context(context)
            , m_regType(regType)
            , m_variableType(variableType)
            , m_options(options)
            , m_valueCount(count)
        {
            AssertFatal(context != nullptr);

            setRegisterCount();
            if(options.contiguousChunkWidth == Register::FULLY_CONTIGUOUS)
            {
                m_options.contiguousChunkWidth = m_registerCount;
            }
            else if(options.contiguousChunkWidth == Register::VALUE_CONTIGUOUS)
            {
                m_options.contiguousChunkWidth = CeilDivide<int>(variableType.getElementSize(), 4);
            }

            if(options.alignment <= 0)
            {
                m_options.alignment = m_variableType.registerAlignment(
                    m_regType, m_options.contiguousChunkWidth, context->targetArchitecture());
            }

            if(options.contiguousChunkWidth != Register::MANUAL)
            {
                AssertFatal(m_options.alignment <= m_options.contiguousChunkWidth,
                            ShowValue(m_options),
                            ShowValue(variableType));
            }
            else
            {
                m_options.contiguousChunkWidth = 1;
            }

            AssertFatal(m_options.contiguousChunkWidth > 0, ShowValue(m_options));
        }

        inline Allocation::~Allocation()
        {
            if(m_allocationState == AllocationState::Allocated)
            {
                auto context = m_context.lock();
                if(context && context->kernelOptions().logLevel > LogLevel::Terse)
                {
                    auto inst = Instruction::Comment(descriptiveComment("Freeing"));
                    context->schedule(inst);
                }
            }
        }

        inline AllocationPtr
            Allocation::SameAs(Value const& val, std::string name, AllocationOptions const& options)
        {
            auto rv = std::make_shared<Allocation>(
                val.m_context.lock(), val.m_regType, val.m_varType, val.valueCount(), options);
            rv->setName(std::move(name));
            return rv;
        }

        // These would need to have the shared_ptr, not just the reference/pointer.
        inline Instruction Allocation::allocate()
        {
            return Instruction::Allocate({shared_from_this()});
        }

        inline bool Allocation::canAllocateNow() const
        {
            if(m_allocationState == AllocationState::Allocated)
                return true;

            if(!IsRegister(m_regType))
                return true;

            auto allocator = m_context.lock()->allocator(m_regType);
            return allocator->canAllocate(shared_from_this());
        }

        inline void Allocation::allocateNow()
        {
            if(m_allocationState == AllocationState::Allocated)
                return;

            if(!IsRegister(m_regType))
                return;

            auto allocator = m_context.lock()->allocator(m_regType);
            allocator->allocate(shared_from_this());
        }

        inline Type Allocation::regType() const
        {
            return m_regType;
        }

        inline AllocationState Allocation::allocationState() const
        {
            return m_allocationState;
        }

        inline ValuePtr Allocation::operator*()
        {
            // auto *ptr = new Value(shared_from_this(),
            //                       m_regType, m_variableType,
            //                       std::ranges::iota_view(0, registerCount()));
            // return {ptr};
            std::vector<int> rge(registerCount());
            std::iota(rge.begin(), rge.end(), 0);
            auto rv = std::make_shared<Value>(shared_from_this(),
                                              m_regType,
                                              m_variableType,
                                              //std::ranges::iota_view(0, registerCount()));
                                              std::move(rge));

            rv->setName(m_name);

            return rv;
        }

        inline AllocationOptions AllocationOptions::FullyContiguous()
        {
            return {.contiguousChunkWidth = Register::FULLY_CONTIGUOUS};
        }

        inline std::string toString(AllocationOptions const& opts)
        {
            return concatenate("alignment: ",
                               opts.alignment,
                               ", phase: ",
                               opts.alignmentPhase,
                               ", contiguousChunkWidth: ",
                               opts.contiguousChunkWidth);
        }

        inline std::ostream& operator<<(std::ostream& stream, AllocationOptions const& opts)
        {
            return stream << toString(opts);
        }

        inline std::string Allocation::descriptiveComment(std::string const& prefix) const
        {
            std::ostringstream msg;

            // e.g.:
            // Allocated A: 4 VGPRs (Float): 0, 1, 2, 3
            auto regCount   = registerCount();
            auto typePrefix = TypePrefix(m_regType);

            msg << prefix << " " << m_name << ": " << regCount << " " << m_regType;
            if(regCount != 1)
                msg << 's';

            msg << " (" << m_variableType << ")";

            if(m_controlOp)
            {
                msg << " (op " << *m_controlOp << ")";
            }

            auto iter = m_registerIndices.begin();
            if(iter != m_registerIndices.end())
            {
                msg << ": " << typePrefix << *iter;
                iter++;

                for(; iter != m_registerIndices.end(); iter++)
                    msg << ", " << typePrefix << *iter;
            }

            msg << std::endl;

            return msg.str();
        }

        inline void Allocation::setReadOnly()
        {
            m_readOnly = true;
        }

        inline bool Allocation::readOnly() const
        {
            return m_readOnly;
        }

        inline int Allocation::registerCount() const
        {
            return m_registerCount;
        }

        inline void Allocation::setRegisterCount()
        {
            m_registerCount
                = m_valueCount
                  * static_cast<size_t>(DataTypeInfo::Get(m_variableType).registerCount);
        }

        inline AllocationOptions Allocation::options() const
        {
            return m_options;
        }

        inline void Allocation::setAllocation(std::shared_ptr<Allocator> allocator,
                                              std::vector<int> const&    registers)
        {
            auto copy = registers;

            setAllocation(allocator, std::move(copy));
        }

        inline void Allocation::setAllocation(std::shared_ptr<Allocator> allocator,
                                              std::vector<int>&&         registers)
        {
            AssertFatal(m_allocationState != AllocationState::Allocated,
                        ShowValue(m_allocationState),
                        ShowValue(registers.size()),
                        ShowValue(registerCount()));

            m_allocator       = allocator;
            m_registerIndices = std::move(registers);

            m_allocationState = AllocationState::Allocated;
        }

        inline std::vector<int> const& Allocation::registerIndices() const
        {
            return m_registerIndices;
        }

        inline void Allocation::free()
        {
            AssertFatal(m_allocationState == AllocationState::Allocated,
                        ShowValue(m_allocationState),
                        ShowValue(AllocationState::Allocated));
            AssertFatal(m_allocator != nullptr, ShowValue(m_allocator));

            m_allocator->free(m_registerIndices);
            m_registerIndices.clear();
            m_allocationState = AllocationState::Freed;
        }

        inline std::string Allocation::name() const
        {
            return m_name;
        }

        inline void Allocation::setName(std::string name)
        {
            m_name = std::move(name);
        }
    }

}
