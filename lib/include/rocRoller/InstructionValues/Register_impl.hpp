#pragma once

#include <algorithm>
#include <concepts>
#include <numeric>
#include <ranges>

#include "RegisterAllocator.hpp"

#include "../CodeGen/Instruction.hpp"
#include "../Context.hpp"
#include "../DataTypes/DataTypes.hpp"
#include "../Operations/CommandArgument.hpp"
#include "../Scheduling/Scheduling.hpp"
#include "../Utilities/Error.hpp"
#include "../Utilities/Generator.hpp"
#include "../Utilities/Settings_fwd.hpp"
#include "../Utilities/Utils.hpp"

namespace rocRoller
{

    namespace Register
    {
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

        constexpr inline Type PromoteType(Type lhs, Type rhs)
        {
            if(lhs == rhs)
                return lhs;

            if(IsRegister(lhs) && !IsRegister(rhs))
                return lhs;
            if(!IsRegister(lhs) && IsRegister(rhs))
                return rhs;

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

        inline std::string ToString(Type t)
        {
            switch(t)
            {
            case Type::Literal:
                return "Literal";
            case Type::Label:
                return "Label";
            case Type::Special:
                return "Special";
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

            default:
                throw std::runtime_error("Invalid register type!");
            }
        }

        inline std::ostream& operator<<(std::ostream& stream, Type t)
        {
            return stream << ToString(t);
        }

        inline std::string ToString(AllocationState state)
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
            return stream << ToString(state);
        }

        inline Value::Value() = default;

        inline Value::~Value() = default;

        inline Value::Value(std::shared_ptr<Context> ctx,
                            Type                     regType,
                            VariableType             variableType,
                            int                      count)
            : m_context(ctx)
            , m_regType(regType)
            , m_varType(variableType)
        {
            AssertFatal(ctx != nullptr);

            auto const   info             = DataTypeInfo::Get(variableType);
            size_t const bytesPerRegister = 4;
            m_allocationCoord             = std::vector<int>(
                count * std::max(static_cast<size_t>(1), info.elementSize / bytesPerRegister));
            std::iota(m_allocationCoord.begin(), m_allocationCoord.end(), 0);
        }

        template <std::ranges::input_range T>
        inline Value::Value(std::shared_ptr<Context> ctx,
                            Type                     regType,
                            VariableType             variableType,
                            T const&                 coord)
            : m_context(ctx)
            , m_regType(regType)
            , m_varType(variableType)
            , m_allocationCoord(coord.begin(), coord.end())
        {
            AssertFatal(ctx != nullptr);
        }

        inline Value::Value(std::shared_ptr<Context> ctx,
                            Type                     regType,
                            VariableType             variableType,
                            std::vector<int>&&       coord)
            : m_context(ctx)
            , m_regType(regType)
            , m_varType(variableType)
            , m_allocationCoord(coord)
        {
            AssertFatal(ctx != nullptr);
        }

        inline Value::Value(std::shared_ptr<Allocation> alloc,
                            Type                        regType,
                            VariableType                variableType,
                            int                         count)
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
        inline Value::Value(std::shared_ptr<Allocation> alloc,
                            Type                        regType,
                            VariableType                variableType,
                            T&                          coord)
            : m_context(alloc->m_context)
            , m_allocation(alloc)
            , m_regType(regType)
            , m_varType(variableType)
            , m_allocationCoord(coord.begin(), coord.end())
        {
            AssertFatal(m_context.lock() != nullptr);
        }

        inline Value::Value(std::shared_ptr<Allocation> alloc,
                            Type                        regType,
                            VariableType                variableType,
                            std::vector<int>&&          coord)
            : m_context(alloc->m_context)
            , m_allocation(alloc)
            , m_regType(regType)
            , m_varType(variableType)
            , m_allocationCoord(coord)
        {
            AssertFatal(m_context.lock() != nullptr);
        }

        template <CCommandArgumentValue T>
        inline std::shared_ptr<Value> Value::Literal(T const& value)
        {
            return Literal(CommandArgumentValue(value));
        }

        inline std::shared_ptr<Value> Value::Literal(CommandArgumentValue const& value)
        {
            auto v = std::make_shared<Value>();

            v->m_literalValue = value;
            v->m_varType      = rocRoller::variableType(value);
            v->m_regType      = Type::Literal;

            return v;
        }

        inline ValuePtr Value::Special(std::string const& name)
        {
            AssertFatal(!name.empty(), "Special register must have a name!");
            auto v = std::make_shared<Value>();

            v->m_specialName = name;
            v->m_regType     = Type::Special;
            v->m_varType     = {DataType::Raw32, PointerType::Value};

            return v;
        }

        inline ValuePtr Value::Special(std::string const& name, ContextPtr ctx)
        {
            AssertFatal(!name.empty(), "Special register must have a name!");
            auto v = std::make_shared<Value>();

            v->m_specialName = name;
            v->m_regType     = Type::Special;
            v->m_varType     = {DataType::Raw32, PointerType::Value};
            v->m_context     = ctx;

            return v;
        }

        inline std::shared_ptr<Value> Value::Label(const std::string& label)
        {
            auto v       = std::make_shared<Value>();
            v->m_regType = Type::Label;
            v->m_label   = label;

            return v;
        }

        inline std::shared_ptr<Value> Value::AllocateLDS(ContextPtr   ctx,
                                                         VariableType variableType,
                                                         int          count,
                                                         unsigned int alignment)
        {
            auto v               = std::make_shared<Value>();
            v->m_regType         = Type::LocalData;
            v->m_varType         = variableType;
            auto       allocator = ctx->ldsAllocator();
            auto const info      = DataTypeInfo::Get(variableType);
            v->m_ldsAllocation   = allocator->allocate(info.elementSize * count, alignment);

            return v;
        }

        inline std::shared_ptr<Value> Value::Placeholder(std::shared_ptr<Context> ctx,
                                                         Type                     regType,
                                                         VariableType             variableType,
                                                         int                      count)
        {
            AssertFatal(ctx != nullptr);
            return std::make_shared<Value>(ctx, regType, variableType, count);
        }

        inline AllocationState Value::allocationState() const
        {
            if(!IsRegister(m_regType))
                return AllocationState::NoAllocation;

            if(!m_allocation)
                return AllocationState::Unallocated;

            return m_allocation->allocationState();
        }

        inline std::shared_ptr<Allocation> Value::allocation() const
        {
            return m_allocation;
        }

        inline std::vector<int> Value::allocationCoord() const
        {
            return m_allocationCoord;
        }

        inline Instruction Value::allocate()
        {
            if(allocationState() != AllocationState::Unallocated)
                throw std::runtime_error("Unnecessary allocation");

            if(!m_allocation)
                m_allocation = Allocation::SameAs(*this, m_name);

            return Instruction::Allocate(shared_from_this());
        }

        inline void Value::allocate(Instruction& inst)
        {
            if(allocationState() != AllocationState::Unallocated)
                return;

            if(!m_allocation)
                m_allocation = Allocation::SameAs(*this, m_name);

            inst.addAllocation(m_allocation);
        }

        inline bool Value::canAllocateNow() const
        {
            auto allocation = m_allocation;
            if(allocation == nullptr)
                allocation = Allocation::SameAs(*this, m_name);

            return allocation->canAllocateNow();
        }

        inline void Value::allocateNow()
        {
            if(m_allocation == nullptr)
                m_allocation = Allocation::SameAs(*this, m_name);

            m_allocation->allocateNow();
            m_contiguousIndices.reset();
        }

        inline void Value::freeNow()
        {
            m_allocation.reset();
            updateContiguousIndices();
        }

        inline bool Value::isPlaceholder() const
        {
            return allocationState() == Register::AllocationState::Unallocated;
        }

        inline bool Value::isZeroLiteral() const
        {
            if(m_regType != Type::Literal)
                return false;

            return std::visit([](auto const& val) { return val == 0; }, m_literalValue);
        }

        inline bool Value::isSpecial() const
        {
            return m_regType == Type::Special;
        }

        inline bool Value::isSCC() const
        {
            return m_regType == Type::Special && m_specialName == "scc";
        }

        inline bool Value::isExec() const
        {
            return m_regType == Type::Special && m_specialName == "exec";
        }

        inline std::shared_ptr<Value> Value::placeholder() const
        {
            return Placeholder(m_context.lock(), m_regType, m_varType, valueCount());
        }

        inline std::shared_ptr<Value> Value::placeholder(Type regType) const
        {
            return Placeholder(m_context.lock(), regType, m_varType, valueCount());
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
            if(!m_allocation || m_allocationCoord.empty())
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

        inline void Value::gprString(std::ostream& os) const
        {
            AssertFatal(m_allocation != nullptr
                            && m_allocation->allocationState() == AllocationState::Allocated
                            && !m_allocationCoord.empty(),
                        "Tried to use unallocated register value!");

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
                os << m_label;
                return;
            case Type::Special:
                os << m_specialName;
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
            auto rv = concatenate(regType(), " ", variableType(), ": ");

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
            if(m_regType == Type::Literal)
                throw std::runtime_error("Literal values have no register indices.");

            if(m_regType == Type::Label)
                throw std::runtime_error("Label values have no register indices.");

            if(!m_allocation || m_allocation->allocationState() != AllocationState::Allocated)
                throw std::runtime_error("Can't get indices for unallocated registers.");

            for(int coord : m_allocationCoord)
                co_yield m_allocation->m_registerIndices.at(coord);
        }

        /**
         * @brief Yields RegisterId for the registers associated with this allocation
         *
         * Note: This function does not yield any ids for Literals, Labels, or unallocated registers
         *
         * @return Generator<RegisterId>
         */
        inline Generator<RegisterId> Value::getRegisterIds() const
        {
            if(m_regType == Type::Literal || m_regType == Type::Label)
            {
                co_return;
            }
            if(!m_allocation || m_allocation->allocationState() != AllocationState::Allocated)
            {
                co_return;
            }
            for(int coord : m_allocationCoord)
            {
                co_yield RegisterId(m_regType, m_allocation->m_registerIndices.at(coord));
            }
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

        inline std::shared_ptr<Context> Value::context() const
        {
            return m_context.lock();
        }

        inline size_t Value::registerCount() const
        {
            return m_allocationCoord.size();
        }

        inline size_t Value::valueCount() const
        {
            return m_allocationCoord.size() * 4 / DataTypeInfo::Get(m_varType).elementSize;
        }

        inline std::string Value::name() const
        {
            return m_name;
        }

        inline void Value::setName(std::string const& name)
        {
            m_name = name;
        }

        inline void Value::setName(std::string&& name)
        {
            m_name = std::move(name);
        }

        inline std::shared_ptr<Value> Value::negate() const
        {
            AssertFatal(IsRegister(m_regType) && hasContiguousIndices());
            auto r = std::make_shared<Value>(
                allocation(), regType(), variableType(), m_allocationCoord);
            r->m_negate = true;
            return r;
        }

        template <std::ranges::forward_range T>
        inline std::shared_ptr<Value> Value::subset(T const& indices)
        {
            AssertFatal(allocationState() == AllocationState::Allocated,
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
        inline std::shared_ptr<Value> Value::subset(std::initializer_list<T> indices)
        {
            return subset<std::initializer_list<T>>(indices);
        }

        inline bool Value::intersects(std::shared_ptr<Register::Value> input) const
        {
            if(regType() != input->regType())
                return false;

            if(allocationState() != AllocationState::Allocated
               || input->allocationState() != AllocationState::Allocated)
                return false;

            for(int a : registerIndices())
            {
                for(int b : input->registerIndices())
                {
                    if(a == b)
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        template <std::ranges::forward_range T>
        inline std::shared_ptr<Value> Value::element(T const& indices)
        {
            AssertFatal(!m_allocationCoord.empty(), ShowValue(m_allocationCoord.size()));
            auto const   info                = DataTypeInfo::Get(m_varType);
            size_t const bytesPerRegister    = 4;
            size_t const elementsPerRegister = info.elementSize / bytesPerRegister;

            std::vector<int> coords;
            for(auto i : indices)
                for(size_t j = 0; j < elementsPerRegister; ++j)
                    coords.push_back(m_allocationCoord.at(i * elementsPerRegister + j));
            return std::make_shared<Value>(m_allocation, m_regType, m_varType, coords);
        }

        template <typename T>
        std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>,
                         std::shared_ptr<Value>> inline Value::element(std::initializer_list<T>
                                                                           indices)
        {
            return element<std::initializer_list<T>>(indices);
        }

        //> TODO: Get default options from data type + target info
        inline Allocation::Allocation(std::shared_ptr<Context> context,
                                      Type                     regType,
                                      VariableType             variableType)
            : Allocation(context, regType, variableType, 1)
        {
        }

        inline Allocation::Allocation(std::shared_ptr<Context> context,
                                      Type                     regType,
                                      VariableType             variableType,
                                      int                      count)
            : m_context(context)
            , m_regType(regType)
            , m_variableType(variableType)
            , m_valueCount(count)
        {
            AssertFatal(context != nullptr);

            setRegisterCount();

            auto elementSize = DataTypeInfo::Get(variableType).elementSize;

            // TODO: Use variabletype to calculate whether alignment is needed
            if(m_variableType.pointerType == PointerType::Buffer)
            {
                m_options.alignment = 4;
                return;
            }

            if(m_regType == Type::Vector && count * elementSize > 4
               && !context->targetArchitecture().HasCapability(GPUCapability::UnalignedVGPRs))
            {
                m_options.alignment = 2;
            }
            else if(m_regType == Type::Scalar && count * elementSize > 4
                    && !context->targetArchitecture().HasCapability(GPUCapability::UnalignedSGPRs))
            {
                m_options.alignment = 2;
            }
        }

        inline Allocation::Allocation(std::shared_ptr<Context> context,
                                      Type                     regType,
                                      VariableType             variableType,
                                      int                      count,
                                      Options const&           options)
            : m_context(context)
            , m_regType(regType)
            , m_variableType(variableType)
            , m_valueCount(count)
            , m_options(options)
        {
            AssertFatal(context != nullptr);
            setRegisterCount();
        }

        inline Allocation::Allocation(std::shared_ptr<Context> context,
                                      Type                     regType,
                                      VariableType             variableType,
                                      int                      count,
                                      Options&&                options)
            : m_context(context)
            , m_regType(regType)
            , m_variableType(variableType)
            , m_valueCount(count)
            , m_options(options)
        {
            AssertFatal(context != nullptr);
            setRegisterCount();
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

        inline std::shared_ptr<Allocation> Allocation::SameAs(Value const&       val,
                                                              std::string const& name)
        {
            auto rv = std::make_shared<Allocation>(
                val.m_context.lock(), val.m_regType, val.m_varType, val.valueCount());
            rv->setName(name);
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

        //void Allocation::allocate(Instruction & inst)
        //{
        //    inst.addAllocation(*this);
        //}

        inline AllocationState Allocation::allocationState() const
        {
            return m_allocationState;
        }

        inline std::shared_ptr<Value> Allocation::operator*()
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

        inline int Allocation::registerCount() const
        {
            return m_registerCount;
        }

        inline void Allocation::setRegisterCount()
        {
            // TODO: Register count oriented data in DataTypeInfo.
            m_registerCount = m_valueCount * DataTypeInfo::Get(m_variableType).elementSize / 4;
        }

        inline Allocation::Options Allocation::options() const
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

        inline void Allocation::setName(std::string const& name)
        {
            m_name = name;
        }

        inline void Allocation::setName(std::string&& name)
        {
            m_name = std::move(name);
        }
    }

}
