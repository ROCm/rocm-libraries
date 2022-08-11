#pragma once

#include <cassert>
#include <concepts>

#include "LDSAllocator.hpp"
#include "RegisterAllocator_fwd.hpp"
#include "Register_fwd.hpp"

#include "../CodeGen/Instruction_fwd.hpp"
#include "../Context_fwd.hpp"
#include "../DataTypes/DataTypes.hpp"
#include "../Expression_fwd.hpp"
#include "../Operations/CommandArgument_fwd.hpp"
#include "../Scheduling/Scheduling.hpp"
#include "../Utilities/Generator.hpp"

namespace rocRoller
{

    /**
     * @brief
     *
     * TODO: Figure out how to handle multi-dimensional arrays of registers
     */
    namespace Register
    {
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
        std::string    ToString(Type t);
        std::ostream&  operator<<(std::ostream& stream, Type t);

        std::string   ToString(AllocationState state);
        std::ostream& operator<<(std::ostream& stream, AllocationState state);

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
            static std::shared_ptr<Value> Literal(T const& value);

            static std::shared_ptr<Value> Literal(CommandArgumentValue const& value);

            /**
             * Special-purpose register such as vcc or exec.
             */
            static ValuePtr Special(std::string const& name);
            static ValuePtr Special(std::string const& name, ContextPtr ctx);

            static std::shared_ptr<Value> Label(const std::string& label);

            /**
             * Placeholder value to be filled in later.
             */
            static std::shared_ptr<Value>
                Placeholder(ContextPtr ctx, Type regType, VariableType varType, int count);

            static ValuePtr WavefrontPlaceholder(ContextPtr context);

            static std::shared_ptr<Value> AllocateLDS(ContextPtr   ctx,
                                                      VariableType varType,
                                                      int          count,
                                                      unsigned int alignment = 4);

            AllocationState allocationState() const;

            std::shared_ptr<Allocation> allocation() const;

            //> Returns a new instruction that only allocates registers for this value.
            Instruction allocate();
            void        allocate(Instruction& inst);

            bool canAllocateNow() const;
            void allocateNow();
            void freeNow();

            bool isPlaceholder() const;
            bool isZeroLiteral() const;

            bool isSCC() const;
            bool isVCC() const;

            /**
             * Returns a new unallocated RegisterValue with the same characteristics (register type,
             * data type, count, etc.)
             */
            std::shared_ptr<Value> placeholder() const;

            /**
             * Returns a new unallocated Value with the specified register type but
             * the same other properties.
             */
            std::shared_ptr<Value> placeholder(Type regType) const;

            Type         regType() const;
            VariableType variableType() const;

            void setVariableType(VariableType value);

            void        toStream(std::ostream& os) const;
            std::string toString() const;
            std::string description() const;

            Value(std::shared_ptr<Context> ctx, Type regType, VariableType variableType, int count);
            Value(std::shared_ptr<Context> ctx,
                  Type                     regType,
                  VariableType             variableType,
                  std::vector<int>&&       coord);
            template <std::ranges::input_range T>
            Value(std::shared_ptr<Context> ctx,
                  Type                     regType,
                  VariableType             variableType,
                  T const&                 coord);

            Value(std::shared_ptr<Allocation> alloc,
                  Type                        regType,
                  VariableType                variableType,
                  int                         count);
            Value(std::shared_ptr<Allocation> alloc,
                  Type                        regType,
                  VariableType                variableType,
                  std::vector<int>&&          coord);
            template <std::ranges::input_range T>
            Value(std::shared_ptr<Allocation> alloc,
                  Type                        regType,
                  VariableType                variableType,
                  T&                          coord);

            std::string name() const;
            void        setName(std::string const& name);
            void        setName(std::string&& name);

            /**
             * Return negated copy.
             */
            std::shared_ptr<Value> negate() const;

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
            std::shared_ptr<Value> subset(T const& indices);

            template <std::integral T>
            std::shared_ptr<Value> subset(std::initializer_list<T> indices);

            bool intersects(std::shared_ptr<Register::Value>) const;

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
            std::shared_ptr<Value> element(T const& indices);
            template <typename T>
            std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>,
                             std::shared_ptr<Value>>
                element(std::initializer_list<T> indices);

            size_t registerCount() const;
            size_t valueCount() const;

            bool           hasContiguousIndices() const;
            Generator<int> registerIndices() const;

            std::string getLiteral() const;

            /**
             * Return a literal's actual value.
             */
            CommandArgumentValue getLiteralValue() const;

            std::string getLabel() const;

            std::shared_ptr<LDSAllocation> getLDSAllocation() const;

            rocRoller::Expression::ExpressionPtr expression();

            std::shared_ptr<Context> context() const;

        private:
            /**
             * Implementation of toString() for general-purpose registers.
             */
            void gprString(std::ostream& os) const;

            friend class Allocation;

            std::string m_name;

            std::weak_ptr<Context> m_context;

            CommandArgumentValue m_literalValue;

            std::string m_specialName;

            std::string m_label;

            std::shared_ptr<Allocation> m_allocation;

            std::shared_ptr<LDSAllocation> m_ldsAllocation;

            Type         m_regType;
            VariableType m_varType;
            bool         m_negate = false;

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

        std::shared_ptr<Value> Representative(std::initializer_list<std::shared_ptr<Value>> values);

        /**
         * Represents one (possible) allocation of register(s) that are thought of collectively.
         *
         * TODO: Make not copyable, enforce construction through shared_ptr
         */
        struct Allocation : public std::enable_shared_from_this<Allocation>
        {
            struct Options
            {
                /// For
                bool contiguous = true;

                /// Allocation x must have (x % alignment) == alignmentPhase
                int alignment      = 1;
                int alignmentPhase = 0;
            };

            /// One value of the given type.
            Allocation(std::shared_ptr<Context> context, Type regType, VariableType variableType);

            Allocation(std::shared_ptr<Context> context,
                       Type                     regType,
                       VariableType             variableType,
                       int                      count);

            Allocation(std::shared_ptr<Context> context,
                       Type                     regType,
                       VariableType             variableType,
                       int                      count,
                       Options const&           options);
            Allocation(std::shared_ptr<Context> context,
                       Type                     regType,
                       VariableType             variableType,
                       int                      count,
                       Options&&                options);

            ~Allocation();

            static std::shared_ptr<Allocation> SameAs(Value const& val, std::string const& name);

            Instruction allocate();
            void        allocate(Instruction& inst);

            bool canAllocateNow() const;
            void allocateNow();

            AllocationState allocationState() const;

            std::shared_ptr<Value> operator*();

            std::string descriptiveComment(std::string const& prefix) const;

            int     registerCount() const;
            Options options() const;

            std::vector<int> const& registerIndices() const;

            void setAllocation(std::shared_ptr<Allocator> allocator,
                               std::vector<int> const&    registers);
            void setAllocation(std::shared_ptr<Allocator> allocator, std::vector<int>&& registers);

            void free();

            std::string name() const;
            void        setName(std::string const& name);
            void        setName(std::string&& name);

        private:
            friend class Value;

            std::weak_ptr<Context> m_context;

            Type         m_regType;
            VariableType m_variableType;

            Options m_options;

            int m_valueCount;
            int m_registerCount;

            AllocationState m_allocationState = AllocationState::Unallocated;

            std::shared_ptr<Allocator> m_allocator;
            std::vector<int>           m_registerIndices;

            std::string m_name;

            void setRegisterCount();
        };
    }
}

#include "Register_impl.hpp"
