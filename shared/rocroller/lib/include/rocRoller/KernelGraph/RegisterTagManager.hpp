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

#include <map>
#include <memory>

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/KernelGraph/KernelGraph_fwd.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager_fwd.hpp>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>

namespace rocRoller
{
    /**
     * @brief Expression attributes (meta data).
     *
     * See also: LoadStoreTileGenerator::generateStride.
     */
    struct RegisterExpressionAttributes
    {
        DataType dataType   = DataType::None; //< Desired result type of the expression
        bool     unitStride = false; //< Expression corresponds to a unitary (=1) element-stride.
        uint     elementBlockSize = 0; //< If non-zero, elements are loaded in blocks.
        Expression::ExpressionPtr
            elementBlockStride; //< If non-null, stride between element blocks.
        Expression::ExpressionPtr
            trLoadPairStride; //< If non-null, stride between element blocks of transpose loads of a wavetile.

        auto operator<=>(RegisterExpressionAttributes const& other) const = default;
    };
    std::string toString(RegisterExpressionAttributes const& attrs);

    /**
     * @brief Register Tag Manager - Keeps track of data flow tags
     * that have been previously calculated.
     *
     * The manager tracks data flow tags that have been previously seen
     * during code generation. It can track either Register::Values or
     * Expressions.
     *
     * It is generally used when one control node calculates a value that
     * needs to be used by another control node during code generation. The
     * node that calculates the value can store it by adding the value to
     * the tag manager. The preceding nodes can then retrieve that value
     * using the associated data flow tag.
     *
     */
    class RegisterTagManager
    {
    public:
        RegisterTagManager(ContextPtr context);
        ~RegisterTagManager();

        /**
         * Copies any necessary info from the kernel graph.  Currently registers
         * Alias and Index edges.
         */
        void initialize(KernelGraph::KernelGraph const& kgraph);

        /**
         * Causes `src` to borrow the register the allocation from `dst`.  This will
         * require that:
         *  - `src` and `dst` will use the same number, type, and alignment
         *    requirements.
         *  - `dst` will not be accessed while `src` is active.
         */
        void addAlias(int src, int dst);

        /**
         * Causes `src` to index the register allocation from `dst`.
         */
        void addIndex(int src, int dst, int index);

        /**
         * @brief Get the Register::Value associated with the provided tag.
         *
         * An exception will be thrown if the tag is not present in the
         * tag manager or if the tag is present, but is not associated with
         * a Register::Value.
         *
         * @param tag
         * @return Register::ValuePtr
         */
        Register::ValuePtr getRegister(int tag) const;

        /**
         * @brief Get the Expression associated with the provided tag.
         *
         * An exception will be thrown if the tag is not present in the
         * tag manager or if the tag is present, but is not associated with
         * an Expression.
         *
         * @param tag
         * @return The expression and the expression's datatype.
         */
        std::pair<Expression::ExpressionPtr, RegisterExpressionAttributes>
            getExpression(int tag) const;

        /**
         * @brief Get the Register::Value associated with the provided tag.
         *
         * If there is no Register::Value already associated with the tag,
         * create a new Register::Value using the provided typing information.
         *
         * Throws an exception if a non-Register::Value is already associated
         * with the tag.
         *
         * @param tag
         * @param regType
         * @param varType
         * @param ValueCount
         * @return Register::ValuePtr
         */
        Register::ValuePtr getRegister(int                         tag,
                                       Register::Type              regType,
                                       VariableType                varType,
                                       size_t                      ValueCount   = 1,
                                       Register::AllocationOptions allocOptions = {});

        /**
         * @brief Get the Register::Value associated with the provided tag.
         *
         * If there is no Register::Value already associated with the tag,
         * create a new Register::Value using the provided register template.
         *
         * Throws an exception if a non-Register::Value is already associated
         * with the tag.
         *
         * @param tag
         * @param tmpl
         * @return Register::ValuePtr
         */
        Register::ValuePtr getRegister(int tag, Register::ValuePtr tmpl);

        /**
         * @brief Add a register to the RegisterTagManager with the provided tag.
         *
         * @param tag The tag the of the register
         * @param value The register value to be added
         */
        void addRegister(int tag, Register::ValuePtr value);

        /**
         * @brief Add an expression to the RegisterTagManager with the provided tag.
         *
         * @param tag The tag the of the register
         * @param value The expression that represents the value within tag.
         * @param dt The DataType of the provided expression.
         */
        void addExpression(int                          tag,
                           Expression::ExpressionPtr    value,
                           RegisterExpressionAttributes attrs);

        /**
         * @brief Delete the value associated with the provided tag.
         *
         * @param tag
         */
        void deleteTag(int tag);

        /**
         * Returns whether or not a register has already been added to the
         * Register Manager.
         */
        bool hasRegister(int tag) const;

        std::optional<int> findRegister(Register::ValuePtr reg) const;

        /**
         * Returns whether or not an expression has already been added to the
         * Register Manager.
         */
        bool hasExpression(int tag) const;

        /**
         * Indicates that the register for `tag` may be borrowed to be used
         * for a different tag's data.
         */
        bool isAliased(int tag) const;

        /**
         * Indicates that the register for `tag` will borrow the allocation
         * from a different tag's register.
         */
        bool hasAlias(int tag) const;

        /**
         * If `hasAlias(tag)`, then returns the tag we will borrow from.
         */
        std::optional<int> getAlias(int tag) const;

        /**
         * Indicates that `tag` is unavailable because `isAliased(tag)` and
         * another tag is currently using `tag`'s allocation.
         */
        bool isBorrowed(int tag) const;

        /**
         * If `getIndex(tag)`, then returns the pair <tag, index>.
         */
        std::optional<std::pair<int, int>> getIndex(int tag) const;

    private:
        std::weak_ptr<Context>            m_context;
        std::map<int, Register::ValuePtr> m_registers;
        std::map<int, std::pair<Expression::ExpressionPtr, RegisterExpressionAttributes>>
            m_expressions;

        inline static constexpr int ALIAS_DEST = -1;

        std::map<int, int> m_borrowedTags;

        std::map<int, int>                 m_aliases;
        std::map<int, std::pair<int, int>> m_indexes;
    };
}

#include <rocRoller/KernelGraph/RegisterTagManager_impl.hpp>
