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
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager_fwd.hpp>

namespace rocRoller
{
    /**
     * @brief Expression attributes (meta data).
     *
     * See also: LoadStoreTileGenerator::generateStride.
     */
    struct RegisterExpressionAttributes
    {
        DataType dataType; //< Desired result type of the expression
        bool     unitStride = false; //< Expression corresponds to a unitary (=1) element-stride.
        uint     elementBlockSize = 0; //< If non-zero, elements are loaded in blocks.
        Expression::ExpressionPtr
            elementBlockStride; //< If non-null, stride between element blocks.
        Expression::ExpressionPtr
            trLoadPairStride; //< If non-null, stride between element blocks of transpose loads of a wavetile.
    };
    std::string toString(RegisterExpressionAttributes t);

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
         * @brief Get the Register::Value associated with the provided tag.
         *
         * An exception will be thrown if the tag is not present in the
         * tag manager or if the tag is present, but is not associated with
         * a Register::Value.
         *
         * @param tag
         * @return Register::ValuePtr
         */
        Register::ValuePtr getRegister(int tag);

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
         * @brief Returns whether or not a register has already been added to the
         *        Register Manager.
         *
         * @param tag
         * @return true
         * @return false
         */
        bool hasRegister(int tag) const;

        /**
         * @brief Returns whether or not an expression has already been added to the
         *        Register Manager.
         *
         * @param tag
         * @return true
         * @return false
         */
        bool hasExpression(int tag) const;

    private:
        std::weak_ptr<Context>            m_context;
        std::map<int, Register::ValuePtr> m_registers;
        std::map<int, std::pair<Expression::ExpressionPtr, RegisterExpressionAttributes>>
            m_expressions;
    };
}

#include <rocRoller/KernelGraph/RegisterTagManager_impl.hpp>
