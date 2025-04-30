/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstddef>
#include <string>

#include <Tensile/Utils.hpp>

namespace TensileLite
{
    /**
 * \ingroup Tensile
 * \defgroup Properties Properties and Predicates
 *
 * @brief Abstract expression evaluation
 *
 * Property: \copydoc TensileLite::Property
 *
 * Predicate: \copydoc TensileLite::Predicates::Predicate
 */

    /**
 * \addtogroup Properties
 * @{
 */

    /**
 * @brief Simplifies implementation of `ToString()` for Property subclasses
 * which may have `index` and/or `value` members.
 */
    template <typename Class, bool HasIndex = Class::HasIndex, bool HasValue = Class::HasValue>
    struct PropertyHelper
    {
    };

    template <typename Class>
    struct PropertyHelper<Class, false, false>
    {
        static std::string ToString(Class const& obj)
        {
            return obj.type();
        }
    };

    template <typename Class>
    struct PropertyHelper<Class, false, true>
    {
        static std::string ToString(Class const& obj)
        {
            return concatenate(obj.type(), "(", obj.value, ")");
        }
    };

    template <typename Class>
    struct PropertyHelper<Class, true, false>
    {
        static std::string ToString(Class const& obj)
        {
            return concatenate(obj.type(), "(", obj.index, ")");
        }
    };

    template <typename Class>
    struct PropertyHelper<Class, true, true>
    {
        static std::string ToString(Class const& obj)
        {
            return concatenate(obj.type(), "(index=", obj.index, ", value=", obj.value, ")");
        }
    };

    /**
 * Abstract object which retrieves a value from another object.
 */
    template <typename Object, typename Value = size_t>
    class Property
    {
    public:
        /**
   * Name which uniquely identifies each subclass.
   */
        virtual std::string type() const = 0;
        virtual ~Property()              = default;

        /**
   * Retrieve the value from the specified object.
   */
        virtual Value operator()(Object const& object) const = 0;

        virtual std::string toString() const = 0;

        /**
   * Retrieve the value from the specified object, while printing
   * relevant debug information to the specified stream.
   */
        virtual Value debugEval(Object const& object, std::ostream& stream) const
        {
            Value rv = (*this)(object);
            stream << *this << ": " << rv;
            return rv;
        }

        template<typename Tp, typename Tpv, typename Tc, typename Ts, typename Tsv, typename... T>
        void debugEvalCmp(std::ostream& stream, Tp prob, Tpv prob_val, Tc cmp, Ts sol, Tsv sol_val, T... args) const
        {
            if (cmp == "==")
            {
                if (prob_val != sol_val)
                    stream << "((" << prob << "=" << prob_val << ") != (" << sol << "=" << sol_val <<  ")), ";
            }
            else if (cmp == ">")
            {
                if (prob_val <= sol_val)
                    stream << "((" << prob << "=" << prob_val << ") <= (" << sol << "=" << sol_val <<  ")), ";
            }
            else if (cmp == "<")
            {
                if (prob_val >= sol_val)
                    stream << "((" << prob << "=" << prob_val << ") >= (" << sol << "=" << sol_val <<  ")), ";
            }
            else if (cmp == ">=")
            {
                if (prob_val < sol_val)
                    stream << "((" << prob << "=" << prob_val << ") < (" << sol << "=" << sol_val <<  ")), ";
            }
            else if (cmp == "<=")
            {
                if (prob_val > sol_val)
                    stream << "((" << prob << "=" << prob_val << ") > (" << sol << "=" << sol_val <<  ")), ";
            }
            else if (cmp == "%")
            {
                if constexpr (std::is_same<Tpv, size_t>::value)
                    if (prob_val % sol_val != 0)
                        stream << "((" << prob << "=" << prob_val << ") % (" << sol << "=" << sol_val <<  ") != 0), ";
            }
            else
            {
                stream << "(invalid statement), ";
            }
            if constexpr (sizeof...(args) >= 5)
                debugEvalCmp(stream, args...);
        }

        template<typename... T>
        Value debugEvalCmp(Object const& object, std::ostream& stream, T... args) const
        {
            Value rv = (*this)(object);
            stream << rv << ": " << this->type() << " (";
            debugEvalCmp(stream, args...);
            stream << ")" << std::endl;
            return rv;
        }
    };

    /**
 * @brief CRTP helper class which simplifies implementation of Property
 * subclasses.
 *
 * Implements the `type()` and `toString()` methods automatically.
 *
 * The subclass must:
 *  - Implement a `static std::string Type()` function which returns a
 *  unique name for the class.
 *  - Have `HasIndex` and `HasValue` value definitions which match the
 *  reality of if it has `index` and/or `value` members.
 *
 * \see https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
 */
    template <typename Class, typename Object, typename Value = size_t>
    class Property_CRTP : public Property<Object, Value>
    {
    public:
        virtual std::string type() const final
        {
            return Class::Type();
        }

        virtual std::string toString() const
        {
            Class const& this_typed = dynamic_cast<Class const&>(*this);
            return PropertyHelper<Class>::ToString(this_typed);
        }
    };

    template <typename Object, typename Value>
    inline std::ostream& operator<<(std::ostream& stream, Property<Object, Value> const& prop)
    {
        return stream << prop.toString();
    }

    template <typename Object, typename Value>
    inline std::ostream&
        operator<<(std::ostream&                                                stream,
                   std::vector<std::shared_ptr<Property<Object, Value>>> const& props)
    {
        stream << "(";

        bool first = true;
        for(auto const& v : props)
        {
            if(!first)
                stream << ", ";
            first = false;

            stream << *v;
        }

        stream << ")";

        return stream;
    }

    /**
 * @}
 */

    /**
 * \ingroup Properties
 * \defgroup PropertyClasses Property Classes
 *
 * @brief Individual Property classes
 */
} // namespace TensileLite
