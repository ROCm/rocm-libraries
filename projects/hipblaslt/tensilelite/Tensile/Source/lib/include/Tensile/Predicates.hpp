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

#include <algorithm>
#include <string>
#include <vector>

#include <Tensile/Properties.hpp>
#include <Tensile/Utils.hpp>

namespace TensileLite
{
    /**
 * \addtogroup Properties
 *
 * @brief A Predicate is a Property whose value is of type `bool`.
 *
 * This allows it to represent a condition of an object, such as `object.size >
 * 4`.
 */
    namespace Predicates
    {
        /**
 * @brief A Property whose value is of type `bool`.
 *
 * Predicates should generally implement better `debugEval()`
 * functionality than other Properties since it's often helpful to know
 * the value of both sides of a comparison.
 */
        template <typename Object>
        using Predicate = Property<Object, bool>;

        /**
 * \copydoc TensileLite::Property_CRTP
 */
        template <typename Class, typename Object>
        using Predicate_CRTP = Property_CRTP<Class, Object, bool>;

        /**
 * \ingroup Properties
 * \defgroup Predicates Predicate Classes
 *
 * @brief Individual Predicate classes.
 */

        /**
 * \addtogroup Predicates
 * @{
 */

        template <typename Object>
        class True : public Predicate_CRTP<True<Object>, Object>
        {
        public:
            enum
            {
                HasIndex = false,
                HasValue = false
            };
            static std::string Type()
            {
                return "TruePred";
            }

            virtual bool operator()(Object const&) const
            {
                return true;
            }

            virtual bool debugEval(Object const& obj, std::ostream& stream) const
            {
                bool rv = (*this)(obj);
                stream << this->type() << " {" << std::endl;
                stream << "}: " << rv << std::endl;
                return rv;
            }
        };

        template <typename Object>
        class False : public Predicate_CRTP<False<Object>, Object>
        {
        public:
            enum
            {
                HasIndex = false,
                HasValue = false
            };
            static std::string Type()
            {
                return "FalsePred";
            }

            virtual bool operator()(Object const&) const
            {
                return false;
            }

            virtual bool debugEval(Object const& obj, std::ostream& stream) const
            {
                bool rv = (*this)(obj);
                stream << this->type() << " {" << std::endl;
                stream << "}: " << rv << std::endl;
                return rv;
            }
        };

        template <typename Object>
        class And : public Predicate_CRTP<And<Object>, Object>
        {
        public:
            enum
            {
                HasIndex = false,
                HasValue = true
            };
            std::vector<std::shared_ptr<Predicate<Object>>> value;

            And() = default;
            And(std::initializer_list<std::shared_ptr<Predicate<Object>>> init)
                : value(init)
            {
            }
            And(std::vector<std::shared_ptr<Predicate<Object>>> const& init)
                : value(init)
            {
            }

            static std::string Type()
            {
                return "And";
            }

            virtual bool operator()(Object const& obj) const
            {
                return std::all_of(
                    value.begin(), value.end(), [&obj](std::shared_ptr<Predicate<Object>> pred) {
                        return (*pred)(obj);
                    });
            }

            virtual bool debugEval(Object const& obj, std::ostream& stream) const
            {
                bool rv = (*this)(obj);
                stream << this->type() << " {" << std::endl;
                for(auto const& term : value)
                {
                    if (!(*term)(obj))
                        term->debugEval(obj, stream);
                }
                stream << "}: " << rv << std::endl;
                return rv;
            }
        };

        template <typename Object>
        class Or : public Predicate_CRTP<Or<Object>, Object>
        {
        public:
            enum
            {
                HasIndex = false,
                HasValue = true
            };
            std::vector<std::shared_ptr<Predicate<Object>>> value;

            Or() = default;
            Or(std::initializer_list<std::shared_ptr<Predicate<Object>>> init)
                : value(init)
            {
            }
            Or(std::vector<std::shared_ptr<Predicate<Object>>> const& init)
                : value(init)
            {
            }

            static std::string Type()
            {
                return "Or";
            }

            virtual bool operator()(Object const& obj) const
            {
                return std::any_of(
                    value.begin(), value.end(), [&obj](std::shared_ptr<Predicate<Object>> pred) {
                        return (*pred)(obj);
                    });
            }

            virtual bool debugEval(Object const& obj, std::ostream& stream) const
            {
                bool rv = (*this)(obj);
                stream << this->type() << " {" << std::endl;
                for(auto const& term : value)
                {
                    if (!(*term)(obj))
                        term->debugEval(obj, stream);
                }
                stream << "}: " << rv << std::endl;
                return rv;
            }
        };

        template <typename Object>
        class Not : public Predicate_CRTP<Not<Object>, Object>
        {
        public:
            enum
            {
                HasIndex = false,
                HasValue = true
            };
            std::shared_ptr<Predicate<Object>> value;
            Not() = default;
            Not(std::shared_ptr<Predicate<Object>> init)
                : value(init)
            {
            }

            static std::string Type()
            {
                return "Not";
            }

            virtual bool operator()(Object const& obj) const
            {
                return !(*value)(obj);
            }

            virtual bool debugEval(Object const& obj, std::ostream& stream) const
            {
                bool rv = (*this)(obj);
                stream << this->type() << " {" << std::endl;
                value->debugEval(obj, stream);
                stream << "}: " << rv << std::endl;
                return rv;
            }
        };

        /**
 * @brief Predicate which returns `true` if the object is of type Subclass.
 *
 * @param Object
 * @param Subclass Must be a subclass of Object.
 */
        template <typename Object, typename Subclass>
        class IsSubclass : public Predicate_CRTP<IsSubclass<Object, Subclass>, Object>
        {
        public:
            enum
            {
                HasIndex = false,
                HasValue = true
            };
            static_assert(std::is_base_of<Object, Subclass>::value,
                          "Subclass must be derived from Object.");

            std::shared_ptr<Predicate<Subclass>> value;

            IsSubclass() = default;
            IsSubclass(std::shared_ptr<Predicate<Subclass>> init)
                : value(init)
            {
            }

            static std::string Type()
            {
                return Subclass::Type();
            }

            virtual bool operator()(Object const& obj) const override
            {
                auto const* sc = dynamic_cast<Subclass const*>(&obj);
                if(!sc)
                    return false;

                return (*value)(*sc);
            }

            virtual bool debugEval(Object const& obj, std::ostream& stream) const override
            {
                bool rv = (*this)(obj);
                stream << this->type() << " {" << std::endl;
                auto const* sc = dynamic_cast<Subclass const*>(&obj);
                if(sc)
                {
                    value->debugEval(*sc, stream);
                }
                else
                {
                    stream << "no match. actual type: " << typeid(obj).hash_code() << ", expected "
                           << typeid(Subclass).hash_code() << std::endl;
                }
                stream << "}: " << rv << std::endl;
                return rv;
            }
        };

        /**
 * @}
 */
    } // namespace Predicates
} // namespace TensileLite
