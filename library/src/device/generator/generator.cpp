// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "generator.h"

#define CONSTRUCT_OPER(NAME)                                \
    NAME::NAME(const std::initializer_list<Expression>& il) \
        : args(il){};                                       \
    NAME::NAME(const std::vector<Expression>& il)           \
        : args(il){};

#define MAKE_BINARY_METHODS(NAME)                                  \
    std::string NAME::render() const                               \
    {                                                              \
        std::string s;                                             \
        if(get_precedence(args[0]) > precedence)                   \
            s += "(" + vrender(args[0]) + ")";                     \
        else                                                       \
            s += vrender(args[0]);                                 \
        for(auto arg = args.begin() + 1; arg != args.end(); ++arg) \
        {                                                          \
            s += oper;                                             \
            if(get_precedence(*arg) >= precedence)                 \
                s += "(" + vrender(*arg) + ")";                    \
            else                                                   \
                s += vrender(*arg);                                \
        }                                                          \
        return s;                                                  \
    }

#define MAKE_UNARY_PREFIX_METHODS(NAME)               \
    std::string NAME::render() const                  \
    {                                                 \
        std::string s = oper;                         \
        if(get_precedence(args.front()) > precedence) \
            s += "(" + vrender(args.front()) + ")";   \
        else                                          \
            s += vrender(args.front());               \
        return s;                                     \
    }

std::string ArgumentList::render() const
{
    std::string f;
    if(!arguments.empty())
    {
        f = arguments[0].render();
        for(unsigned int i = 1; i < arguments.size(); ++i)
        {
            f += ",";
            f += arguments[i].render();
        }
    }
    return f;
}

ArgumentList::operator bool() const
{
    return !arguments.empty();
}

void ArgumentList::append(Variable&& v)
{
    arguments.push_back(std::move(v));
}

void ArgumentList::append(const Variable& v)
{
    arguments.push_back(v);
}

void ArgumentList::set_value(const std::string& name, const std::string& value)
{
    for(auto& arg : arguments)
    {
        if(arg.name == name)
        {
            arg.name = value;
            return;
        }
    }
    // didn't find the argument - that should be a programmer error
    throw std::runtime_error("ArgumentList::set_value failed to find " + name);
}

CONSTRUCT_OPER(Add);
CONSTRUCT_OPER(Subtract);
CONSTRUCT_OPER(Multiply);
CONSTRUCT_OPER(Divide);
CONSTRUCT_OPER(Modulus);

CONSTRUCT_OPER(Less);
CONSTRUCT_OPER(LessEqual);
CONSTRUCT_OPER(Greater);
CONSTRUCT_OPER(GreaterEqual);
CONSTRUCT_OPER(Equal);
CONSTRUCT_OPER(NotEqual);
CONSTRUCT_OPER(ShiftLeft);
CONSTRUCT_OPER(ShiftRight);
CONSTRUCT_OPER(And);
CONSTRUCT_OPER(BitAnd);
CONSTRUCT_OPER(Or);

CONSTRUCT_OPER(UnaryMinus);
CONSTRUCT_OPER(Not);
CONSTRUCT_OPER(PreIncrement);
CONSTRUCT_OPER(PreDecrement);

CONSTRUCT_OPER(ComplexLiteral);

MAKE_BINARY_METHODS(Add);
MAKE_BINARY_METHODS(Subtract);
MAKE_BINARY_METHODS(Multiply);
MAKE_BINARY_METHODS(Divide);
MAKE_BINARY_METHODS(Modulus);

MAKE_BINARY_METHODS(Less);
MAKE_BINARY_METHODS(LessEqual);
MAKE_BINARY_METHODS(Greater);
MAKE_BINARY_METHODS(GreaterEqual);
MAKE_BINARY_METHODS(Equal);
MAKE_BINARY_METHODS(NotEqual);
MAKE_BINARY_METHODS(ShiftLeft);
MAKE_BINARY_METHODS(ShiftRight);
MAKE_BINARY_METHODS(And);
MAKE_BINARY_METHODS(BitAnd);
MAKE_BINARY_METHODS(Or);

MAKE_UNARY_PREFIX_METHODS(UnaryMinus);
MAKE_UNARY_PREFIX_METHODS(Not);
MAKE_UNARY_PREFIX_METHODS(PreIncrement);
MAKE_UNARY_PREFIX_METHODS(PreDecrement);

Ternary::Ternary(Expression&& cond, Expression&& true_result, Expression&& false_result)
    : args{std::move(cond), std::move(true_result), std::move(false_result)}
{
}

Ternary::Ternary(std::vector<Expression>&& args)
    : args(std::move(args))
{
}

std::string Ternary::render() const
{
    return vrender(args[0]) + " ? " + vrender(args[1]) + " : " + vrender(args[2]);
}

LoadGlobal::LoadGlobal(const Expression& ptr, const Expression& index)
    : args{ptr, index}
{
}

LoadGlobal::LoadGlobal(const std::vector<Expression>& args)
    : args(args)
{
}

std::string LoadGlobal::render() const
{
    return "load_cb(" + vrender(args[0]) + "," + vrender(args[1]) + ", load_cb_data, nullptr)";
}

LoadGlobalPlanar::LoadGlobalPlanar(const std::vector<Expression>& args)
    : args(args)
{
}

std::string LoadGlobalPlanar::render() const
{
    return "load_planar(" + vrender(args[0]) + "," + vrender(args[1]) + "," + vrender(args[2])
           + ")";
}

std::string ArgumentList::render_decl() const
{
    std::string f;
    const char* separator = "";
    const char* comma     = ",";
    for(const auto& arg : arguments)
    {
        f += separator;
        f += arg.type;
        // arrays (i.e. where size != 0) are passed as pointers
        if(arg.pointer || arg.size)
            f += "*";
        if(arg.restrict)
            f += " __restrict__";
        f += " " + arg.name;
        if(arg.decl_default)
            f += " = " + vrender(*arg.decl_default);
        separator = comma;
    }
    return f;
}

Variable::Variable(const std::string& _name,
                   const std::string& _type,
                   bool               pointer,
                   bool restrict,
                   unsigned int size)
    : name(_name)
    , type(_type)
    , pointer(pointer)
    , restrict(restrict)
{
    if(size > 0)
        this->size = Expression{size};
}

// NOTE: cppcheck doesn't realize all of the members are actually
// initialized here
//
// cppcheck-suppress uninitMemberVar
Variable::Variable(const Variable& v)
    : name(v.name)
    , type(v.type)
    , pointer(v.pointer)
    , restrict(v.restrict)
    , component(v.component)
    , index(v.index)
    , index2D(v.index2D)
    , size(v.size)
    , size2D(v.size2D)
    , decl_default(v.decl_default)
{
}

Variable::Variable(const Variable& v, const Expression& _index)
    : name(v.name)
    , type(v.type)
    , pointer(v.pointer)
    , restrict(v.restrict)
    , component(v.component)
    , index(_index)
{
    size         = v.size;
    size2D       = v.size2D;
    decl_default = v.decl_default;
}

Variable::Variable(const Variable& v, const Expression& _index, const Expression& _index2D)
    : Variable(v, _index)
{
    index2D = _index2D;
}

Variable Variable::address() const
{
    if(index)
    {
        return Variable("&" + name + "[" + vrender(*index) + "]", type + "*");
    }
    return Variable("&" + name, type + "*");
}

Variable Variable::x() const
{
    Variable ret{*this};
    ret.component = Component::REAL;
    return ret;
}

Variable Variable::y() const
{
    Variable ret{*this};
    ret.component = Component::IMAG;
    return ret;
}

std::string Variable::render() const
{
    std::string output;
    if(index)
    {
        output = name + "[" + vrender(*index) + "]";
        if(index2D)
            output += "[" + vrender(*index2D) + "]";
    }
    else
        output = name;

    if(component == Component::REAL)
        output += ".x";
    else if(component == Component::IMAG)
        output += ".y";
    return output;
}

Variable Variable::operator[](const Expression& index) const
{
    return Variable(*this, index);
}

Variable Variable::at(const Expression& index, const Expression& index2D) const
{
    return Variable(*this, index, index2D);
}

OptionalExpression::OptionalExpression() {}

OptionalExpression::OptionalExpression(OptionalExpression&& o)
    : expr(std::move(o.expr))
{
}
OptionalExpression::OptionalExpression(const OptionalExpression& o)
{
    if(o.expr)
        this->expr = std::make_unique<Expression>(*o.expr);
}

OptionalExpression& OptionalExpression::operator=(OptionalExpression&& o)
{
    std::swap(expr, o.expr);
    return *this;
}

OptionalExpression& OptionalExpression::operator=(const OptionalExpression& o)
{
    if(o.expr)
        this->expr = std::make_unique<Expression>(*o.expr);
    return *this;
}

OptionalExpression::operator bool() const
{
    return expr != nullptr;
}

Expression OptionalExpression::operator*() const
{
    return *expr;
}

OptionalExpression::OptionalExpression(const Expression& expr)
{
    this->expr = std::make_unique<Expression>(expr);
}
OptionalExpression& OptionalExpression::operator=(const Expression& in_expr)
{
    this->expr = std::make_unique<Expression>(in_expr);
    return *this;
}

std::string ComplexLiteral::render() const
{
    std::string ret       = "{";
    const char* separator = nullptr;
    for(const auto& arg : args)
    {
        if(separator)
            ret += separator;
        ret += vrender(arg);
        separator = oper;
    }
    ret += "}";
    return ret;
}

std::string TwiddleMultiply::render() const
{
    auto& a = vars[0];
    auto& b = vars[1];
    return ComplexLiteral{a.x() * b.x() - a.y() * b.y(), a.y() * b.x() + a.x() * b.y()}.render();
}

std::string TwiddleMultiplyConjugate::render() const
{
    auto& a = vars[0];
    auto& b = vars[1];
    return ComplexLiteral{a.x() * b.x() + a.y() * b.y(), a.y() * b.x() - a.x() * b.y()}.render();
}

Parens::Parens(Expression&& inside)
    : args{std::move(inside)}
{
}

Parens::Parens(std::vector<Expression>&& args)
    : args(std::move(args))
{
}

std::string Parens::render() const
{
    return "(" + vrender(args.front()) + ")";
}

CallExpr::CallExpr(const std::string& name, const std::vector<Expression>& arguments)
    : name(name)
    , arguments(arguments){};

CallExpr::CallExpr(const std::string&             name,
                   const TemplateList&            templates,
                   const std::vector<Expression>& arguments)
    : name(name)
    , templates(templates)
    , arguments(arguments){};

std::string CallExpr::render() const
{
    std::string f;
    f += name;
    const char* separator = nullptr;
    const char* comma     = ",";
    if(!templates.arguments.empty())
    {
        f += "<";
        // template args just have the names, not types
        for(const auto& arg : templates.arguments)
        {
            if(separator)
                f += separator;
            f += arg.name;
            separator = comma;
        }
        f += ">";
    }
    f += "(";
    separator = nullptr;
    for(const auto& arg : arguments)
    {
        if(separator)
            f += separator;
        f += vrender(arg);
        separator = comma;
    }
    f += ")";
    return f;
}

IntrinsicLoad::IntrinsicLoad(const std::vector<Expression>& args)
    : args(args)
{
}

IntrinsicLoadPlanar::IntrinsicLoadPlanar(const std::vector<Expression>& args)
    : args(args)
{
}

std::string IntrinsicLoad::render() const
{
    // intrinsic_load(const T* data, unsigned int voffset, unsigned int soffset,
    // bool rw)
    return "intrinsic_load(" + vrender(args[0]) + "," + vrender(args[1]) + "," + vrender(args[2])
           + "," + vrender(args[3]) + ")";
}

std::string IntrinsicLoadPlanar::render() const
{
    return "{"
           "intrinsic_load("
           + vrender(args[0]) + "," + vrender(args[2]) + "," + vrender(args[3]) + ","
           + vrender(args[4]) + ")" + "," + "intrinsic_load(" + vrender(args[1]) + ","
           + vrender(args[2]) + "," + vrender(args[3]) + "," + vrender(args[4]) + ")" + "}";
}

std::string Declaration::render() const
{
    std::string s;
    s = var.type;
    if(var.pointer)
        s += "*";
    s += " " + var.name;
    if(var.size)
        s += "[" + vrender(*var.size) + "]";
    if(var.size2D)
        s += "[" + vrender(*var.size2D) + "]";
    if(value)
        s += " = " + vrender(*value);
    s += ";";
    return s;
}

std::string StoreGlobalPlanar::render() const
{
    return "store_planar(" + realPtr.render() + "," + imagPtr.render() + "," + vrender(index) + ","
           + vrender(value) + ");";
}

std::string Butterfly::render() const
{
    std::string func;
    if(forward)
    {
        func += "FwdRad" + std::to_string(args.size()) + "B1";
    }
    else
    {
        func += "InvRad" + std::to_string(args.size()) + "B1";
    }
    return Call{func, args}.render();
}

StatementList::StatementList() {}
StatementList::StatementList(const std::initializer_list<Statement>& il)
    : statements(il){};
std::string StatementList::render() const
{
    std::string r;
    for(auto s : statements)
        r += vrender(s) + "\n";
    return r;
}

For::For(const Variable&      var,
         const Expression&    initial,
         const Expression&    condition,
         const Expression&    increment,
         const StatementList& body,
         bool                 pragma_unroll)
    : var(var)
    , initial(initial)
    , condition(condition)
    , increment(increment)
    , body(body)
    , pragma_unroll(pragma_unroll){};

std::string For::render() const
{
    std::string s;
    if(pragma_unroll)
        s += "#pragma unroll\n";
    s += "for(";
    s += var.type + " " + var.name + " = ";
    s += vrender(initial) + "; ";
    s += vrender(condition) + "; ";

    // ++ and -- are nicer to read, so render those as a special case
    if(std::holds_alternative<Literal>(increment) && std::get<Literal>(increment).value == "1")
        s += "++" + var.name;
    else if(std::holds_alternative<Literal>(increment)
            && std::get<Literal>(increment).value == "-1")
        s += "--" + var.name;
    else
        s += var.name + " += " + vrender(increment);
    s += ") {\n ";
    s += body.render();
    s += "\n}";
    return s;
}

While::While(const Expression& condition, const StatementList& body)
    : condition(condition)
    , body(body){};
std::string While::render() const
{
    std::string s;
    s += "while(";
    s += vrender(condition) + ") {\n";
    s += body.render();
    s += "\n}";
    return s;
}

If::If(const Expression& condition, const StatementList& body)
    : condition(condition)
    , body(body){};
std::string If::render() const
{
    std::string s;
    s += "if(";
    s += vrender(condition);
    s += ") {\n";
    s += body.render();
    s += "\n}\n";
    return s;
}

ElseIf::ElseIf(const Expression& condition, const StatementList& body)
    : condition(condition)
    , body(body){};
std::string ElseIf::render() const
{
    std::string s;
    s += "else if(";
    s += vrender(condition);
    s += ") {\n";
    s += body.render();
    s += "\n}\n";
    return s;
}

Else::Else(const StatementList& body)
    : body(body){};
std::string Else::render() const
{
    std::string s;
    s += "else {\n";
    s += body.render();
    s += "\n}\n";
    return s;
}

std::string Function::render() const
{
    std::string f;
    if(templates)
    {
        f += "template<" + templates.render_decl() + ">";
    }
    f += qualifier + " ";
    if(launch_bounds)
        f += "__launch_bounds__(" + std::to_string(launch_bounds) + ") ";
    f += return_type + " " + name;
    f += "(" + arguments.render_decl() + ") {\n";
    f += body.render();
    f += "}\n";
    return f;
}
