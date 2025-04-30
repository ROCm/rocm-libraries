################################################################################
#
# Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import ast
from Tensile.Configuration import ReadWriteTransformDict
from Tensile.Configuration import Parameter
from Tensile.Configuration import CallableParameter
from Tensile.Configuration import ExpressionEvaluator
from Tensile.Configuration import ProjectConfig

def test_ReadWriteTransformDict():
    def readXForm(obj, key):
        return obj.readNoTransform(key+"-Mod")

    def writeXForm(obj, key, value):
        obj.writeNoTransform(key+"-Mod", value)

    test = ReadWriteTransformDict(readXForm, writeXForm)

    # Test with the re-direction of transform funcs
    test["Key"] = 10
    assert test["Key"] == 10 and not "Key" in test
    assert test["Key"] == test.readWithTransform("Key")
    assert "Key-Mod" in test and test.readNoTransform("Key-Mod") == test["Key"]

    # Test bypass
    test.writeNoTransform("Key", 25)
    assert "Key" in test and test.readNoTransform("Key") == 25
    assert test["Key"] != test.readNoTransform("Key")

    # Test removal
    test.setReadTransform(None)
    test.setWriteTransform(None)

    # Test restoration of default behaviour
    test["Key"] = -5
    assert test["Key"] == -5 and test.readNoTransform("Key") == -5

def test_Parameter():
    name = "banana"
    initVal = 15
    defaultVal = 10
    descr = "I am a banana"
    initType = type(initVal)
    b = Parameter(name, initVal, defaultVal, descr)

    #attr
    assert b.value == initVal
    assert b.defaultValue == defaultVal
    assert b.name == name
    assert b.description == descr
    assert b.type == initType

    b.createAttr("color", "yellow")
    assert b.color == "yellow"

    # Accessor funcs
    assert b.getDefault() == defaultVal
    assert b.getValue() == initVal
    assert b.getDescription() == descr
    assert b.resetToDefault() == defaultVal

    # Write Xform
    try:
        # Unknown attr should not be writable
        b.peel
        raise Exception("Should not get here")
    except KeyError:
        pass
    except :
        assert 0, "Failed mutability test"

    try:
        # This should not be writable
        b.type = int
        raise Exception("Should not get here")
    except AttributeError:
        pass
    except :
        assert 0, "Failed type mutability test"

    try:
        # Writing different type should throw
        # b.value initialized with int
        b.value = "Dude"
        raise Exception("Should not get here")
    except AttributeError:
        pass
    except :
        assert 0, "Failed type assignment test"

    try:
        # Writing same type should succeed
        b.value = 55
    except :
        assert 0, "Failed assignment test"

    b.resetToDefault()

    # Comparators const, rhs
    assert (5 < b) == (5 < defaultVal)
    assert (5 <= b) == (5 <= defaultVal)
    assert (5 == b) == (5 == defaultVal)
    assert (5 != b) == (5 != defaultVal)
    assert (5 > b) == (5 > defaultVal)
    assert (5 >= b) == (5 >= defaultVal)

    # Comparators const, lhs
    assert (b < 5) == (defaultVal < 5)
    assert (b <= 5) == (defaultVal <= 5)
    assert (b == 5) == (defaultVal == 5)
    assert (b != 5) == (defaultVal != 5)
    assert (b > 5) == (defaultVal > 5)
    assert (b >= 5) == (defaultVal >= 5)

    defaultVal2 = 1
    a = Parameter("Apple", defaultVal2, defaultVal2, "I am an apple")
    # Comparators param, lhs
    assert (a < b) == (defaultVal2 < defaultVal)
    assert (a <= b) == (defaultVal2 <= defaultVal)
    assert (a == b) == (defaultVal2 == defaultVal)
    assert (a != b) == (defaultVal2 != defaultVal)
    assert (a > b) == (defaultVal2 > defaultVal)
    assert (a >= b) == (defaultVal2 >= defaultVal)

    assert (b < a) == (defaultVal < defaultVal2)
    assert (b <= a) == (defaultVal <= defaultVal2)
    assert (b == a) == (defaultVal == defaultVal2)
    assert (b != a) == (defaultVal != defaultVal2)
    assert (b > a) == (defaultVal > defaultVal2)
    assert (b >= a) == (defaultVal >= defaultVal2)

    # Bin op const, rhs
    assert (5 + b) == (5 + defaultVal)
    assert (5 - b) == (5 - defaultVal)
    assert (5 * b) == (5 * defaultVal)
    assert (5 / b) == (5 / defaultVal)
    assert (5 // b) == (5 // defaultVal)
    assert (5 % b) == (5 % defaultVal)
    assert (5 ** b) == (5 ** defaultVal)
    assert (5 >> b) == (5 >> defaultVal)
    assert (5 << b) == (5 << defaultVal)
    assert (5 & b) == (5 & defaultVal)
    assert (5 | b) == (5 | defaultVal)
    assert (5 ^ b) == (5 ^ defaultVal)

    # Bin op const, lhs
    assert (b + 5) == (defaultVal + 5)
    assert (b - 5) == (defaultVal - 5)
    assert (b * 5) == (defaultVal * 5)
    assert (b / 5) == (defaultVal / 5)
    assert (b // 5) == (defaultVal // 5)
    assert (b % 5) == (defaultVal % 5)
    assert (b ** 5) == (defaultVal ** 5)
    assert (b >> 5) == (defaultVal >> 5)
    assert (b << 5) == (defaultVal << 5)
    assert (b & 5) == (defaultVal & 5)
    assert (b | 5) == (defaultVal | 5)
    assert (b ^ 5) == (defaultVal ^ 5)

    # Bin op param, rhs
    assert (a + b) == (defaultVal2 + defaultVal)
    assert (a - b) == (defaultVal2 - defaultVal)
    assert (a * b) == (defaultVal2 * defaultVal)
    assert (a / b) == (defaultVal2 / defaultVal)
    assert (a // b) == (defaultVal2 // defaultVal)
    assert (a % b) == (defaultVal2 % defaultVal)
    assert (a ** b) == (defaultVal2 ** defaultVal)
    assert (a >> b) == (defaultVal2 >> defaultVal)
    assert (a << b) == (defaultVal2 << defaultVal)
    assert (a & b) == (defaultVal2 & defaultVal)
    assert (a | b) == (defaultVal2 | defaultVal)
    assert (a ^ b) == (defaultVal2 ^ defaultVal)

    # Bin op param, lhs
    assert (b + a) == (defaultVal + defaultVal2)
    assert (b - a) == (defaultVal - defaultVal2)
    assert (b * a) == (defaultVal * defaultVal2)
    assert (b / a) == (defaultVal / defaultVal2)
    assert (b // a) == (defaultVal // defaultVal2)
    assert (b % a) == (defaultVal % defaultVal2)
    assert (b ** a) == (defaultVal ** defaultVal2)
    assert (b >> a) == (defaultVal >> defaultVal2)
    assert (b << a) == (defaultVal << defaultVal2)
    assert (b & a) == (defaultVal & defaultVal2)
    assert (b | a) == (defaultVal | defaultVal2)
    assert (b ^ a) == (defaultVal ^ defaultVal2)

    # Unary
    assert (~b) == (~defaultVal)
    assert (not b) == (not defaultVal)
    assert (-b) == (-defaultVal)
    assert (+b) == (+defaultVal)
    assert bool(b) == bool(defaultVal)

def test_CallableParameter():
    # Test with function
    def bananaFunc(obj):
        return obj.testVal
    b = CallableParameter("banana", bananaFunc, "I am a banana")
    b.createAttr("testVal", -99)

    assert b.__call__() == -99
    assert b.value == -99

    # Test with lambda
    appleFunc = lambda obj: obj.testVal
    a = CallableParameter("apple", appleFunc, "I am an apple")
    a.createAttr("testVal", 101)

    assert a.__call__() == 101
    assert a.value == 101

    # Test write
    a.value = 55
    assert a.value == 101

def test_BinaryOp():
    # Test 2 const
    lhs = -3
    rhs = 5

    p0 = CallableParameter.createBinaryOp(lhs, rhs, "And")
    p1 = CallableParameter.createBinaryOp(lhs, rhs, "Or")
    p2 = CallableParameter.createBinaryOp(lhs, rhs, "Lt")
    p3 = CallableParameter.createBinaryOp(lhs, rhs, "LtE")
    p4 = CallableParameter.createBinaryOp(lhs, rhs, "Eq")
    p5 = CallableParameter.createBinaryOp(lhs, rhs, "NotEq")
    p6 = CallableParameter.createBinaryOp(lhs, rhs, "Gt")
    p7 = CallableParameter.createBinaryOp(lhs, rhs, "GtE")
    p8 = CallableParameter.createBinaryOp(lhs, rhs, "Mult")
    p9 = CallableParameter.createBinaryOp(lhs, rhs, "Pow")
    p10 = CallableParameter.createBinaryOp(lhs, rhs, "Div")
    p11 = CallableParameter.createBinaryOp(lhs, rhs, "FloorDiv")
    p12 = CallableParameter.createBinaryOp(lhs, rhs, "Mod")
    p13 = CallableParameter.createBinaryOp(lhs, rhs, "Add")
    p14 = CallableParameter.createBinaryOp(lhs, rhs, "Sub")
    p15 = CallableParameter.createBinaryOp(lhs, rhs, "BitAnd")
    p16 = CallableParameter.createBinaryOp(lhs, rhs, "BitOr")
    p17 = CallableParameter.createBinaryOp(lhs, rhs, "min")
    p18 = CallableParameter.createBinaryOp(lhs, rhs, "max")

    def customOp0(lhs, rhs):
        return rhs

    customOp1 = lambda lhs, rhs : rhs

    def badOp(lhs):
        return lhs

    p19 = CallableParameter.createBinaryOp(lhs, rhs, customOp0)
    assert p19.name == "CustomBinaryOp"

    p20 = CallableParameter.createBinaryOp(lhs, rhs, customOp1)
    assert p20.name == "CustomBinaryOp"

    try:
        p21 = CallableParameter.createBinaryOp(lhs, rhs, badOp)
        p21.value # Dummy to satisfy linter
    except CallableParameter.BadFunc:
        pass
    except :
        assert 0, "Failed bad binary op function handling"

    assert p0.value == (lhs and rhs)
    assert p1.value == (lhs or rhs)
    assert p2.value == (lhs < rhs)
    assert p3.value == (lhs <= rhs)
    assert p4.value == (lhs == rhs)
    assert p5.value == (lhs != rhs)
    assert p6.value == (lhs > rhs)
    assert p7.value == (lhs >= rhs)
    assert p8.value == (lhs * rhs)
    assert p9.value == (lhs ** rhs)
    assert p10.value == (lhs / rhs)
    assert p11.value == (lhs // rhs)
    assert p12.value == (lhs % rhs)
    assert p13.value == (lhs + rhs)
    assert p14.value == (lhs - rhs)
    assert p15.value == (lhs & rhs)
    assert p16.value == (lhs | rhs)
    assert p17.value == min(lhs, rhs)
    assert p18.value == max(lhs, rhs)
    assert p19.value == rhs
    assert p20.value == rhs

    # Test lhs const
    lhs = -3
    rhs = Parameter("banana", 5, 5, "I am a banana")

    p0 = CallableParameter.createBinaryOp(lhs, rhs, "And")
    p1 = CallableParameter.createBinaryOp(lhs, rhs, "Or")
    p2 = CallableParameter.createBinaryOp(lhs, rhs, "Lt")
    p3 = CallableParameter.createBinaryOp(lhs, rhs, "LtE")
    p4 = CallableParameter.createBinaryOp(lhs, rhs, "Eq")
    p5 = CallableParameter.createBinaryOp(lhs, rhs, "NotEq")
    p6 = CallableParameter.createBinaryOp(lhs, rhs, "Gt")
    p7 = CallableParameter.createBinaryOp(lhs, rhs, "GtE")
    p8 = CallableParameter.createBinaryOp(lhs, rhs, "Mult")
    p9 = CallableParameter.createBinaryOp(lhs, rhs, "Pow")
    p10 = CallableParameter.createBinaryOp(lhs, rhs, "Div")
    p11 = CallableParameter.createBinaryOp(lhs, rhs, "FloorDiv")
    p12 = CallableParameter.createBinaryOp(lhs, rhs, "Mod")
    p13 = CallableParameter.createBinaryOp(lhs, rhs, "Add")
    p14 = CallableParameter.createBinaryOp(lhs, rhs, "Sub")
    p15 = CallableParameter.createBinaryOp(lhs, rhs, "BitAnd")
    p16 = CallableParameter.createBinaryOp(lhs, rhs, "BitOr")
    p17 = CallableParameter.createBinaryOp(lhs, rhs, "min")
    p18 = CallableParameter.createBinaryOp(lhs, rhs, "max")

    def customOp2(lhs, rhs):
        return lhs

    customOp3 = lambda lhs, rhs : lhs

    p19 = CallableParameter.createBinaryOp(lhs, rhs, customOp2)
    p20 = CallableParameter.createBinaryOp(lhs, rhs, customOp3)

    assert p0.value == (lhs and rhs)
    assert p1.value == (lhs or rhs)
    assert p2.value == (lhs < rhs)
    assert p3.value == (lhs <= rhs)
    assert p4.value == (lhs == rhs)
    assert p5.value == (lhs != rhs)
    assert p6.value == (lhs > rhs)
    assert p7.value == (lhs >= rhs)
    assert p8.value == (lhs * rhs)
    assert p9.value == (lhs ** rhs)
    assert p10.value == (lhs / rhs)
    assert p11.value == (lhs // rhs)
    assert p12.value == (lhs % rhs)
    assert p13.value == (lhs + rhs)
    assert p14.value == (lhs - rhs)
    assert p15.value == (lhs & rhs)
    assert p16.value == (lhs | rhs)
    assert p17.value == min(lhs, rhs)
    assert p18.value == max(lhs, rhs)
    assert p19.value == lhs
    assert p20.value == lhs

    # Test rhs const
    lhs = Parameter("banana", -3, -3, "I am a banana")
    rhs = 5

    p0 = CallableParameter.createBinaryOp(lhs, rhs, "And")
    p1 = CallableParameter.createBinaryOp(lhs, rhs, "Or")
    p2 = CallableParameter.createBinaryOp(lhs, rhs, "Lt")
    p3 = CallableParameter.createBinaryOp(lhs, rhs, "LtE")
    p4 = CallableParameter.createBinaryOp(lhs, rhs, "Eq")
    p5 = CallableParameter.createBinaryOp(lhs, rhs, "NotEq")
    p6 = CallableParameter.createBinaryOp(lhs, rhs, "Gt")
    p7 = CallableParameter.createBinaryOp(lhs, rhs, "GtE")
    p8 = CallableParameter.createBinaryOp(lhs, rhs, "Mult")
    p9 = CallableParameter.createBinaryOp(lhs, rhs, "Pow")
    p10 = CallableParameter.createBinaryOp(lhs, rhs, "Div")
    p11 = CallableParameter.createBinaryOp(lhs, rhs, "FloorDiv")
    p12 = CallableParameter.createBinaryOp(lhs, rhs, "Mod")
    p13 = CallableParameter.createBinaryOp(lhs, rhs, "Add")
    p14 = CallableParameter.createBinaryOp(lhs, rhs, "Sub")
    p15 = CallableParameter.createBinaryOp(lhs, rhs, "BitAnd")
    p16 = CallableParameter.createBinaryOp(lhs, rhs, "BitOr")
    p17 = CallableParameter.createBinaryOp(lhs, rhs, "min")
    p18 = CallableParameter.createBinaryOp(lhs, rhs, "max")

    def customOp4(lhs, rhs):
        return lhs+rhs*lhs

    customOp5 = lambda lhs, rhs : lhs+rhs*lhs

    p19 = CallableParameter.createBinaryOp(lhs, rhs, customOp4)
    p20 = CallableParameter.createBinaryOp(lhs, rhs, customOp5)

    assert p0.value == (lhs and rhs)
    assert p1.value == (lhs or rhs)
    assert p2.value == (lhs < rhs)
    assert p3.value == (lhs <= rhs)
    assert p4.value == (lhs == rhs)
    assert p5.value == (lhs != rhs)
    assert p6.value == (lhs > rhs)
    assert p7.value == (lhs >= rhs)
    assert p8.value == (lhs * rhs)
    assert p9.value == (lhs ** rhs)
    assert p10.value == (lhs / rhs)
    assert p11.value == (lhs // rhs)
    assert p12.value == (lhs % rhs)
    assert p13.value == (lhs + rhs)
    assert p14.value == (lhs - rhs)
    assert p15.value == (lhs & rhs)
    assert p16.value == (lhs | rhs)
    assert p17.value == min(lhs, rhs)
    assert p18.value == max(lhs, rhs)
    assert p19.value == (lhs+lhs*rhs)
    assert p20.value == (lhs+lhs*rhs)

    # Test lhs, rhs params
    lhs = Parameter("banana", -3, -3, "I am a banana")
    rhs = Parameter("apple", 5, 5, "I am an apple")

    p0 = CallableParameter.createBinaryOp(lhs, rhs, "And")
    p1 = CallableParameter.createBinaryOp(lhs, rhs, "Or")
    p2 = CallableParameter.createBinaryOp(lhs, rhs, "Lt")
    p3 = CallableParameter.createBinaryOp(lhs, rhs, "LtE")
    p4 = CallableParameter.createBinaryOp(lhs, rhs, "Eq")
    p5 = CallableParameter.createBinaryOp(lhs, rhs, "NotEq")
    p6 = CallableParameter.createBinaryOp(lhs, rhs, "Gt")
    p7 = CallableParameter.createBinaryOp(lhs, rhs, "GtE")
    p8 = CallableParameter.createBinaryOp(lhs, rhs, "Mult")
    p9 = CallableParameter.createBinaryOp(lhs, rhs, "Pow")
    p10 = CallableParameter.createBinaryOp(lhs, rhs, "Div")
    p11 = CallableParameter.createBinaryOp(lhs, rhs, "FloorDiv")
    p12 = CallableParameter.createBinaryOp(lhs, rhs, "Mod")
    p13 = CallableParameter.createBinaryOp(lhs, rhs, "Add")
    p14 = CallableParameter.createBinaryOp(lhs, rhs, "Sub")
    p15 = CallableParameter.createBinaryOp(lhs, rhs, "BitAnd")
    p16 = CallableParameter.createBinaryOp(lhs, rhs, "BitOr")
    p17 = CallableParameter.createBinaryOp(lhs, rhs, "min")
    p18 = CallableParameter.createBinaryOp(lhs, rhs, "max")

    def customOp6(lhs, rhs):
        return lhs-rhs*lhs

    customOp7 = lambda lhs, rhs : lhs-rhs*lhs

    p19 = CallableParameter.createBinaryOp(lhs, rhs, customOp6)
    p20 = CallableParameter.createBinaryOp(lhs, rhs, customOp7)

    assert p0.value == (lhs and rhs)
    assert p1.value == (lhs or rhs)
    assert p2.value == (lhs < rhs)
    assert p3.value == (lhs <= rhs)
    assert p4.value == (lhs == rhs)
    assert p5.value == (lhs != rhs)
    assert p6.value == (lhs > rhs)
    assert p7.value == (lhs >= rhs)
    assert p8.value == (lhs * rhs)
    assert p9.value == (lhs ** rhs)
    assert p10.value == (lhs / rhs)
    assert p11.value == (lhs // rhs)
    assert p12.value == (lhs % rhs)
    assert p13.value == (lhs + rhs)
    assert p14.value == (lhs - rhs)
    assert p15.value == (lhs & rhs)
    assert p16.value == (lhs | rhs)
    assert p17.value == min(lhs, rhs)
    assert p18.value == max(lhs, rhs)
    assert p19.value == (lhs-lhs*rhs)
    assert p20.value == (lhs-lhs*rhs)

def test_UnaryOp():
    # Test const
    rhs = 5

    p0 = CallableParameter.createUnaryOp(rhs, "Not")
    p1 = CallableParameter.createUnaryOp(rhs, "Invert")
    p2 = CallableParameter.createUnaryOp(rhs, "None")

    def customOp(rhs):
        return rhs*5

    customOp1 = lambda rhs : rhs*5

    def badOp(lhs, rhs):
        return lhs + rhs

    p3 = CallableParameter.createUnaryOp(rhs, customOp)
    assert p3.name == "CustomUnaryOp"

    p4 = CallableParameter.createUnaryOp(rhs, customOp1)
    assert p4.name == "CustomUnaryOp"

    try:
        p5 = CallableParameter.createUnaryOp(rhs, badOp)
        p5.value # Dummy to satisfy linter
    except CallableParameter.BadFunc:
        pass
    except :
        assert 0, "Failed bad binary op function"

    assert p0.value == (not rhs)
    assert p1.value == ~rhs
    assert p2.value == rhs
    assert p3.value == rhs*5
    assert p4.value == rhs*5

    # Test param
    rhs = Parameter("banana", 5, 5, "I am a banana")

    p0 = CallableParameter.createUnaryOp(rhs, "Not")
    p1 = CallableParameter.createUnaryOp(rhs, "Invert")
    p2 = CallableParameter.createUnaryOp(rhs, "None")

    def customOp(rhs):
        return rhs*5

    customOp1 = lambda rhs : rhs*5

    p3 = CallableParameter.createUnaryOp(rhs, customOp)
    assert p3.name == "CustomUnaryOp"

    p4 = CallableParameter.createUnaryOp(rhs, customOp1)
    assert p4.name == "CustomUnaryOp"

    assert p0.value == (not rhs)
    assert p1.value == ~rhs
    assert p2.value == rhs
    assert p3.value == rhs*5
    assert p4.value == rhs*5

def test_ExpressionEvaluator():
    # Context
    a = 5
    b = -5
    c = 1
    d = -2
    context = {"a": a, "b": b, "c": c, "d": d}

    # Binary ops
    expr = "a + (b + (c*d) % a - 25)//d + 20"

    tree = ast.parse(expr)
    exprEval = ExpressionEvaluator().evaluate(tree, context)
    assert  exprEval == (a + (b + (c*d) % a - 25)//d + 20)

    # Mixed ops
    expr = "+a < (-b ^ d + ((c << 1) * (d >> 5)) > ~20 >= (not a & c | d) and c <= d or a == d)"
    tree = ast.parse(expr)
    exprEval = ExpressionEvaluator().evaluate(tree, context)
    assert  exprEval == (+a < (-b ^ d + ((c << 1) * (d >> 5)) > ~20 >= (not a & c | d) and c <= d or a ==d))

    # Assignment to value
    expr = "a=20"
    tree = ast.parse(expr)
    exprEval = ExpressionEvaluator().evaluate(tree, context)
    assert(context["a"] == 20) and (exprEval == 20)

    # Assignment to name
    expr = "a=b"
    tree = ast.parse(expr)
    exprEval = ExpressionEvaluator().evaluate(tree, context)
    assert(context["a"] == context["b"]) and (exprEval == context["b"])

    # Multiple assignment
    expr = "a=b=c"
    tree = ast.parse(expr)
    exprEval = ExpressionEvaluator().evaluate(tree, context)
    assert(context["a"] == context["c"]) and (context["b"] == context["c"]) and (exprEval == context["c"])

    # Assignment to expression
    expr = "b = max(5, 10)*10 / (2**4) - min(b, 4 % 2//1) if (1 != 2) else 0"
    compareVal = max(5, 10)*10 / (2**4) - min(context["b"], 4 % 2//1) if (1 != 2) else 0
    tree = ast.parse(expr)
    exprEval = ExpressionEvaluator().evaluate(tree, context)
    assert (context["b"] == compareVal) and (exprEval == compareVal)


def test_ProjectConfig():
    proj = ProjectConfig()
    proj.createValue("tVal", 75, -99, "TestDescription")
    subA = proj.createSection("SubA")
    subA.createValue("tVal", 0)
    subA.createValue("tMin", 0)
    subA.createValue("tMax", 99)
    subB = proj.createSection("SubB")
    subB.createValue("tVal", 0)
    subB.createValue("tMin", 78)
    subB.createValue("tMax", 100)
    subC = subB.createSection("SubC")
    subC.createValue("tVal", 0, 500, "Test")
    subC.createValue("tMin", -10000)
    subC.createValue("tMax", -500)

    # Assertions with different hierarchy levels
    proj.addConstraint("tVal >= SubA.tMin and tVal >= SubB.SubC.tMax")

    # Hard assignments for constraints
    proj.addConstraint("SubA.tVal = max(SubA.tMin, min(SubA.tMax, tVal))")
    proj.addConstraint("SubB.tVal = max(SubB.tMin, min(SubB.tMax, tVal))")
    proj.addConstraint("SubB.SubC.tVal = max(SubB.SubC.tMin, min(SubB.SubC.tMax, tVal))")
    proj.addConstraint("tVal = max(SubA.tVal, max(SubB.tVal, SubB.SubC.tVal))")
    proj.checkConstraints()

    assert subA.tVal == 75
    assert subB.tVal == 78
    assert subC.tVal == -500
    assert proj.tVal == 78

    # Access patterns:
    # Attribute chains. A.B.C
    # Item chains ["A"]["B"]["C"]
    # Item attribute chains ["A.B.C"]
    proj.SubA.tVal = 50
    assert subA.tVal == 50
    proj["SubA.tVal"] = 29
    assert subA.tVal == 29

    proj.tVal = -1
    assert proj.tVal == -1
    proj["tVal"] = +2
    assert proj.tVal == +2

    proj.SubB.SubC.tVal = 44
    assert subC.tVal == 44
    proj["SubB.SubC.tVal"] = 21
    assert subC.tVal == 21
    proj["SubB"]["SubC"]["tVal"] = 42
    assert subC.tVal == 42

    assert proj.getDefaultValue("tVal") == -99
    assert proj.getDescription("tVal") == "TestDescription"

    assert proj.getDefaultValue("SubB.SubC.tVal") == 500
    assert proj.getDescription("SubB.SubC.tVal") == "Test"
