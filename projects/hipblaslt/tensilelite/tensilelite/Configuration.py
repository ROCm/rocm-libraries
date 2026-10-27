################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

from copy import deepcopy
import ast

class ReadWriteTransformDict(dict):  # dicts take a mapping or iterable as their optional first argument
    """
    The focus of this class is to allow explicit customization of
    reading and writing to attributes.
    Users of this class might restrict read or write access to particular
    variables, or perform some customization on those functionalities.
    """

    __slots__ = () # no __dict__ - that would be redundant

    readTransformFuncKey = '_readTransformFunc'
    writeTransformFuncKey = '_writeTransformFunc'

    def __init__(self, readTransformFunc = None, writeTransformFunc = None):
        super().__init__()
        if readTransformFunc is not None and callable(readTransformFunc):
            self.setReadTransform(readTransformFunc)
        if writeTransformFunc is not None and callable(writeTransformFunc):
            self.setWriteTransform(writeTransformFunc)

    def __getitem__(self, key):
        return self.readWithTransform(key) if self.hasReadTransform() else self.readNoTransform(key)

    def __setitem__(self, key, value):
        self.writeWithTransform(key, value) if self.hasWriteTransform() else self.writeNoTransform(key, value)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __copy__(self): # don't delegate w/ super - dict.copy() -> dict :(
        cls = self.__class__
        result = cls.__new__(cls)
        result.update(self)
        return result

    def __deepcopy__(self, memo): # don't delegate w/ super - dict.copy() -> dict :(
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key in self.keys():
            result.writeNoTransform(key, deepcopy(self.readNoTransform(key)))
        return result

    def __repr__(self):
        return "\n".join(self.__toPrettyLines())

    def __toPrettyLines(self):
        """
        Embedded values are sometimes hard to read and understand
        with so many layers, so add some tabs and line breaks
        to help distinguish hierarchy trees of objects in the dict.
        """
        prettyLines = []
        prettyLines += ["<{0}(".format(type(self).__name__)]
        for key, value in self.items():
            if isinstance(value, ReadWriteTransformDict):
                valuePrettyLines = value.__toPrettyLines()
                prettyLines += ["\t{0}: {1}".format(key, valuePrettyLines.pop(0))]
                for line in valuePrettyLines:
                    prettyLines += ["\t{0}".format(line)]
            else:
                prettyLines += ["\t{0}: {1}".format(key, value)]
        prettyLines += [")>"]
        return prettyLines

    def get(self, key, default=None):
        try:
            return self[key]
        except:
            return default

    def set(self, key, value):
        try:
            self[key] = value
        except:
            pass

    # Transform tests
    def hasReadTransform(self):
        return self.readTransformFuncKey in self

    def hasWriteTransform(self):
        return self.writeTransformFuncKey in self

    # Omit transform
    def readNoTransform(self, name):
        return super().__getitem__(name)

    def writeNoTransform(self, name, value):
        super().__setitem__(name, value)

    # Use transform
    def readWithTransform(self, name):
        return self.getReadTransform()(self, name)

    def writeWithTransform(self, name, value):
        self.getWriteTransform()(self, name, value)

    # Transform get/set
    def getReadTransform(self):
        return self.readNoTransform(self.readTransformFuncKey) if self.hasReadTransform() else None

    def setReadTransform(self, readTransformFunc):
        self.writeNoTransform(self.readTransformFuncKey, readTransformFunc)
        if readTransformFunc is None:
            self.pop(self.readTransformFuncKey, None)

    def getWriteTransform(self):
        return self.readNoTransform(self.writeTransformFuncKey) if self.hasWriteTransform() else None

    def setWriteTransform(self, writeTransformFunc):
        self.writeNoTransform(self.writeTransformFuncKey, writeTransformFunc)
        if writeTransformFunc is None:
            self.pop(self.writeTransformFuncKey, None)

    def toDict(self):
        return dict(zip(self.keys(), self.values()))

    @staticmethod
    def flattenDict(obj, prefix = "", separator="."):
        """
        Sometimes it is favorable to reduce to a single hierarchy
        dict for readability or debugging.
        """
        result = ReadWriteTransformDict()
        for key in obj.keys():
            attribute = obj.readNoTransform(key)
            attribute = attribute.readNoTransform("value") if isinstance(attribute, ReadWriteTransformDict) else attribute
            if isinstance(attribute, dict):
                subDict = ReadWriteTransformDict.toFlattenedDict(attribute, key, separator)
                for subKey in subDict.keys():
                    newKey = separator.join([prefix, subKey]) if prefix != "" else subKey
                    result[newKey] = deepcopy(subDict[subKey])
            else:
                newKey = separator.join([prefix, key]) if prefix != "" else key
                result[newKey] = deepcopy(obj[key])
        return result

    def toFlattenedDict(self, prefix = "", separator="."):
        return ReadWriteTransformDict.flattenDict(self, prefix, separator)

class Parameter(ReadWriteTransformDict):
    """
    The Parameter class will be used as the main
    container for configuration properties.
    For this class we want to group configurations
    by name, value, default value and descriptions.
    No other attributes may be set.
    All operators  and reverse operators are overloaded
    to use value attribute as the operand as this is
    usually the one we are interested in.
    """
    __slots__ = ()

    # Simplification of creating attributes
    def createAttr(self, name, value):
        super().writeNoTransform(name, value)

    def __init__(self, name, initialValue, defaultValue=None, description=""):
        self.createAttr("type", type(initialValue))
        self.createAttr("name", name)
        self.createAttr("value", initialValue)
        self.createAttr("defaultValue", defaultValue)
        self.createAttr("description", description)

#        if defaultValue is None:
#            self.createAttr("defaultValue", initialValue)
#        else:
#            self.createAttr("defaultValue", defaultValue)

        # Restrict writing to only known fields
        def writeXForm(self, key, value):
            if key not in self or key == "type":
                raise AttributeError("Cannot write attribute: {0}".format(key))
            else:
                # Ensure incoming values are same type to allow assignment
                objType = self.readNoTransform("type")
                incomingType = type(value)
                if key == "value" and objType is not incomingType:
                    raise AttributeError("Type preservation: stored {0} != incoming {1}".format(objType, incomingType))
                else:
                    self.writeNoTransform(key, value)

        super().__init__(None, writeXForm)

    # Boolean ops
    def __lt__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value < rhs.value
        else:
            return self.value < rhs

    def __rlt__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value < self.value
        else:
            return lhs < self.value

    def __le__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value <= rhs.value
        else:
            return self.value <= rhs

    def __rle__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value <= self.value
        else:
            return lhs <= self.value

    def __eq__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value == rhs.value
        else:
            return self.value == rhs

    def __req__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value == self.value
        else:
            return lhs == self.value

    def __ne__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value != rhs.value
        else:
            return self.value != rhs

    def __rne__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value != self.value
        else:
            return lhs != self.value

    def __gt__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value > rhs.value
        else:
            return self.value > rhs

    def __rgt__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value > self.value
        else:
            return lhs > self.value

    def __ge__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value >= rhs.value
        else:
            return self.value >= rhs

    def __rge__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value >= self.value
        else:
            return lhs >= self.value

    # Binary Ops with their reverses
    def __add__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value + rhs.value
        else:
            return self.value + rhs

    def __radd__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value + self.value
        else:
            return lhs + self.value

    def __sub__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value - rhs.value
        else:
            return self.value - rhs

    def __rsub__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value - self.value
        else:
            return lhs - self.value

    def __mul__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value * rhs.value
        else:
            return self.value * rhs

    def __rmul__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value * self.value
        else:
            return lhs * self.value

    def __truediv__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value / rhs.value
        else:
            return self.value / rhs

    def __rtruediv__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value / self.value
        else:
            return lhs / self.value

    def __floordiv__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value // rhs.value
        else:
            return self.value // rhs

    def __rfloordiv__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value // self.value
        else:
            return lhs // self.value

    def __mod__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value % rhs.value
        else:
            return self.value % rhs

    def __rmod__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value % self.value
        else:
            return lhs % self.value

    def __pow__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value ** rhs.value
        else:
            return self.value ** rhs

    def __rpow__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value ** self.value
        else:
            return lhs ** self.value

    def __rshift__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value >> rhs.value
        else:
            return self.value >> rhs

    def __rrshift__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value >> self.value
        else:
            return lhs >> self.value

    def __lshift__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value << rhs.value
        else:
            return self.value << rhs

    def __rlshift__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value << self.value
        else:
            return lhs << self.value

    def __and__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value & rhs.value
        else:
            return self.value & rhs

    def __rand__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value & self.value
        else:
            return lhs & self.value

    def __or__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value | rhs.value
        else:
            return self.value | rhs

    def __ror__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value | self.value
        else:
            return lhs | self.value

    def __xor__(self, rhs):
        if isinstance(rhs, Parameter):
            return self.value ^ rhs.value
        else:
            return self.value ^ rhs

    def __rxor__(self, lhs):
        if isinstance(lhs, Parameter):
            return lhs.value ^ self.value
        else:
            return lhs ^ self.value

    def __bool__(self):
        return bool(self.value)

    # Unary ops
    def __neg__(self):
        return -self.value

    def __pos__(self):
        return +self.value

    def __invert__(self):
        return ~self.value

    # Interface
    def resetToDefault(self):
        self.value = self.defaultValue
        return self.value

    def getValue(self):
        return self.value

    def getDefault(self):
        return self.defaultValue

    def getDescription(self):
        return self.description

class CallableParameter(Parameter):
    """
    This class is a Parameter who gets its value
    dynamically at read time. It stores a function
    that gets called with self as context, which the
    result will replace the value attribute.
    This is useful for retrieving values that are not
    constants and must be retrieved from another function
    calls
    """
    __slots__ = ()

    class BadFunc(Exception):
        pass

    def __init__(self, name, callFunc, description=""):
        super().__init__(name, 0, 0, description)
        self.createAttr("callFunc", callFunc)

        # Before each read, call on self to update value
        rXForm = self.getReadTransform()
        def readXForm(obj, key):
            if key == "value":
                obj.writeNoTransform(key, obj.__call__())
            return rXForm(obj, key) if rXForm is not None else obj.readNoTransform(key)
        self.setReadTransform(readXForm)

        # Prevent external writing to value attribute
        # Will be updated by reads as callable is updated.
        wXForm = self.getWriteTransform()
        def writeXForm(obj, key, value):
            if key == "value":
                pass
            else:
                wXForm(obj, key, value) if wXForm is not None else obj.writeNoTransform(key, value)
        self.setWriteTransform(writeXForm)

    def __call__(self):
        return self.callFunc(self)

    @classmethod
    def createBinaryOp(cls, lhs, rhs, op):
        """
        This function will build a binary operation
        function to call with two operands that are
        bound to the parameter
        """
        FuncMap = {
        "And"      : lambda lhs, rhs : lhs and rhs,
        "Or"       : lambda lhs, rhs : lhs or rhs,
        "Lt"       : lambda lhs, rhs : lhs <  rhs,
        "LtE"      : lambda lhs, rhs : lhs <= rhs,
        "Eq"       : lambda lhs, rhs : lhs == rhs,
        "NotEq"    : lambda lhs, rhs : lhs != rhs,
        "Gt"       : lambda lhs, rhs : lhs >  rhs,
        "GtE"      : lambda lhs, rhs : lhs >= rhs,
        "Mult"     : lambda lhs, rhs : lhs * rhs,
        "Pow"      : lambda lhs, rhs : lhs ** rhs,
        "Div"      : lambda lhs, rhs : lhs / rhs,
        "FloorDiv" : lambda lhs, rhs : lhs // rhs,
        "Mod"      : lambda lhs, rhs : lhs % rhs,
        "Add"      : lambda lhs, rhs : lhs + rhs,
        "Sub"      : lambda lhs, rhs : lhs - rhs,
        "BitAnd"   : lambda lhs, rhs : lhs & rhs,
        "BitOr"    : lambda lhs, rhs : lhs | rhs,
        "BitXor"   : lambda lhs, rhs : lhs ^ rhs,
        "LShift"   : lambda lhs, rhs : lhs << rhs,
        "RShift"   : lambda lhs, rhs : lhs >> rhs,
        "min"      : lambda lhs, rhs : min(lhs, rhs),
        "max"      : lambda lhs, rhs : max(lhs, rhs),
        }

        # Incoming op is a name
        if isinstance(op, str):
            assert op in FuncMap, "Missing operation in funcMap: {0}".format(op)
            name = op
            func = FuncMap[op]
        # Incoming op assumed as a function
        else:
            name = "CustomBinaryOp"
            func = op

        # Verify that the function is callable with 2 params
        try:
            func(lhs, rhs)
        except Exception as ex:
            raise CallableParameter.BadFunc from ex

        # Capture function and attached values
        opKey = next((key for key in FuncMap if FuncMap[key] == op), None)
        if opKey is not None and opKey in ["And", "Or", "Lt", "Le", "Eq", "NotEq", "Gt", "Ge"]:
            callFunc = lambda obj : bool(func(obj.lhs, obj.rhs))
        else:
            callFunc = lambda obj : func(obj.lhs, obj.rhs)

        binOp = cls(name, callFunc, "Binary operaton with two operands")
        binOp.createAttr("lhs", lhs)
        binOp.createAttr("rhs", rhs)

        return binOp

    @classmethod
    def createUnaryOp(cls, rhs, op):
        """
        This function will build a unary operation
        function to call with one operand that is
        bound to the parameter
        """

        FuncMap = {
        "Not"    : lambda val : not val,
        "Invert" : lambda val : ~val,
        "USub"   : lambda val : -val,
        "UAdd"   : lambda val : +val,
        "None"   : lambda val : val
        }

        # Incoming op is a name
        if isinstance(op, str):
            assert op in FuncMap, "Missing operation in funcMap: {0}".format(op)
            name = op
            func = FuncMap[op]
        # Incoming op assumed as a function
        else:
            name = "CustomUnaryOp"
            func = op

        # Verify that the function is callable with 1 param
        try:
            func(rhs)
        except Exception as ex:
            raise CallableParameter.BadFunc from ex

        # Capture function and attached values
        opKey = next((key for key in FuncMap if FuncMap[key] == op), None)
        if opKey is not None and opKey in ["Not"]:
            callFunc = lambda obj : bool(func(obj.rhs))
        else:
            callFunc = lambda obj : func(obj.rhs)

        unOp = cls(name, callFunc, "Unary operaton with one operand")
        unOp.createAttr("rhs", rhs)

        return unOp

class ExpressionEvaluator(object):
    """
    This class is a recursive visitor of an ast module tree.
    The ast tree is generated from a text expression (e.g. a > 5).
    Given a context object to resolve names to values, the expression
    evaluator will return a chain of callable parameters that can be
    invoked to get the result of the expression.
    Example:
        context = {a : 10}
        tree = ast.parse("a > 5", mode='exec') # ast library
        exprEval = ExpressionEvaluator.evaluate(tree, context)
        result = exprEval.__call__() # = True
    """

    __slots__ = ()

    def evaluate(self, node, namesContext):

        #print("Type: {0}".format(type(node).__name__))
        #print(node)

        nodeType = type(node).__name__

        # Root Level 0 node (required)
        if nodeType == "Module":
            # fields: ('body')
            assert len(node.body) == 1, "Expecting only one expression"
            return self.evaluate(node.body[0], namesContext)

        # Level 1 node (required)
        elif nodeType == "Expr":
            # fields: ('value')
            result = self.evaluate(node.value, namesContext)
            return result.value if hasattr(result, "value") else result

        # Binary operations nodes which take 2 operands
        elif nodeType == "BinOp":
            # fields: ('op', 'left', 'right')
            lhs = self.evaluate(node.left, namesContext)
            rhs = self.evaluate(node.right, namesContext)
            op = type(node.op).__name__
            return CallableParameter.createBinaryOp(lhs, rhs, op)

        elif nodeType == "BoolOp":
            # fields: ('op', 'values')
            lhs = self.evaluate(node.values[0], namesContext)
            rhs = self.evaluate(node.values[1], namesContext)
            op = type(node.op).__name__
            return CallableParameter.createBinaryOp(lhs, rhs, op)

        elif nodeType == "Compare":
            # fields: ('left', 'ops', 'comparators')
            lhs = self.evaluate(node.left, namesContext)
            for i in range(len(node.ops)):
                rhs = self.evaluate(node.comparators[i], namesContext)
                op = type(node.ops[i]).__name__
                result = CallableParameter.createBinaryOp(lhs, rhs, op)
                lhs = result
            return lhs

        # Unary operations which take 1 operand
        elif nodeType == "UnaryOp":
            # fields: ('op', 'operand')
            value = self.evaluate(node.operand, namesContext)
            op = type(node.op).__name__
            return CallableParameter.createUnaryOp(value, op)

        elif nodeType == "Call":
            # fields: ('func', 'args', 'keywords')
            if len(node.args) == 2:
                lhs = self.evaluate(node.args[0], namesContext)
                rhs = self.evaluate(node.args[1], namesContext)
                # For the op, give empty context so we get a string
                # from the name node. Then pass it into the factory
                op = self.evaluate(node.func, {})
                return CallableParameter.createBinaryOp(lhs, rhs, op)

            elif len(node.args) == 1:
                rhs = self.evaluate(node.args[0], namesContext)
                # For the op, give empty context so we get a string
                # from the name node. Then pass it into the factory
                op = self.evaluate(node.func, {})
                return CallableParameter.createUnaryOp(rhs, op)

            assert 0, "Unknown function call with {0} parameters".format(len(node.args))

        elif nodeType == "Assign":
            # fields: ('targets', 'value')

            value = self.evaluate(node.value, namesContext)
            valueToAssign = value.value if hasattr(value, "value") else value

            for target in node.targets:
                targetType = type(target).__name__

                # If we are assigning to a name, use
                # current context as write object.
                if targetType == "Name":
                    assignObj = namesContext
                    assignAttr = target.id

                # If we are assigning to an attribute,
                # get the embedded context to write to
                elif targetType == "Attribute":
                    assignObj = self.evaluate(target, namesContext)
                    assignAttr = target.attr

                else: assert 0, "Don't know how to handle target node type: {0}".format(targetType)

                # The assignment value might be an expression,
                # so if it is callable, then evaluate it
                assignObj[assignAttr] = valueToAssign

            return valueToAssign

        # Leaf node types
        elif nodeType == "Name":
            # fields:  ('id')
            if node.id in namesContext:
                return namesContext[node.id]
            else:
                 print("No context for named variable: {0}".format(node.id))
                 return str(node.id)

        elif nodeType == "Attribute":
            # fields:  ('value', 'attr', 'ctx')

            # node.attr has the leaf attribute. We need to recurse and get
            # the leaf attribute container so we can access the attribute.
            # node.value has the next chained node, which is either another Attribute
            # or Name node
            newContext = self.evaluate(node.value, namesContext)
            assert node.attr in newContext, "No attribute for named variable: {0}".format(node.attr)

            # For storing context, return the target value's container so we can
            # setattr on it. See: Assign case.
            # Otherwise return the actual stored value
            return newContext if type(node.ctx).__name__ == "Store" else newContext[node.attr]

        elif nodeType == "IfExp":
            # fields ('test', 'body', 'orelse')
            theTest = self.evaluate(node.test, namesContext)
            trueCond = self.evaluate(node.body, namesContext)
            falseCond = self.evaluate(node.orelse, namesContext)
            return trueCond if theTest() else falseCond

        elif nodeType == "Num":
            # fields ('n')
            return node.n

        elif nodeType == "Constant":
            # fields ('kind', 'value')
            return node.value

        elif nodeType == "Str":
            # fields ('s')
            return node.s

        assert 0, "Unknown node type: {0}".format(nodeType)

# Everything stored as a Parameter object, which has
# value, default value and type attributes
class ProjectConfig(ReadWriteTransformDict):
    """
    Adds support for hierarchical configuration designs.
    Allows grouping of configurations together.
    Also allows dynamic constraints.

    Each Config entry is a Parameter type object that
    encapsulates attributes of Name, Type, Value, DefaultValue
    and Description as seen above.

    Example:

    Program:
        Network:
            IP: X.X.X.X
            PORT: XXXX
            MaxPort: YYYY
        UI:
            WindowSize: 2000
            Antialiasing: True

    # Creation patterns:
    Program = ProjectConfig()
    networkConfig = Program.createSection("Network")
    networkConfig.createValue("IP", X.X.X.X, "This is the IP we connect to")
    networkConfig.createValue("PORT", XXXX, "Port we want to access")
    networkConfig.createValue("PORT", YYYY, "Maximum port value")

    uiConfig = Program.createSection("UI")
    uiConfig.createValue("WindowSize", 2000)
    uiConfig.createValue("Antialiasing", True)

    # Access patterns:
    IP = Program.Network.IP
    PORT = Program["Network"]["IP"]
    networkConfig = Program["Network"]

    # Constraints assertions: e.g. limit port numbers or antialiasing.
    Program.addConstraint("Program.Network.PORT >= MaxPort")
    Program.addConstraint("False if Program.UI.WindowSize > 2000 and Program.UI.Antialiasing else True")

    # Overrides: e.g. enforce port numbers
    Program.addConstraint("Program.Network.PORT = Program.Network.MaxPort if Program.Network.PORT > Program.Network.MaxPort")

    """

    __slots__ = ()

    ContainerType = Parameter
    SubConfigType : 'ProjectConfig'
    ConstraintsKey = "_Constraints"

    def __init__(self):

        # Sub-configs will be stored inside ContainerType,
        # so when reading them as an attribute, forward the config
        # stored in the container's value
        def readXForm(obj, key):

            attribute = obj.readNoTransform(key)

            # Attribute is a Container, which has a "value"
            # and should be forwarded to get around awkward
            # value forwarding. We are mainly interested
            # in accessing the value attribute, so we can
            # now access it with config["Attrib"] instead
            # of config["Attrib"]["value"]
            if isinstance(attribute, self.ContainerType):
                return attribute.readNoTransform("value")
            else:
                return attribute

        def writeXForm(obj, key, value):
            if key in super(obj.__class__, obj).keys() and isinstance(obj.readNoTransform(key), self.ContainerType):
                obj.readNoTransform(key).writeNoTransform("value", value)
            else:
                obj.writeNoTransform(key, value)

        super().__init__(readXForm, writeXForm)

    def __contains__(self, key):
        return True if super().__contains__(key) else self.toFlattenedDict().__contains__(key)

    def __getContainer(self, key):
        # In case we are accessing with keys such as [a.b.c]
        # Then we need to recurse to get the embedded container.
        # Assumes that a and b for example are embedded configs
        if "." in key:
            try:
                levels = key.split(".")
                currentValue = self
                for level in levels[:-1]:
                    currentValue = currentValue[level]
                return currentValue.readNoTransform(levels[-1])
            except:
               pass

        # Fallback, in case there is a "." in the key
        return self.readNoTransform(key)

    def __getitem__(self, key):

        if "." in key:
            try:
                levels = key.split(".")
                currentValue = self
                for level in levels:
                    currentValue = currentValue[level]
                return currentValue
            except:
                pass

        # Fallback is to check if the full key lives
        # in the top level, in case it just happens
        # to have a '.' in it.
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        # In case we are accessing with keys such as [a.b.c]
        # Then we need to recurse to get the embedded value.
        # Assumes that a and b for example are embedded configs
        if "." in key:
            try:
                levels = key.split(".")
                currentValue = self
                for level in levels[:-1]:
                    currentValue = currentValue[level]
                setattr(currentValue, levels[-1], value)
                return
            except:
                pass

        # Fallback is to check if the full key lives
        # in the top level, in case it just happens
        # to have a '.' in it.
        super().__setitem__(key, value)

    # Simplification of creating attributes
    def createValue(self, name, value, defaultValue = None, description=""):
        self[name] = self.ContainerType(name, value, defaultValue, description)
        return self[name]

    def createSection(self, name):
        return self.createValue(name, ProjectConfig())

    def resetToDefaults(self):
        for key in self.keys():
            attribute = self.readNoTransform(key)
            if isinstance(attribute, ProjectConfig):
                attribute.resetToDefaults()
            elif isinstance(attribute, self.ContainerType):
                attribute.writeNoTransform("value", attribute.readNoTransform("defaultValue"))

    def addConstraint(self, expressionStr):
        if self.ConstraintsKey not in self:
            self.createSection(self.ConstraintsKey)

        self[self.ConstraintsKey].createValue(expressionStr, ast.parse(expressionStr, mode='exec'))

    def checkConstraints(self):
        result = True
        if self.ConstraintsKey in self:
            constraints = self.readNoTransform(self.ConstraintsKey).value
            for expression, tree in [(x, y) for (x, y) in constraints.items() if "_" not in x]:
                result = ExpressionEvaluator().evaluate(tree.value, self)
                value = result.value if isinstance(result, Parameter) else result
                assert value, "Constraint evaluation failed: {0}".format(expression)
        return value

    def getDefaultValue(self, name):
        return self.__getContainer(name).getDefault()

    def getDescription(self, name):
        return self.__getContainer(name).getDescription()
