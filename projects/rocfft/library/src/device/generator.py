# Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""HIP code generator."""

import inspect
import os
import subprocess
import types

from pathlib import Path as path
from typing import List, Any

#
# Helpers
#


def get_file_and_line(up=2):
    """Get file and file number of frame 'up'-steps up in the stack."""
    frame = inspect.currentframe()
    for _ in range(up):
        frame = frame.f_back
        if frame is None:
            return None, None
    file_name, line_number, *_ = inspect.getframeinfo(frame)
    return path(file_name).name, line_number


def join(sep, n):
    """Coerce elements in `n` to strings and join them seperator `sep`."""
    if isinstance(n, BaseNode):
        return sep + str(n)
    return sep.join(str(x) for x in n)


def sjoin(n):
    """Join with spaces."""
    return join(' ', n)


def njoin(n):
    """Join with newlines."""
    return join(os.linesep, n)


def cjoin(n):
    """Join with commas."""
    return join(', ', n)


def clang_format(code):
    """Format code using clang-format."""
    try:
        p = subprocess.run(['/opt/rocm/llvm/bin/clang-format', '-style=file'],
                           stdout=subprocess.PIPE,
                           input=str(code),
                           encoding='ascii',
                           check=True)
        return p.stdout
    except FileNotFoundError:
        # code formatting doesn't affect functionality, so just assume
        # default ROCm path and ignore errors if it's not there.
        pass
    return str(code)


def clang_format_file(filename):
    """Format a file using clang-format.  Ignores errors so the file
       remains unformatted if clang-format isn't runnable."""
    try:
        p = subprocess.run(['/opt/rocm/llvm/bin/clang-format', '-i', filename])
    except:
        # code formatting doesn't affect functionality, so just assume
        # default ROCm path and ignore errors if it's not there.
        pass


def write(fname, code, format=False):
    """Format code and write to `fname`.

    If `fname` already exists, only write if the contents changed.
    """
    if format:
        code = clang_format(code)
    else:
        code = str(code)
    f = path(fname)
    if f.exists():
        existing = f.read_text()
        if existing == code:
            return
    f.write_text(code)


def walk(x):
    """Tree traversal."""
    if isinstance(x, BaseNode):
        yield x
        for a in x.args:
            if isinstance(a, BaseNode):
                yield from a
            else:
                yield a


def sanity_check(y):
    """Optional sanity check to avoid common pitfalls."""
    failed = False
    for x in y:
        if isinstance(x, list) and len(x) > 1:
            failed = True
            print(
                f'Sanity check: '
                f'list object found in nodes won\'t be traversed and can lead to undesirable effects.\n'
                f'Node type = {type(x)}' + '\n' + f'Node contents:\n' +
                f'{njoin(x)}')
        # elif:
        #     add some other checks
    if failed:
        raise RuntimeError


def depth_first(x, f):
    """Apply `f` to each node of the AST `x`.

    Nodes are traversed in depth-first order.
    """
    if isinstance(x, BaseNode):
        y = type(x)(file_name=x.file_name, line_number=x.line_number)
        y.args = [depth_first(a, f) for a in x.args]
        return f(y)
    return f(x)


def copy(x):
    """Return a deep copy of the AST node `x`."""
    return depth_first(x, lambda y: y)


#
# Code generator base classes
#


def make_raw(s):
    """Make a simple AST node whose string representation is `s`."""

    def decorator(target):
        target.__str__ = lambda self: s
        return target

    return decorator


# XXX there is some redundancy between name_args and the constructor
#     for BaseNode


def name_args(names):
    """Make an AST node with named arguments."""

    def name_args_decorator(target):

        # attach setters & getters for each name
        for i, name in enumerate(names):

            def fset(self, val, idx=i):
                self.args[idx] = val

            setattr(target, name,
                    property(lambda self, idx=i: self.args[idx], fset))

        # define a new init that takes args and kwargs using names
        def new_init(self, *args, **kwargs):
            self.args = [None for x in names]
            for i, arg in enumerate(args):
                self.args[i] = arg
            for i, name in enumerate(names):
                if name in kwargs:
                    self.args[i] = kwargs[name]

            # self
            try:
                self.file_name = kwargs['file_name']
                self.line_number = kwargs['line_number']
            except KeyError:
                self.file_name, self.line_number = get_file_and_line()

            if hasattr(self, '__post_init__'):
                getattr(self, '__post_init__')()

        target.__init__ = new_init
        return target

    return name_args_decorator


class BaseNode:
    """Base AST Node."""

    args: List[Any]
    sep: str = None

    def __init__(self, *args, **kwargs):
        try:
            self.file_name = kwargs['file_name']
            self.line_number = kwargs['line_number']
        except KeyError:
            self.file_name, self.line_number = get_file_and_line()
        self.args = list(args)
        if hasattr(self, '__post_init__'):
            getattr(self, '__post_init__')(self)

    def __str__(self):
        if self.sep is not None:
            return join(self.sep, self.args)
        return str(self.args[0])

    def __iter__(self):
        return walk(self)

    def provenance(self):
        return '/* ' + self.file_name + ':' + str(self.line_number) + ' */'


class BaseNodeOps(BaseNode):
    """BaseNode with basic math operations added."""

    def __add__(self, a):
        return Add(self, a)

    def __radd__(self, a):
        return Add(a, self)

    def __sub__(self, a):
        return Sub(self, a)

    def __rsub__(self, a):
        return Sub(a, self)

    def __mul__(self, a):
        return Multiply(self, a)

    def __rmul__(self, a):
        return Multiply(a, self)

    def __mod__(self, a):
        return Mod(self, a)

    def __rmod__(self, a):
        return Mod(a, self)

    def __truediv__(self, a):
        return Divide(self, a)

    def __rtruediv__(self, a):
        return Divide(a, self)

    def __eq__(self, a):
        return Equal(self, a)

    def __ne__(self, a):
        return NotEqual(self, a)

    def __ge__(self, a):
        return GreaterEqual(self, a)

    def __gt__(self, a):
        return Greater(self, a)

    def __le__(self, a):
        return LessEqual(self, a)

    def __lt__(self, a):
        return Less(self, a)

    def __shl__(self, a):
        return ShiftLeft(self, a)

    def __shr__(self, a):
        return ShiftRight(self, a)


class ArgumentList(BaseNode):

    def __add__(self, lst):
        if isinstance(lst, list):
            self.args.extend(lst)
        elif isinstance(lst, ArgumentList):
            self.args.extend(lst.args)
        else:
            self.args.append(lst)
        return self

    def __str__(self):
        args = []
        for x in self.args:
            if isinstance(x, Variable):
                args.append(x.argument())
            else:
                args.append(str(x))
        return cjoin(args)

    def set_value(self, name, value):
        for i, arg in enumerate(self.args):
            if hasattr(arg, 'name'):
                if arg.name == name:
                    self.args[i] = value

    def callexpr(self):
        args = []
        for x in self.args:
            if isinstance(x, Variable):
                args.append(x.name)
            else:
                args.append(str(x))
        return cjoin(args)


class StatementList(BaseNode):

    def __add__(self, lst):
        if isinstance(lst, list):
            self.args.extend(lst)
        elif isinstance(lst, StatementList):
            self.args.extend(lst.args)
        else:
            self.args.append(lst)
        return self

    def __str__(self):
        return njoin(self.args)

    def __getitem__(self, idx):
        return StatementList() + self.args[idx]

    def __iter__(self):
        idx = 0
        while idx < len(self.args):
            yield self.args[idx]
            idx += 1

    def __len__(self):
        return len(self.args)


@name_args(['name', 'type', 'value'])
class ForwardDeclaration(BaseNode):

    def __str__(self):
        s = f'{self.type} {self.name}'
        if self.value is not None:
            s += f'({self.value});'
        return s


@name_args(['name', 'type', 'value'])
class InlineDeclaration(BaseNode):

    def __str__(self):
        s = f'{self.type} {self.name}'
        if self.value is not None:
            s += f' = {self.value}'
        return s


@name_args(
    ['name', 'type', 'size', 'value', 'shared', 'pointer', 'post_qualifier'])
class Declaration(BaseNode):

    def __str__(self):
        s = ''
        if self.size == 'dynamic':
            s += 'extern '
        if self.shared:
            s += '__shared__ '
        s += f'{self.type}'

        if self.pointer:
            s += f' *'

        if self.post_qualifier is not None:
            s += f' {self.post_qualifier}'

        s += f' {self.name}'

        if self.size is not None:
            if self.size == 'dynamic':
                s += f'[]'
            else:
                s += f'[{self.size}]'
        if self.value is not None:
            s += f' = {self.value}'
        s += ';'
        return s


def Declarations(*args):
    return [x.declaration() for x in args]


class CallbackDeclaration(BaseNode):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.append(Variable('scalar_type', 'typename'))
        self.args.append(Variable('cbtype', 'CallbackType'))

    def __str__(self):
        ret = f'auto load_cb = get_load_cb<{str(self.args[0])},{str(self.args[1])}>(load_cb_fn);'
        ret += f'auto store_cb = get_store_cb<{str(self.args[0])},{str(self.args[1])}>(store_cb_fn);'
        return ret


class TemplateList(ArgumentList):
    pass


class CommentBlock(BaseNode):

    def __str__(self):
        return njoin(['/*'] + [' * ' + str(a) for a in self.args] + [' */'])


class CommentLines(BaseNode):

    def __str__(self):
        return njoin(' // ' + str(a) for a in self.args)


class Pragma(BaseNode):

    def __str__(self):
        return '#pragma ' + sjoin(self.args)


class Include(BaseNode):

    def __str__(self):
        return '#include ' + sjoin(self.args)


class ExternC(BaseNode):

    def __str__(self):
        return 'extern "C" { ' + njoin(self.args) + ' }'


@make_raw(os.linesep + os.linesep)
class LineBreak(BaseNode):
    pass


@make_raw('return;')
class ReturnStatement(BaseNode):
    pass


@make_raw('break;')
class BreakStatement(BaseNode):
    pass


@make_raw('__syncthreads();')
class SyncThreads(BaseNode):
    pass


#
# Operators
#


def make_unary(prefix):

    def decorator(target):
        target.__str__ = lambda self: prefix + str(self.args[0])
        return target

    return decorator


def make_binary(separator):

    def decorator(target):
        target.sep = separator
        return target

    return decorator


@make_unary('&')
class Address(BaseNode):
    pass


@make_unary('-')
class Negate(BaseNode):
    pass


@make_unary('!')
class Not(BaseNode):
    pass


@make_unary('++')
class Increment(BaseNode):
    pass


@make_unary('--')
class Decrement(BaseNode):
    pass


@name_args(['lhs', 'rhs', 'sep'])
class BaseAssign(BaseNode):

    def __str__(self):
        return str(self.args[0]) + str(self.sep) + str(self.args[1]) + ';' \
            + self.provenance()


@name_args(['lhs', 'rhs'])
@make_binary(' = ')
class Assign(BaseAssign):
    pass


@name_args(['lhs', 'cond', 'true_rhs', 'false_rhs'])
class ConditionalAssign(BaseNode):

    def __str__(self):
        return (str(self.lhs) + ' = (' + str(self.cond) + ') ? ' +
                str(self.true_rhs) + ' : ' + str(self.false_rhs) + ';' +
                self.provenance())


@name_args(['cond', 'true_rhs', 'false_rhs'])
class Ternary(BaseNode):

    def __str__(self):
        return f'({str(self.cond)}) ? ({str(self.true_rhs)}) : ({str(self.false_rhs)})'


@name_args(['lhs', 'rhs'])
class InlineAssign(BaseNode):

    def __str__(self):
        return str(self.args[0]) + ' = ' + str(self.args[1])


@name_args(['buf', 'offset'])
class LoadGlobal(BaseNode):

    def __str__(self):
        return f'load_cb({self.args[0]}, {self.args[1]}, load_cb_data, nullptr)'


@name_args(['buf', 'offset', 'element'])
class StoreGlobal(BaseNode):

    def __str__(self):
        return f'store_cb({self.args[0]}, {self.args[1]}, {self.args[2]}, store_cb_data, nullptr);'


@make_binary('&&')
class And(BaseNodeOps):
    pass


@make_binary('||')
class Or(BaseNodeOps):
    pass


@make_binary('.')
class Component(BaseNodeOps):
    pass


@make_binary(' + ')
class Add(BaseNodeOps):
    pass


@make_binary(' - ')
class Sub(BaseNodeOps):
    pass


@make_binary(' / ')
class Divide(BaseNodeOps):
    pass


@make_binary(' * ')
class Multiply(BaseNodeOps):
    pass


class TwiddleMultiply(BaseNode):
    # complex a * b
    def __str__(self):
        a, b = self.args
        return str(ComplexLiteral(a.x * b.x - a.y * b.y,
                                  a.y * b.x + a.x * b.y))


class TwiddleMultiplyConjugate(BaseNode):
    # complex a * conj(b)
    def __str__(self):
        a, b = self.args
        return str(ComplexLiteral(a.x * b.x + a.y * b.y,
                                  a.y * b.x - a.x * b.y))


@make_binary(' % ')
class Mod(BaseNodeOps):
    pass


@name_args(['lhs', 'rhs'])
@make_binary(' += ')
class AddAssign(BaseAssign):
    pass


@name_args(['lhs', 'rhs'])
@make_binary(' -= ')
class SubAssign(BaseAssign):
    pass


@name_args(['lhs', 'rhs'])
@make_binary(' /= ')
class DivideAssign(BaseAssign):
    pass


@name_args(['lhs', 'rhs'])
@make_binary(' *= ')
class MultiplyAssign(BaseAssign):
    pass


@name_args(['lhs', 'rhs'])
@make_binary(' %= ')
class ModAssign(BaseAssign):
    pass


@make_binary(' == ')
class Equal(BaseNodeOps):
    pass


@make_binary(' != ')
class NotEqual(BaseNodeOps):
    pass


@make_binary(' > ')
class Greater(BaseNodeOps):
    pass


@make_binary(' >= ')
class GreaterEqual(BaseNodeOps):
    pass


@make_binary(' < ')
class Less(BaseNodeOps):
    pass


@make_binary(' <= ')
class LessEqual(BaseNodeOps):
    pass


@make_binary(' << ')
class ShiftLeft(BaseNodeOps):
    pass


@make_binary(' >> ')
class ShiftRight(BaseNodeOps):
    pass


#
# Variables
#


@name_args(['variable', 'index'])
class ArrayElement(BaseNodeOps):

    @property
    def x(self):
        return Component(str(self), 'x')

    @property
    def y(self):
        return Component(str(self), 'y')

    def address(self):
        return Address(str(self))

    def __str__(self) -> str:
        return str(self.variable) + '[' + str(self.index) + ']'


@name_args([
    'name', 'type', 'size', 'array', 'restrict', 'value', 'post_qualifier',
    'shared', 'pointer'
])
class Variable(BaseNodeOps):

    @property
    def x(self):
        return Component(self.name, 'x')

    @property
    def y(self):
        return Component(self.name, 'y')

    def address(self):
        return Address(self.name)

    def declaration(self):
        if self.size is not None:
            return Declaration(self.name,
                               self.type,
                               size=self.size,
                               value=self.value,
                               shared=self.shared,
                               pointer=self.pointer,
                               post_qualifier=self.post_qualifier)
        return Declaration(self.name,
                           self.type,
                           value=self.value,
                           shared=self.shared,
                           pointer=self.pointer,
                           post_qualifier=self.post_qualifier)

    def inline(self, value):
        return InlineDeclaration(self.name, self.type, value)

    def argument(self):
        if self.array:
            return f'{self.type} * {self.post_qualifier} {self.name}'
        if self.value is not None:
            return f'{self.type} {self.post_qualifier} {self.name} = {self.value}'
        return f'{self.type} {self.post_qualifier} {self.name}'

    def inline(self, value):
        return InlineDeclaration(self.name, self.type, value)

    def __str__(self):
        return str(self.name)

    def __getitem__(self, idx):
        return ArrayElement(self.name, idx)

    def __post_init__(self):
        if self.post_qualifier is None:
            self.post_qualifier = ''
        if self.restrict:
            self.post_qualifier += ' __restrict__'


@name_args(['name', 'type'])
class Map(BaseNodeOps):

    def address(self):
        return Address(self.name)

    def __str__(self):
        return str(self.name)

    def emplace(self, key, value):
        return Call(self.name + '.emplace', arguments=ArgumentList(key, value))

    def assert_emplace(self, key, value, what_error):
        emplace = Call(self.name + '.emplace',
                       arguments=ArgumentList(key, value)).inline()
        status = Call(name='std::get<1>',
                      arguments=ArgumentList(emplace)).inline()
        throw = StatementList(
            Throw('std::runtime_error("' + str(what_error) + '")'))
        return If(Equal(status, "false"), throw)

    def insert(self, key, value, def_key_pool, function_map, lds_size_bytes):
        return Call('insert_default_entry',
                    arguments=ArgumentList(key, value, def_key_pool,
                                           function_map, lds_size_bytes))

    def insert_pp(self, key, value_1, value_2, def_key_pool, function_map,
                  lds_size_bytes):
        return Call('insert_default_entry',
                    arguments=ArgumentList(key, value_1, value_2, def_key_pool,
                                           function_map, lds_size_bytes))

    # def __getitem__(self, idx):
    #     return ArrayElement(self.name, idx)


class ComplexLiteral(BaseNodeOps):

    def __str__(self):
        return '{' + str(self.args[0]) + ', ' + str(self.args[1]) + '}'


class Group(BaseNodeOps):

    def __str__(self):
        return '(' + str(self.args[0]) + ')'


B = Group

#
# Control flow
#


@name_args(['value'])
class Throw(BaseNode):

    def __str__(self):
        return 'throw ' + str(self.value) + ';'


class Block(BaseNode):

    def __str__(self):
        return '{' + njoin(self.args) + '}'


@name_args(['condition', 'body'])
class If(BaseNode):

    def __str__(self):
        return 'if(' + str(self.condition) + ') {' + njoin(self.body) + '}'


@name_args(['condition', 'bodyif', 'bodyelse'])
class IfElse(BaseNode):

    def __str__(self):
        return 'if(' + str(self.condition) + ') {' + njoin(
            self.bodyif) + '} else {' + njoin(self.bodyelse) + '}'


@name_args(['condition', 'body'])
class While(BaseNode):

    def __str__(self):
        return 'while(' + str(self.condition) + ') {' + njoin(self.body) + '}'


@name_args(['initial', 'condition', 'iteration', 'body'])
class For(BaseNode):

    def __str__(self):
        return 'for(' + join('; ',
                             [self.initial, self.condition, self.iteration
                              ]) + ') {' + njoin(self.body) + '}'


#
# Functions
#


@name_args(['name', 'spec'])
class Using(BaseNode):

    def __str__(self):
        return f'using {self.name} = {self.spec};'


@name_args([
    'name', 'value', 'arguments', 'templates', 'qualifier', 'launch_bounds',
    'body', 'meta'
])
class Function(BaseNode):

    def __str__(self) -> str:
        f = self.provenance() + os.linesep
        if self.templates:
            f += 'template<' + str(self.templates) + '>'
        if self.qualifier is not None:
            f += self.qualifier + ' '
        if self.launch_bounds is not None:
            f += '__launch_bounds__(' + str(self.launch_bounds) + ') '
        if self.value is None:
            f += ' void '
        elif self.value:
            f += ' ' + str(self.value) + ' '
        f += self.name
        f += '(' + str(self.arguments) + ')'
        f += '{' + njoin(self.body) + '}'
        return f

    def address(self):
        return Address(self.name)

    def instantiate(self, name, *targs):
        return Using(name, self.name + '<' + cjoin(*targs) + '>')

    def call(self, arguments, templates=None):
        return Call(name=self.name, arguments=arguments, templates=templates)


@name_args(['name', 'arguments', 'templates', 'launch_params'])
class Call(BaseNode):

    def __str__(self) -> str:
        f = self.name
        if self.templates:
            f += '<' + self.templates.callexpr() + '>'
        if self.launch_params:
            f += '<<<' + self.launch_params.callexpr() + '>>>'
        f += '(' + self.arguments.callexpr() + ');'
        f += self.provenance()
        return f

    def inline(self):
        return InlineCall(*self.args)


@name_args(['name', 'arguments', 'templates', 'launch_params'])
class InlineCall(BaseNodeOps):

    def __str__(self) -> str:
        f = self.name
        if self.templates:
            f += '<' + self.templates.callexpr() + '>'
        f += '(' + self.arguments.callexpr() + ')'
        return f


#
# Re-writing helpers
#


def make_planar(kernel, varname):
    """Rewrite 'kernel' to use planar i/o instead of interleaved i/o.

    The interleaved array 'varname' is replaced with planar arrays.
    We assume that, in the body of the kernel, the i/o array is only
    used in assignments (that is, typically loaded to and from LDS).

    For example, suppose we want to make the 'inout' array planar.
    Assignments like

       lds[idx] = inout[idx];

    become

       lds[idx] = { inoutre[idx], inoutim[idx] };

    Assignments like

       inout[idx] = lds[idx];

    become

       inoutre[idx] = lds[idx].x;
       inoutim[idx] = lds[idx].y;

    Finally, argument lists like:

       device_kernel(scalar_type *inout);

    become

       device_kernel(real_type_t<scalar_type> *inoutre, real_type_t<scalar_type> *inoutim);

    """

    rname = varname + 're'
    iname = varname + 'im'

    def visitor(x):
        if isinstance(x, BaseAssign):
            lhs, rhs = x.lhs, x.rhs

            # on rhs
            if isinstance(rhs, ArrayElement):
                name, index = rhs.args
                if name == varname:
                    return Assign(
                        lhs,
                        ComplexLiteral(ArrayElement(rname, index),
                                       ArrayElement(iname, index)))

            # on lhs
            if isinstance(lhs, ArrayElement):
                name, index = lhs.args
                if name == varname:
                    return StatementList(
                        Assign(ArrayElement(rname, index), Component(rhs,
                                                                     'x')),
                        Assign(ArrayElement(iname, index), Component(rhs,
                                                                     'y')))

        if isinstance(x, ArgumentList):
            args = []
            for arg in x.args:
                if isinstance(arg, Variable):
                    if arg.name == varname:
                        real_type = f'real_type_t<{arg.type}>'
                        args.append(
                            Variable(rname,
                                     type=real_type,
                                     array=True,
                                     restrict=True))
                        args.append(
                            Variable(iname,
                                     type=real_type,
                                     array=True,
                                     restrict=True))
                    else:
                        args.append(arg)
                else:
                    args.append(arg)
            return ArgumentList(*args)

        # callbacks don't support planar, so loads and stores are
        # instead just direct memory accesses
        if isinstance(x, LoadGlobal):
            if x.args[0].name == varname:
                return ArrayElement(x.args[0], x.args[1])

        if isinstance(x, StoreGlobal):
            if x.args[0].name == varname:
                return StatementList(
                    Assign(ArrayElement(rname, x.args[1]),
                           Component(x.args[2], 'x')),
                    Assign(ArrayElement(iname, x.args[1]),
                           Component(x.args[2], 'y')))

        return x

    return depth_first(kernel, visitor)


def make_out_of_place(kernel, names):
    """Rewrite 'kernel' to use separate input and output buffers.

    The input/output array 'varname' is replaced with separate input
    and output arrays 'inname' and 'outname'.  We assume that, in the
    body of the kernel, the i/o array is only used in assignments
    (that is, typically loaded to and from LDS).

    For example, suppose we want to make the in-place 'inout' array
    into out-of-place arrays.  Assignments like

       lds[idx] = inout[idx];

    become

       lds[idx] = in[idx];

    Assignments like

       inout[idx] = lds[idx];

    become

       out[idx] = lds[idx];

    Finally, argument lists like:

       device_kernel(scalar_type *inout);

    become

       device_kernel(scalar_type *in, scalar_type *out);

    """

    def input_visitor(x):
        if isinstance(x, (Variable, ArrayElement)):
            name = x.args[0]
            if name in names:
                y = copy(x)
                y.args[0] = name + '_in'
                return y

        return x

    def output_visitor(x):
        if isinstance(x, (Variable, ArrayElement, StoreGlobal)):
            name = x.args[0]
            if name in names:
                y = copy(x)
                y.args[0] = name + '_out'
                return y
        return x

    def duplicate_visitor(x):
        if str(getattr(x, 'name', None)) in names:
            xi, xo = copy(x), copy(x)
            xi.args[0] = x.name + '_in'
            xo.args[0] = x.name + '_out'
            if xi.value is not None:
                xi.args[3] = depth_first(xi.args[3], input_visitor)
            if xo.value is not None:
                xo.args[3] = depth_first(xo.args[3], output_visitor)
            return StatementList(xi, xo)
        return x

    def visitor(x):
        if isinstance(x, Declaration):
            return duplicate_visitor(x)

        if isinstance(x, BaseAssign):
            lhs, rhs = x.lhs, x.rhs

            # on lhs, plain variable
            if isinstance(lhs, Variable):
                if lhs.name in names:
                    return StatementList(
                        Assign(input_visitor(lhs),
                               depth_first(rhs, input_visitor)),
                        Assign(output_visitor(lhs),
                               depth_first(rhs, output_visitor)))

            # traverse rhs
            if isinstance(rhs, ArrayElement):
                if rhs.variable in names:
                    nrhs = depth_first(rhs, input_visitor)
                    nrhs.args[0] = rhs.variable + '_in'
                    return Assign(lhs, nrhs)

            # traverse lhs
            if isinstance(lhs, ArrayElement):
                if lhs.variable in names:
                    nlhs = depth_first(lhs, output_visitor)
                    nlhs.args[0] = lhs.variable + '_out'
                    return Assign(nlhs, rhs)

        if isinstance(x, ArgumentList):
            args = []
            for arg in x.args:
                if isinstance(arg, (Variable, ArrayElement)):
                    name = arg.args[0]
                    if name in names:
                        ai, ao = copy(arg), copy(arg)
                        ai.args[0] = name + '_in'
                        ao.args[0] = name + '_out'
                        args.extend([ai, ao])
                    else:
                        args.append(arg)
                else:
                    args.append(arg)
            return ArgumentList(*args)

        if isinstance(x, LoadGlobal):
            x.args[0] = depth_first(x.args[0], input_visitor)
            x.args[1] = depth_first(x.args[1], input_visitor)

        if isinstance(x, StoreGlobal):
            x.args[0] = depth_first(x.args[0], output_visitor)
            x.args[1] = depth_first(x.args[1], output_visitor)

        return x

    return depth_first(kernel, visitor)


def make_inverse(kernel):
    """Rewrite forward 'kernel' to be an inverse kernel.

    Forward butterfly calls like this

       FwdRadX(...)

    are re-written to backward butterfly calls like this

       InvRadX(...);

    and instances of TwiddleMultiply are changed to TwiddleMultiplyConjugate.
    """

    kernel = rename_functions(kernel,
                              lambda x: x.replace('forward', 'inverse'))
    kernel = rename_functions(kernel, lambda x: x.replace('FwdRad', 'InvRad'))

    def visitor(x):
        if isinstance(x, TwiddleMultiply):
            y = TwiddleMultiplyConjugate()
            y.args = x.args
            return y
        return x

    return depth_first(kernel, visitor)


def rename_functions(kernel, sub):
    """Rename..."""

    def visitor(x):
        if isinstance(x, (Function, Call)):
            y = copy(x)
            y.args[0] = sub(x.args[0])
            return y
        return x

    return depth_first(kernel, visitor)


def make_rtc(kernel, specs):
    """Turn a global function into a runtime-compile-able function.
    """

    real_type = specs['real_type']
    stridebin = specs['stridebin']
    apply_large_twiddle = specs['apply_large_twiddle']
    large_twiddle_base = specs['large_twiddle_base']
    ebtype = specs['ebtype']
    cbtype = specs['cbtype']

    complex_type = real_type + '2'

    def visitor(x):
        if isinstance(x, Function):
            y = copy(x)
            # give it "C" linkage so we don't need C++ name mangling
            y.qualifier = 'extern "C" __global__'
            y.args[0] = specs['kernel_name']
            # de-templatize
            y.templates = None
            return y
        elif isinstance(x, Variable):
            # change templated variables to concrete types

            # scalar type variables + template params
            if x.args[1] is not None and 'scalar_type' in x.args[1]:
                y = copy(x)
                y.args[1] = x.args[1].replace('scalar_type', complex_type)
                return y
            # scalar type template params
            elif x.args[0] == 'scalar_type':
                y = copy(x)
                y.args[0] = complex_type
                return y
            # other template params
            elif x.args[0] == 'sb':
                y = copy(x)
                y.args[0] = stridebin
                return y
            elif x.args[0] == 'apply_large_twiddle':
                y = copy(x)
                y.args[0] = 'true' if apply_large_twiddle else 'false'
                return y
            elif x.args[0] == 'large_twiddle_base':
                y = copy(x)
                y.args[0] = large_twiddle_base
                return y
            elif x.args[0] == 'ebtype':
                y = copy(x)
                y.args[0] = ebtype
                return y
            elif x.args[0] == 'cbtype':
                y = copy(x)
                y.args[0] = cbtype
                return y
        # declarations
        elif isinstance(x, str):
            if 'scalar_type' in x:
                return x.replace('scalar_type', complex_type)

        return x

    return depth_first(kernel, visitor)
