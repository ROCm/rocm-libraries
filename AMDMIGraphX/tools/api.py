#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#####################################################################################
import string
import sys
import re
import runpy
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

type_map: Dict[str, Callable[['Parameter'], None]] = {}
cpp_type_map: Dict[str, str] = {}
functions: List['Function'] = []
cpp_classes: List['CPPClass'] = []
error_type = ''
success_type = ''
try_wrap = ''

export_c_macro = 'MIGRAPHX_C_EXPORT'

c_header_preamble: List[str] = []
c_api_body_preamble: List[str] = []
cpp_header_preamble: List[str] = []


def bad_param_error(msg: str):
    return 'throw std::runtime_error("{}")'.format(msg)


class Template(string.Template):
    idpattern = '[_a-zA-Z0-9@]+'


class Type:
    def __init__(self, name: str) -> None:
        self.name = name.strip()

    def is_pointer(self) -> bool:
        return self.name.endswith('*')

    def is_reference(self) -> bool:
        return self.name.endswith('&')

    def is_const(self) -> bool:
        return self.name.startswith('const ')

    def is_variadic(self):
        return self.name.startswith('...')

    def add_pointer(self) -> 'Type':
        return Type(self.name + '*')

    def add_reference(self):
        return Type(self.name + '&')

    def add_const(self) -> 'Type':
        return Type('const ' + self.name)

    def inner_type(self) -> Optional['Type']:
        i = self.name.find('<')
        j = self.name.rfind('>')
        if i > 0 and j > 0:
            return Type(self.name[i + 1:j])
        else:
            return None

    def remove_generic(self) -> 'Type':
        i = self.name.find('<')
        j = self.name.rfind('>')
        if i > 0 and j > 0:
            return Type(self.name[0:i] + self.name[j + 1:])
        else:
            return self

    def remove_pointer(self) -> 'Type':
        if self.is_pointer():
            return Type(self.name[0:-1])
        return self

    def remove_reference(self) -> 'Type':
        if self.is_reference():
            return Type(self.name[0:-1])
        return self

    def remove_const(self) -> 'Type':
        if self.is_const():
            return Type(self.name[6:])
        return self

    def basic(self) -> 'Type':
        return self.remove_pointer().remove_const().remove_reference()

    def decay(self) -> 'Type':
        t = self.remove_reference()
        if t.is_pointer():
            return t
        else:
            return t.remove_const()

    def const_compatible(self, t: 'Type'):
        if t.is_const():
            return self.add_const()
        return self

    def str(self) -> str:
        return self.name


header_function = Template('''
${export_c_macro} ${error_type} ${name}(${params});
''')

function_pointer_typedef = Template('''
typedef ${error_type} (*${fname})(${params});
''')

c_api_impl = Template('''
extern "C" ${error_type} ${name}(${params})
{
    ${va_start}auto api_error_result = ${try_wrap}([&] {
        ${body};
    });
    ${va_end}return api_error_result;
}
''')


class CFunction:
    def __init__(self, name: str) -> None:
        self.name = name
        self.params: List[str] = []
        self.body: List[str] = []
        self.va_start: List[str] = []
        self.va_end: List[str] = []

    def add_param(self, type: str, pname: str) -> None:
        self.params.append('{} {}'.format(type, pname))

    def add_statement(self, stmt: str) -> None:
        self.body.append(stmt)

    def add_vlist(self, name: str) -> None:
        last_param = self.params[-1].split()[-1]
        self.va_start = [
            'va_list {};'.format(name),
            'va_start({}, {});'.format(name, last_param)
        ]
        self.va_end = ['va_end({});'.format(name)]
        self.add_param('...', '')

    def substitute(self, form: Template, **kwargs) -> str:
        return form.substitute(error_type=error_type,
                               try_wrap=try_wrap,
                               name=self.name,
                               params=', '.join(self.params),
                               body=";\n        ".join(self.body),
                               va_start="\n    ".join(self.va_start),
                               va_end="\n    ".join(self.va_end),
                               **kwargs)

    def generate_header(self) -> str:
        return self.substitute(header_function, export_c_macro=export_c_macro)

    def generate_function_pointer(self, name: Optional[str] = None) -> str:
        return self.substitute(function_pointer_typedef,
                               fname=name or self.name)

    def generate_body(self) -> str:
        return self.substitute(c_api_impl)


class BadParam:
    def __init__(self, cond: str, msg: str) -> None:
        self.cond = cond
        self.msg = msg


class Parameter:
    def __init__(self,
                 name: str,
                 type: str,
                 optional: bool = False,
                 returns: bool = False,
                 virtual: bool = False,
                 this: bool = False,
                 hidden: bool = False) -> None:
        self.name = name
        self.type = Type(type)
        self.optional = optional
        self.cparams: List[Tuple[str, str]] = []
        self.size_cparam = -1
        self.size_name = ''
        self.read = '${name}'
        self.write = ['*${name} = ${result}']
        self.cpp_read = '${name}'
        self.cpp_write = '${name}'
        self.returns = returns
        self.virtual = virtual
        self.this = this
        self.hidden = hidden
        self.bad_param_check: Optional[BadParam] = None
        self.virtual_read: Optional[List[str]] = None
        self.virtual_write: Optional[str] = None

    def get_name(self, prefix: Optional[str] = None) -> str:
        if prefix:
            return prefix + self.name
        else:
            return self.name

    def get_cpp_type(self) -> str:
        if self.type.str() in cpp_type_map:
            return cpp_type_map[self.type.basic().str()]
        elif self.type.basic().str() in cpp_type_map:
            return cpp_type_map[self.type.basic().str()]
        elif self.returns:
            return self.type.decay().str()
        else:
            return self.type.str()

    def substitute(self,
                   s: str,
                   prefix: Optional[str] = None,
                   result: Optional[str] = None) -> str:
        ctype = None
        if len(self.cparams) > 0:
            ctype = Type(self.cparams[0][0]).basic().str()
        return Template(s).safe_substitute(name=self.get_name(prefix),
                                           type=self.type.str(),
                                           ctype=ctype or '',
                                           cpptype=self.get_cpp_type(),
                                           size=self.size_name,
                                           result=result or '')

    def add_param(self, t: Union[str, Type],
                  name: Optional[str] = None) -> None:
        if not isinstance(t, str):
            t = t.str()
        self.cparams.append((t, name or self.name))

    def add_size_param(self, name: Optional[str] = None) -> None:
        self.size_cparam = len(self.cparams)
        self.size_name = name or self.name + '_size'
        if self.returns:
            self.add_param('size_t *', self.size_name)
        else:
            self.add_param('size_t', self.size_name)

    def bad_param(self, cond: str, msg: str) -> None:
        self.bad_param_check = BadParam(cond, msg)

    def remove_size_param(self, name):
        p = None
        if self.size_cparam >= 0:
            p = self.cparams[self.size_cparam]
            del self.cparams[self.size_cparam]
            self.size_name = name
        return p

    def update(self) -> None:
        t = self.type.basic().str()
        g = self.type.remove_generic().basic().str()
        if t in type_map:
            type_map[t](self)
        elif g in type_map:
            type_map[g](self)
        else:
            if self.returns:
                self.add_param(self.type.remove_reference().add_pointer())
            else:
                self.add_param(self.type.remove_reference())
        if isinstance(self.write, str):
            raise ValueError("Error for {}: write cannot be a string".format(
                self.type.str()))

    def virtual_arg(self, prefix: Optional[str] = None) -> List[str]:
        read = self.virtual_read
        if not read and len(self.write) >= len(self.cparams):
            read = [
                Template(w.partition('=')[2]).safe_substitute(result='${name}')
                for w in self.write
            ]
        if not read:
            raise ValueError("No virtual_read parameter provided for: " +
                             self.type.str())
        if isinstance(read, str):
            raise ValueError(
                "Error for {}: virtual_read cannot be a string".format(
                    self.type.str()))
        return [self.substitute(r, prefix=prefix) for r in read]

    def virtual_param(self, prefix: Optional[str] = None) -> str:
        return self.substitute('${type} ${name}', prefix=prefix)

    def virtual_output_args(self, prefix: Optional[str] = None) -> List[str]:
        container_type = self.type.remove_generic().basic().str()
        decl_list: List[str] = []
        container = (container_type == "std::vector"
                     or container_type == "vector")
        for t, n, in self.cparams:
            if not decl_list and container:
                decl_list.append('{prefix}{n}.data()'.format(prefix=prefix
                                                             or '',
                                                             n=n))
            else:
                decl_list.append('&{prefix}{n}'.format(prefix=prefix or '',
                                                       n=n))
        return decl_list

    def virtual_output_declarations(self,
                                    prefix: Optional[str] = None) -> List[str]:
        container_type = self.type.remove_generic().basic().str()
        container = (container_type == "std::vector"
                     or container_type == "vector")
        decl_list: List[str] = []
        for t, n, in self.cparams:
            if not decl_list and container:
                inner_t = self.type.inner_type()
                if inner_t:
                    decl_list.append(
                        'std::array<{inner_t}, 1024> {prefix}{n};'.format(
                            inner_t=inner_t.str(), prefix=prefix or '', n=n))
            else:
                decl_list.append(
                    'std::remove_pointer_t<{type}> {prefix}{n}'.format(
                        type=Type(t).str(), prefix=prefix or '', n=n))
                decl_list[-1] += '=1024;' if container else ';'
        return decl_list

    def virtual_output(self, prefix: Optional[str] = None) -> str:
        write = self.virtual_write
        if not write:
            if '*' in self.read or '->' in self.read:
                write = Template(self.read).safe_substitute(name='(&${name})')
            else:
                write = self.read
        return self.substitute(write, prefix=prefix)

    def cpp_param(self, prefix: Optional[str] = None) -> str:
        return self.substitute('${cpptype} ${name}', prefix=prefix)

    def cpp_arg(self, prefix: Optional[str] = None) -> str:
        return self.substitute(self.cpp_read, prefix=prefix)

    def cpp_output_args(self, prefix: Optional[str] = None) -> List[str]:
        return [
            '&{prefix}{n}'.format(prefix=prefix, n=n) for t, n in self.cparams
        ]

    def output_declarations(self, prefix: Optional[str] = None) -> List[str]:
        return [
            '{type} {prefix}{n};'.format(type=Type(t).remove_pointer().str(),
                                         prefix=prefix,
                                         n=n) for t, n in self.cparams
        ]

    def output_args(self, prefix=None):
        return [
            '&{prefix}{n};'.format(prefix=prefix, n=n) for t, n in self.cparams
        ]

    def cpp_output(self, prefix: Optional[str] = None) -> str:
        return self.substitute(self.cpp_write, prefix=prefix)

    def input(self, prefix: Optional[str] = None) -> str:
        return '(' + self.substitute(self.read, prefix=prefix) + ')'

    def outputs(self, result: Optional[str] = None) -> List[str]:
        return [self.substitute(w, result=result) for w in self.write]

    def add_to_cfunction(self, cfunction: CFunction) -> None:
        for t, name in self.cparams:
            if t.startswith('...'):
                cfunction.add_vlist(name)
            else:
                cfunction.add_param(self.substitute(t), self.substitute(name))
        if self.bad_param_check:
            msg = 'Bad parameter {name}: {msg}'.format(
                name=self.name, msg=self.bad_param_check.msg)
            cfunction.add_statement('if ({cond}) {body}'.format(
                cond=self.substitute(self.bad_param_check.cond),
                body=bad_param_error(msg)))


def template_var(s: str) -> str:
    return '${' + s + '}'


def to_template_vars(params: List[Union[Any, Parameter]]) -> str:
    return ', '.join([template_var(p.name) for p in params])


class Function:
    def __init__(self,
                 name: str,
                 params: Optional[List[Parameter]] = None,
                 shared_size: bool = False,
                 returns: Optional[str] = None,
                 invoke: Optional[str] = None,
                 fname: Optional[str] = None,
                 return_name: Optional[str] = None,
                 virtual: bool = False,
                 **kwargs) -> None:
        self.name = name
        self.params = params or []
        self.shared_size = False
        self.cfunction: Optional[CFunction] = None
        self.fname = fname
        self.invoke = invoke or '${__fname__}($@)'
        self.return_name = return_name or 'out'
        self.returns = Parameter(self.return_name, returns,
                                 returns=True) if returns else None
        for p in self.params:
            p.virtual = virtual
        if self.returns:
            self.returns.virtual = virtual

    def share_params(self) -> None:
        if self.shared_size == True:
            size_param_name = 'size'
            size_type = Type('size_t')
            for param in self.params:
                p = param.remove_size_param(size_param_name)
                if p:
                    size_type = Type(p[0])
            self.params.append(Parameter(size_param_name, size_type.str()))

    def update(self) -> None:
        self.share_params()
        for param in self.params:
            param.update()
        if self.returns:
            self.returns.update()
        self.create_cfunction()

    def inputs(self) -> str:
        return ', '.join([p.input() for p in self.params])

    # TODO: Shoule we remove Optional?
    def input_map(self) -> Dict[str, Optional[str]]:
        m: Dict[str, Optional[str]] = {}
        for p in self.params:
            m[p.name] = p.input()
        m['return'] = self.return_name
        m['@'] = self.inputs()
        m['__fname__'] = self.fname
        return m

    def get_invoke(self) -> str:
        return Template(self.invoke).safe_substitute(self.input_map())

    def write_to_tmp_var(self) -> bool:
        if not self.returns:
            return False
        return len(self.returns.write) > 1 or self.returns.write[0].count(
            '${result}') > 1

    def get_cfunction(self) -> CFunction:
        if self.cfunction:
            return self.cfunction
        raise Exception(
            "self.cfunction is None: self.update() needs to be called.")

    def create_cfunction(self) -> None:
        self.cfunction = CFunction(self.name)
        # Add the return as a parameter
        if self.returns:
            self.returns.add_to_cfunction(self.cfunction)
        # Add the input parameters
        for param in self.params:
            param.add_to_cfunction(self.cfunction)
        f: Optional[str] = self.get_invoke()
        # Write the assignments
        assigns = []
        if self.returns:
            result = f
            if self.write_to_tmp_var() and f:
                f = 'auto&& api_result = ' + f
                result = 'api_result'
            else:
                f = None
            assigns = self.returns.outputs(result)
        if f:
            self.cfunction.add_statement(f)
        for assign in assigns:
            self.cfunction.add_statement(assign)


cpp_class_template = Template('''

struct ${name} : handle_base<${ctype}, decltype(&${destroy}), ${destroy}>
{
    ${name}(${ctype} p, bool own = true)
    : m_handle(nullptr)
    {
        this->set_handle(p, own);
    }
    ${constructors}

    ${methods}
};
''')

cpp_class_method_template = Template('''
    ${return_type} ${name}(${params}) const
    {
        ${outputs}
        this->call_handle(${args});
        return ${result};
    }
''')

cpp_class_void_method_template = Template('''
    void ${name}(${params}) const
    {
        this->call_handle(${args});
    }
''')

cpp_class_constructor_template = Template('''
    ${name}(${params})
    : m_handle(nullptr)
    {
        m_handle = this->make_handle(${args});
    }
''')


class CPPMember:
    def __init__(self,
                 name: str,
                 function: Function,
                 prefix: str,
                 method: bool = True) -> None:
        self.name = name
        self.function = function
        self.prefix = prefix
        self.method = method

    def get_function_params(self) -> List[Union[Any, Parameter]]:
        if self.method:
            return self.function.params[1:]
        else:
            return self.function.params

    def get_args(self) -> str:
        output_args = []
        if self.function.returns:
            output_args = self.function.returns.cpp_output_args(self.prefix)
        if not self.function.cfunction:
            raise Exception('self.function.update() must be called')
        return ', '.join(
            ['&{}'.format(self.function.cfunction.name)] + output_args +
            [p.cpp_arg(self.prefix) for p in self.get_function_params()])

    def get_params(self) -> str:
        return ', '.join(
            [p.cpp_param(self.prefix) for p in self.get_function_params()])

    def get_return_declarations(self) -> str:
        if self.function.returns:
            return '\n        '.join([
                d
                for d in self.function.returns.output_declarations(self.prefix)
            ])
        else:
            return ''

    def get_result(self):
        return self.function.returns.input(self.prefix)

    def generate_method(self) -> str:
        if not self.function.cfunction:
            raise Exception('self.function.update() must be called')
        if self.function.returns:
            return_type = self.function.returns.get_cpp_type()
            return cpp_class_method_template.safe_substitute(
                return_type=return_type,
                name=self.name,
                cfunction=self.function.cfunction.name,
                result=self.function.returns.cpp_output(self.prefix),
                params=self.get_params(),
                outputs=self.get_return_declarations(),
                args=self.get_args(),
                success=success_type)
        else:
            return cpp_class_void_method_template.safe_substitute(
                name=self.name,
                cfunction=self.function.cfunction.name,
                params=self.get_params(),
                args=self.get_args(),
                success=success_type)

    def generate_constructor(self, name: str) -> str:
        if not self.function.cfunction:
            raise Exception('self.function.update() must be called')
        return cpp_class_constructor_template.safe_substitute(
            name=name,
            cfunction=self.function.cfunction.name,
            params=self.get_params(),
            args=self.get_args(),
            success=success_type)


class CPPClass:
    def __init__(self, name: str, ctype: str) -> None:
        self.name = name
        self.ctype = ctype
        self.constructors: List[CPPMember] = []
        self.methods: List[CPPMember] = []
        self.prefix = 'p'

    def add_method(self, name: str, f: Function) -> None:
        self.methods.append(CPPMember(name, f, self.prefix, method=True))

    def add_constructor(self, name: str, f: Function) -> None:
        self.constructors.append(CPPMember(name, f, self.prefix, method=True))

    def generate_methods(self) -> str:
        return '\n    '.join([m.generate_method() for m in self.methods])

    def generate_constructors(self) -> str:
        return '\n    '.join(
            [m.generate_constructor(self.name) for m in self.constructors])

    def substitute(self, s: Union[string.Template, str], **kwargs) -> str:
        t = string.Template(s) if isinstance(s, str) else s
        destroy = self.ctype + '_destroy'
        return t.safe_substitute(name=self.name,
                                 ctype=self.ctype,
                                 destroy=destroy,
                                 **kwargs)

    def generate(self) -> str:
        return self.substitute(
            cpp_class_template,
            constructors=self.substitute(self.generate_constructors()),
            methods=self.substitute(self.generate_methods()))


def params(virtual: Optional[Dict[str, str]] = None,
           **kwargs) -> List[Parameter]:
    result = []
    v: Dict[str, str] = virtual or {}
    for name in v:
        result.append(Parameter(name, v[name]))
    for name in kwargs:
        result.append(Parameter(name, kwargs[name]))
    return result


gparams = params


def add_function(name: str, *args, **kwargs) -> Function:
    f = Function(name, *args, **kwargs)
    functions.append(f)
    return f


def once(f: Callable) -> Any:
    @wraps(f)
    def decorated(*args, **kwargs):
        if not decorated.has_run:
            decorated.has_run = True
            return f(*args, **kwargs)

    d: Any = decorated
    d.has_run = False
    return d


@once
def process_functions() -> None:
    for f in functions:
        f.update()


def generate_lines(p: List[str]) -> str:
    return '\n'.join(p)


def generate_c_header() -> str:
    process_functions()
    return generate_lines(
        c_header_preamble +
        [f.get_cfunction().generate_header() for f in functions])


def generate_c_api_body() -> str:
    process_functions()
    return generate_lines(
        c_api_body_preamble +
        [f.get_cfunction().generate_body() for f in functions])


def generate_cpp_header() -> str:
    process_functions()
    return generate_lines(cpp_header_preamble +
                          [c.generate() for c in cpp_classes])


c_type_map: Dict[str, Type] = {}


def cwrap(name: str, c_type: Optional[str] = None) -> Callable:
    def with_cwrap(f):
        type_map[name] = f
        if c_type:
            c_type_map[name] = Type(c_type)

        @wraps(f)
        def decorated(*args, **kwargs):
            return f(*args, **kwargs)

        return decorated

    return with_cwrap


handle_typedef = Template('''
typedef struct ${ctype} * ${ctype}_t;
typedef const struct ${ctype} * const_${ctype}_t;
''')

handle_definition = Template('''
extern "C" struct ${ctype};
struct ${ctype} {
    template<class... Ts>
    ${ctype}(Ts&&... xs)
    : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {}
    ${cpptype} object;
};
''')

handle_preamble = '''
template<class T, class U, class Target=std::remove_pointer_t<T>>
static Target* object_cast(U* x)
{
    return reinterpret_cast<Target*>(x);
}
template<class T, class U, class Target=std::remove_pointer_t<T>>
static const Target* object_cast(const U* x)
{
    return reinterpret_cast<const Target*>(x);
}

template <class T, class... Ts, class Target = std::remove_pointer_t<T>>
static Target* allocate(Ts&&... xs)
{
    if constexpr(std::is_aggregate<Target>{})
        return new Target{std::forward<Ts>(xs)...}; // NOLINT
    else
        return new Target(std::forward<Ts>(xs)...); // NOLINT
}

template<class T>
static void destroy(T* x)
{
    delete x; // NOLINT
}


// TODO: Move to interface preamble
template <class C, class D>
struct manage_generic_ptr
{
    manage_generic_ptr() = default;
    
    manage_generic_ptr(std::nullptr_t)
    {
    }

    manage_generic_ptr(void* pdata, const char* obj_tname, C pcopier, D pdeleter)
        : data(nullptr), obj_typename(obj_tname), copier(pcopier), deleter(pdeleter)
    {
        copier(&data, pdata);
    }

    manage_generic_ptr(const manage_generic_ptr& rhs)
        : data(nullptr), obj_typename(rhs.obj_typename), copier(rhs.copier), deleter(rhs.deleter)
    {
        if(copier)
            copier(&data, rhs.data);
    }

    manage_generic_ptr(manage_generic_ptr&& other) noexcept
        : data(other.data), obj_typename(other.obj_typename), copier(other.copier), deleter(other.deleter)
    {
        other.data    = nullptr;
        other.obj_typename = "";
        other.copier  = nullptr;
        other.deleter = nullptr;
    }

    manage_generic_ptr& operator=(manage_generic_ptr rhs)
    {
        std::swap(data, rhs.data);
        std::swap(obj_typename, rhs.obj_typename);
        std::swap(copier, rhs.copier);
        std::swap(deleter, rhs.deleter);
        return *this;
    }

    ~manage_generic_ptr()
    {
        if(data != nullptr)
            deleter(data);
    }

    void* data = nullptr;
    const char* obj_typename = "";
    C copier   = nullptr;
    D deleter  = nullptr;
};
'''

cpp_handle_preamble = '''
template<class T, class Deleter, Deleter deleter>
struct handle_base
{

    template<class F, class... Ts>
    void make_handle(F f, Ts&&... xs)
    {
        T* result = nullptr;
        auto e = F(&result, std::forward<Ts>(xs)...);
        if (e != ${success})
            throw std::runtime_error("Failed to call function");
        set_handle(result);
    }

    template<class F, class... Ts>
    void call_handle(F f, Ts&&... xs)
    {
        auto e = F(this->get_handle_ptr(), std::forward<Ts>(xs)...);
        if (e != ${success})
            throw std::runtime_error("Failed to call function");
    }

    const std::shared_ptr<T>& get_handle() const
    {
        return m_handle;
    }

    T* get_handle_ptr() const
    {
        assert(m_handle != nullptr);
        return get_handle().get();
    }

    void set_handle(T* ptr, bool own = true)
    {
        if (own)
            m_handle = std::shared_ptr<T>{ptr, deleter};
        else
            m_handle = std::shared_ptr<T>{ptr, [](T*) {}};
    }

protected:
    std::shared_ptr<T> m_handle;
};

'''


@once
def add_handle_preamble() -> None:
    c_api_body_preamble.append(handle_preamble)
    cpp_header_preamble.append(
        string.Template(cpp_handle_preamble).substitute(success=success_type))


def add_handle(name: str,
               ctype: str,
               cpptype: str,
               destroy: Optional[str] = None,
               ref=False,
               skip_def=False) -> None:
    opaque_type = ctype + '_t'
    const_opaque_type = 'const_' + opaque_type

    def handle_wrap(p: Parameter):
        t = Type(opaque_type)
        if p.type.is_const():
            t = Type('const_' + opaque_type)
        # p.read = 'object_cast<${ctype}>(&(${name}))'
        if p.virtual:
            p.add_param(t)
        elif p.returns:
            p.add_param(t.add_pointer())
        else:
            p.add_param(t)
            p.bad_param('${name} == nullptr', 'Null pointer')
        if p.type.is_reference():
            p.virtual_read = ['object_cast<${ctype}>(&(${name}))']
            p.cpp_write = '${cpptype}(${name}, false)'
            p.write = ['*${name} = object_cast<${ctype}>(&(${result}))']
        elif p.type.is_pointer():
            p.virtual_read = ['object_cast<${ctype}>(${result})']
            p.cpp_write = '${cpptype}(${name}, false)'
            p.write = ['*${name} = object_cast<${ctype}>(${result})']
        else:
            p.virtual_read = ['object_cast<${ctype}>(&(${name}))']
            p.cpp_write = '${cpptype}(${name})'
            p.write = ['*${name} = allocate<${ctype}>(${result})']
        if skip_def:
            p.read = '*${name}'
        else:
            p.read = '${name}->object'
        p.cpp_read = '${name}.get_handle_ptr()'

    type_map[cpptype] = handle_wrap
    if not ref:
        add_function(destroy or ctype + '_' + 'destroy',
                     params({name: opaque_type}),
                     fname='destroy')
        add_function(ctype + '_' + 'assign_to',
                     params(output=opaque_type, input=const_opaque_type),
                     invoke='*output = *input')
    add_handle_preamble()
    c_header_preamble.append(handle_typedef.substitute(locals()))
    if not skip_def:
        c_api_body_preamble.append(handle_definition.substitute(locals()))


@cwrap('std::vector')
def vector_c_wrap(p: Parameter) -> None:
    inner = p.type.inner_type()
    # Not a generic type
    if not inner:
        return
    if inner.str() in c_type_map:
        inner = c_type_map[inner.str()]

    t = inner.add_pointer()
    if p.type.is_reference():
        if p.type.is_const():
            t = t.add_const()
    if p.returns:
        if p.type.is_reference():
            p.add_param(t.add_pointer())
            p.add_size_param()
            p.bad_param('${name} == nullptr or ${size} == nullptr',
                        'Null pointer')
        elif p.virtual:
            p.add_param(t)
            p.add_size_param()
            p.bad_param('${name} == nullptr or ${size} == nullptr',
                        'Null pointer')
            p.virtual_write = '{${name}.begin(), ${name}.begin()+${size}}; // cppcheck-suppress returnDanglingLifetime'
        else:
            p.add_param(t)
            p.bad_param('${name} == nullptr', 'Null pointer')
    else:
        p.add_param(t)
        p.add_size_param()
        p.bad_param('${name} == nullptr and ${size} != 0', 'Null pointer')

    p.read = '${type}(${name}, ${name}+${size})'
    p.cpp_write = '${type}(${name}, ${name}+${size})'
    p.virtual_read = ['${name}.data()', '${name}.size()']
    if p.type.is_reference():
        p.write = [
            '*${name} = ${result}.data()', '*${size} = ${result}.size()'
        ]
    else:
        p.write = ['std::copy(${result}.begin(), ${result}.end(), ${name})']


@cwrap('std::string', 'char*')
def string_c_wrap(p: Parameter) -> None:
    t = Type('char*')
    if p.returns:
        if p.type.is_reference():
            p.add_param(t.add_pointer())
            p.bad_param('${name} == nullptr', 'Null pointer')
        else:
            p.add_param(t)
            p.add_param('size_t', p.name + '_size')
            p.bad_param('${name} == nullptr', 'Null pointer')
    else:
        p.add_param(t)
        p.bad_param('${name} == nullptr', 'Null pointer')

    p.read = '${type}(${name})'
    p.cpp_write = '${type}(${name})'
    p.virtual_read = ['${name}.c_str()']
    if p.type.is_reference():
        p.write = ['*${name} = ${result}.c_str()']
    else:
        p.write = [
            'auto* it = std::copy_n(${result}.begin(), std::min(${result}.size(), ${name}_size - 1), ${name});'
            '*it = \'\\0\''
        ]


class Handle:
    def __init__(self, name: str, ctype: str, cpptype: str, **kwargs) -> None:
        self.name = name
        self.ctype = ctype
        self.cpptype = cpptype
        self.opaque_type = self.ctype + '_t'
        self.cpp_class = CPPClass(name, ctype)
        add_handle(name, ctype, cpptype, **kwargs)
        cpp_type_map[cpptype] = name

    def cname(self, name: str) -> str:
        return self.ctype + '_' + name

    def substitute(self, s: str, **kwargs) -> str:
        return Template(s).safe_substitute(name=self.name,
                                           ctype=self.ctype,
                                           cpptype=self.cpptype,
                                           opaque_type=self.opaque_type,
                                           **kwargs)

    def constructor(self,
                    name: str,
                    params: Optional[List[Parameter]] = None,
                    fname: Optional[str] = None,
                    invoke: Optional[str] = None,
                    **kwargs) -> 'Handle':
        create = self.substitute('allocate<${cpptype}>($@)')
        if fname:
            create = self.substitute('allocate<${cpptype}>(${fname}($@))',
                                     fname=fname)

        f = add_function(self.cname(name),
                         params=params,
                         invoke=invoke or create,
                         returns=self.cpptype + '*',
                         return_name=self.name,
                         **kwargs)
        self.cpp_class.add_constructor(name, f)
        return self

    def method(self,
               name: str,
               params: Optional[List[Parameter]] = None,
               fname: Optional[str] = None,
               invoke: Optional[str] = None,
               cpp_name: Optional[str] = None,
               const: Optional[bool] = None,
               **kwargs) -> 'Handle':
        cpptype = self.cpptype
        if const:
            cpptype = Type(cpptype).add_const().str()
        p = Parameter(self.name, cpptype)
        args = to_template_vars(params or [])
        f = add_function(self.cname(name),
                         params=[p] + (params or []),
                         invoke=invoke
                         or self.substitute('${var}.${fname}(${args})',
                                            var=template_var(self.name),
                                            fname=fname or name,
                                            args=args),
                         **kwargs)
        self.cpp_class.add_method(cpp_name or name, f)
        return self

    def function(self, name, params=None, **kwargs):
        add_function(self.cname(name), params=params, **kwargs)
        return self

    def add_cpp_class(self) -> None:
        cpp_classes.append(self.cpp_class)


interface_handle_definition = Template('''
extern "C" struct ${ctype};
struct ${ctype} {
    template<class... Ts>
    ${ctype}(void* p, ${copier} c, ${deleter} d,  const char* obj_typename, Ts&&... xs)
    : object_ptr(p, obj_typename, c, d), xobject(std::forward<Ts>(xs)...)
    {}
    manage_generic_ptr<${copier}, ${deleter}> object_ptr = nullptr;
    ${cpptype} xobject;
    ${functions}
};
''')

c_api_virtual_impl = Template('''
${return_type} ${name}(${params}) const
{
    if (${fname} == nullptr)
        throw std::runtime_error("${name} function is missing.");
    ${output_decls}
    std::array<char, 256> exception_msg;
    exception_msg.front() = '\\0';
    auto api_error_result = ${fname}(${args});
    if (api_error_result != ${success}) {
        const std::string exception_str(exception_msg.data()); 
        throw std::runtime_error("Error in ${name} of: " + std::string(object_ptr.obj_typename) + ": " + exception_str);
    }
    return ${output};
}
''')


def generate_virtual_impl(f: Function, fname: str) -> str:
    success = success_type
    name = f.name
    return_type = 'void'
    output_decls = ''
    output = ''
    largs = []
    lparams = []
    if f.returns:
        return_type = f.returns.type.str()
        output_decls = '\n'.join(f.returns.virtual_output_declarations())
        largs += f.returns.virtual_output_args()
        output = f.returns.virtual_output()
    largs += [arg for p in f.params for arg in p.virtual_arg()]
    lparams += [
        p.virtual_param() for p in f.params if not (p.this or p.hidden)
    ]
    args = ', '.join(largs)
    params = ', '.join(lparams)
    return c_api_virtual_impl.substitute(locals())


class Interface(Handle):
    def __init__(self, name: str, ctype: str, cpptype: str) -> None:
        super().__init__(name, ctype, cpptype, skip_def=True)
        self.ifunctions: List[Function] = []
        self.members: List[str] = []

    def mname(self, name: str) -> str:
        return name + "_f"

    def constructor(  # type: ignore
            self,
            name: str,
            params: Optional[List[Parameter]] = None,
            **kwargs) -> 'Interface':
        create = self.substitute('allocate<${opaque_type}>($@)')

        initial_params = gparams(obj='void*',
                                 c=self.cname('copy'),
                                 d=self.cname('delete'))

        add_function(self.cname(name),
                     params=initial_params + (params or []),
                     invoke=create,
                     returns=self.opaque_type,
                     return_name=self.name,
                     **kwargs)
        return self

    def method(self, *args, **kwargs) -> 'Interface':
        super().method(*args, **kwargs)
        return self

    def virtual(self,
                name: str,
                params: Optional[List[Parameter]] = None,
                const: Optional[bool] = None,
                **kwargs) -> 'Interface':

        # Add this parameter to the function
        this = Parameter('obj', 'void*', this=True)
        this.virtual_read = ['object_ptr.data']
        exception_msg = Parameter('exception_msg', 'char*', hidden=True)
        exception_msg.virtual_read = ['${name}.data()']
        exception_msg_size = Parameter('exception_msg_size',
                                       'size_t',
                                       hidden=True)
        exception_msg_size.virtual_read = ['exception_msg.size()']
        f = Function(name,
                     params=[this, exception_msg, exception_msg_size] +
                     (params or []),
                     virtual=True,
                     **kwargs)
        self.ifunctions.append(f)

        add_function(self.cname('set_' + name),
                     params=gparams(obj=self.opaque_type,
                                    input=self.cname(name)),
                     invoke='${{obj}}->{name} = ${{input}}'.format(
                         name=self.mname(name)))
        return self

    def generate_function(self, f: Function):
        cname = self.cname(f.name)
        mname = self.mname(f.name)
        function = generate_virtual_impl(f, fname=mname)
        return f"{cname} {mname} = nullptr;{function}"

    def generate(self):
        required_functions = [
            Function('copy',
                     params=gparams(out='void**', input='void*'),
                     virtual=True),
            Function('delete', params=gparams(input='void*'), virtual=True)
        ]
        for f in self.ifunctions + required_functions:
            f.update()
        c_header_preamble.extend([
            f.get_cfunction().generate_function_pointer(self.cname(f.name))
            for f in self.ifunctions + required_functions
        ])
        function_list = [self.generate_function(f) for f in self.ifunctions]
        ctype = self.ctype
        cpptype = self.cpptype
        copier = self.cname('copy')
        deleter = self.cname('delete')
        functions = '\n'.join(function_list)

        c_api_body_preamble.append(
            interface_handle_definition.substitute(locals()))


def handle(ctype: str,
           cpptype: str,
           name: Optional[str] = None,
           ref: Optional[bool] = None) -> Callable:
    def with_handle(f):
        n = name or f.__name__
        h = Handle(n, ctype, cpptype, ref=ref)
        f(h)
        h.add_cpp_class()

        @wraps(f)
        def decorated(*args, **kwargs):
            return f(*args, **kwargs)

        return decorated

    return with_handle


def interface(ctype: str, cpptype: str,
              name: Optional[str] = None) -> Callable:
    def with_interface(f):
        n = name or f.__name__
        h = Interface(n, ctype, cpptype)
        f(h)
        h.generate()

        @wraps(f)
        def decorated(*args, **kwargs):
            return f(*args, **kwargs)

        return decorated

    return with_interface


def template_eval(template, **kwargs):
    start = '<%'
    end = '%>'
    escaped = (re.escape(start), re.escape(end))
    mark = re.compile('%s(.*?)%s' % escaped, re.DOTALL)
    for key in kwargs:
        exec('%s = %s' % (key, kwargs[key]))
    for item in mark.findall(template):
        e = eval(item.strip())
        template = template.replace(start + item + end, str(e))
    return template


def run(path: Union[Path, str]) -> str:
    return template_eval(open(path).read())


if __name__ == "__main__":
    sys.modules['api'] = sys.modules['__main__']
    runpy.run_path(sys.argv[1])
    if len(sys.argv) > 2:
        r = run(sys.argv[2])
        sys.stdout.write(r)
    else:
        sys.stdout.write(generate_c_header())
        sys.stdout.write(generate_c_api_body())
        # sys.stdout.write(generate_cpp_header())
