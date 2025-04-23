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
import string, sys, re

trivial = [
    'std::size_t', 'instruction_ref', 'support_metric', 'const_module_ref',
    'bool', 'any_ptr'
]

export_macro = 'MIGRAPHX_EXPORT'

headers = '''
#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
'''

form = string.Template('''
#ifdef TYPE_ERASED_DECLARATION

// Type-erased interface for:
struct ${export_macro} ${struct_name}
{
${decl_members}
};

#else
// NOLINTBEGIN(performance-unnecessary-value-param)
struct ${struct_name}
{
private:
    
    ${default_members}

    template <class PrivateDetailTypeErasedT>
    struct private_te_unwrap_reference
    {
        using type = PrivateDetailTypeErasedT;
    };
    template <class PrivateDetailTypeErasedT>
    struct private_te_unwrap_reference<std::reference_wrapper<PrivateDetailTypeErasedT>>
    {
        using type = PrivateDetailTypeErasedT;
    };
    template <class PrivateDetailTypeErasedT>
    using private_te_pure = typename std::remove_cv<typename std::remove_reference<PrivateDetailTypeErasedT>::type>::type;

    template <class PrivateDetailTypeErasedT>
    using private_te_constraints_impl = decltype(${constraint_members}, void());

    template <class PrivateDetailTypeErasedT>
    using private_te_constraints = private_te_constraints_impl<typename private_te_unwrap_reference<private_te_pure<PrivateDetailTypeErasedT>>::type>;

public:
    // Constructors
    ${struct_name} () = default;

    template <typename PrivateDetailTypeErasedT, 
        typename = private_te_constraints<PrivateDetailTypeErasedT>,
        typename = typename std::enable_if<not std::is_same<private_te_pure<PrivateDetailTypeErasedT>, ${struct_name}>{}>::type>
    ${struct_name} (PrivateDetailTypeErasedT&& value) :
        private_detail_te_handle_mem_var (
            std::make_shared<
                private_detail_te_handle_type<private_te_pure<PrivateDetailTypeErasedT>>
            >(std::forward<PrivateDetailTypeErasedT>(value))
        )
    {}

    // Assignment
    template <typename PrivateDetailTypeErasedT,
        typename = private_te_constraints<PrivateDetailTypeErasedT>,
        typename = typename std::enable_if<not std::is_same<private_te_pure<PrivateDetailTypeErasedT>, ${struct_name}>{}>::type>
    ${struct_name} & operator= (PrivateDetailTypeErasedT&& value)
    {
        using std::swap;
        auto * derived = this->any_cast<private_te_pure<PrivateDetailTypeErasedT>>();
        if(derived and private_detail_te_handle_mem_var.use_count() == 1)
        {
            *derived = std::forward<PrivateDetailTypeErasedT>(value);
        }
        else 
        {
            ${struct_name} rhs(value);
            swap(private_detail_te_handle_mem_var, rhs.private_detail_te_handle_mem_var); 
        }
        return *this;
    }

    // Cast
    template<typename PrivateDetailTypeErasedT>
    PrivateDetailTypeErasedT * any_cast()
    {
        return this->type_id() == typeid(PrivateDetailTypeErasedT) ?
        std::addressof(static_cast<private_detail_te_handle_type<typename std::remove_cv<PrivateDetailTypeErasedT>::type> &>(private_detail_te_get_handle()).private_detail_te_value) :
        nullptr;
    }

    template<typename PrivateDetailTypeErasedT>
    const typename std::remove_cv<PrivateDetailTypeErasedT>::type * any_cast() const
    {
        return this->type_id() == typeid(PrivateDetailTypeErasedT) ?
        std::addressof(static_cast<const private_detail_te_handle_type<typename std::remove_cv<PrivateDetailTypeErasedT>::type> &>(private_detail_te_get_handle()).private_detail_te_value) :
        nullptr;
    }

    const std::type_info& type_id() const
    {
        if(private_detail_te_handle_empty()) return typeid(std::nullptr_t);
        else return private_detail_te_get_handle().type();
    }

    ${nonvirtual_members}

    friend bool is_shared(const ${struct_name} & private_detail_x, const ${struct_name} & private_detail_y)
    {
        return private_detail_x.private_detail_te_handle_mem_var == private_detail_y.private_detail_te_handle_mem_var;
    }

private:
    struct private_detail_te_handle_base_type
    {
        virtual ~private_detail_te_handle_base_type () {}
        virtual std::shared_ptr<private_detail_te_handle_base_type> clone () const = 0;
        virtual const std::type_info& type() const = 0;

        ${pure_virtual_members}
    };

    template <typename PrivateDetailTypeErasedT>
    struct private_detail_te_handle_type :
        private_detail_te_handle_base_type
    {
        template <typename PrivateDetailTypeErasedU = PrivateDetailTypeErasedT>
        private_detail_te_handle_type (PrivateDetailTypeErasedT value,
                typename std::enable_if<
                    std::is_reference<PrivateDetailTypeErasedU>::value
                >::type * = nullptr) :
            private_detail_te_value (value)
        {}

        template <typename PrivateDetailTypeErasedU = PrivateDetailTypeErasedT>
        private_detail_te_handle_type (PrivateDetailTypeErasedT value,
                typename std::enable_if<
                    not std::is_reference<PrivateDetailTypeErasedU>::value,
                    int
                >::type * = nullptr) noexcept :
            private_detail_te_value (std::move(value))
        {}

        std::shared_ptr<private_detail_te_handle_base_type> clone () const override
        { return std::make_shared<private_detail_te_handle_type>(private_detail_te_value); }

        const std::type_info& type() const override
        {
            return typeid(private_detail_te_value);
        }

        ${virtual_members}

        PrivateDetailTypeErasedT private_detail_te_value;
    };

    template <typename PrivateDetailTypeErasedT>
    struct private_detail_te_handle_type<std::reference_wrapper<PrivateDetailTypeErasedT>> :
        private_detail_te_handle_type<PrivateDetailTypeErasedT &>
    {
        private_detail_te_handle_type (std::reference_wrapper<PrivateDetailTypeErasedT> ref) :
            private_detail_te_handle_type<PrivateDetailTypeErasedT &> (ref.get())
        {}
    };

    bool private_detail_te_handle_empty() const
    {
        return private_detail_te_handle_mem_var == nullptr;
    }

    const private_detail_te_handle_base_type & private_detail_te_get_handle () const
    {
        assert(private_detail_te_handle_mem_var != nullptr); 
        return *private_detail_te_handle_mem_var; 
    }

    private_detail_te_handle_base_type & private_detail_te_get_handle ()
    {
        assert(private_detail_te_handle_mem_var != nullptr); 
        if (private_detail_te_handle_mem_var.use_count() > 1)
            private_detail_te_handle_mem_var = private_detail_te_handle_mem_var->clone();
        return *private_detail_te_handle_mem_var;
    }

    std::shared_ptr<private_detail_te_handle_base_type> private_detail_te_handle_mem_var;
};

template<typename ValueType>
inline const ValueType * any_cast(const ${struct_name} * x)
{
    return x->any_cast<ValueType>();
}

template<typename ValueType>
inline ValueType * any_cast(${struct_name} * x)
{
    return x->any_cast<ValueType>();
}

template<typename ValueType>
inline ValueType & any_cast(${struct_name} & x)
{
    auto * y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if (y == nullptr) throw std::bad_cast();
    return *y;
}

template<typename ValueType>
inline const ValueType & any_cast(const ${struct_name} & x)
{
    const auto * y = x.any_cast<typename std::remove_reference<ValueType>::type>();
    if (y == nullptr) throw std::bad_cast();
    return *y;
}
// NOLINTEND(performance-unnecessary-value-param)
#endif
''')

nonvirtual_member = string.Template('''
${friend} ${return_type} ${name}(${params}) ${const}
{
    assert(${this}.private_detail_te_handle_mem_var);
    ${return_} ${this}.private_detail_te_get_handle().${internal_name}(${member_args});
}
''')

pure_virtual_member = string.Template(
    "virtual ${return_type} ${internal_name}(${member_params}) ${member_const} = 0;\n"
)

virtual_member = string.Template('''
${return_type} ${internal_name}(${member_params}) ${member_const} override
{
    ${using}
    ${return_} ${call};
}
''')

comment_member = string.Template(
    '''*     ${friend} ${return_type} ${name}(${params}) ${const};''')

decl_member = string.Template('''    ${comment}
    ${friend} ${return_type} ${name}(${params}) ${const};
''')

default_member = string.Template('''
template<class T>
static auto private_detail_te_default_${name}(char, T&& private_detail_te_self ${comma} ${member_params})
-> decltype(private_detail_te_self.${name}(${args}))
{
    ${return_} private_detail_te_self.${name}(${args});
}

template<class T>
static ${return_type} private_detail_te_default_${internal_name}(float, T&& private_detail_te_self ${comma} ${member_params})
{
    ${return_} ${default}(private_detail_te_self ${comma} ${args});
}
''')


def trim_type_name(name):
    n = name.strip()
    if n.startswith('const'):
        return trim_type_name(n[5:])
    if n.endswith(('&', '*')):
        return trim_type_name(n[0:-1])
    return n


def internal_name(name):
    internal_names = {
        'operator<<': 'operator_shift_left',
        'operator>>': 'operator_shift_right',
    }
    if name in internal_names:
        return internal_names[name]
    else:
        return name


def generate_constraint(m, friend, indirect):
    if m['name'].startswith('operator'):
        return None
    if friend:
        return None
    if indirect:
        return string.Template(
            'private_detail_te_default_${internal_name}(char(0), std::declval<PrivateDetailTypeErasedT>() ${comma} ${param_constraints})'
        ).substitute(m)
    return string.Template(
        'std::declval<PrivateDetailTypeErasedT>().${name}(${param_constraints})'
    ).substitute(m)


def generate_call(m, friend, indirect):
    if m['name'].startswith('operator'):
        op = m['name'][8:]
        args = m['args']
        if ',' in args:
            return args.replace(',', op)
        else:
            return string.Template('${op}${args}').substitute(op=op, args=args)
    if friend:
        return string.Template('${name}(${args})').substitute(m)
    if indirect:
        return string.Template(
            'private_detail_te_default_${internal_name}(char(0), private_detail_te_value ${comma} ${args})'
        ).substitute(m)
    return string.Template(
        'private_detail_te_value.${name}(${args})').substitute(m)


def convert_member(d, struct_name):
    for name in d:
        member = {
            'name': name,
            'internal_name': internal_name(name),
            'const': '',
            'member_const': '',
            'friend': '',
            'this': '(*this)',
            'using': '',
            'brief': '',
            'return_': '',
            'comment': '// '
        }
        args = []
        params = []
        param_constraints = []
        member_args = []
        member_params = []
        skip = False
        friend = False
        indirect = False
        if 'friend' in d[name]:
            friend = True
            skip = True
        if 'default' in d[name]:
            indirect = True
        for x in d[name]:
            t = d[name][x]
            if x == 'return':
                member['return_type'] = t if t else 'void'
                if member['return_type'] != 'void':
                    member['return_'] = 'return'
            elif x == 'const':
                member['const'] = 'const'
                member['member_const'] = 'const'
            elif x == 'friend':
                member['friend'] = 'friend'
            elif x == 'default':
                member['default'] = t
                member['comment'] = member['comment'] + '(optional)'
            elif x == 'using':
                member['using'] = 'using {};'.format(d[name]['using'])
            elif x == '__brief__':
                member['doc'] = '/// ' + t
            elif x.startswith('__') and x.endswith('__'):
                continue
            else:
                use_member = not (skip and struct_name == trim_type_name(t))
                arg_name = x
                if not use_member:
                    arg_name = 'private_detail_te_value'
                    member['this'] = x
                    if 'const' in t:
                        member['member_const'] = 'const'
                if t.endswith(('&', '*')) or t in trivial:
                    if use_member: member_args.append(x)
                    args.append(arg_name)
                else:
                    if use_member:
                        member_args.append('std::move({})'.format(x))
                    args.append('std::move({})'.format(arg_name))
                params.append(t + ' ' + x)
                param_constraints.append('std::declval<{}>()'.format(t))
                if use_member: member_params.append(t + ' ' + x)
                else: skip = False
        member['args'] = ','.join(args)
        member['member_args'] = ','.join(member_args)
        member['params'] = ','.join(params)
        member['params'] = ','.join(params)
        member['member_params'] = ','.join(member_params)
        member['param_constraints'] = ','.join(param_constraints)
        member['comma'] = ',' if len(args) > 0 else ''
        member['call'] = generate_call(member, friend, indirect)
        member['constraint'] = generate_constraint(member, friend, indirect)
        return member
    return None


def generate_form(name, members):
    nonvirtual_members = []
    pure_virtual_members = []
    virtual_members = []
    comment_members = []
    default_members = []
    decl_members = []
    constraint_members = []
    for member in members:
        m = convert_member(member, name)
        nonvirtual_members.append(nonvirtual_member.substitute(m))
        pure_virtual_members.append(pure_virtual_member.substitute(m))
        virtual_members.append(virtual_member.substitute(m))
        comment_members.append(comment_member.substitute(m))
        decl_members.append(decl_member.substitute(m))
        m_constraint = m['constraint']
        if m_constraint:
            constraint_members.append(m_constraint)
        if 'default' in m:
            default_members.append(default_member.substitute(m))
    return form.substitute(nonvirtual_members=''.join(nonvirtual_members),
                           pure_virtual_members=''.join(pure_virtual_members),
                           virtual_members=''.join(virtual_members),
                           default_members=''.join(default_members),
                           decl_members=''.join(decl_members),
                           constraint_members=','.join(constraint_members),
                           comment_members='\n'.join(comment_members),
                           struct_name=name,
                           export_macro=export_macro)


def virtual(name, returns=None, **kwargs):
    args = kwargs
    args['return'] = returns
    return {name: args}


def friend(name, returns=None, **kwargs):
    args = kwargs
    args['return'] = returns
    args['friend'] = 'friend'
    return {name: args}


def interface(name, *members):
    return generate_form(name, members)


def template_eval(template, **kwargs):
    start = '<%'
    end = '%>'
    escaped = (re.escape(start), re.escape(end))
    mark = re.compile('%s(.*?)%s' % escaped, re.DOTALL)
    for key in kwargs:
        exec('%s = %s' % (key, kwargs[key]))
    for item in mark.findall(template):
        template = template.replace(start + item + end,
                                    str(eval(item.strip())))
    return template


def run(p):
    return template_eval(open(p).read())


if __name__ == '__main__':
    sys.stdout.write(run(sys.argv[1]))
