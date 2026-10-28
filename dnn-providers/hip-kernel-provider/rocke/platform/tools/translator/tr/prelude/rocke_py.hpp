// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke_py.hpp -- Python semantics for C++ translated from the Python kernel
 * builders. The translator emits calls into it so every translated file gets
 * the same floor division, string compare, repr and range behaviour as CPython.
 *
 * PRIVATE: support for translated sources in this directory; not installed,
 * not API; EXPERIMENTAL. tools/translator/tr/prelude/ holds the master copy;
 * tr.regen copies it next to the sources that need it and never over an
 * existing copy, so a copy in the tree is maintained by hand like its sources.
 */
#ifndef ROCKE_ROCKE_PY_HPP
#define ROCKE_ROCKE_PY_HPP

#include <optional>
#include <stdint.h>
#include <string.h>
#include <variant>

#include <algorithm>
#include <array>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <stdlib.h>

#include "rocke/error.hpp"
#include "rocke/ir.h"

namespace py
{

/* ---------------------------------------------------------------- integers */

/* a // b and a % b with Python's floor semantics; b == 0 raises. */
inline int64_t floordiv(int64_t a, int64_t b)
{
    if(b == 0)
        throw ckc::ValueError("integer division or modulo by zero");
    int64_t q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
        q -= 1;
    return q;
}

inline int64_t mod(int64_t a, int64_t b)
{
    return a - floordiv(a, b) * b;
}

/* range(start, stop, step): iterable like Python's; step 0 raises. */
class range
{
public:
    range(int64_t stop)
        : start_(0)
        , stop_(stop)
        , step_(1)
    {
    }
    range(int64_t start, int64_t stop, int64_t step = 1)
        : start_(start)
        , stop_(stop)
        , step_(step)
    {
        if(step == 0)
            throw ckc::ValueError("range() arg 3 must not be zero");
    }
    class iterator
    {
    public:
        iterator(int64_t v, int64_t step)
            : v_(v)
            , step_(step)
        {
        }
        int64_t operator*() const
        {
            return v_;
        }
        iterator& operator++()
        {
            v_ += step_;
            return *this;
        }
        bool operator!=(const iterator& o) const
        {
            return v_ != o.v_;
        }

    private:
        int64_t v_;
        int64_t step_;
    };
    iterator begin() const
    {
        return iterator(start_, step_);
    }
    iterator end() const
    {
        return iterator(start_ + len() * step_, step_);
    }
    int64_t len() const
    {
        if(step_ > 0)
            return stop_ > start_ ? (stop_ - start_ + step_ - 1) / step_ : 0;
        return start_ > stop_ ? (start_ - stop_ - step_ - 1) / (-step_) : 0;
    }

private:
    int64_t start_;
    int64_t stop_;
    int64_t step_;
};

/* ----------------------------------------------------------- sequences */

/* A Python tuple of homogeneous items: a vector that prints as "(a, b)". */
template <class T>
struct tuple : std::vector<T>
{
    using std::vector<T>::vector;
    tuple() = default;
    tuple(const std::vector<T>& v)
        : std::vector<T>(v)
    {
    }
};

/* x[i] with Python's negative-index rule; out of range raises IndexError
 * (a ValueError here: no IndexError class in ckc). */
template <class S>
auto at(S& s, int64_t i) -> decltype(s[0])
{
    const int64_t n = (int64_t)s.size();
    if(i < 0)
        i += n;
    if(i < 0 || i >= n)
        throw ckc::ValueError("index out of range");
    return s[(size_t)i];
}

template <class S>
int64_t len(const S& s)
{
    return (int64_t)s.size();
}

/* A non-owning view of a C (pointer, count) pair: a Python tuple field. */
template <class T>
struct span
{
    const T* p;
    int64_t n;
    const T* begin() const
    {
        return p;
    }
    const T* end() const
    {
        return p + n;
    }
    size_t size() const
    {
        return (size_t)n;
    }
    const T& operator[](size_t i) const
    {
        return p[i];
    }
};

/* at() of a span made in place (`py::at(py::span<..>{p, n}, 0)`): the
 * elements live in the C array, so the reference outlives the view. */
template <class T>
const T& at(const span<T>& s, int64_t i)
{
    return at<const span<T>>(s, i);
}

/* list.append on a C (pointer, count, capacity) field the struct owns. */
template <class T, class V>
void opush(T*& p, int& n, int& cap, V v)
{
    if(n == cap)
    {
        const int c = cap ? 2 * cap : 16;
        T* q = (T*)realloc((void*)p, (size_t)c * sizeof(T));
        if(q == NULL)
            throw ckc::OOMError("out of memory");
        p = q;
        cap = c;
    }
    p[n++] = (T)v;
}

/* iter(seq) / next(it) over a sequence the iterator copies. */
template <class T>
class iter
{
public:
    template <class S>
    explicit iter(const S& s)
        : v_(s.begin(), s.end())
    {
    }
    T next()
    {
        if(i_ >= v_.size())
            throw ckc::ValueError("StopIteration");
        return v_[i_++];
    }

private:
    std::vector<T> v_;
    size_t i_ = 0;
};

inline std::string str(const rocke_type_t* t)
{
    return t && t->name ? std::string(t->name) : std::string("None");
}

/* x[a:b] of a sequence, with Python's clamping and negative indices. */
template <class S>
auto slice(const S& s,
           int64_t a,
           int64_t b) -> std::vector<typename std::decay<decltype(s[0])>::type>
{
    const int64_t n = (int64_t)s.size();
    if(a < 0)
        a += n;
    if(b < 0)
        b += n;
    a = a < 0 ? 0 : (a > n ? n : a);
    b = b < 0 ? 0 : (b > n ? n : b);
    std::vector<typename std::decay<decltype(s[0])>::type> out;
    for(int64_t i = a; i < b; ++i)
        out.push_back(s[(size_t)i]);
    return out;
}

/* The kernel-argument dict {name: Value}: insertion order kept, KeyError on a miss. */
class kargs
{
public:
    rocke_value_t* at(const std::string& k) const
    {
        for(const auto& kv : kv_)
            if(kv.first == k)
                return kv.second;
        throw ckc::KeyError("'" + k + "'");
    }
    void set(const std::string& k, rocke_value_t* v)
    {
        for(auto& kv : kv_)
            if(kv.first == k)
            {
                kv.second = v;
                return;
            }
        kv_.emplace_back(k, v);
    }
    size_t size() const
    {
        return kv_.size();
    }

private:
    std::vector<std::pair<std::string, rocke_value_t*>> kv_;
};

/* Keyword attributes {"readonly": True, ...} passed as **attrs. */
struct attrs
{
    std::vector<std::pair<std::string, int64_t>> kv;
};

/* ------------------------------------------------------------- strings */

/* Python ==: two absent (NULL) strings are equal. */
inline bool eq(const char* a, const char* b)
{
    if(a == NULL || b == NULL)
        return a == b;
    return strcmp(a, b) == 0;
}
inline bool eq(const std::string& a, const char* b)
{
    return b != NULL && a == b;
}

/* A str a function returns as const char*: the std::string it was built in
 * dies with the function, so its text is interned (one copy per distinct
 * value, kept for the life of the process, like a Python str constant). */
inline const char* keep(const std::string& s)
{
    static std::mutex m;
    static std::set<std::string> pool;
    std::lock_guard<std::mutex> lock(m);
    return pool.insert(s).first->c_str();
}

/* str() and repr() of the values generated code formats. */
inline std::string str(int64_t v)
{
    return std::to_string((long long)v);
}
inline std::string str(int v)
{
    return std::to_string(v);
}
inline std::string str(bool v)
{
    return v ? "True" : "False";
}
inline std::string str(const char* s)
{
    return s ? std::string(s) : std::string("None");
}
inline std::string str(const std::string& s)
{
    return s;
}

inline std::string repr(const char* s)
{
    if(s == NULL)
        return "None";
    /* CPython quotes with ' unless the text holds a ' and no " */
    const bool dq = strchr(s, '\'') != NULL && strchr(s, '"') == NULL;
    const char q = dq ? '"' : '\'';
    std::string out(1, q);
    for(const char* p = s; *p; ++p)
    {
        if(*p == '\\' || *p == q)
            out += '\\';
        out += *p;
    }
    out += q;
    return out;
}
inline std::string repr(const std::string& s)
{
    return repr(s.c_str());
}
inline std::string repr(int64_t v)
{
    return str(v);
}
inline std::string repr(int v)
{
    return str(v);
}
inline std::string repr(bool v)
{
    return str(v);
}

template <class T, size_t N>
std::string repr(const std::array<T, N>& a)
{
    std::string out = "(";
    for(size_t i = 0; i < N; ++i)
    {
        if(i)
            out += ", ";
        out += repr(a[i]);
    }
    out += N == 1 ? ",)" : ")";
    return out;
}
template <class T>
std::string repr(const tuple<T>& t)
{
    std::string out = "(";
    for(size_t i = 0; i < t.size(); ++i)
    {
        if(i)
            out += ", ";
        out += repr(t[i]);
    }
    out += t.size() == 1 ? ",)" : ")";
    return out;
}
template <class T>
std::string repr(const std::vector<T>& v)
{
    std::string out = "[";
    for(size_t i = 0; i < v.size(); ++i)
    {
        if(i)
            out += ", ";
        out += repr(v[i]);
    }
    return out + "]";
}
template <class T, size_t N>
std::string str(const std::array<T, N>& a)
{
    return repr(a);
}
template <class T>
std::string str(const tuple<T>& t)
{
    return repr(t);
}
template <class T>
std::string str(const std::vector<T>& v)
{
    return repr(v);
}

/* sep.join(parts) */
template <class S>
std::string join(const char* sep, const S& parts)
{
    std::string out;
    bool first = true;
    for(const auto& p : parts)
    {
        if(!first)
            out += sep;
        out += p;
        first = false;
    }
    return out;
}

/* x in seq, with Python == on strings */
template <class S, class X>
bool contains(const S& s, const X& x)
{
    for(const auto& e : s)
    {
        if constexpr(std::is_convertible<X, const char*>::value)
        {
            if(eq(e, x))
                return true;
        }
        else if(e == x)
            return true;
    }
    return false;
}

/* A Python dict: insertion order kept; assigning an existing key keeps its
 * position (as in Python). Iterating yields (key, value) pairs. */
template <class K, class V>
class dict
{
public:
    using item = std::pair<K, V>;
    void set(const K& k, const V& v)
    {
        for(auto& kv : kv_)
            if(kv.first == k)
            {
                kv.second = v;
                return;
            }
        kv_.emplace_back(k, v);
    }
    const V& at(const K& k) const
    {
        for(const auto& kv : kv_)
            if(kv.first == k)
                return kv.second;
        throw ckc::KeyError(repr(k));
    }
    std::vector<K> keys() const
    {
        std::vector<K> out;
        for(const auto& kv : kv_)
            out.push_back(kv.first);
        return out;
    }
    std::vector<V> values() const
    {
        std::vector<V> out;
        for(const auto& kv : kv_)
            out.push_back(kv.second);
        return out;
    }
    V setdefault(const K& k, const V& dflt)
    {
        for(const auto& kv : kv_)
            if(kv.first == k)
                return kv.second;
        kv_.emplace_back(k, dflt);
        return dflt;
    }
    V get(const K& k, const V& dflt) const
    {
        for(const auto& kv : kv_)
            if(kv.first == k)
                return kv.second;
        return dflt;
    }
    bool has(const K& k) const
    {
        for(const auto& kv : kv_)
            if(kv.first == k)
                return true;
        return false;
    }
    const std::vector<item>& items() const
    {
        return kv_;
    }
    typename std::vector<item>::const_iterator begin() const
    {
        return kv_.begin();
    }
    typename std::vector<item>::const_iterator end() const
    {
        return kv_.end();
    }
    size_t size() const
    {
        return kv_.size();
    }

private:
    std::vector<item> kv_;
};

/* A malloc'd copy of a sequence: the storage of a list field a C struct owns
 * (pointer, count, capacity). At least one slot, so a NULL means no memory. */
template <class T, class S>
T* own_copy(const S& s)
{
    T* p = (T*)malloc(sizeof(T) * (s.size() ? s.size() : 1));
    if(p == nullptr)
        throw ckc::OOMError("out of memory");
    size_t i = 0;
    for(const auto& x : s)
        p[i++] = (T)x;
    return p;
}

/* s[lo:] / s[lo:hi] with Python's clamping (non-negative bounds as written) */
inline std::string str_slice(const std::string& s, int64_t lo, int64_t hi = INT64_MAX)
{
    const int64_t n = (int64_t)s.size();
    if(lo < 0)
        lo += n;
    if(hi < 0)
        hi += n;
    lo = lo < 0 ? 0 : (lo > n ? n : lo);
    hi = hi < 0 ? 0 : (hi > n ? n : hi);
    return hi > lo ? s.substr((size_t)lo, (size_t)(hi - lo)) : std::string();
}

/* s[i] of a str: the one-character string at i (Python's negative-index rule;
 * out of range raises, as at() does) */
inline std::string str_at(const std::string& s, int64_t i)
{
    return std::string(1, at(s, i));
}

/* int(s) of a decimal string; ValueError otherwise */
inline int64_t int_of(const std::string& s)
{
    size_t i = 0;
    while(i < s.size() && (s[i] == ' ' || s[i] == '\t'))
        ++i;
    size_t j = s.size();
    while(j > i && (s[j - 1] == ' ' || s[j - 1] == '\t'))
        --j;
    const std::string t = s.substr(i, j - i);
    size_t k = (!t.empty() && (t[0] == '-' || t[0] == '+')) ? 1 : 0;
    if(k == t.size())
        throw ckc::ValueError("invalid literal for int() with base 10: " + repr(s));
    for(size_t q = k; q < t.size(); ++q)
        if(t[q] < '0' || t[q] > '9')
            throw ckc::ValueError("invalid literal for int() with base 10: " + repr(s));
    return std::stoll(t);
}

/* max(xs) / min(xs) of a sequence: the first extremum; ValueError when empty */
template <class S>
auto max_of(const S& s) -> std::decay_t<decltype(*s.begin())>
{
    if(s.begin() == s.end())
        throw ckc::ValueError("max() arg is an empty sequence");
    auto best = *s.begin();
    for(const auto& x : s)
        if(best < x)
            best = x;
    return best;
}

template <class S>
auto min_of(const S& s) -> std::decay_t<decltype(*s.begin())>
{
    if(s.begin() == s.end())
        throw ckc::ValueError("min() arg is an empty sequence");
    auto best = *s.begin();
    for(const auto& x : s)
        if(x < best)
            best = x;
    return best;
}

/* own_copy with an item conversion (std::array -> a generated tuple struct) */
template <class T, class S, class F>
T* own_copy_map(const S& s, F f)
{
    T* p = (T*)malloc(sizeof(T) * (s.size() ? s.size() : 1));
    if(p == nullptr)
        throw ckc::OOMError("out of memory");
    size_t i = 0;
    for(const auto& x : s)
        p[i++] = f(x);
    return p;
}

/* rocke.helpers.spec.ceil_div: -(-a // b) */
inline int64_t ceil_div(int64_t a, int64_t b)
{
    return -floordiv(-a, b);
}

/* math.gcd(a, b): non-negative, gcd(0, 0) == 0 */
inline int64_t gcd(int64_t a, int64_t b)
{
    a = a < 0 ? -a : a;
    b = b < 0 ? -b : b;
    while(b != 0)
    {
        const int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/* A Python set of ints: ascending iteration (CPython's order for small
 * non-negative ints); pop() takes the first. */
template <class T>
class set
{
public:
    void add(const T& x)
    {
        auto it = std::lower_bound(v_.begin(), v_.end(), x);
        if(it == v_.end() || !(*it == x))
            v_.insert(it, x);
    }
    void discard(const T& x)
    {
        auto it = std::lower_bound(v_.begin(), v_.end(), x);
        if(it != v_.end() && *it == x)
            v_.erase(it);
    }
    T pop()
    {
        if(v_.empty())
            throw ckc::KeyError("'pop from an empty set'");
        T x = v_.front();
        v_.erase(v_.begin());
        return x;
    }
    size_t size() const
    {
        return v_.size();
    }
    bool empty() const
    {
        return v_.empty();
    }
    typename std::vector<T>::const_iterator begin() const
    {
        return v_.begin();
    }
    typename std::vector<T>::const_iterator end() const
    {
        return v_.end();
    }

private:
    std::vector<T> v_;
};

/* seq.index(x): the first position; ValueError when absent */
template <class S, class X>
int64_t index_of(const S& s, const X& x)
{
    int64_t i = 0;
    for(const auto& e : s)
    {
        if(eq(e, x))
            return i;
        ++i;
    }
    throw ckc::ValueError("tuple.index(x): x not in tuple");
}

/* list(d.items()): (key, value) tuples */
template <class D>
auto items_list(const D& d)
{
    using KV = typename std::decay_t<decltype(*d.begin())>;
    std::vector<std::tuple<typename KV::first_type, typename KV::second_type>> out;
    for(const auto& kv : d)
        out.emplace_back(kv.first, kv.second);
    return out;
}

/* list.pop(i): removes and returns item i (Python's negative indices) */
template <class T>
T pop(std::vector<T>& v, int64_t i = -1)
{
    const int64_t n = (int64_t)v.size();
    if(i < 0)
        i += n;
    if(i < 0 || i >= n)
        throw ckc::ValueError("pop index out of range");
    T x = v[(size_t)i];
    v.erase(v.begin() + i);
    return x;
}

/* itertools.pairwise(xs) */
template <class S>
auto pairwise(const S& s)
{
    using T = std::decay_t<decltype(*s.begin())>;
    std::vector<std::array<T, 2>> out;
    auto it = s.begin();
    if(it == s.end())
        return out;
    auto prev = *it++;
    for(; it != s.end(); ++it)
    {
        out.push_back({prev, *it});
        prev = *it;
    }
    return out;
}

/* list(range(...)) */
inline std::vector<int64_t> range_list(int64_t stop)
{
    std::vector<int64_t> v;
    for(int64_t i = 0; i < stop; ++i)
        v.push_back(i);
    return v;
}

inline std::vector<int64_t> range_list(int64_t start, int64_t stop, int64_t step = 1)
{
    if(step == 0)
        throw ckc::ValueError("range() arg 3 must not be zero");
    std::vector<int64_t> v;
    for(int64_t i = start; step > 0 ? i < stop : i > stop; i += step)
        v.push_back(i);
    return v;
}

/* the value of an optional; None raises TypeError as Python does on use */
template <class T>
const T& deref(const std::optional<T>& o)
{
    if(!o.has_value())
        throw ckc::TypeError("'NoneType' object is not subscriptable");
    return *o;
}

template <class T>
T& deref(std::optional<T>& o)
{
    if(!o.has_value())
        throw ckc::TypeError("'NoneType' object is not subscriptable");
    return *o;
}

/* == of two sequences with an item comparison (items of different spellings) */
template <class A, class B, class F>
bool seq_eq(const A& a, const B& b, F eq)
{
    if(a.size() != b.size())
        return false;
    auto ib = b.begin();
    for(const auto& x : a)
        if(!eq(x, *ib++))
            return false;
    return true;
}

template <class T>
struct is_std_array : std::false_type
{
};
template <class T, size_t N>
struct is_std_array<std::array<T, N>> : std::true_type
{
};

/* a list (of lists) of None into the element type it is later filled with */
template <class To, class From>
To retype(const From& f)
{
    if constexpr(std::is_same_v<std::decay_t<From>, std::nullptr_t>)
        return To(nullptr);
    else if constexpr(std::is_convertible_v<const From&, To>)
        return To(f); // a leaf (or an already matching container)
    else if constexpr(is_std_array<To>::value)
    {
        // a sequence into a fixed tuple (ValueError on another length)
        if(f.size() != std::tuple_size_v<To>)
            throw ckc::ValueError("wrong number of values to unpack");
        To out{};
        size_t i = 0;
        for(const auto& x : f)
            out[i++] = retype<typename To::value_type>(x);
        return out;
    }
    else
    {
        To out;
        for(const auto& x : f)
            out.push_back(retype<typename To::value_type>(x));
        return out;
    }
}

/* a sequence into another sequence type, item by item through f */
template <class To, class S, class F>
To map_to(const S& s, F f)
{
    To out;
    for(const auto& x : s)
        out.push_back(f(x));
    return out;
}

/* a sequence into a fixed tuple of N items (ValueError on another length) */
template <class T, size_t N, class S>
std::array<T, N> to_array(const S& s)
{
    if(s.size() != N)
        throw ckc::ValueError("wrong number of values to unpack");
    std::array<T, N> out{};
    size_t i = 0;
    for(const auto& x : s)
        out[i++] = x;
    return out;
}

/* int.bit_length(); f"{x:#x}" / f"{x:x}" */
inline int64_t bit_length(int64_t x)
{
    uint64_t u = x < 0 ? (uint64_t)(-(x + 1)) + 1 : (uint64_t)x;
    int64_t n = 0;
    while(u)
    {
        ++n;
        u >>= 1;
    }
    return n;
}

inline std::string hex(int64_t x, bool prefix)
{
    const bool neg = x < 0;
    uint64_t u = neg ? (uint64_t)(-(x + 1)) + 1 : (uint64_t)x;
    std::string d;
    do
    {
        d.insert(d.begin(), "0123456789abcdef"[u & 15]);
        u >>= 4;
    } while(u);
    return std::string(neg ? "-" : "") + (prefix ? "0x" : "") + d;
}

/* str.strip / lower / upper / split(sep) / partition(sep) */
inline std::string strip(const std::string& s)
{
    const char* ws = " \t\n\r\f\v";
    const size_t b = s.find_first_not_of(ws);
    if(b == std::string::npos)
        return std::string();
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

inline std::string lower(std::string s)
{
    for(auto& c : s)
        if(c >= 'A' && c <= 'Z')
            c = (char)(c - 'A' + 'a');
    return s;
}

inline std::string upper(std::string s)
{
    for(auto& c : s)
        if(c >= 'a' && c <= 'z')
            c = (char)(c - 'a' + 'A');
    return s;
}

inline std::vector<std::string> split(const std::string& s, const std::string& sep)
{
    if(sep.empty())
        throw ckc::ValueError("empty separator");
    std::vector<std::string> out;
    size_t at = 0;
    for(;;)
    {
        const size_t k = s.find(sep, at);
        if(k == std::string::npos)
        {
            out.push_back(s.substr(at));
            return out;
        }
        out.push_back(s.substr(at, k - at));
        at = k + sep.size();
    }
}

inline std::array<std::string, 3> partition(const std::string& s, const std::string& sep)
{
    const size_t k = s.find(sep);
    if(k == std::string::npos)
        return {s, std::string(), std::string()};
    return {s.substr(0, k), sep, s.substr(k + sep.size())};
}

/* str.startswith / str.endswith */
inline bool startswith(const std::string& s, const std::string& p)
{
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

inline bool endswith(const std::string& s, const std::string& p)
{
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

} // namespace py

#endif /* ROCKE_ROCKE_PY_HPP */
