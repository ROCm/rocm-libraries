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

#include <string>

#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    inline std::string toString(GeneratorState s)
    {
        switch(s)
        {
        case GeneratorState::NoValue:
            return "NoValue";
        case GeneratorState::HasValue:
            return "HasValue";
        case GeneratorState::HasRange:
            return "HasRange";
        case GeneratorState::HasRangeValue:
            return "HasRangeValue";
        case GeneratorState::HasCopiedValue:
            return "HasCopiedValue";
        case GeneratorState::Done:
            return "Done";

        case GeneratorState::Count:
        default:
            break;
        }
        throw std::runtime_error("Invalid GeneratorState");
    }

    inline std::ostream& operator<<(std::ostream& stream, GeneratorState const& s)
    {
        return stream << toString(s);
    }

    template <typename T, CInputRangeOf<T> TheRange>
    template <std::convertible_to<TheRange> ARange>
    ConcreteRange<T, TheRange>::ConcreteRange(ARange&& r)
        : m_range(std::forward<ARange>(r))
        , m_iter(std::begin(m_range))
    {
    }

    template <typename T, CInputRangeOf<T> TheRange>
    constexpr std::optional<T> ConcreteRange<T, TheRange>::take_value()
    {
        if(m_iter == m_range.end())
            return {};
        return *m_iter;
    }

    template <typename T, CInputRangeOf<T> TheRange>
    constexpr void ConcreteRange<T, TheRange>::increment()
    {
        if(m_iter != m_range.end())
            ++m_iter;
    }

    template <std::movable T>
    Generator<T> Generator<T>::promise_type::get_return_object()
    {
        return Generator<T>{Handle::from_promise(*this)};
    }

    template <std::movable T>
    constexpr void Generator<T>::promise_type::unhandled_exception() noexcept
    {
        m_exception = std::current_exception();
        m_value.reset();
        m_range.reset();
    }

    template <std::movable T>
    inline void Generator<T>::promise_type::check_exception() const
    {
        // NOTE: A thread-safe version of this would be similar to
        //     std::exception_ptr exc = nullptr;
        //     std::swap(exc, m_exception);
        //     if(exc)
        //       std::rethrow_exception(exc);
        //
        // Thread-safety is not necessary (RR generates within a single thread); and
        // conditionally swapping improves performance.
        if(m_exception)
        {
            std::exception_ptr exc = nullptr;
            std::swap(exc, m_exception);
            std::rethrow_exception(exc);
        }
    }

    template <std::movable T>
    void Generator<T>::promise_type::advance_range()
    {
        m_range->increment();
        m_value = m_range->take_value();
        if(!m_value)
            m_range.reset();
    }

    template <std::movable T>
    constexpr std::suspend_always Generator<T>::promise_type::yield_value(T v) noexcept
    {
        m_value = std::move(v);
        m_range.reset();

        return {};
    }

    template <std::movable T>
    template <CInputRangeOf<T> ARange>
    constexpr std::suspend_always Generator<T>::promise_type::yield_value(ARange&& r) noexcept
    {
        using MyRange = ConcreteRange<T, std::remove_reference_t<ARange>>;
        m_range.reset(new MyRange(std::forward<ARange>(r)));

        try
        {
            // The C++ runtime will not catch an exception from here, so we
            // must handle it ourselves.
            m_value = m_range->take_value();
            if(!m_value)
                m_range.reset();
        }
        catch(...)
        {
            unhandled_exception();
        }

        return {};
    }

    template <std::movable T>
    constexpr std::suspend_always
        Generator<T>::promise_type::yield_value(std::initializer_list<T> r) noexcept
    {
        return yield_value<std::initializer_list<T>>(std::move(r));
    }

    template <std::movable T>
    constexpr GeneratorState Generator<T>::promise_type::state() const
    {
        check_exception();

        if(m_range)
        {
            if(m_value.has_value())
                return GeneratorState::HasRangeValue;
            else
                return GeneratorState::HasRange;
        }
        else
        {
            if(m_value.has_value())
                return GeneratorState::HasValue;

            return GeneratorState::NoValue;
        }
    }

    template <std::movable T>
    inline constexpr GeneratorState Generator<T>::state() const
    {
        if(m_coroutine)
            m_coroutine.promise().check_exception();

        if(!m_coroutine || m_coroutine.done())
            return GeneratorState::Done;

        return m_coroutine.promise().state();
    }

    template <std::movable T>
    constexpr std::optional<T> const& Generator<T>::promise_type::value() const
    {
        check_exception();

        return m_value;
    }

    template <std::movable T>
    void Generator<T>::promise_type::discard_value()
    {
        check_exception();

        if(m_value.has_value())
            m_value.reset();
    }

    template <std::movable T>
    auto Generator<T>::Iterator::operator++() -> Iterator&
    {
        // If the iterator was previously incremented but not dereferenced, pretend we've dereferenced it for consistency.
        switch(state())
        {
        case GeneratorState::HasRange:
        case GeneratorState::NoValue:
            get();
        default:
            break;
        }

        switch(state())
        {

        case GeneratorState::HasValue:
        case GeneratorState::HasRangeValue:
            m_coroutine.promise().discard_value();
            break;

        // The call to get() from the top of this function should advance to a
        // point where we have a value, or to the end of the coroutine.
        case GeneratorState::NoValue:
        case GeneratorState::HasRange:
            throw std::runtime_error("Invalid generator state!");

        case GeneratorState::HasCopiedValue:
            m_value.reset();
            break;

        case GeneratorState::Done:
        default:
            break;
        }
        return *this;
    }

    template <std::movable T>
    auto Generator<T>::Iterator::operator++(int) -> Iterator
    {
        if(!m_coroutine || m_coroutine.done())
            return *this;

        Iterator tmp(*get());

        ++(*this);

        return tmp;
    }

    template <std::movable T>
    T const* Generator<T>::Iterator::get() const
    {
        switch(state())
        {
        case GeneratorState::HasCopiedValue:
            return &*m_value;

        case GeneratorState::HasValue:
        case GeneratorState::HasRangeValue:
            return &*(m_coroutine.promise().value());

        case GeneratorState::HasRange:
            m_coroutine.promise().advance_range();
            return get();

        case GeneratorState::NoValue:
            m_coroutine.resume();
            return get();

        case GeneratorState::Done:
        default:
            return nullptr;
        }
    }

    template <std::movable T>
    T const& Generator<T>::Iterator::operator*() const
    {
        auto rv = get();
        if(rv)
            return *rv;

        throw std::runtime_error("nullptr!");
    }

    template <std::movable T>
    T const* Generator<T>::Iterator::operator->() const
    {
        return get();
    }

    template <std::movable T>
    bool Generator<T>::Iterator::isDone() const
    {
        return get() == nullptr;
    }

    template <std::movable T>
    bool Generator<T>::Iterator::operator==(std::default_sentinel_t) const
    {
        return isDone();
    }

    template <std::movable T>
    bool Generator<T>::Iterator::operator!=(std::default_sentinel_t) const
    {
        return !isDone();
    }

    template <std::movable T>
    GeneratorState Generator<T>::Iterator::state() const
    {
        if(m_coroutine)
            m_coroutine.promise().check_exception();

        if(m_value)
            return GeneratorState::HasCopiedValue;

        if(!m_coroutine || m_coroutine.done())
            return GeneratorState::Done;

        return m_coroutine.promise().state();
    }

    template <std::movable T>
    Generator<T>::Iterator::Iterator(Handle const& coroutine)
        : m_coroutine{coroutine}
    {
    }

    template <std::movable T>
    Generator<T>::Iterator::Iterator()
    {
    }

    template <std::movable T>
    Generator<T>::Iterator::Iterator(T value)
        : m_value(value)
    {
    }

    template <std::movable T>
    Generator<T>::Generator(Handle const& coroutine)
        : m_coroutine{coroutine}
    {
    }

    template <std::movable T>
    Generator<T>::~Generator()
    {
        if(m_coroutine)
        {
            m_coroutine.destroy();
        }
    }

    template <std::movable T>
    Generator<T>::Generator(Generator&& other) noexcept
        : m_coroutine{other.m_coroutine}
    {
        other.m_coroutine = {};
    }

    template <std::movable T>
    auto Generator<T>::operator=(Generator&& rhs) noexcept -> Generator&
    {
        if(this != &rhs)
        {
            std::swap(m_coroutine, rhs.m_coroutine);
        }
        return *this;
    }

    template <std::movable T>
    constexpr auto Generator<T>::begin() -> iterator
    {
        return iterator{m_coroutine};
    }

    template <std::movable T>
    constexpr auto Generator<T>::end() -> iterator
    {
        return iterator{std::default_sentinel_t{}};
    }

    template <std::movable T>
    template <template <typename...> typename Container>
    constexpr Container<T> Generator<T>::to()
    {
        auto b = begin();
        auto e = end();
        return Container<T>(b, e);
    }

    template <std::movable T>
    template <std::predicate<T> Predicate>
    Generator<T> Generator<T>::filter(Predicate predicate)
    {
        return rocRoller::filter(predicate, std::move(*this));
    }

    template <std::movable T>
    template <std::invocable<T> Func>
    Generator<std::invoke_result_t<Func, T>> Generator<T>::map(Func func)
    {
        return rocRoller::map(func, std::move(*this));
    }

    template <std::movable T>
    Generator<T> Generator<T>::take(size_t n)
    {
        return rocRoller::take(n, std::move(*this));
    }

    template <std::ranges::input_range Range, typename Predicate>
    requires(std::predicate<Predicate, std::ranges::range_value_t<Range>>)
        Generator<std::ranges::range_value_t<Range>> filter(Predicate predicate, Range range)
    {
        for(auto val : range)
        {
            if(predicate(val))
                co_yield val;
        }
    }

    template <std::ranges::input_range Range, typename Func>
    requires(std::invocable<Func, std::ranges::range_value_t<Range>>)
        Generator<std::invoke_result_t<Func, std::ranges::range_value_t<Range>>> map(Func  func,
                                                                                     Range range)
    {
        for(auto val : range)
            co_yield func(val);
    }

    template <typename T>
    Generator<T> take(size_t n, Generator<T> gen)
    {
        auto it = gen.begin();
        for(size_t i = 0; i < n && it != gen.end(); ++i, ++it)
        {
            co_yield *it;
        }
    }

    template <std::movable T>
    std::optional<T> Generator<T>::only()
    {
        return rocRoller::only(std::move(*this));
    }

    template <std::ranges::input_range Range>
    inline std::optional<std::ranges::range_value_t<Range>> only(Range range)
    {
        auto iter = range.begin();
        if(iter == range.end())
            return {};

        auto value = *iter;

        ++iter;
        if(iter == range.end())
            return value;

        return {};
    }

    template <std::movable T>
    constexpr bool Generator<T>::empty()
    {
        return begin() == end();
    }

    template <std::ranges::input_range Range>
    inline constexpr bool empty(Range range)
    {
        return range.begin() == range.end();
    }

}
