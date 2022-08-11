#pragma once

#include <string>

#include "Generator.hpp"
// #include "Logging.hpp"

namespace rocRoller
{
    // ConcreteRange
    template <typename T, typename TheRange>
    ConcreteRange<T, TheRange>::ConcreteRange(TheRange&& r)
        : m_range(std::move(r))
        , m_iter(std::begin(m_range))
    {
    }

    template <typename T, typename TheRange>
    auto ConcreteRange<T, TheRange>::take_value() -> std::optional<T>
    {
        if(m_iter == m_range.end())
            return {};
        return *m_iter;
    }

    template <typename T, typename TheRange>
    auto ConcreteRange<T, TheRange>::increment() -> void
    {
        ++m_iter;
    }

    // Generator<T>::promise_type
    template <std::movable T>
    auto Generator<T>::promise_type::get_return_object() -> Generator<T>
    {
        return Generator<T>{Handle::from_promise(*this)};
    }

    template <std::movable T>
    auto Generator<T>::promise_type::unhandled_exception() noexcept -> void
    {
        this->m_exception = std::current_exception();
        this->m_value.reset();
        this->m_range.reset();
    }

    template <std::movable T>
    void Generator<T>::promise_type::check_exception() const
    {
        std::exception_ptr exc = nullptr;
        std::swap(exc, m_exception);
        if(exc)
            std::rethrow_exception(exc);
    }

    template <std::movable T>
    auto Generator<T>::promise_type::yield_value(T v) noexcept -> std::suspend_always
    {
        // Log::debug("{} taking value {}", this, v);
        this->m_value = std::move(v);
        this->m_range.reset();

        return {};
    }

    template <std::movable T>
    template <std::ranges::input_range ARange>
    auto Generator<T>::promise_type::yield_value(ARange&& r) noexcept -> std::suspend_always
    {
        // Log::debug("{} taking range.", this);
        this->m_range.reset(new ConcreteRange<T, std::remove_reference_t<ARange>>(std::move(r)));

        return {};
    }

    template <std::movable T>
    auto Generator<T>::promise_type::has_value() -> bool
    {
        check_exception();

        if(m_value.has_value())
            return true;

        if(this->m_range)
        {
            m_value = this->m_range->take_value();

            return m_value.has_value();
        }

        return false;
    }
    template <std::movable T>
    auto Generator<T>::promise_type::value() const -> T const&
    {
        check_exception();

        if(this->m_value.has_value())
            return *this->m_value;

        if(this->m_range)
            return *(m_value = this->m_range->take_value());

        throw std::runtime_error("");
    }
    template <std::movable T>
    auto Generator<T>::promise_type::discard_value() -> void
    {
        check_exception();

        if(this->m_value.has_value())
            m_value.reset();

        if(this->m_range)
            this->m_range->increment();
    }

    // Generator<T>::LazyIter
    template <std::movable T>
    auto Generator<T>::LazyIter::operator++() -> LazyIter&
    {
        if(!m_coroutine || m_coroutine.done())
            return *this;

        if(m_value.has_value())
        {
            m_value.reset();
        }
        else
        {
            auto& p = m_coroutine.promise();
            p.discard_value();
        }

        return *this;
    }

    template <std::movable T>
    auto Generator<T>::LazyIter::operator++(int) -> LazyIter
    {
        if(!m_coroutine || m_coroutine.done())
            return *this;

        LazyIter tmp(*get());

        ++(*this);

        return tmp;
    }

    template <std::movable T>
    auto Generator<T>::LazyIter::get() const -> T const*
    {
        if(m_coroutine)
            m_coroutine.promise().check_exception();

        if(m_value.has_value())
            return &*m_value;

        if(!m_coroutine || m_coroutine.done())
            return nullptr;

        auto& p = m_coroutine.promise();

        if(p.has_value())
            return &p.value();

        m_coroutine.resume();
        return get();
    }

    template <std::movable T>
    auto Generator<T>::LazyIter::operator*() const -> T const&
    {
        auto rv = get();
        if(rv)
            return *rv;

        throw std::runtime_error("nullptr!");
    }

    template <std::movable T>
    auto Generator<T>::LazyIter::operator->() const -> T const*
    {
        return get();
    }

    template <std::movable T>
    bool Generator<T>::LazyIter::isDone() const
    {
        return get() == nullptr;
    }

    template <std::movable T>
    auto Generator<T>::LazyIter::operator==(std::default_sentinel_t) const -> bool
    {
        return isDone();
    }

    template <std::movable T>
    auto Generator<T>::LazyIter::operator!=(std::default_sentinel_t) const -> bool
    {
        return !isDone();
    }

    template <std::movable T>
    Generator<T>::LazyIter::LazyIter(Handle coroutine)
        : m_coroutine{coroutine}
    {
        m_coroutine.resume();
    }

    template <std::movable T>
    Generator<T>::LazyIter::LazyIter()
        : m_isEnd(true)
    {
    }

    template <std::movable T>
    Generator<T>::LazyIter::LazyIter(T value)
        : m_value(value)
    {
    }

    // Generator<T>
    template <std::movable T>
    Generator<T>::Generator(const Handle coroutine)
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
    auto Generator<T>::begin() -> iterator
    {
        return iterator{m_coroutine};
    }

    template <std::movable T>
    auto Generator<T>::end() -> iterator
    {
        return iterator{std::default_sentinel_t{}};
    }

    template <std::movable T>
    template <template <typename...> typename Container>
    Container<T> Generator<T>::to()
    {
        auto b = begin();
        auto e = end();
        return Container<T>(b, e);
    }

}
