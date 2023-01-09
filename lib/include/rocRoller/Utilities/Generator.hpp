#pragma once

/**
 * Compiler-specific tricks to get coroutines working
 *
 *
 * Clang: Alias experimental objects to std space.
 * TODO: Remove when coroutine is moved out of experimental phase
 *
 * gcc: yield a temporary variable to workaround internal compiler error.
 * TODO: Remove when compiler bug has been fixed.

 * NOTE: The compiler bug is ONLY present when yielding the result of calling a constructor:
 * co_yield Instruction(...) // fails on GCC, use co_yield_()
 * NOT the result of a normal function call:
 * co_yield Instruction::Comment(...) // works fine.
 */
#if defined(__clang__)
#include <experimental/coroutine>

namespace std
{
    using suspend_always = experimental::suspend_always;
    template <typename T>
    using coroutine_handle = experimental::coroutine_handle<T>;
}

#define co_yield_(arg) co_yield arg

#else
#include <coroutine>

#define co_yield_(arg)                    \
    do                                    \
    {                                     \
        auto generator_tmp_object_ = arg; \
        co_yield generator_tmp_object_;   \
    } while(0)
#endif

#include <iterator>
#include <optional>
#include <ranges>

#include "../InstructionValues/Register_fwd.hpp"

namespace rocRoller
{

    template <typename T>
    struct Range
    {
        virtual ~Range() = default;

        virtual std::optional<T> take_value() = 0;
        virtual void             increment()  = 0;
    };

    template <typename T, typename TheRange>
    struct ConcreteRange : public Range<T>
    {
        explicit ConcreteRange(TheRange&& r);

        virtual std::optional<T> take_value() override;
        virtual void             increment() override;

        TheRange m_range;
        using iter = decltype(m_range.begin());
        iter m_iter;
    };

    template <std::movable T>
    class Generator
    {
    public:
        class promise_type
        {
        public:
            Generator<T> get_return_object();

            static void return_void() noexcept {}
            void        unhandled_exception() noexcept;

            void check_exception() const;

            static std::suspend_always initial_suspend() noexcept
            {
                return {};
            }
            static std::suspend_always final_suspend() noexcept
            {
                return {};
            }

            std::suspend_always yield_value(T v) noexcept;

            template <std::ranges::input_range ARange>
            std::suspend_always yield_value(ARange&& r) noexcept;

            bool     has_value();
            T const& value() const;
            void     discard_value();

        private:
            mutable std::optional<T>  m_value;
            std::unique_ptr<Range<T>> m_range;

            mutable std::exception_ptr m_exception = nullptr;
        };

        class LazyIter
        {
        public:
            // Required for STL iterator adaptors.  Not actually required to provide the - operator though.
            using difference_type = std::ptrdiff_t;
            using value_type      = T;

            using Handle = std::coroutine_handle<promise_type>;

            LazyIter& operator++();
            LazyIter  operator++(int);

            T const* get() const;
            T const& operator*() const;
            T const* operator->() const;

            bool isDone() const;
            bool operator==(std::default_sentinel_t) const;
            bool operator!=(std::default_sentinel_t) const;

            LazyIter();
            // cppcheck-suppress noExplicitConstructor
            LazyIter(Handle const& coroutine);
            // cppcheck-suppress noExplicitConstructor
            LazyIter(T value);

        private:
            std::optional<T> m_value;
            mutable Handle   m_coroutine;
            bool             m_isEnd = false;
        };

        using iterator   = std::common_iterator<LazyIter, std::default_sentinel_t>;
        using Handle     = std::coroutine_handle<promise_type>;
        using value_type = T;

        // cppcheck-suppress noExplicitConstructor
        Generator(Handle const& coroutine);
        Generator() {}
        ~Generator();

        Generator(const Generator&) = delete;
        Generator(Generator&& other) noexcept;

        Generator& operator=(const Generator&) = delete;
        Generator& operator                    =(Generator&& rhs) noexcept;

        // cppcheck-suppress functionConst
        // cppcheck-suppress functionStatic
        iterator begin();
        // cppcheck-suppress functionConst
        // cppcheck-suppress functionStatic
        iterator end();

        /**
         * Returns a `Container<T>` constructed with `begin()` and `end()` as arguments.
         * Useful for storing the output of a Generator into e.g. a std::vector or std::set.
         * Be careful that the Generator will exit and will not yield an infinite sequence.
         */
        template <template <typename...> typename Container>
        Container<T> to();

    private:
        Handle m_coroutine;
    };

    template <typename T, std::predicate<T> Predicate>
    Generator<T> filter(Predicate predicate, Generator<T> gen);
}

#include "Generator_impl.hpp"
