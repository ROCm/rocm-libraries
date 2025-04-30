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
#include <coroutine>

#define co_yield_(arg) co_yield arg

#else
#include <coroutine>

#define co_yield_(arg)                             \
    do                                             \
    {                                              \
        auto generator_tmp_object_ = arg;          \
        co_yield std::move(generator_tmp_object_); \
    } while(0)
#endif

#include <iterator>
#include <optional>
#include <ranges>
#include <utility>

#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/Utilities/Concepts.hpp>

namespace rocRoller
{
    enum class GeneratorState
    {
        /// Does not have a value.  Will have to resume the coroutine in order to obtain a value.
        NoValue = 0,

        /// Has a single value.
        HasValue,

        /// Has a range which may or may not contain at least one more value.
        HasRange,

        /// Has a value from a range.
        HasRangeValue,

        /// Has a value which has been copied into the iterator. Only possible for an
        /// iterator that has been returned from the postfix increment operator.
        /// e.g. auto x = iter++; -> x will be in the HasCopiedValue state.
        HasCopiedValue,

        // Has completed the coroutine, and has no value.
        Done,
        Count
    };
    std::string   toString(GeneratorState s);
    std::ostream& operator<<(std::ostream&, GeneratorState const&);

    /**
     * Range/ConcreteRange wraps a collection object behind a virtual interface,
     * which allows the Generator implementation to store a yielded collection
     * and provide access to its items.
     */
    template <typename T>
    struct Range
    {
        virtual ~Range() = default;

        /// @brief  Takes the next value from the range.  Will not return a value
        /// if we have reached the end of the range.
        virtual constexpr std::optional<T> take_value() = 0;
        virtual constexpr void             increment()  = 0;
    };

    template <typename T, CInputRangeOf<T> TheRange>
    struct ConcreteRange : public Range<T>
    {
        template <std::convertible_to<TheRange> ARange>
        explicit ConcreteRange(ARange&& r);

        virtual constexpr std::optional<T> take_value() override;
        virtual constexpr void             increment() override;

    private:
        TheRange m_range;
        using iter = decltype(m_range.begin());
        iter m_iter;
    };

    /**
     * @brief Return type to make a lazy sequence ("generator") from a coroutine.
     *
     * ## Using Generator collections
     *
     * A `Generator<T>` adheres to the `std::ranges::input_range` concept and
     * so it behaves in many ways the same as other collections in C++:
     *
     * ```
     * Generator<int> someCollection();
     *
     * // ...
     *
     * for(int val: someCollection())
     *   std::cout << val << std::endl;
     *
     * auto g = someCollection();
     * for(auto iter = g.begin(); iter != g.end(); ++iter)
     *   std::cout << val << std::endl;
     *
     * auto g2 = someCollection();
     * std::vector v(g2.begin(), g2.end());
     * ```
     * Because it's a lazy collection with no storage, there are a few important
     * differences:
     *
     *  - It's impossible to know how many elements are in a `Generator` until
     *      it has finished executing, at which point there are 0 elements in it.
     *  - All iterators of the same `Generator` other than the `end()` iterator
     *      are effectively at the same location (the `Generator`'s current
     *      location), and incrementing any of them will increment all of them.
     *  - It's impossible to copy a Generator.
     *  - Deleting a Generator will cancel any further execution.
     *      - This can be useful if we're searching for a particular value.
     *          Once it's found we can cancel the search.
     *  - It's possible to create (and use) a Generator that never finishes.
     *      - Be careful that the code that calls it will not try to run until
     *          the generator is empty.
     *  - The prefix increment operator (`++iter`) is more efficient than the
     *      postfix increment operator (`iter++`).
     *      - The postfix version will create a copy of the current value if
     *          this is needed.
     *
     * A Generator will not begin executing until the first value is needed.
     * This will happen the first time:
     *
     *  - An iterator is compared with `end()`, or
     *  - An iterator is dereferenced.
     *
     * This means that there's no way to make a Generator take a reference to a
     * temporary and have it behave correctly. The temporary will have been
     * deleted by the time it's needed, even if the first line of the function
     * makes a copy of it.
     *      - As a rule, coroutines should take arguments by value.  By
     *        (non-const) reference can also work for output arguments.
     *      - A coroutine should **never** take an argument by const-reference.
     *
     * ### Convenience functions:
     *
     * - If you need a more standard collection, most can be created with
     *    `.to<>()` (e.g. `someCollection().to<std::vector>()`).
     * - `gen.map(func)` or `map(func, gen)` will transform each element.
     * - `gen.filter(pred)` or `filter(pred, gen)` will filter elements.
     * - `gen.take(n)` or `take(n, gen)` will limit the number of values (e.g.
     *    `fibonacci<int>().take(4)`).
     * - `gen.only()` or `only(gen)` will return the first value if and only if
     * the collection has exactly one more element.
     *      - The free function version (`only(c)`) will work on other
     * collections as well.
     * - `gen.empty()` or `empty(gen)` will return true if `gen` is empty.
     *      - The free function version also works on other collections.
     *
     * ## Implementing Generator functions
     *
     * A coroutine in C++ is a function that has at least one of `co_yield`,
     * `co_await`, and `co_return`. A coroutine with return type `Generator<T>`
     * supports:
     *
     *  - `co_return (void);` if you want to return early or create an empty
     *  sequence.
     *  - `co_yield t;` where `t` is of type `T`
     *  - `co_yield r;` where `r` is an input range of `T`. This includes:
     *      - `Generator<T>`
     *      - `std::vector<T>`
     *      - `std::set<t>`
     *      - `std::initializer_list<T>`
     *  -  It does *not* support `co_await`.
     *
     * Note that Generator<T> is movable but not copyable (what would it mean to
     * copy a function in the middle of execution, anyway?) This means that if
     * you `co_yield` an lvalue Generator (i.e. a named variable) you will have
     * to move it. Omitting this will result in a compiler error.
     *
     * Does not compile:
     * ```
     * Generator<int> collectionTwo()
     * {
     *     auto gen = someCollection();
     *     co_yield gen;
     * }
     * ```
     * This compiles:
     * ```
     * Generator<int> collectionTwo()
     * {
     *     auto gen = someCollection();
     *     co_yield std::move(gen);
     * }
     * ```
     * So does this:
     * ```
     * Generator<int> collectionTwo()
     * {
     *     co_yield someCollection();
     * }
     * ```
     *
     * Also note that if you `co_yield` a copyable lvalue (e.g. `std::vector`),
     * it will copy by default, which may not be what you want for performance.
     *
     * Makes a copy of `vec`:
     * ```
     * Generator<int> collectionTwo()
     * {
     *     std::vector<int> vec(4, 0);
     *     co_yield vec;
     * }
     * ```
     * Moves from `vec`:
     * ```
     * Generator<int> collectionTwo()
     * {
     *     std::vector<int> vec(4, 0);
     *     co_yield std::move(vec);
     * }
     * ```
     * No extra copies:
     * ```
     * Generator<int> collectionTwo()
     * {
     *     co_yield std::vector<int>(4, 0);
     * }
     * ```
     *
     * ## Implementation of Generator class template
     *
     * The `promise` object of a coroutine is the primary interface with the
     * C++ runtime.  The promise object of a coroutine with return type `X` is
     *  `X::promise_type` (or another type that is configured via traits).
     *
     * The `state()` functions of the Generator, promise_type, and Iterator
     * classes will report what state the generator is currently in.  Note:
     *
     *  - The `HasCopiedValue` state is particular to one iterator. It is also not
     *    externally visible since the `std::common_iterator` adaptor does not
     *    expose it.
     *  - Generally, the coroutine handle (and the promise object with it) are
     *    destroyed once the coroutine returns, so it does not see the Done
     *    state.
     *
     * \dot
     * digraph {
     *     rankdir=LR
     *
     *     Call [label="Generator Called" shape=box];
     *
     *     NoValue [label="No Value"];
     *     HasValue [label="Has Value"];
     *     RangeEx [label="Executing Range" shape=diamond]
     *     HasRange [ label="Has Range"];
     *     HasRangeValue[label="Has Range Value"];
     *     Executing [shape=diamond]
     *
     *     HasRangeValue -> Executing [style=invis]
     *
     *     Done[label=Done shape=box];
     *
     *     {
     *         rank=max
     *         RangeEx
     *         Done
     *     }
     *
     *     {
     *         rank=same
     *         Executing
     *         HasRangeValue
     *         HasRange
     *         HasValue
     *     }
     *
     *     Call -> NoValue;
     *     NoValue  -> Executing [label="it != gen.end(), *it, ++it"];
     *     HasValue -> NoValue [label="++it"];
     *     Executing -> RangeEx [label="co_yield range"];
     *     HasRangeValue -> HasRange[label="++it"];
     *
     *     HasRange -> RangeEx [label="it != gen.end(), *it, ++it"]
     *     RangeEx -> HasRangeValue [label="rangeIt != range.end()"]
     *     RangeEx -> Executing [label="rangeIt == range.end()"]
     *     Executing -> HasValue [label="co_yield value"];
     *     Executing -> Done [label="co_return"];
     * }
     * \enddot
     *
     */
    template <std::movable T>
    class Generator
    {
    public:
        /**
         * \brief The type of the promise object for a Generator coroutine.
         *
         * This class acts as the interface to the C++ coroutine runtime.  It
         * receives and stores yielded values (and ranges) from the coroutine,
         * and provides them to the iterator objects.  It maintains the primary
         * state of whether we currently have a value or not, and whether we
         * currently have a range or not.
         */
        class promise_type
        {
        public:
            /****
             * Required interface with C++ runtime
             ****/

            Generator<T> get_return_object();

            static constexpr void return_void() noexcept {}
            constexpr void        unhandled_exception() noexcept;

            static constexpr std::suspend_always initial_suspend() noexcept
            {
                return {};
            }
            static constexpr std::suspend_always final_suspend() noexcept
            {
                return {};
            }

            constexpr std::suspend_always yield_value(T v) noexcept;

            template <CInputRangeOf<T> ARange>
            constexpr std::suspend_always yield_value(ARange&& r) noexcept;

            constexpr std::suspend_always yield_value(std::initializer_list<T> r) noexcept;

            /****
             * Implementation & Interface
             ****/

            void check_exception() const;

            constexpr GeneratorState state() const;

            constexpr std::optional<T> const& value() const;

            void discard_value();
            void advance_range();

        private:
            mutable std::optional<T>  m_value;
            std::unique_ptr<Range<T>> m_range;

            mutable std::exception_ptr m_exception = nullptr;
        };

        /**
         * \brief Iterator logic for Generator coroutines.
         *
         * Implements the basic logic of:
         *  - When to discard the promise's currently held value (if there is one)
         *  - When to resume the coroutine
         *  - When to increment a range held in the promise
         *  - When we have finished the coroutine.
         *
         * Note that this is not exactly like iterators in e.g. std::vector,
         * as from this class's perspective, the end() iterator is of type
         * `std::default_sentinel_t`, not `Iterator`.  The
         * `std::common_iterator<>` adaptor provides that functionality.
         */
        class Iterator
        {
        public:
            // Required for STL iterator adaptors.  Not actually required to provide the - operator though.
            using difference_type = std::ptrdiff_t;
            using value_type      = T;

            using Handle = std::coroutine_handle<promise_type>;

            Iterator& operator++();
            Iterator  operator++(int);

            T const* get() const;
            T const& operator*() const;
            T const* operator->() const;

            bool operator==(std::default_sentinel_t) const;
            bool operator!=(std::default_sentinel_t) const;

            friend bool operator==(std::default_sentinel_t const& a, Iterator const& b)
            {
                return b == a;
            }

            Iterator();
            // cppcheck-suppress noExplicitConstructor
            Iterator(Handle const& coroutine);
            // cppcheck-suppress noExplicitConstructor
            Iterator(T value);

        private:
            bool           isDone() const;
            GeneratorState state() const;

            std::optional<T> m_value;
            mutable Handle   m_coroutine;
        };

        using iterator   = std::common_iterator<Iterator, std::default_sentinel_t>;
        using Handle     = std::coroutine_handle<promise_type>;
        using value_type = T;

        // cppcheck-suppress noExplicitConstructor
        explicit Generator(Handle const& coroutine);
        Generator() {}
        ~Generator();

        Generator(const Generator&) = delete;
        Generator(Generator&& other) noexcept;

        Generator& operator=(const Generator&) = delete;

        Generator& operator=(Generator&& rhs) noexcept;

        // cppcheck-suppress functionConst
        // cppcheck-suppress functionStatic
        constexpr iterator begin();
        // cppcheck-suppress functionConst
        // cppcheck-suppress functionStatic
        constexpr iterator end();

        constexpr GeneratorState state() const;

        /**
         * Returns a `Container<T>` constructed with `begin()` and `end()` as arguments.
         * Useful for storing the output of a Generator into e.g. a std::vector or std::set.
         * Be careful that the Generator will exit and will not yield an infinite sequence.
         */
        template <template <typename...> typename Container>
        constexpr Container<T> to();

        /**
         * Yields values that satisfy the predicate.
         */
        template <std::predicate<T> Predicate>
        Generator<T> filter(Predicate predicate);

        /**
         * Yields the result of invoking 'func' on each value.
         */
        template <std::invocable<T> Func>
        Generator<std::invoke_result_t<Func, T>> map(Func func);

        /**
         * Yields the first `n` values from `this`.
         */
        Generator<T> take(size_t n);

        /**
         * Returns first value from *this, if *this only contains one value.
         * Will return empty otherwise.
         *
         * *this will contain an empty generator afterwards.
         */
        std::optional<T> only();

        /**
         * Returns true if `*this` is an empty generator.
         */
        constexpr bool empty();

    private:
        Handle m_coroutine;
    };

    template <typename T, std::predicate<T> Predicate>
    Generator<T> filter(Predicate predicate, Generator<T> gen);

    template <typename T, std::invocable<T> Func>
    Generator<std::invoke_result_t<Func, T>> map(Func func, Generator<T> gen);

    /**
     * Yields the first `n` values from `gen`
     */
    template <typename T>
    Generator<T> take(size_t n, Generator<T> gen);

    /**
     * @brief Return first value of range.
     * If the range does not contain a single value, returns empty.
     *
     * Will consume items if called on a Generator.
     *
     * If called on a Generator, will consume items from it.
     * The implementation consumes items from the generator; so
     * will not work on persistent generators.  Currently the copy
     * constructor of Generator is deleted, and therefore you can
     * not call this on persistent generators.
     */
    template <std::ranges::input_range Range>
    inline std::optional<std::ranges::range_value_t<Range>> only(Range range);

    /**
     * @brief True if the generator is empty.
     *
     * The implementation consumes items from the generator, so
     * will not work on persistent generators.
     *
     * Generator is non-copyable and so this cannot be called on an lvalue
     * generator. Currently the copy constructor of Generator is deleted, and
     * therefore you cannot call this on persistent generators.
     */
    template <std::ranges::input_range Range>
    inline constexpr bool empty(Range range);

}

#include <rocRoller/Utilities/Generator_impl.hpp>
