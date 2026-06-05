// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef STINKYTOFU_ERROR_HANDLING_HPP
#define STINKYTOFU_ERROR_HANDLING_HPP

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

/// @file ErrorHandling.hpp
/// @brief Three-tier error handling system for StinkyTofu
///
/// This file provides three error handling mechanisms:
/// 1. assert() - Debug-time precondition checks (disabled in release builds)
/// 2. Expected<T> - Type-safe error handling for recoverable errors
/// 3. STINKY_UNREACHABLE() - Mark impossible code paths
///
/// Decision Tree:
/// - Is this a bug in StinkyTofu? → STINKY_UNREACHABLE()
/// - Is this a development-time check? → assert()
/// - Is this a recoverable error? → Expected<T>

namespace stinkytofu {

// ============================================================================
// STINKY_UNREACHABLE - For code paths that should never execute
// ============================================================================

/// @brief Marks code paths that should never execute due to internal logic.
/// Unlike assert(), this remains in release builds and terminates immediately.
///
/// Use this when:
/// - Code should be unreachable due to prior validation
/// - Internal consistency checks fail (indicates a bug in StinkyTofu)
/// - After exhaustive switch/if-else (but compiler can't prove it)
///
/// Example:
/// @code
///   const HwInstDesc* desc = getMCIDByUOp(opcode, arch);
///   if (!desc) {
///       return Expected<...>::Error("Instruction not supported");
///   }
///   StinkyInstruction* inst = createInstruction(desc, ...);
///   if (!inst) {
///       // This should never happen after desc validation
///       STINKY_UNREACHABLE("createInstruction failed after getMCIDByUOp check");
///   }
/// @endcode
[[noreturn]] inline void stinky_unreachable_internal(const char* msg, const char* file,
                                                     unsigned line) {
    std::cerr << file << ":" << line << ": UNREACHABLE executed";
    if (msg) std::cerr << ": " << msg;
    std::cerr << "!\n";
    std::cerr.flush();
    std::abort();
}

#define STINKY_UNREACHABLE(msg) ::stinkytofu::stinky_unreachable_internal(msg, __FILE__, __LINE__)

/// @brief Report a fatal error and abort. Unlike assert(), this remains in
/// release builds and flushes stderr so the message is captured by callers
/// (e.g. stinkytofu-check via popen).
///
/// Use for input validation errors that cannot be recovered from:
/// @code
///   if (inconsistent_state) {
///       std::cerr << "details...\n";
///       report_fatal_error("inconsistent memory tokens in basic block");
///   }
/// @endcode
[[noreturn]] inline void report_fatal_error(const std::string& msg) {
    std::cerr << "FATAL ERROR: " << msg << "\n";
    std::cerr.flush();
    std::abort();
}

// ============================================================================
// Expected<T> - Type-safe error handling for recoverable errors
// ============================================================================

/// @brief A type that represents either a value (T) or an error (std::string).
/// Inspired by Rust's Result<T, E> and LLVM's Expected<T>.
///
/// Use this for:
/// - Architecture doesn't support an instruction
/// - User-provided invalid parameters
/// - Features not yet implemented
/// - Any recoverable error condition
///
/// Example:
/// @code
///   Expected<StinkyInstruction*> createInstruction(...) {
///       if (!arch_supports) {
///           return Expected<StinkyInstruction*>::Error("Not supported");
///       }
///       return inst;  // Success
///   }
///
///   // Usage:
///   auto result = createInstruction(...);
///   if (!result) {
///       std::cerr << "Error: " << result.getError() << "\n";
///       return;
///   }
///   use(*result);  // Guaranteed valid
/// @endcode
///
/// @tparam T The type of the value on success
template <typename T>
class Expected {
   public:
    /// @brief Construct a successful Expected from a value
    Expected(T value) : has_value_(true) {
        new (&storage_.value) T(std::move(value));
    }

    /// @brief Create a failed Expected with an error message
    /// @param msg Human-readable error description
    /// @return Expected in error state
    static Expected Error(std::string msg) {
        Expected result;
        result.has_value_ = false;
        new (&result.storage_.error) std::string(std::move(msg));
        return result;
    }

    /// @brief Destructor - properly destroys either value or error
    ~Expected() {
        if (has_value_) {
            storage_.value.~T();
        } else {
            storage_.error.~basic_string();
        }
    }

    /// @brief Move constructor
    Expected(Expected&& other) noexcept : has_value_(other.has_value_) {
        if (has_value_) {
            new (&storage_.value) T(std::move(other.storage_.value));
        } else {
            new (&storage_.error) std::string(std::move(other.storage_.error));
        }
    }

    /// @brief Move assignment operator
    Expected& operator=(Expected&& other) noexcept {
        if (this != &other) {
            this->~Expected();
            has_value_ = other.has_value_;
            if (has_value_) {
                new (&storage_.value) T(std::move(other.storage_.value));
            } else {
                new (&storage_.error) std::string(std::move(other.storage_.error));
            }
        }
        return *this;
    }

    // Deleted copy operations (move-only for performance)
    Expected(const Expected&) = delete;
    Expected& operator=(const Expected&) = delete;

    /// @brief Check if Expected contains a value (not an error)
    /// @return true if successful, false if error
    explicit operator bool() const {
        return has_value_;
    }

    /// @brief Check if Expected contains a value
    bool hasValue() const {
        return has_value_;
    }

    /// @brief Check if Expected contains an error
    bool hasError() const {
        return !has_value_;
    }

    /// @brief Access the value (lvalue reference)
    /// @warning Undefined behavior if Expected contains an error
    T& operator*() & {
        assert(has_value_ && "Accessing value of Expected in error state");
        return storage_.value;
    }

    /// @brief Access the value (const lvalue reference)
    /// @warning Undefined behavior if Expected contains an error
    const T& operator*() const& {
        assert(has_value_ && "Accessing value of Expected in error state");
        return storage_.value;
    }

    /// @brief Access the value (rvalue reference, for moving)
    /// @warning Undefined behavior if Expected contains an error
    T&& operator*() && {
        assert(has_value_ && "Accessing value of Expected in error state");
        return std::move(storage_.value);
    }

    /// @brief Access the value via pointer (for member access)
    /// @warning Undefined behavior if Expected contains an error
    T* operator->() {
        assert(has_value_ && "Accessing value of Expected in error state");
        return &storage_.value;
    }

    /// @brief Access the value via pointer (const, for member access)
    /// @warning Undefined behavior if Expected contains an error
    const T* operator->() const {
        assert(has_value_ && "Accessing value of Expected in error state");
        return &storage_.value;
    }

    /// @brief Get the error message
    /// @warning Undefined behavior if Expected contains a value
    /// @return The error message string
    const std::string& getError() const {
        assert(!has_value_ && "Accessing error of Expected in success state");
        return storage_.error;
    }

   private:
    /// @brief Private default constructor (used by Error() factory)
    Expected() = default;

    /// @brief Whether this Expected contains a value (true) or error (false)
    bool has_value_;

    /// @brief Storage for either value or error (union for space efficiency)
    union Storage {
        T value;
        std::string error;
        Storage() {}
        ~Storage() {}
    } storage_;
};

// ============================================================================
// Usage Guidelines
// ============================================================================

/// @example assert() - Debug-time precondition checks
/// @code
///   Expected<StinkyInstruction*> createInstruction(StinkyRegister dst) {
///       // Check preconditions in debug builds
///       assert(dst.isValid() && "Invalid destination register");
///
///       // ... implementation
///   }
/// @endcode
///
/// Cost: 0 cycles in release builds (-DNDEBUG)
/// Use for: Null pointers, invalid parameters that indicate caller bugs

/// @example Expected<T> - Recoverable errors
/// @code
///   Expected<std::vector<StinkyInstruction*>> SMulLOU32(...) {
///       const HwInstDesc* desc = getMCIDByUOp(GFX::s_mul_lo_u32, arch);
///       if (!desc) {
///           return Expected<...>::Error(
///               "s_mul_lo_u32 not supported (requires gfx1250+)");
///       }
///       return result;
///   }
///
///   // Caller handles error:
///   auto insts = tofu.SMulLOU32(...);
///   if (!insts) {
///       // Graceful fallback
///       return emitFallbackSequence();
///   }
/// @endcode
///
/// Cost: ~1-3 cycles (flag check)
/// Use for: Architecture limitations, unsupported features, user errors

/// @example STINKY_UNREACHABLE() - Impossible code paths
/// @code
///   const HwInstDesc* desc = getMCIDByUOp(opcode, arch);
///   if (!desc) {
///       return Expected<...>::Error("Instruction not supported");
///   }
///
///   // After validation above, this should never fail
///   StinkyInstruction* inst = createInstruction(desc, ...);
///   if (!inst) {
///       STINKY_UNREACHABLE("createInstruction failed after getMCIDByUOp");
///   }
/// @endcode
///
/// Cost: 0 cycles (compiles to trap instruction)
/// Use for: Internal consistency checks, unreachable code after validation

}  // namespace stinkytofu

#endif  // STINKYTOFU_ERROR_HANDLING_HPP
