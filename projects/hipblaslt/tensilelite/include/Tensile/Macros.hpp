// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// TENSILE_API controls C++ symbol visibility for TensileLite types and free
// functions whose declarations appear in this directory's public headers.
//
// TensileLite is built as an OBJECT library (`tensilelite-host`) and is
// statically composed into consumer shared libraries -- libhipblaslt.so today,
// and historically libhipsparselt.so as well. The exported ABI of those
// consumer DSOs is their own C surface (e.g. hipblasLt*); TensileLite's class
// and template definitions are deliberately NOT part of that ABI.
//
// CMake's `CXX_VISIBILITY_PRESET hidden` on `tensilelite-host` only governs
// the visibility of symbols emitted from TensileLite's own translation units.
// It does NOT govern what consumer translation units emit when they `#include`
// these headers. Inline methods, implicit template instantiations, and the
// vtable / typeinfo for any class used as a complete type in a consumer TU
// will be emitted into the consumer's object files under the CONSUMER's
// per-TU `-fvisibility=` setting.
//
// When two consumer DSOs both export the same TensileLite-internal C++
// mangled name with incompatible class layouts, ELF flat-namespace
// interposition causes one DSO's calls to resolve into the other DSO's
// definition and crash. This is the failure mode that was observed in
// March 2026 when libhipsparselt.so first shipped alongside libhipblaslt.so
// with a divergent ContractionProblemGemm layout.
//
// To prevent this at the source, TENSILE_API is defined as a class- and
// function-level hidden-visibility attribute. Class-level visibility
// attributes travel through the header into every consuming TU and OVERRIDE
// the consumer's per-TU `-fvisibility=` fallback for the marked type --
// including its vtable, typeinfo, inline method bodies, and implicit
// template instantiations.
//
// On compilers without the GCC visibility attribute (notably MSVC), this
// expands to nothing and visibility falls back to the platform default; on
// those platforms the relevant ABI hazard does not exist in the same form.

#if defined(__GNUC__) || defined(__clang__)
#define TENSILE_API __attribute__((visibility("hidden")))
#else
#define TENSILE_API
#endif

// TENSILE_HIDDEN_BEGIN / TENSILE_HIDDEN_END wrap a region of declarations
// (typically a `namespace TensileLite { ... }` block in a public header) and
// apply hidden visibility to every declaration inside, regardless of whether
// the declaration carries an explicit attribute. This is belt-and-braces for
// the per-symbol TENSILE_API marker above: it ensures that types added in the
// future without explicit annotation -- and the implicit weak symbols every
// class with virtual functions emits (vtable, typeinfo, typeinfo name) --
// also receive hidden visibility in consumer translation units.
//
// Use as:
//
//     TENSILE_HIDDEN_BEGIN
//     namespace TensileLite { /* declarations */ }
//     TENSILE_HIDDEN_END
//
// On compilers without GCC visibility pragmas, both macros expand to nothing.

#if defined(__GNUC__) || defined(__clang__)
#define TENSILE_HIDDEN_BEGIN _Pragma("GCC visibility push(hidden)")
#define TENSILE_HIDDEN_END   _Pragma("GCC visibility pop")
#else
#define TENSILE_HIDDEN_BEGIN
#define TENSILE_HIDDEN_END
#endif
