// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Security-aware environment-variable access for code-object load paths.
//
// ROCM-26729 / SEC-00896 (Untrusted Search Path): hipBLASLt reads
// HIPBLASLT_TENSILE_LIBPATH and HIPBLASLT_EXT_OP_LIBRARY_PATH and uses them,
// unmodified, as the directory it loads GPU code objects (.hsaco/.co) and
// msgpack solution libraries from. If a hipBLASLt-using process runs with
// elevated privileges (set-user-ID / set-group-ID) and inherits a hostile
// environment, an attacker who can only set env vars can redirect kernel
// loading to an attacker-controlled directory and execute arbitrary GPU ISA.
//
// The remediation mirrors glibc's secure_getenv: honor these overrides for a
// normal process (the documented, sanctioned dev/deploy workflow is unchanged)
// but refuse them when the process was started in a kernel-designated secure
// context, i.e. exactly the escalation vector the finding describes. This is
// implemented header-only (rather than in a .cpp) so that the library sites and
// the CI regression test share one definition without depending on the
// library's hidden-visibility symbols.

#include <cstdlib>

#if defined(_WIN32)
// No POSIX credential headers on Windows.
#elif defined(__GLIBC__)
#include <sys/auxv.h>
#else
#include <unistd.h>
#endif

// True when the process is running in a security-sensitive context in which
// untrusted environment must not be honored.
//
//   - On glibc this is getauxval(AT_SECURE), the exact signal glibc's
//     secure_getenv keys on: the kernel sets AT_SECURE for set-user-ID /
//     set-group-ID execs AND other credential-changing execs such as file
//     capabilities, so this covers more than a bare real/effective ID mismatch.
//   - On other POSIX platforms we fall back to comparing real and effective
//     user/group IDs (the set-uid/set-gid case).
//   - Windows has no set-uid/set-gid concept, so this is always false.
//
// In an ordinary (non-privileged) process this returns false, which is what
// makes it testable without special privileges. This is the one place that
// needs a platform guard; the helpers below are plain C++ expressed in terms
// of it.
inline bool rocblaslt_process_is_privileged()
{
#if defined(_WIN32)
    return false;
#elif defined(__GLIBC__)
    return ::getauxval(AT_SECURE) != 0;
#else
    return ::getuid() != ::geteuid() || ::getgid() != ::getegid();
#endif
}

// Pure decision helpers, parameterized on the privilege state.
//
// The suppression policy ("refuse the override when privileged") is separated
// from the live OS privilege probe (rocblaslt_process_is_privileged) so the
// policy can be unit-tested for both privilege states in an ordinary,
// non-privileged CI process -- passing is_privileged=true exercises the
// security branch without needing an actual set-uid/set-gid harness. The
// production entry points below bind is_privileged to the real probe; only the
// probe itself (a thin getauxval/getuid call) then remains untestable without
// real privilege, and that is OS behavior rather than our logic.
inline const char* rocblaslt_secure_getenv_impl(const char* name, bool is_privileged)
{
    if(name == nullptr)
        return nullptr;
    if(is_privileged)
        return nullptr;
    return std::getenv(name);
}

inline bool rocblaslt_env_suppressed_for_security_impl(const char* name, bool is_privileged)
{
    if(name == nullptr)
        return false;
    if(!is_privileged)
        return false;
    return std::getenv(name) != nullptr;
}

// Like std::getenv, but returns nullptr when the process is privileged (see
// rocblaslt_process_is_privileged). Use this for any environment variable that
// selects a filesystem path from which code objects or libraries are loaded.
//
// A call site that also emits the suppression diagnostic should instead probe
// once with rocblaslt_process_is_privileged() and pass the result to both
// _impl helpers, so the common (override unset) path does not repeat the probe.
inline const char* rocblaslt_secure_getenv(const char* name)
{
    return rocblaslt_secure_getenv_impl(name, rocblaslt_process_is_privileged());
}

// True when `name` is present in the environment but rocblaslt_secure_getenv is
// deliberately refusing it because the process is privileged. This is intended
// only for producing a diagnostic log so the security-driven behavior change is
// discoverable; callers must not use it to actually honor the override.
inline bool rocblaslt_env_suppressed_for_security(const char* name)
{
    return rocblaslt_env_suppressed_for_security_impl(name, rocblaslt_process_is_privileged());
}
