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
// normal process (the documented, sanctioned dev/deploy workflow is
// unchanged) but refuse them when the real and effective user/group IDs
// differ, i.e. exactly the set-uid/set-gid escalation vector the finding
// describes. This is implemented header-only (rather than in a .cpp) so that
// the library sites and the CI regression test share one definition without
// depending on the library's hidden-visibility symbols.

#include <cstdlib>

#if !defined(_WIN32)
#include <unistd.h>
#endif

// True when the process is running with elevated privileges (set-user-ID or
// set-group-ID), such that untrusted environment must not be honored for
// security-sensitive lookups. Windows has no set-uid/set-gid concept, so this
// is always false there. In an ordinary (non-privileged) process this returns
// false, which is what makes it testable without special privileges.
//
// This is the one place that needs a platform guard; the helpers below are
// plain C++ expressed in terms of it.
inline bool rocblaslt_process_is_privileged()
{
#if defined(_WIN32)
    return false;
#else
    return ::getuid() != ::geteuid() || ::getgid() != ::getegid();
#endif
}

// Like std::getenv, but returns nullptr when the process is privileged (see
// rocblaslt_process_is_privileged). Use this for any environment variable that
// selects a filesystem path from which code objects or libraries are loaded.
inline const char* rocblaslt_secure_getenv(const char* name)
{
    if(name == nullptr)
        return nullptr;
    if(rocblaslt_process_is_privileged())
        return nullptr;
    return std::getenv(name);
}

// True when `name` is present in the environment but rocblaslt_secure_getenv is
// deliberately refusing it because the process is privileged. This is intended
// only for producing a diagnostic log so the security-driven behavior change is
// discoverable; callers must not use it to actually honor the override.
inline bool rocblaslt_env_suppressed_for_security(const char* name)
{
    if(name == nullptr)
        return false;
    if(!rocblaslt_process_is_privileged())
        return false;
    return std::getenv(name) != nullptr;
}
