// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "unique_path.hpp"
#include "unique_path_features.hpp"

namespace {

static void fail(int err, std::error_code* ec)
{
    if(ec == nullptr)
    {
        throw std::system_error(err, std::system_category(), "miopen::unique_path");
    }

    ec->assign(err, std::system_category());
}

#if defined(MIOPEN_WINDOWS_API) && defined(MIOPEN_FILESYSTEM_HAS_BCRYPT)
//! Converts NTSTATUS error codes to Win32 error codes for reporting
static DWORD translate_ntstatus(NTSTATUS status) noexcept
{
    // We have to cast to unsigned integral type to avoid signed overflow and narrowing conversion
    // in the constants.
    switch(static_cast<ULONG>(status))
    {
    case static_cast<ULONG>(STATUS_NO_MEMORY): return ERROR_OUTOFMEMORY;
    case static_cast<ULONG>(STATUS_BUFFER_OVERFLOW): return ERROR_BUFFER_OVERFLOW;
    case static_cast<ULONG>(STATUS_INVALID_HANDLE): return ERROR_INVALID_HANDLE;
    case static_cast<ULONG>(STATUS_INVALID_PARAMETER): return ERROR_INVALID_PARAMETER;
    case static_cast<ULONG>(STATUS_NO_MORE_FILES): return ERROR_NO_MORE_FILES;
    case static_cast<ULONG>(STATUS_NO_SUCH_DEVICE):
    case static_cast<ULONG>(STATUS_DEVICE_DOES_NOT_EXIST): return ERROR_DEV_NOT_EXIST;
    case static_cast<ULONG>(STATUS_NO_SUCH_FILE):
    case static_cast<ULONG>(STATUS_OBJECT_NAME_NOT_FOUND):
    case static_cast<ULONG>(STATUS_OBJECT_PATH_NOT_FOUND): return ERROR_FILE_NOT_FOUND;
    case static_cast<ULONG>(STATUS_SHARING_VIOLATION): return ERROR_SHARING_VIOLATION;
    case static_cast<ULONG>(STATUS_EAS_NOT_SUPPORTED): return ERROR_EAS_NOT_SUPPORTED;
    case static_cast<ULONG>(STATUS_ACCESS_DENIED): return ERROR_ACCESS_DENIED;
    case static_cast<ULONG>(STATUS_BAD_NETWORK_PATH): return ERROR_BAD_NETPATH;
    case static_cast<ULONG>(STATUS_BAD_NETWORK_NAME): return ERROR_BAD_NET_NAME;
    case static_cast<ULONG>(STATUS_DIRECTORY_NOT_EMPTY): return ERROR_DIR_NOT_EMPTY;
    case static_cast<ULONG>(STATUS_NOT_A_DIRECTORY):
        return ERROR_DIRECTORY; // The directory name is invalid
    case static_cast<ULONG>(STATUS_NOT_FOUND): return ERROR_NOT_FOUND;
    // map "invalid info class" to "not supported" as this error likely indicates that the kernel
    // does not support what we request
    case static_cast<ULONG>(STATUS_INVALID_INFO_CLASS):
    default: return ERROR_NOT_SUPPORTED;
    }
}
#endif // defined(MIOPEN_WINDOWS_API) && defined(MIOPEN_FILESYSTEM_HAS_BCRYPT)

#if defined(MIOPEN_POSIX_API) && !defined(MIOPEN_FILESYSTEM_HAS_ARC4RANDOM)

#define RETRY_IF_INTERRUPTED(result, error, operation, ...) \
    {                                                       \
        do                                                  \
        {                                                   \
            result = operation(__VA_ARGS__);                \
            error  = errno;                                 \
        } while(result == -1 && error == EINTR);            \
    }

//! Fills buffer with cryptographically random data obtained from /dev/(u)random
static int fill_random_dev_random(void* buf, size_t len)
{
    int file{};
    int err{};

    RETRY_IF_INTERRUPTED(file, err, ::open, "/dev/urandom", O_RDONLY | O_CLOEXEC);

    if(file == -1)
    {
        RETRY_IF_INTERRUPTED(file, err, ::open, "/dev/random", O_RDONLY | O_CLOEXEC);

        if(file == -1)
        {
            return err;
        }
    }

    size_t bytes_read = 0;

    while(bytes_read < len)
    {
        ssize_t n;

        RETRY_IF_INTERRUPTED(n, err, ::read, file, buf, len - bytes_read);

        if(n == -1) [[unlikely]]
        {
            close(file);
            return err;
        }

        bytes_read += n;
        buf = static_cast<char*>(buf) + n;
    }

    close(file);

    return 0;
}

#if defined(MIOPEN_FILESYSTEM_HAS_GETRANDOM) || defined(MIOPEN_FILESYSTEM_HAS_GETRANDOM_SYSCALL)

typedef int fill_random_t(void* buf, size_t len);

//! Pointer to the implementation of fill_random.
static std::atomic<fill_random_t*> fill_random = &fill_random_dev_random;

//! Fills buffer with cryptographically random data obtained from getrandom()
static int fill_random_getrandom(void* buf, size_t len)
{
    size_t bytes_read = 0u;

    while(bytes_read < len)
    {
#if defined(MIOPEN_FILESYSTEM_HAS_GETRANDOM)
        const ssize_t n = ::getrandom(buf, len - bytes_read, 0u);
#else
        const ssize_t n = ::syscall(SYS_getrandom, buf, len - bytes_read, 0u);
#endif
        if(n < 0) [[unlikely]]
        {
            const int err = errno;

            if(err == EINTR)
            {
                continue;
            }

            if(err == ENOSYS && bytes_read == 0u)
            {
                std::atomic_store_explicit(
                    &fill_random, fill_random_dev_random, std::memory_order_relaxed);
                return fill_random_dev_random(buf, len);
            }

            return err;
        }

        bytes_read += n;
        buf = static_cast<char*>(buf) + n;
    }

    return 0;
}

#endif // defined(MIOPEN_FILESYSTEM_HAS_GETRANDOM) ||
       // defined(MIOPEN_FILESYSTEM_HAS_GETRANDOM_SYSCALL)

#endif // defined(MIOPEN_POSIX_API) && !defined(MIOPEN_FILESYSTEM_HAS_ARC4RANDOM)

static void system_crypt_random(void* buf, size_t len, std::error_code* ec)
{
#if defined(MIOPEN_POSIX_API)

#if defined(MIOPEN_FILESYSTEM_HAS_GETRANDOM) || defined(MIOPEN_FILESYSTEM_HAS_GETRANDOM_SYSCALL)

    const int err = std::atomic_load_explicit(&fill_random, std::memory_order_relaxed)(buf, len);

    if(err != 0) [[unlikely]]
    {
        fail(err, ec);
    }

#elif defined(MIOPEN_FILESYSTEM_HAS_ARC4RANDOM)

    arc4random_buf(buf, len);

#else

    const int err = fill_random_dev_random(buf, len);

    if(err != 0) [[unlikely]]
    {
        fail(err, ec);
    }

#endif

#else // defined(MIOPEN_POSIX_API)

#if defined(MIOPEN_FILESYSTEM_HAS_BCRYPT)

    BCRYPT_ALG_HANDLE handle;
    NTSTATUS status = BCryptOpenAlgorithmProvider(&handle, BCRYPT_RNG_ALGORITHM, nullptr, 0);

    if(status != STATUS_SUCCESS) [[unlikely]]
    {
        fail(translate_ntstatus(status), ec);
        return;
    }

    status = BCryptGenRandom(handle, static_cast<PUCHAR>(buf), static_cast<ULONG>(len), 0);

    BCryptCloseAlgorithmProvider(handle, 0);

    if(status != STATUS_SUCCESS) [[unlikely]]
    {
        fail(translate_ntstatus(status), ec);
        return;
    }

#else // defined(MIOPEN_FILESYSTEM_HAS_BCRYPT)

    HCRYPTPROV handle;
    DWORD err = 0u;
    if(!CryptAcquireContextW(
           &handle, nullptr, nullptr, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT | CRYPT_SILENT))
        [[unlikely]]
    {
        err = GetLastError();
        fail(err, ec);
        return;
    }

    const BOOL gen_ok = CryptGenRandom(handle, static_cast<DWORD>(len), static_cast<BYTE*>(buf));

    if(!gen_ok) [[unlikely]]
    {
        err = GetLastError();
    }

    CryptReleaseContext(handle, 0);

    if(!gen_ok) [[unlikely]]
    {
        fail(err, ec);
        return;
    }

#endif // defined(MIOPEN_FILESYSTEM_HAS_BCRYPT)

#endif // defined(MIOPEN_POSIX_API)
}

#ifdef MIOPEN_WINDOWS_API
const constexpr wchar_t hex[]   = L"0123456789abcdef";
const constexpr wchar_t percent = L'%';
#else
const constexpr char hex[] = "0123456789abcdef";
const constexpr char percent = '%';
#endif

} // unnamed namespace

#if defined(linux) || defined(__linux) || defined(__linux__)

//! Initializes fill_random implementation pointer
static void
init_fill_random_impl(unsigned int major_ver, unsigned int minor_ver, unsigned int patch_ver)
{
#if defined(MIOPEN_FILESYSTEM_HAS_GETRANDOM) || defined(MIOPEN_FILESYSTEM_HAS_GETRANDOM_SYSCALL)
    fill_random_t* fr = &fill_random_dev_random;

    if(major_ver > 3u || (major_ver == 3u && minor_ver >= 17u))
    {
        fr = &fill_random_getrandom;
    }

    std::atomic_store_explicit(&fill_random, fr, std::memory_order_relaxed);
#endif
}

struct syscall_initializer
{
    syscall_initializer()
    {
        struct ::utsname system_info;

        if(uname(&system_info) == 0)
        {
            unsigned int major_ver = 0u, minor_ver = 0u, patch_ver = 0u;
            const int count =
                std::sscanf(system_info.release, "%u.%u.%u", &major_ver, &minor_ver, &patch_ver);

            if(count >= 3)
            {
                init_fill_random_impl(major_ver, minor_ver, patch_ver);
            }
        }
    }
};

MIOPEN_ATTRIBUTE_UNUSED MIOPEN_FILESYSTEM_ATTRIBUTE_RETAIN static const syscall_initializer
    syscall_init;

#endif // defined(linux) || defined(__linux) || defined(__linux__)

namespace miopen::detail {

fs::path unique_path(fs::path const& model, std::error_code* ec)
{
    // This function used wstring for fear of misidentifying
    // a part of a multibyte character as a percent sign.
    // However, double byte encodings only have 80-FF as lead
    // bytes and 40-7F as trailing bytes, whereas % is 25.
    // So, use string on POSIX and avoid conversions.

    fs::path::string_type s(model.native());

    char ran[16] = {}; // init to avoid clang static analyzer message

    const constexpr unsigned int max_nibbles = 2u * sizeof(ran); // 4-bits per nibble
    unsigned int nibbles_used                = max_nibbles;

    for(fs::path::string_type::size_type i = 0, n = s.size(); i < n; ++i)
    {
        if(s[i] == percent) // digit request
        {
            if(nibbles_used == max_nibbles)
            {
                system_crypt_random(ran, sizeof(ran), ec);

                if(ec && *ec)
                {
                    return fs::path();
                }

                nibbles_used = 0;
            }

            unsigned int c = ran[nibbles_used / 2u];
            c >>= 4u * (nibbles_used++ & 1u); // if odd, shift right 1 nibble
            s[i] = hex[c & 0xf];              // convert to hex digit and replace
        }
    }

    if(ec)
    {
        ec->clear();
    }

    return s;
}

} // namespace miopen::detail
