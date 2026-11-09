// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Windows-only build-time check: verifies that each DLL named on the command line
// has an embedded PE VERSIONINFO resource whose FILEVERSION matches the configured project
// version, whose PRODUCTVERSION agrees with it, and whose descriptive string fields are non-empty.
// Standalone (no gtest) since it has no HIP/GPU dependency and only needs to run once per link.
//
// Reads the resource directory off disk via the version API, which maps the image as data: the
// DLL's DllMain never runs and its imports are never resolved, so no HIP runtime, driver, or GPU
// is required. That is what lets this run on the CPU-only Windows build machines.
//
// Uses wmain/wchar_t paths throughout: build directories routinely contain non-ASCII characters,
// and narrowing them would corrupt the path and surface as a bogus "no VERSIONINFO" failure.
//
// Silent on success: this runs as a POST_BUILD step on every DLL link, so printing the whole
// resource each time would bury the build output. On failure it dumps every field it read, which
// is what you actually want for diagnosis. Pass --verbose to dump them on success too.
//
// Usage: win_versioninfo_check.exe [--verbose] <expected-major.minor.patch> <dll-path> [...]

#include <windows.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

// Holds "65535.65535.65535" (17 chars) with room to spare; the components are 16-bit fields of
// VS_FIXEDFILEINFO, so the formatted length is bounded and cannot overflow this.
constexpr size_t kVersionBufferChars = 64;

const wchar_t* const kRequiredFields[] = {
    L"CompanyName", L"FileDescription", L"FileVersion", L"ProductName", L"ProductVersion"};

// Returns the StringFileInfo field's value, or an empty string if it is absent or blank.
std::wstring QueryStringField(void* block, const wchar_t* field_name)
{
    void* value            = nullptr;
    UINT value_len         = 0;
    std::wstring sub_block = L"\\StringFileInfo\\040904B0\\" + std::wstring(field_name);
    if(!VerQueryValueW(block, sub_block.c_str(), &value, &value_len) || value_len == 0)
    {
        return {};
    }
    return std::wstring(static_cast<const wchar_t*>(value));
}

void FormatVersion(wchar_t (&out)[kVersionBufferChars], DWORD ms, DWORD ls)
{
    swprintf(out, kVersionBufferChars, L"%u.%u.%u", HIWORD(ms), LOWORD(ms), HIWORD(ls));
}

// Dumps everything read from the resource. Used on failure, and on success under --verbose.
void DumpResource(const std::wstring& dll_path,
                  const wchar_t* actual_version,
                  const wchar_t* expected_version,
                  const std::vector<std::wstring>& values,
                  FILE* out)
{
    fwprintf(out,
             L"%ls: FILEVERSION = %ls (expected %ls)\n",
             dll_path.c_str(),
             actual_version,
             expected_version);
    for(size_t i = 0; i < values.size(); ++i)
    {
        fwprintf(out,
                 L"  %ls = %ls\n",
                 kRequiredFields[i],
                 values[i].empty() ? L"<missing or empty>" : values[i].c_str());
    }
}

bool CheckOneDll(const std::wstring& dll_path, const std::wstring& expected_version, bool verbose)
{
    DWORD handle    = 0;
    DWORD info_size = GetFileVersionInfoSizeW(dll_path.c_str(), &handle);
    if(info_size == 0)
    {
        fwprintf(stderr,
                 L"FAIL %ls: GetFileVersionInfoSizeW failed (error %lu) - no VERSIONINFO "
                 L"resource embedded\n",
                 dll_path.c_str(),
                 GetLastError());
        return false;
    }

    std::vector<char> buffer(info_size);
    if(!GetFileVersionInfoW(dll_path.c_str(), handle, info_size, buffer.data()))
    {
        fwprintf(stderr,
                 L"FAIL %ls: GetFileVersionInfoW failed (error %lu)\n",
                 dll_path.c_str(),
                 GetLastError());
        return false;
    }

    VS_FIXEDFILEINFO* fixed_info = nullptr;
    UINT fixed_info_len          = 0;
    if(!VerQueryValueW(
           buffer.data(), L"\\", reinterpret_cast<void**>(&fixed_info), &fixed_info_len) ||
       fixed_info == nullptr)
    {
        fwprintf(stderr,
                 L"FAIL %ls: VerQueryValueW(root) failed - no VS_FIXEDFILEINFO\n",
                 dll_path.c_str());
        return false;
    }

    wchar_t actual_version[kVersionBufferChars];
    FormatVersion(actual_version, fixed_info->dwFileVersionMS, fixed_info->dwFileVersionLS);

    std::vector<std::wstring> values;
    values.reserve(std::size(kRequiredFields));
    for(const wchar_t* field : kRequiredFields)
    {
        values.push_back(QueryStringField(buffer.data(), field));
    }

    // Collect every problem before reporting, so one run shows all of them rather than making the
    // caller rebuild to discover the next.
    std::vector<std::wstring> problems;
    if(expected_version != actual_version)
    {
        problems.push_back(L"FILEVERSION does not match the configured project version");
    }

    // PRODUCTVERSION is a separate pair of DWORDs in VS_FIXEDFILEINFO and nothing in the .rc
    // template guarantees the two stay in step, so a DLL could carry a correct FILEVERSION
    // alongside a stale or zeroed PRODUCTVERSION. Pin them together.
    if(fixed_info->dwProductVersionMS != fixed_info->dwFileVersionMS ||
       fixed_info->dwProductVersionLS != fixed_info->dwFileVersionLS)
    {
        wchar_t product_version[kVersionBufferChars];
        FormatVersion(
            product_version, fixed_info->dwProductVersionMS, fixed_info->dwProductVersionLS);
        problems.push_back(std::wstring(L"PRODUCTVERSION (") + product_version +
                           L") does not match FILEVERSION");
    }

    for(size_t i = 0; i < values.size(); ++i)
    {
        if(values[i].empty())
        {
            problems.push_back(std::wstring(L"StringFileInfo field '") + kRequiredFields[i] +
                               L"' is missing or empty");
        }
    }

    if(problems.empty())
    {
        if(verbose)
        {
            DumpResource(dll_path, actual_version, expected_version.c_str(), values, stdout);
        }
        return true;
    }

    fwprintf(stderr, L"FAIL %ls: embedded VERSIONINFO is not valid\n", dll_path.c_str());
    for(const std::wstring& problem : problems)
    {
        fwprintf(stderr, L"  - %ls\n", problem.c_str());
    }
    DumpResource(dll_path, actual_version, expected_version.c_str(), values, stderr);
    return false;
}

} // namespace

int wmain(int argc, wchar_t** argv)
{
    int arg      = 1;
    bool verbose = false;
    if(arg < argc && std::wstring(argv[arg]) == L"--verbose")
    {
        verbose = true;
        ++arg;
    }

    if(argc - arg < 2)
    {
        fwprintf(
            stderr,
            L"usage: %ls [--verbose] <expected-major.minor.patch> <dll-path> [<dll-path> ...]\n"
            L"(need at least one DLL path - zero DLLs is a configuration bug, not a pass)\n",
            argc > 0 ? argv[0] : L"win_versioninfo_check");
        return 1;
    }

    const std::wstring expected_version = argv[arg++];
    bool any_failed                     = false;
    int checked                         = 0;

    for(int i = arg; i < argc; ++i)
    {
        if(!CheckOneDll(argv[i], expected_version, verbose))
        {
            any_failed = true;
        }
        ++checked;
    }

    if(any_failed)
    {
        fwprintf(stderr, L"win_versioninfo_check: FAILED (%d file(s) checked)\n", checked);
        return 1;
    }

    if(verbose)
    {
        fwprintf(stdout, L"win_versioninfo_check: PASSED (%d file(s) checked)\n", checked);
    }
    return 0;
}
