// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Windows-only build-time check (ROCM-20937): verifies that each DLL named on the command line
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
// Usage: win_versioninfo_check.exe <expected-major.minor.patch> <dll-path> [<dll-path> ...]

#include <windows.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

// Holds "65535.65535.65535" (17 chars) with room to spare; the components are 16-bit fields of
// VS_FIXEDFILEINFO, so the formatted length is bounded and cannot overflow this.
constexpr size_t kVersionBufferChars = 64;

bool CheckNonEmptyStringField(void* block,
                              const wchar_t* field_name,
                              const std::wstring& dll_path,
                              bool& ok)
{
    void* value            = nullptr;
    UINT value_len         = 0;
    std::wstring sub_block = L"\\StringFileInfo\\040904B0\\" + std::wstring(field_name);
    if(!VerQueryValueW(block, sub_block.c_str(), &value, &value_len) || value_len == 0 ||
       static_cast<const wchar_t*>(value)[0] == L'\0')
    {
        fwprintf(stderr,
                 L"FAIL %ls: missing or empty StringFileInfo field '%ls'\n",
                 dll_path.c_str(),
                 field_name);
        ok = false;
        return false;
    }
    fwprintf(stdout, L"  %ls = %ls\n", field_name, static_cast<const wchar_t*>(value));
    return true;
}

bool CheckOneDll(const std::wstring& dll_path,
                 const std::wstring& expected_version,
                 bool& any_failed)
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
        any_failed = true;
        return false;
    }

    std::vector<char> buffer(info_size);
    if(!GetFileVersionInfoW(dll_path.c_str(), handle, info_size, buffer.data()))
    {
        fwprintf(stderr,
                 L"FAIL %ls: GetFileVersionInfoW failed (error %lu)\n",
                 dll_path.c_str(),
                 GetLastError());
        any_failed = true;
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
        any_failed = true;
        return false;
    }

    wchar_t actual_version[kVersionBufferChars];
    swprintf(actual_version,
             kVersionBufferChars,
             L"%u.%u.%u",
             HIWORD(fixed_info->dwFileVersionMS),
             LOWORD(fixed_info->dwFileVersionMS),
             HIWORD(fixed_info->dwFileVersionLS));

    fwprintf(stdout,
             L"%ls: FILEVERSION = %ls (expected %ls)\n",
             dll_path.c_str(),
             actual_version,
             expected_version.c_str());

    bool ok = true;
    if(expected_version != actual_version)
    {
        fwprintf(stderr,
                 L"FAIL %ls: FILEVERSION '%ls' does not match expected '%ls'\n",
                 dll_path.c_str(),
                 actual_version,
                 expected_version.c_str());
        ok = false;
    }

    // PRODUCTVERSION is a separate pair of DWORDs in VS_FIXEDFILEINFO and nothing in the .rc
    // template guarantees the two stay in step, so a DLL could carry a correct FILEVERSION
    // alongside a stale or zeroed PRODUCTVERSION. Pin them together.
    if(fixed_info->dwProductVersionMS != fixed_info->dwFileVersionMS ||
       fixed_info->dwProductVersionLS != fixed_info->dwFileVersionLS)
    {
        fwprintf(stderr,
                 L"FAIL %ls: PRODUCTVERSION %u.%u.%u.%u does not match FILEVERSION %u.%u.%u.%u\n",
                 dll_path.c_str(),
                 HIWORD(fixed_info->dwProductVersionMS),
                 LOWORD(fixed_info->dwProductVersionMS),
                 HIWORD(fixed_info->dwProductVersionLS),
                 LOWORD(fixed_info->dwProductVersionLS),
                 HIWORD(fixed_info->dwFileVersionMS),
                 LOWORD(fixed_info->dwFileVersionMS),
                 HIWORD(fixed_info->dwFileVersionLS),
                 LOWORD(fixed_info->dwFileVersionLS));
        ok = false;
    }

    static const wchar_t* kRequiredFields[] = {
        L"CompanyName", L"FileDescription", L"FileVersion", L"ProductName", L"ProductVersion"};
    for(const wchar_t* field : kRequiredFields)
    {
        CheckNonEmptyStringField(buffer.data(), field, dll_path, ok);
    }

    if(!ok)
    {
        any_failed = true;
    }
    return ok;
}

} // namespace

int wmain(int argc, wchar_t** argv)
{
    if(argc < 3)
    {
        fwprintf(stderr,
                 L"usage: %ls <expected-major.minor.patch> <dll-path> [<dll-path> ...]\n"
                 L"(need at least one DLL path - zero DLLs is a configuration bug, not a pass)\n",
                 argc > 0 ? argv[0] : L"win_versioninfo_check");
        return 1;
    }

    const std::wstring expected_version = argv[1];
    bool any_failed                     = false;
    int checked                         = 0;

    for(int i = 2; i < argc; ++i)
    {
        CheckOneDll(argv[i], expected_version, any_failed);
        ++checked;
    }

    if(any_failed)
    {
        fwprintf(stderr, L"win_versioninfo_check: FAILED (%d file(s) checked)\n", checked);
        return 1;
    }

    fwprintf(stdout, L"win_versioninfo_check: PASSED (%d file(s) checked)\n", checked);
    return 0;
}
