// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Windows-only build-time check (ROCM-20937): verifies that each DLL named on the command line
// has an embedded PE VERSIONINFO resource with a plausible FileVersion/ProductVersion and
// non-empty descriptive fields. Standalone (no gtest) since it has no HIP/GPU dependency and only
// needs to run once per build, not per-GPU.
//
// Usage: win_versioninfo_check.exe <expected-major.minor.patch> <dll-path> [<dll-path> ...]

#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {

bool CheckNonEmptyStringField(void* block,
                               const wchar_t* field_name,
                               const std::wstring& dll_path,
                               bool& ok)
{
    void* value    = nullptr;
    UINT value_len  = 0;
    std::wstring sub_block = L"\\StringFileInfo\\040904B0\\" + std::wstring(field_name);
    if(!VerQueryValueW(block, sub_block.c_str(), &value, &value_len) || value_len == 0 ||
       static_cast<const wchar_t*>(value)[0] == L'\0')
    {
        fwprintf(stderr, L"FAIL %ls: missing or empty StringFileInfo field '%ls'\n", dll_path.c_str(),
                 field_name);
        ok = false;
        return false;
    }
    fwprintf(stdout, L"  %ls = %ls\n", field_name, static_cast<const wchar_t*>(value));
    return true;
}

bool CheckOneDll(const std::wstring& dll_path, const std::string& expected_version, bool& any_failed)
{
    DWORD handle    = 0;
    DWORD info_size = GetFileVersionInfoSizeW(dll_path.c_str(), &handle);
    if(info_size == 0)
    {
        fwprintf(stderr, L"FAIL %ls: GetFileVersionInfoSizeW failed (error %lu) - no VERSIONINFO "
                         L"resource embedded\n",
                 dll_path.c_str(), GetLastError());
        any_failed = true;
        return false;
    }

    std::vector<char> buffer(info_size);
    if(!GetFileVersionInfoW(dll_path.c_str(), handle, info_size, buffer.data()))
    {
        fwprintf(stderr, L"FAIL %ls: GetFileVersionInfoW failed (error %lu)\n", dll_path.c_str(),
                 GetLastError());
        any_failed = true;
        return false;
    }

    VS_FIXEDFILEINFO* fixed_info = nullptr;
    UINT fixed_info_len          = 0;
    if(!VerQueryValueW(buffer.data(), L"\\", reinterpret_cast<void**>(&fixed_info), &fixed_info_len) ||
       fixed_info == nullptr)
    {
        fwprintf(stderr, L"FAIL %ls: VerQueryValueW(root) failed - no VS_FIXEDFILEINFO\n",
                 dll_path.c_str());
        any_failed = true;
        return false;
    }

    char actual_version[64];
    std::snprintf(actual_version, sizeof(actual_version), "%u.%u.%u",
                  HIWORD(fixed_info->dwFileVersionMS), LOWORD(fixed_info->dwFileVersionMS),
                  HIWORD(fixed_info->dwFileVersionLS));

    fwprintf(stdout, L"%ls: FILEVERSION = %hs (expected prefix %hs)\n", dll_path.c_str(),
             actual_version, expected_version.c_str());

    bool ok = true;
    if(expected_version != actual_version)
    {
        fwprintf(stderr, L"FAIL %ls: FILEVERSION '%hs' does not match expected '%hs'\n",
                 dll_path.c_str(), actual_version, expected_version.c_str());
        ok = false;
    }

    static const wchar_t* kRequiredFields[] = {L"CompanyName",    L"FileDescription", L"FileVersion",
                                                L"ProductName",   L"ProductVersion"};
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

std::wstring ToWide(const std::string& s)
{
    return std::wstring(s.begin(), s.end());
}

} // namespace

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        fprintf(stderr,
                "usage: %s <expected-major.minor.patch> <dll-path> [<dll-path> ...]\n"
                "(need at least one DLL path - zero DLLs is a configuration bug, not a pass)\n",
                argc > 0 ? argv[0] : "win_versioninfo_check");
        return 1;
    }

    const std::string expected_version = argv[1];
    bool any_failed                    = false;
    int checked                        = 0;

    for(int i = 2; i < argc; ++i)
    {
        CheckOneDll(ToWide(argv[i]), expected_version, any_failed);
        ++checked;
    }

    if(any_failed)
    {
        fprintf(stderr, "win_versioninfo_check: FAILED (%d file(s) checked)\n", checked);
        return 1;
    }

    fprintf(stdout, "win_versioninfo_check: PASSED (%d file(s) checked)\n", checked);
    return 0;
}
