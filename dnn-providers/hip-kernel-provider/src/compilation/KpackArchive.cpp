// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "KpackArchive.hpp"

#include <utility>

#include <rocm_kpack/kpack.h>

namespace hip_kernel_provider::compilation
{

namespace
{

kpack_archive_t asArchive(void* handle)
{
    return static_cast<kpack_archive_t>(handle);
}

/// Spelling of a kpack_error_t, so a diagnostic reads "KPACK_ERROR_ARCH_NOT_FOUND (14)"
/// rather than a bare number. Unknown values fall through to the numeric form alone.
std::string errorName(kpack_error_t code)
{
    switch(code)
    {
    case KPACK_SUCCESS:
        return "KPACK_SUCCESS";
    case KPACK_ERROR_INVALID_ARGUMENT:
        return "KPACK_ERROR_INVALID_ARGUMENT";
    case KPACK_ERROR_FILE_NOT_FOUND:
        return "KPACK_ERROR_FILE_NOT_FOUND";
    case KPACK_ERROR_INVALID_FORMAT:
        return "KPACK_ERROR_INVALID_FORMAT";
    case KPACK_ERROR_UNSUPPORTED_VERSION:
        return "KPACK_ERROR_UNSUPPORTED_VERSION";
    case KPACK_ERROR_KERNEL_NOT_FOUND:
        return "KPACK_ERROR_KERNEL_NOT_FOUND";
    case KPACK_ERROR_DECOMPRESSION_FAILED:
        return "KPACK_ERROR_DECOMPRESSION_FAILED";
    case KPACK_ERROR_OUT_OF_MEMORY:
        return "KPACK_ERROR_OUT_OF_MEMORY";
    case KPACK_ERROR_NOT_IMPLEMENTED:
        return "KPACK_ERROR_NOT_IMPLEMENTED";
    case KPACK_ERROR_IO_ERROR:
        return "KPACK_ERROR_IO_ERROR";
    case KPACK_ERROR_MSGPACK_PARSE_FAILED:
        return "KPACK_ERROR_MSGPACK_PARSE_FAILED";
    case KPACK_ERROR_PATH_DISCOVERY_FAILED:
        return "KPACK_ERROR_PATH_DISCOVERY_FAILED";
    case KPACK_ERROR_INVALID_METADATA:
        return "KPACK_ERROR_INVALID_METADATA";
    case KPACK_ERROR_ARCHIVE_NOT_FOUND:
        return "KPACK_ERROR_ARCHIVE_NOT_FOUND";
    case KPACK_ERROR_ARCH_NOT_FOUND:
        return "KPACK_ERROR_ARCH_NOT_FOUND";
    // A code from a newer reader than the one pinned here still has to print; the numeric
    // form travels alongside this string at every call site.
    default:
        break;
    }
    return "unknown kpack error";
}

KpackError makeError(KpackLoadStage stage, kpack_error_t code)
{
    KpackError error;
    error.stage = stage;
    error.code = static_cast<int>(code);
    error.codeName = errorName(code);
    error.archiveAbsent
        = code == KPACK_ERROR_FILE_NOT_FOUND || code == KPACK_ERROR_ARCHIVE_NOT_FOUND;
    return error;
}

} // namespace

KpackCodeObject::KpackCodeObject(void* data, size_t size)
    : _data(data)
    , _size(size)
{
}

KpackCodeObject::~KpackCodeObject()
{
    if(_data != nullptr)
    {
        kpack_free_kernel(_data);
    }
}

KpackCodeObject::KpackCodeObject(KpackCodeObject&& other) noexcept
    : _data(std::exchange(other._data, nullptr))
    , _size(std::exchange(other._size, 0))
{
}

KpackCodeObject& KpackCodeObject::operator=(KpackCodeObject&& other) noexcept
{
    if(this != &other)
    {
        if(_data != nullptr)
        {
            kpack_free_kernel(_data);
        }
        _data = std::exchange(other._data, nullptr);
        _size = std::exchange(other._size, 0);
    }
    return *this;
}

KpackArchive::~KpackArchive()
{
    close();
}

KpackArchive::KpackArchive(KpackArchive&& other) noexcept
    : _archive(std::exchange(other._archive, nullptr))
{
}

KpackArchive& KpackArchive::operator=(KpackArchive&& other) noexcept
{
    if(this != &other)
    {
        close();
        _archive = std::exchange(other._archive, nullptr);
    }
    return *this;
}

void KpackArchive::close()
{
    if(_archive != nullptr)
    {
        kpack_close(asArchive(_archive));
        _archive = nullptr;
    }
}

bool KpackArchive::open(const std::filesystem::path& path, KpackError& error)
{
    close();

    kpack_archive_t archive = nullptr;
    // string() rather than the native wide form: the reader's C interface takes char*,
    // and the descriptor paths this sees are ASCII by construction.
    const kpack_error_t status = kpack_open(path.string().c_str(), &archive);
    if(status != KPACK_SUCCESS)
    {
        error = makeError(KpackLoadStage::OPEN_ARCHIVE, status);
        return false;
    }

    _archive = archive;
    return true;
}

bool KpackArchive::architectures(std::vector<std::string>& arches, KpackError& error) const
{
    arches.clear();

    size_t count = 0;
    const kpack_error_t status = kpack_get_architecture_count(asArchive(_archive), &count);
    if(status != KPACK_SUCCESS)
    {
        error = makeError(KpackLoadStage::ARCH_LOOKUP, status);
        return false;
    }

    arches.reserve(count);
    for(size_t index = 0; index < count; ++index)
    {
        const char* arch = nullptr;
        const kpack_error_t archStatus = kpack_get_architecture(asArchive(_archive), index, &arch);
        if(archStatus != KPACK_SUCCESS)
        {
            error = makeError(KpackLoadStage::ARCH_LOOKUP, archStatus);
            arches.clear();
            return false;
        }
        arches.emplace_back(arch == nullptr ? "" : arch);
    }

    return true;
}

bool KpackArchive::codeObject(const std::string& tocKey,
                              const std::string& arch,
                              KpackCodeObject& codeObject,
                              KpackError& error) const
{
    void* data = nullptr;
    size_t size = 0;
    const kpack_error_t status
        = kpack_get_kernel(asArchive(_archive), tocKey.c_str(), arch.c_str(), &data, &size);
    if(status != KPACK_SUCCESS)
    {
        // KERNEL_NOT_FOUND / ARCH_NOT_FOUND mean the entry is absent; everything else
        // at this point is the decompressor giving up on an entry it did find.
        const bool absent
            = status == KPACK_ERROR_KERNEL_NOT_FOUND || status == KPACK_ERROR_ARCH_NOT_FOUND;
        error
            = makeError(absent ? KpackLoadStage::ENTRY_LOOKUP : KpackLoadStage::DECOMPRESS, status);
        return false;
    }

    codeObject = KpackCodeObject(data, size);
    if(codeObject.empty())
    {
        // A success return with nothing in it would otherwise reach hipModuleLoadData
        // as a null pointer and fail there, one stage too late to name.
        error = makeError(KpackLoadStage::DECOMPRESS, KPACK_ERROR_DECOMPRESSION_FAILED);
        return false;
    }

    return true;
}

} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
