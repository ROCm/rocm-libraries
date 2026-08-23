// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "KpackArchive.hpp"

#include <algorithm>
#include <string_view>
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

/// Does `data` begin with a container hipModuleLoadData can be expected to accept?
///
/// The reader reports success on a TOC entry whose `offset`/`size`/`ordinal` keys are
/// absent, having read those fields uninitialized, so a well-formed request can come
/// back holding an unrelated entry's bytes. Nothing in the reported status distinguishes
/// that from a correct hit. Checking the magic turns the common case of it -- bytes that
/// are not a code object at all -- into a named DECOMPRESS failure here, rather than an
/// opaque rejection inside HIP or, worse, a successful load of the wrong kernel.
///
/// Two containers are legitimate: a bare ELF, and a clang offload bundle, which is what
/// hipcc emits and what the packer stores.
bool hasCodeObjectMagic(const void* data, size_t size)
{
    static constexpr std::string_view ELF_MAGIC{"\177ELF", 4};
    static constexpr std::string_view BUNDLE_MAGIC{"__CLANG_OFFLOAD_BUNDLE__"};

    const std::string_view bytes(static_cast<const char*>(data),
                                 std::min(size, BUNDLE_MAGIC.size()));
    const auto hasPrefix = [bytes](std::string_view prefix) {
        return bytes.size() >= prefix.size() && bytes.compare(0, prefix.size(), prefix) == 0;
    };
    return hasPrefix(ELF_MAGIC) || hasPrefix(BUNDLE_MAGIC);
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
    //
    // The reader parses the TOC behind this call and casts msgpack values without
    // checking their type first, which throws msgpack::type_error on a wrong-typed
    // field. That derives from std::bad_cast, not std::runtime_error, so no catch
    // upstream intercepts it, and it would otherwise unwind out of a function declared
    // to report failure by returning false.
    kpack_error_t status = KPACK_ERROR_INVALID_FORMAT;
    try
    {
        status = kpack_open(path.string().c_str(), &archive);
    }
    catch(...)
    {
        error = makeError(KpackLoadStage::OPEN_ARCHIVE, KPACK_ERROR_MSGPACK_PARSE_FAILED);
        return false;
    }

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
    // Guarded for the same reason as kpack_open: decompression reads TOC-derived sizes
    // and can allocate against them, so a malformed archive reaches here as an
    // exception rather than a status.
    kpack_error_t status = KPACK_ERROR_DECOMPRESSION_FAILED;
    try
    {
        status = kpack_get_kernel(asArchive(_archive), tocKey.c_str(), arch.c_str(), &data, &size);
    }
    catch(...)
    {
        error = makeError(KpackLoadStage::DECOMPRESS, KPACK_ERROR_DECOMPRESSION_FAILED);
        return false;
    }

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

    if(!hasCodeObjectMagic(codeObject.data(), codeObject.size()))
    {
        error = makeError(KpackLoadStage::DECOMPRESS, KPACK_ERROR_INVALID_METADATA);
        return false;
    }

    return true;
}

} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
