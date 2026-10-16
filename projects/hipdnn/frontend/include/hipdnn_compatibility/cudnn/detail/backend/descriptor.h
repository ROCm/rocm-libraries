// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/detail/backend/attribute_table.h>
#include <hipdnn_compatibility/cudnn/detail/status_translation.h>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
{

// Shim-originated failure. Carries the cuDNN status directly so the entry-point
// catch ladder never has to translate it.
class ShimError : public std::runtime_error
{
public:
    ShimError(cudnnStatus_t status, const std::string& message)
        : std::runtime_error(message)
        , _status(status)
    {
    }

    cudnnStatus_t status() const noexcept
    {
        return _status;
    }

private:
    cudnnStatus_t _status;
};

// Converts a failed hipDNN call into a ShimError. The backend message is drained
// immediately because reading it clears it.
inline void checkHipdnn(hipdnnStatus_t status, const std::string& context)
{
    if(status == HIPDNN_STATUS_SUCCESS)
    {
        return;
    }
    std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> backendMessage{};
    hipdnn_frontend::detail::hipdnnBackend()->getLastErrorString(backendMessage.data(),
                                                                 backendMessage.size());
    throw ShimError(toCudnnStatus(status), context + ": " + backendMessage.data());
}

class Descriptor;

// "HDNNV8" plus a layout revision. A handle minted by a shim revision with a
// different Handle layout fails the magic check instead of being misread. Bump
// the low word whenever Handle or Descriptor changes layout.
inline constexpr std::uint64_t HANDLE_MAGIC = 0x4844'4E4E'5638'0001ULL;

// cudnnBackendDescriptor_t points at one of these. Each caller-visible handle
// owns one reference; descriptor-valued attributes hold further references to the
// same object, so destroying a handle never invalidates a descriptor stored in
// another descriptor.
struct Handle
{
    std::uint64_t magic;
    std::shared_ptr<Descriptor> object;
};

inline cudnnBackendDescriptor_t makeHandle(std::shared_ptr<Descriptor> object)
{
    return new Handle{HANDLE_MAGIC, std::move(object)};
}

// Returns nullptr for null or foreign pointers. The magic is read with memcpy so
// a non-Handle buffer is inspected without type-punning.
inline Handle* liveHandle(cudnnBackendDescriptor_t descriptor) noexcept
{
    if(descriptor == nullptr)
    {
        return nullptr;
    }
    std::uint64_t magic = 0;
    std::memcpy(&magic, descriptor, sizeof(magic));
    if(magic != HANDLE_MAGIC)
    {
        return nullptr;
    }
    auto* handle = static_cast<Handle*>(descriptor);
    return handle->object != nullptr ? handle : nullptr;
}

inline Handle& requireHandle(cudnnBackendDescriptor_t descriptor, const char* role)
{
    Handle* handle = liveHandle(descriptor);
    if(handle == nullptr)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string(role) + " is null or not a live cuDNN shim descriptor");
    }
    return *handle;
}

// Size in bytes of one element of the given attribute type.
inline std::size_t elementSize(cudnnBackendAttributeType_t type)
{
    switch(type)
    {
    case CUDNN_TYPE_BACKEND_DESCRIPTOR:
        return sizeof(cudnnBackendDescriptor_t);
    case CUDNN_TYPE_BOOLEAN:
        return sizeof(bool);
    case CUDNN_TYPE_CHAR:
        return sizeof(char);
    case CUDNN_TYPE_DOUBLE:
        return sizeof(double);
    case CUDNN_TYPE_FLOAT:
        return sizeof(float);
    case CUDNN_TYPE_FRACTION:
        return sizeof(cudnnFraction_t);
    case CUDNN_TYPE_HANDLE:
        return sizeof(cudnnHandle_t);
    case CUDNN_TYPE_INT32:
        return sizeof(std::int32_t);
    case CUDNN_TYPE_INT64:
        return sizeof(std::int64_t);
    case CUDNN_TYPE_VOID_PTR:
        return sizeof(void*);
    default:
        // Every remaining attribute type is a C enum; c_types.h asserts they are
        // all 4 bytes.
        if(isEnumAttributeType(type))
        {
            return sizeof(std::int32_t);
        }
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        "unknown attribute type " + std::to_string(static_cast<int>(type)));
    }
}

// One attribute value. `count` is the element count as set; CHAR values hold the
// bytes before the first NUL and report `bytes.size() + 1` on GET (0 if `count`
// is 0).
struct StoredAttribute
{
    cudnnBackendAttributeType_t type = CUDNN_TYPE_INT64;
    std::int64_t count = 0;
    std::vector<std::byte> bytes;
    std::vector<std::shared_ptr<Descriptor>> descriptors;
};

// Base of every shim descriptor. Owns the generic attribute store and the
// lifecycle; subclasses (one per descriptor type) add validation and lowering
// through the hooks below. Downcast with descriptorCast(), never dynamic_cast:
// consumers may build with RTTI disabled.
class Descriptor : public std::enable_shared_from_this<Descriptor>
{
public:
    explicit Descriptor(cudnnBackendDescriptorType_t type)
        : _type(type)
    {
    }

    virtual ~Descriptor() = default;

    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    Descriptor(Descriptor&&) = delete;
    Descriptor& operator=(Descriptor&&) = delete;

    cudnnBackendDescriptorType_t type() const noexcept
    {
        return _type;
    }

    bool isFinalized() const noexcept
    {
        return _finalized;
    }

    void setAttribute(cudnnBackendAttributeName_t name,
                      cudnnBackendAttributeType_t attributeType,
                      std::int64_t elementCount,
                      const void* arrayOfElements)
    {
        if(_finalized)
        {
            throw ShimError(CUDNN_STATUS_NOT_INITIALIZED,
                            "cannot set " + attributeName(name) + " on finalized "
                                + descriptorTypeName(_type));
        }
        const AttributeInfo info = lookup(name, attributeType);
        if(!info.settable)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            std::string(info.name) + " is computed at finalize and cannot be set");
        }
        // Upstream passes data() of possibly empty vectors, which may be null.
        if(arrayOfElements == nullptr && elementCount != 0)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            std::string(info.name) + ": arrayOfElements is null");
        }
        StoredAttribute value = decode(info, attributeType, elementCount, arrayOfElements);
        validateSet(name, value);
        _attributes.insert_or_assign(name, std::move(value));
    }

    void getAttribute(cudnnBackendAttributeName_t name,
                      cudnnBackendAttributeType_t attributeType,
                      std::int64_t requestedElementCount,
                      std::int64_t* elementCount,
                      void* arrayOfElements) const
    {
        if(!_finalized)
        {
            throw ShimError(CUDNN_STATUS_NOT_INITIALIZED,
                            "cannot get " + attributeName(name) + " before "
                                + descriptorTypeName(_type) + " is finalized");
        }
        const AttributeInfo info = lookup(name, attributeType);

        std::optional<StoredAttribute> defaulted;
        const StoredAttribute* value = findAttribute(name);
        if(value == nullptr)
        {
            defaulted = defaultAttribute(info);
            if(!defaulted)
            {
                throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                std::string(info.name) + " has not been set");
            }
            value = &*defaulted;
        }
        if(value->type != attributeType)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            std::string(info.name) + " holds " + attributeTypeName(value->type)
                                + ", requested " + attributeTypeName(attributeType));
        }

        // An empty CHAR value (never set, or set with count 0) has no NUL to count.
        const std::int64_t available = value->type == CUDNN_TYPE_CHAR && value->count > 0
                                           ? static_cast<std::int64_t>(value->bytes.size() + 1)
                                           : value->count;

        // Null array OR zero request is a count query (upstream fetchNotes passes
        // a non-zero count with a null array).
        if(arrayOfElements == nullptr || requestedElementCount == 0)
        {
            if(elementCount == nullptr)
            {
                throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                std::string(info.name) + ": count query with null elementCount");
            }
            *elementCount = available;
            return;
        }
        if(requestedElementCount < 0)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            std::string(info.name) + ": requestedElementCount is negative");
        }

        const std::int64_t written = std::min(requestedElementCount, available);
        const auto writtenSize = static_cast<std::size_t>(written);
        switch(value->type)
        {
        case CUDNN_TYPE_BACKEND_DESCRIPTOR:
            fillDescriptors(info, *value, writtenSize, arrayOfElements);
            break;
        case CUDNN_TYPE_CHAR:
        {
            if(writtenSize == 0)
            {
                break;
            }
            auto* out = static_cast<char*>(arrayOfElements);
            const std::size_t chars = writtenSize - 1;
            if(chars > 0)
            {
                std::memcpy(out, value->bytes.data(), chars);
            }
            out[chars] = '\0';
            break;
        }
        default:
            if(writtenSize > 0)
            {
                std::memcpy(
                    arrayOfElements, value->bytes.data(), writtenSize * elementSize(value->type));
            }
            break;
        }
        if(elementCount != nullptr)
        {
            *elementCount = written;
        }
    }

    void finalize()
    {
        if(_finalized)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            std::string(descriptorTypeName(_type)) + " is already finalized");
        }
        finalizeImpl();
        _finalized = true;
    }

protected:
    // Called after the generic table checks and decoding, before the value is
    // committed. Throw ShimError to reject.
    virtual void validateSet(cudnnBackendAttributeName_t /*name*/, const StoredAttribute& /*value*/)
    {
    }

    // Validates required attributes and precomputes lowering state. Throw
    // ShimError to reject; the descriptor stays unfinalized.
    virtual void finalizeImpl() = 0;

    // In-place GET of a descriptor-valued attribute: makes the caller-created
    // `target` (already checked to have this descriptor's type) observe this
    // finalized descriptor. Finalized descriptors are immutable, so aliasing is
    // equivalent to a copy and keeps object identity.
    virtual void fillInto(Handle& target)
    {
        target.object = shared_from_this();
    }

    const StoredAttribute* findAttribute(cudnnBackendAttributeName_t name) const
    {
        const auto it = _attributes.find(name);
        return it == _attributes.end() ? nullptr : &it->second;
    }

    bool hasAttribute(cudnnBackendAttributeName_t name) const
    {
        return _attributes.find(name) != _attributes.end();
    }

    // First element of a stored (or defaulted) value. T must match the stored
    // element size (e.g. int64_t for INT64, bool for BOOLEAN, the C enum for enum
    // types).
    template <typename T>
    std::optional<T> scalarAttribute(cudnnBackendAttributeName_t name) const
    {
        static_assert(std::is_trivially_copyable_v<T>);
        std::optional<StoredAttribute> defaulted;
        const StoredAttribute* value = storedOrDefault(name, defaulted);
        if(value == nullptr || value->count < 1)
        {
            return std::nullopt;
        }
        requireElementSize(name, *value, sizeof(T));
        T result;
        std::memcpy(static_cast<void*>(&result), value->bytes.data(), sizeof(T));
        return result;
    }

    template <typename T>
    std::vector<T> arrayAttribute(cudnnBackendAttributeName_t name) const
    {
        static_assert(std::is_trivially_copyable_v<T>);
        const StoredAttribute* value = findAttribute(name);
        if(value == nullptr || value->count == 0)
        {
            return {};
        }
        requireElementSize(name, *value, sizeof(T));
        std::vector<T> result(static_cast<std::size_t>(value->count));
        std::memcpy(result.data(), value->bytes.data(), result.size() * sizeof(T));
        return result;
    }

    // FLOAT or DOUBLE scalar (alpha/beta accept either), widened to double.
    std::optional<double> realAttribute(cudnnBackendAttributeName_t name) const
    {
        const StoredAttribute* value = findAttribute(name);
        if(value == nullptr || value->count < 1)
        {
            return std::nullopt;
        }
        if(value->type == CUDNN_TYPE_FLOAT)
        {
            float result = 0.0F;
            std::memcpy(&result, value->bytes.data(), sizeof(result));
            return static_cast<double>(result);
        }
        requireElementSize(name, *value, sizeof(double));
        double result = 0.0;
        std::memcpy(&result, value->bytes.data(), sizeof(result));
        return result;
    }

    std::optional<std::string> stringAttribute(cudnnBackendAttributeName_t name) const
    {
        const StoredAttribute* value = findAttribute(name);
        if(value == nullptr)
        {
            return std::nullopt;
        }
        return std::string(reinterpret_cast<const char*>(value->bytes.data()), value->bytes.size());
    }

    std::shared_ptr<Descriptor> descriptorAttribute(cudnnBackendAttributeName_t name) const
    {
        const StoredAttribute* value = findAttribute(name);
        if(value == nullptr || value->descriptors.empty())
        {
            return nullptr;
        }
        return value->descriptors.front();
    }

    std::vector<std::shared_ptr<Descriptor>>
        descriptorArrayAttribute(cudnnBackendAttributeName_t name) const
    {
        const StoredAttribute* value = findAttribute(name);
        return value == nullptr ? std::vector<std::shared_ptr<Descriptor>>{} : value->descriptors;
    }

    // Records a value computed during finalizeImpl (including GET-only
    // attributes). Same table checks as SET, minus the settable/finalized rules.
    void storeComputed(cudnnBackendAttributeName_t name,
                       cudnnBackendAttributeType_t attributeType,
                       std::int64_t elementCount,
                       const void* arrayOfElements)
    {
        const AttributeInfo info = lookup(name, attributeType);
        if(attributeType == CUDNN_TYPE_BACKEND_DESCRIPTOR)
        {
            throw ShimError(CUDNN_STATUS_INTERNAL_ERROR,
                            std::string(info.name) + ": use storeComputedDescriptors");
        }
        _attributes.insert_or_assign(name,
                                     decode(info, attributeType, elementCount, arrayOfElements));
    }

    void storeComputedDescriptors(cudnnBackendAttributeName_t name,
                                  std::vector<std::shared_ptr<Descriptor>> descriptors)
    {
        const AttributeInfo info = lookup(name, CUDNN_TYPE_BACKEND_DESCRIPTOR);
        checkCount(info, static_cast<std::int64_t>(descriptors.size()));
        for(std::size_t i = 0; i < descriptors.size(); ++i)
        {
            if(descriptors[i] == nullptr)
            {
                throw ShimError(CUDNN_STATUS_INTERNAL_ERROR,
                                std::string(info.name) + ": null computed descriptor");
            }
            checkReferent(info, *descriptors[i], i);
        }
        StoredAttribute value;
        value.type = CUDNN_TYPE_BACKEND_DESCRIPTOR;
        value.count = static_cast<std::int64_t>(descriptors.size());
        value.descriptors = std::move(descriptors);
        _attributes.insert_or_assign(name, std::move(value));
    }

private:
    // Owner and type checks shared by SET, GET and storeComputed.
    AttributeInfo lookup(cudnnBackendAttributeName_t name,
                         cudnnBackendAttributeType_t attributeType) const
    {
        const std::optional<AttributeInfo> found = attributeInfo(name);
        if(!found)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "unknown attribute " + std::to_string(static_cast<int>(name)));
        }
        const AttributeInfo& info = *found;
        if(info.owner != _type)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            std::string(info.name) + " is not an attribute of "
                                + descriptorTypeName(_type));
        }
        if(!info.accepts(attributeType))
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            std::string(info.name) + " does not accept "
                                + attributeTypeName(attributeType));
        }
        return info;
    }

    static void checkCount(const AttributeInfo& info, std::int64_t elementCount)
    {
        if(info.countRule == CountRule::SCALAR)
        {
            if(elementCount != 1)
            {
                throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                std::string(info.name) + " expects exactly 1 element, got "
                                    + std::to_string(elementCount));
            }
            return;
        }
        if(elementCount < 0 || elementCount > info.maxCount)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            std::string(info.name) + ": element count "
                                + std::to_string(elementCount) + " outside [0, "
                                + std::to_string(info.maxCount) + "]");
        }
    }

    static void
        checkReferent(const AttributeInfo& info, const Descriptor& referent, std::size_t index)
    {
        const std::string element = std::string(info.name) + "[" + std::to_string(index) + "]";
        switch(info.referentKind)
        {
        case ReferentKind::EXACT:
            if(referent.type() != info.referent)
            {
                throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                element + " must be " + descriptorTypeName(info.referent) + ", got "
                                    + descriptorTypeName(referent.type()));
            }
            break;
        case ReferentKind::ANY_OPERATION:
            if(!isOperationDescriptorType(referent.type()))
            {
                throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                element + " must be an operation descriptor, got "
                                    + descriptorTypeName(referent.type()));
            }
            break;
        case ReferentKind::UNAVAILABLE:
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            element + ": the referenced descriptor type is not supported");
        case ReferentKind::NONE:
        default:
            throw ShimError(CUDNN_STATUS_INTERNAL_ERROR,
                            element + ": attribute table has no referent rule");
        }
        if(!referent.isFinalized())
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            element + " (" + descriptorTypeName(referent.type())
                                + ") is not finalized");
        }
    }

    static StoredAttribute decode(const AttributeInfo& info,
                                  cudnnBackendAttributeType_t attributeType,
                                  std::int64_t elementCount,
                                  const void* arrayOfElements)
    {
        checkCount(info, elementCount);
        StoredAttribute value;
        value.type = attributeType;
        value.count = elementCount;
        const auto count = static_cast<std::size_t>(elementCount);

        switch(attributeType)
        {
        case CUDNN_TYPE_BACKEND_DESCRIPTOR:
        {
            const auto* handles = static_cast<const cudnnBackendDescriptor_t*>(arrayOfElements);
            value.descriptors.reserve(count);
            for(std::size_t i = 0; i < count; ++i)
            {
                Handle* handle = liveHandle(handles[i]);
                if(handle == nullptr)
                {
                    throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                    std::string(info.name) + "[" + std::to_string(i)
                                        + "] is null or not a live cuDNN shim descriptor");
                }
                checkReferent(info, *handle->object, i);
                value.descriptors.push_back(handle->object);
            }
            return value;
        }
        case CUDNN_TYPE_CHAR:
        {
            // The count may include a trailing NUL (upstream round-trips the
            // JSON buffer it read, NUL included); keep the bytes before it.
            const auto* chars = static_cast<const char*>(arrayOfElements);
            const std::size_t length
                = static_cast<std::size_t>(std::find(chars, chars + count, '\0') - chars);
            value.bytes.resize(length);
            if(length > 0)
            {
                std::memcpy(value.bytes.data(), chars, length);
            }
            return value;
        }
        case CUDNN_TYPE_BOOLEAN:
        {
            // Normalize so typed reads as bool are always 0 or 1.
            const auto* raw = static_cast<const unsigned char*>(arrayOfElements);
            value.bytes.resize(count);
            for(std::size_t i = 0; i < count; ++i)
            {
                value.bytes[i] = raw[i] != 0 ? std::byte{1} : std::byte{0};
            }
            return value;
        }
        case CUDNN_TYPE_FRACTION:
        {
            const auto* fractions = static_cast<const cudnnFraction_t*>(arrayOfElements);
            for(std::size_t i = 0; i < count; ++i)
            {
                if(fractions[i].denominator == 0)
                {
                    throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                    std::string(info.name) + "[" + std::to_string(i)
                                        + "] has a zero denominator");
                }
            }
            break;
        }
        default:
            if(const std::optional<std::int32_t> valueCount = enumValueCount(attributeType))
            {
                const auto* raw = static_cast<const std::byte*>(arrayOfElements);
                for(std::size_t i = 0; i < count; ++i)
                {
                    std::int32_t element = 0;
                    std::memcpy(&element, raw + i * sizeof(element), sizeof(element));
                    if(element < 0 || element >= *valueCount)
                    {
                        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                        std::string(info.name) + "[" + std::to_string(i)
                                            + "] = " + std::to_string(element) + " is not a valid "
                                            + attributeTypeName(attributeType) + " value");
                    }
                }
            }
            break;
        }

        const std::size_t size = count * elementSize(attributeType);
        value.bytes.resize(size);
        if(size > 0)
        {
            std::memcpy(value.bytes.data(), arrayOfElements, size);
        }
        return value;
    }

    static std::optional<StoredAttribute> defaultAttribute(const AttributeInfo& info)
    {
        if(info.countRule != CountRule::SCALAR)
        {
            // Unset arrays and strings read as empty: upstream GETs optional
            // arrays such as ENGINECFG_KNOB_CHOICES unconditionally.
            StoredAttribute empty;
            empty.type = info.types[0];
            return empty;
        }
        if(!info.hasDefault)
        {
            return std::nullopt;
        }
        StoredAttribute value;
        value.type = info.types[0];
        value.count = 1;
        const std::size_t size = elementSize(value.type);
        value.bytes.resize(size);
        if(value.type == CUDNN_TYPE_BOOLEAN)
        {
            value.bytes[0] = info.defaultValue != 0 ? std::byte{1} : std::byte{0};
        }
        else if(size == sizeof(std::int64_t))
        {
            std::memcpy(value.bytes.data(), &info.defaultValue, size);
        }
        else
        {
            const auto narrow = static_cast<std::int32_t>(info.defaultValue);
            std::memcpy(value.bytes.data(), &narrow, sizeof(narrow));
        }
        return value;
    }

    const StoredAttribute* storedOrDefault(cudnnBackendAttributeName_t name,
                                           std::optional<StoredAttribute>& defaulted) const
    {
        const StoredAttribute* value = findAttribute(name);
        if(value != nullptr)
        {
            return value;
        }
        const std::optional<AttributeInfo> info = attributeInfo(name);
        if(info)
        {
            defaulted = defaultAttribute(*info);
        }
        return defaulted ? &*defaulted : nullptr;
    }

    static void requireElementSize(cudnnBackendAttributeName_t name,
                                   const StoredAttribute& value,
                                   std::size_t size)
    {
        if(value.type == CUDNN_TYPE_BACKEND_DESCRIPTOR || value.type == CUDNN_TYPE_CHAR
           || elementSize(value.type) != size)
        {
            throw ShimError(CUDNN_STATUS_INTERNAL_ERROR,
                            "typed read of " + attributeName(name)
                                + " does not match its stored element size");
        }
    }

    static void fillDescriptors(const AttributeInfo& info,
                                const StoredAttribute& value,
                                std::size_t count,
                                void* arrayOfElements)
    {
        auto* targets = static_cast<cudnnBackendDescriptor_t*>(arrayOfElements);
        // Validate every target before touching any, so a bad element leaves the
        // caller's descriptors unchanged.
        std::vector<Handle*> handles(count);
        for(std::size_t i = 0; i < count; ++i)
        {
            handles[i] = liveHandle(targets[i]);
            if(handles[i] == nullptr)
            {
                throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                std::string(info.name) + ": output element " + std::to_string(i)
                                    + " is not a live cuDNN shim descriptor");
            }
            if(handles[i]->object->type() != value.descriptors[i]->type())
            {
                throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                std::string(info.name) + ": output element " + std::to_string(i)
                                    + " is " + descriptorTypeName(handles[i]->object->type())
                                    + ", expected "
                                    + descriptorTypeName(value.descriptors[i]->type()));
            }
        }
        for(std::size_t i = 0; i < count; ++i)
        {
            value.descriptors[i]->fillInto(*handles[i]);
        }
    }

    cudnnBackendDescriptorType_t _type;
    bool _finalized = false;
    std::map<cudnnBackendAttributeName_t, StoredAttribute> _attributes;
};

// Holds attributes for a descriptor type whose validating class has not landed
// yet; finalize always fails so nothing downstream can consume it.
class StorageOnlyDescriptor final : public Descriptor
{
public:
    using Descriptor::Descriptor;

protected:
    void finalizeImpl() override
    {
        throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                        std::string("finalizing ") + descriptorTypeName(type())
                            + " is not supported by the hipDNN cuDNN shim");
    }
};

// Tag-checked downcast; the factory maps each descriptor type to exactly one
// class, so the tag identifies the dynamic type.
template <typename Derived>
Derived& descriptorCast(Descriptor& descriptor, cudnnBackendDescriptorType_t expected)
{
    static_assert(std::is_base_of_v<Descriptor, Derived>);
    if(descriptor.type() != expected)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string("expected ") + descriptorTypeName(expected) + ", got "
                            + descriptorTypeName(descriptor.type()));
    }
    return static_cast<Derived&>(descriptor);
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
