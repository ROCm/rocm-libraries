// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BackendEnumStringUtils.hpp"
#include "HipdnnException.hpp"
#include "hipdnn_backend.h"
#include <algorithm>
#include <memory>
#include <spdlog/fmt/fmt.h>
#include <type_traits>
#include <vector>

// NOLINTBEGIN(portability-template-virtual-member-function)

namespace hipdnn_backend
{
class IGraphOperation;
}

struct IBackendDescriptor
{
    virtual ~IBackendDescriptor() = default;

    virtual void finalize() = 0;
    virtual bool isFinalized() const = 0;
    virtual void getAttribute(hipdnnBackendAttributeName_t attributeName,
                              hipdnnBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements) const
        = 0;
    virtual void setAttribute(hipdnnBackendAttributeName_t attributeName,
                              hipdnnBackendAttributeType_t attributeType,
                              int64_t elementCount,
                              const void* arrayOfElements)
        = 0;

    virtual hipdnnBackendDescriptorType_t getType() const = 0;
    virtual std::string toString() const = 0;

    virtual hipdnn_backend::IGraphOperation* asGraphOperation()
    {
        return nullptr;
    }
};

// NOLINTEND(portability-template-virtual-member-function)

//NOLINTBEGIN(readability-identifier-naming)
namespace hipdnn_backend
{

class MockDescriptorUtility;

// NOLINTBEGIN(portability-template-virtual-member-function)

template <typename T>
class HipdnnBackendDescriptorImpl : public IBackendDescriptor
{
private:
    bool _finalized = false;
    hipdnnBackendDescriptorType_t _type = HIPDNN_INVALID_TYPE_EXT;

    friend T;

    HipdnnBackendDescriptorImpl()
        : _type(getStaticType())
    {
    }

public:
    void finalize() override
    {
        _finalized = true;
    }
    bool isFinalized() const override
    {
        return _finalized;
    }

    hipdnnBackendDescriptorType_t getType() const override
    {
        return _type;
    }

    std::string toString() const override
    {
        return hipdnn_backend::hipdnnGetBackendDescriptorTypeName(_type);
    }

    static hipdnnBackendDescriptorType_t getStaticType()
    {
        return T::getStaticType();
    }

    IGraphOperation* asGraphOperation() override
    {
        if constexpr(std::is_base_of_v<IGraphOperation, T>)
        {
            return static_cast<T*>(this);
        }
        else
        {
            return nullptr;
        }
    }
};

// NOLINTEND(portability-template-virtual-member-function)

}

namespace
{
// Helper trait for static_asserts in unpack.
template <typename Child_descriptor>
constexpr bool is_valid_backend_child_descriptor_v = std::is_base_of_v<
    hipdnn_backend::HipdnnBackendDescriptorImpl<std::remove_const_t<Child_descriptor>>,
    std::remove_const_t<Child_descriptor>>;
}

struct HipdnnBackendDescriptor : public IBackendDescriptor
{
    ~HipdnnBackendDescriptor() override = default;

    void finalize() override;
    bool isFinalized() const override;
    void getAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements) const override;
    void setAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements) override;

    template <typename ChildDescriptor>
    std::shared_ptr<ChildDescriptor> asDescriptor() const
    {
        return unpackDescriptor<ChildDescriptor>(
            this,
            HIPDNN_STATUS_INTERNAL_ERROR,
            "Failed to cast backend descriptor: Null descriptor provided.");
    }

    bool isValid();

    hipdnnBackendDescriptorType_t getType() const override;
    std::string toString() const override;

    bool operator==(const HipdnnBackendDescriptor& other) const;

    // Returns the wrapped impl as a shared_ptr to IBackendDescriptor.
    std::shared_ptr<IBackendDescriptor> getImpl() const
    {
        return _impl;
    }

    // Returns a shared_ptr to the IGraphOperation interface if the wrapped impl
    // implements it, or nullptr otherwise. Uses the virtual asGraphOperation()
    // method instead of dynamic_cast, so it works with -fno-rtti.
    std::shared_ptr<hipdnn_backend::IGraphOperation> tryAsGraphOperation() const
    {
        auto* graphOp = _impl->asGraphOperation();
        if(graphOp == nullptr)
        {
            return nullptr;
        }
        // The aliasing shared_ptr constructor shares ownership with _impl
        // (increments its ref count) while pointing to the IGraphOperation
        // subobject. This ensures proper lifetime management without dynamic_cast.
        return {_impl, graphOp};
    }

    // Unpacks a HipdnnBackendDescriptor into a shared_ptr of the specified type.
    // Throws an exception if the descriptor is null.
    template <typename ChildDescriptor>
    static std::shared_ptr<ChildDescriptor>
        unpackDescriptor(const HipdnnBackendDescriptor* descriptor,
                         hipdnnStatus_t status,
                         const std::string& message)
    {
        static_assert(is_valid_backend_child_descriptor_v<ChildDescriptor>,
                      "ChildDescriptor must inherit from HipdnnBackendDescriptorImpl");

        THROW_IF_NULL(descriptor, status, message);
        THROW_IF_NULL(descriptor->_impl, status, message);
        THROW_IF_NE(descriptor->getType(),
                    ChildDescriptor::getStaticType(),
                    hipdnnStatus_t::HIPDNN_STATUS_BAD_PARAM,
                    "Unpacking HipdnnBackendDescriptor failed: Descriptor type mismatch.");

        auto childDescriptor = std::static_pointer_cast<ChildDescriptor>(descriptor->_impl);

        return childDescriptor;
    }

    // Unpacks a HipdnnBackendDescriptor from an array of elements.
    // The array of elements is expected to be a pointer to a HipdnnBackendDescriptor.
    // Throws an exception if the array of elements is null or if the descriptor cannot be unpacked.
    template <typename ChildDescriptor>
    static std::shared_ptr<ChildDescriptor> unpackDescriptor(const void* arrayOfElements,
                                                             hipdnnStatus_t status,
                                                             const std::string& message)
    {
        static_assert(is_valid_backend_child_descriptor_v<ChildDescriptor>,
                      "ChildDescriptor must inherit from HipdnnBackendDescriptorImpl");

        THROW_IF_NULL(arrayOfElements, status, message);

        return unpackDescriptor<ChildDescriptor>(
            *static_cast<HipdnnBackendDescriptor* const*>(arrayOfElements), status, message);
    }

    // Packs a shared_ptr into a new HipdnnBackendDescriptor.
    // Note: Ownership of the descriptor is transferred to the caller,
    //       and it is the caller's responsibility to delete it.
    //       You can use Scoped_descriptor to manage the lifetime of the descriptor automatically.
    static HipdnnBackendDescriptor*
        packDescriptor(const std::shared_ptr<const IBackendDescriptor>& impl)
    {
        auto descriptor = new HipdnnBackendDescriptor();
        descriptor->_impl = std::const_pointer_cast<IBackendDescriptor>(impl);
        return descriptor;
    }

    // Packs a shared_ptr into a new HipdnnBackendDescriptor, and stores it into a provided arrayOfElements ptr.
    // Note: Ownership of the descriptor is transferred to the public API caller.
    //       It is the caller's responsibility to delete it.
    static void packDescriptor(const std::shared_ptr<const IBackendDescriptor>& impl,
                               void*& arrayOfElements)
    {
        *static_cast<HipdnnBackendDescriptor**>(arrayOfElements) = packDescriptor(impl);
    }

    /// Packs a container of shared_ptr descriptors into an output array with exception safety.
    /// On success, ownership of all packed descriptors transfers to the caller.
    /// On exception, any already-packed descriptors are deleted before re-throwing.
    template <typename Container>
    static void packDescriptorArray(const Container& descriptors,
                                    HipdnnBackendDescriptor** outputArray)
    {
        std::vector<HipdnnBackendDescriptor*> packed;
        packed.reserve(descriptors.size());
        try
        {
            for(const auto& desc : descriptors)
            {
                packed.push_back(packDescriptor(desc));
            }
        }
        catch(...)
        {
            for(auto* p : packed)
            {
                delete p;
            }
            throw;
        }

        std::copy(packed.begin(), packed.end(), outputArray);
    }

private:
    std::shared_ptr<IBackendDescriptor> _impl;

    // Give access to mock classes for testing purposes only.
    friend class hipdnn_backend::MockDescriptorUtility;
};
//NOLINTEND(readability-identifier-naming)

template <>
struct fmt::formatter<HipdnnBackendDescriptor> : fmt::formatter<std::string>
{
    auto format(const HipdnnBackendDescriptor& descriptor, format_context& ctx) const
    {
        return fmt::formatter<std::string>::format(descriptor.toString(), ctx);
    }
};
