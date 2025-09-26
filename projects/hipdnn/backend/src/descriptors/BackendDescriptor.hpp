// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipdnnException.hpp"
#include "hipdnn_backend.h"
#include <memory>

// NOLINTBEGIN(portability-template-virtual-member-function)

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
    hipdnnBackendDescriptorType_t _type = HIPDNN_INVALID_TYPE;

public:
    HipdnnBackendDescriptorImpl()
        : _type(getStaticType())
    {
    }

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

    static hipdnnBackendDescriptorType_t getStaticType()
    {
        return T::getStaticType();
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

    bool operator==(const HipdnnBackendDescriptor& other) const;

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

private:
    std::shared_ptr<IBackendDescriptor> _impl;

    // Give access to mock classes for testing purposes only.
    friend class hipdnn_backend::MockDescriptorUtility;
};
//NOLINTEND(readability-identifier-naming)
