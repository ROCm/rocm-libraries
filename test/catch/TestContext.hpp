
#pragma once

#include <memory>

#include <rocRoller/Context.hpp>

class TestContext;

/**
 * Helper to create a Context during a test.
 *
 * Will by default name the kernel for the current test name. Since catch2 does not
 * include the test parameterization in the name, the helper functions all can take any
 * number of arguments that will be stringified into the kernel name.
 */
class TestContext
{
public:
    ~TestContext();

    /**
     * Use this when creating a kernel for the current (physically present) GPU, to
     * enable running the kernel.
     */
    template <typename... Params>
    static TestContext ForTestDevice(rocRoller::KernelOptions const& kernelOpts = {},
                                     Params const&... params);

    /**
     * Use this when creating a kernel for a default GPU target (currently gfx90a), when
     * testing features that are not target-specific.
     */
    template <typename... Params>
    static TestContext ForDefaultTarget(rocRoller::KernelOptions const& kernelOpts = {},
                                        Params const&... params);

    /**
     * Use this when creating a kernel for a specific GPU target.
     */
    template <typename... Params>
    static TestContext ForTarget(rocRoller::GPUArchitectureTarget const& arch,
                                 rocRoller::KernelOptions const&         kernelOpts = {},
                                 Params const&... params);

    /**
     * Use this when creating a kernel for a specific GPU target.
     */
    template <typename... Params>
    static TestContext ForTarget(std::string const&              arch,
                                 rocRoller::KernelOptions const& kernelOpts = {},
                                 Params const&... params);

    /**
     * Use this when creating a kernel for a specific GPU target.
     */
    template <typename... Params>
    static TestContext ForTarget(rocRoller::GPUArchitecture const& arch,
                                 rocRoller::KernelOptions const&   kernelOpts = {},
                                 Params const&... params);

    /**
     * Creates a valid kernel name which includes the name of the currently running test
     * as well as stringified versions of all the provided parameters.
     */
    template <typename... Params>
    static std::string KernelName(Params const&... params);

    /**
     * Returns `name` (which may contain spaces or other special characters) converted
     * into a valid C identifier name
     */
    static std::string EscapeKernelName(std::string name);

    rocRoller::ContextPtr get();

    rocRoller::Context& operator*();
    rocRoller::Context* operator->();

    /**
     * Returns the assembly code generated so far as a string.
     */
    std::string output();

protected:
    TestContext(rocRoller::ContextPtr context);

    rocRoller::ContextPtr m_context;
};

#include "TestContext_impl.hpp"
