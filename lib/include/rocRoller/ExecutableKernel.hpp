#pragma once

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Utilities/HIPTimer.hpp>

namespace rocRoller
{
    // KernelInvocation contains the information needed to launch a kernel with
    // the ExecutableKernel class.
    struct KernelInvocation
    {
        std::array<unsigned int, 3> workitemCount  = {1, 1, 1};
        std::array<unsigned int, 3> workgroupSize  = {1, 1, 1};
        unsigned int                sharedMemBytes = 0;
    };

    // The executer class can load a kernel from a string of machine code and
    // then launch the kernel on a GPU.
    class ExecutableKernel
    {
    public:
        ExecutableKernel();
        ~ExecutableKernel() = default;

        /**
         * kernelLoaded returns true if a kernel has already been loaded.
         *
         *  @return bool
         */
        bool kernelLoaded() const;

        /**
         * @brief Loads a kernel from a string of machine code.
         *
         * @param instructions A string containing an entire assembly file
         * @param target The target architecture
         * @param kernelName The name of the kernel
         */
        void loadKernel(std::string const&           instructions,
                        const GPUArchitectureTarget& target,
                        std::string const&           kernelName);

        /**
         * @brief Loads a kernel from a file.
         *
         * @param fileName Assembly file name
         * @param kernelName The name of the kernel
         * @param target The target architecture
         */
        void loadKernelFromFile(std::string const&           fileName,
                                std::string const&           kernelName,
                                const GPUArchitectureTarget& target);

        /**
         * @brief Loads a kernel from a file.
         *
         * @param fileName Code-object file name
         * @param kernelName The name of the kernel
         * @param target The target architecture
         */
        void loadKernelFromCodeObjectFile(std::string const&           fileName,
                                          std::string const&           kernelName,
                                          const GPUArchitectureTarget& target);

        /**
         * @brief Loads a kernel from a string of machine code.
         *
         * @param instructions A string containing an entire assembly file
         * @param target The target architecture
         * @param kernelName The name of the kernel
         */
        void loadKernel(std::string const& instructions, ContextPtr context);

        /**
         * @brief Execute a kernel on a GPU.
         *
         * @param args The arguments to the Kernel
         * @param invocation Other information needed to launch the kernel.
         * @param timer HIPTimer that will record how long the kernel took to execute
         * @param iteration Iteration number within the timer
         */
        void executeKernel(const KernelArguments&    args,
                           const KernelInvocation&   invocation,
                           std::shared_ptr<HIPTimer> timer,
                           int                       iteration);

        /**
         * @brief Execute a kernel on a GPU with stream specified
         *
         * @param args The arguments to the Kernel
         * @param invocation Other information needed to launch the kernel.
         * @param stream The stream the kernel is executed on
         */
        void executeKernel(const KernelArguments&  args,
                           const KernelInvocation& invocation,
                           hipStream_t             stream = 0);

        /**
         * @brief Returns the hipFunction for the kernel
        */
        hipFunction_t getHipFunction() const;

    private:
        struct HIPData;

        std::string              m_kernelName;
        bool                     m_kernelLoaded;
        std::shared_ptr<HIPData> m_hipData;

        /**
         * @brief Execute a kernel on a GPU with optional timer and stream
         *
         * @param args The arguments to the Kernel
         * @param invocation Other information needed to launch the kernel.
         * @param timer HIPTimer that will record how long the kernel took to execute
         * @param iteration Iteration number within the timer
         * @param stream The stream the kernel is executed on
         */
        void executeKernel(const KernelArguments&    args,
                           const KernelInvocation&   invocation,
                           std::shared_ptr<HIPTimer> timer,
                           int                       iteration,
                           hipStream_t               stream);
    };
}

#include <rocRoller/ExecutableKernel_impl.hpp>
