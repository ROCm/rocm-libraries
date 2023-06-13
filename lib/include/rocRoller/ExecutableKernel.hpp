#pragma once

#include "Context_fwd.hpp"

#include "GPUArchitecture/GPUArchitectureTarget.hpp"
#include "KernelArguments.hpp"

namespace rocRoller
{
    // KernelInvocation contains the information needed to launch a kernel with
    // the ExecutableKernel class.
    struct KernelInvocation
    {
        std::array<unsigned int, 3> workitemCount;
        std::array<unsigned int, 3> workgroupSize;
        unsigned int                sharedMemBytes;

        KernelInvocation()
            : workitemCount({1, 1, 1})
            , workgroupSize({1, 1, 1})
            , sharedMemBytes(0)
        {
        }
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
         */
        void executeKernel(const KernelArguments& args, const KernelInvocation& invocation);

        /**
         * @brief Execute a kernel on a GPU.
         *
         * @param args The arguments to the Kernel
         * @param invocation Other information needed to launch the kernel.
         */
        void executeKernel();

    private:
        struct HIPData;

        std::string              m_kernelName;
        bool                     m_kernelLoaded;
        std::shared_ptr<HIPData> m_hipData;
    };
}

#include "ExecutableKernel_impl.hpp"
