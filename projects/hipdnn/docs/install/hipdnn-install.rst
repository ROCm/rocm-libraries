.. meta::
  :description: hipDNN install 
  :keywords: Component, ROCm, install

*******************
hipDNN installation
*******************

Ensure the required dependencies are installed on your system as outlined in [Dependencies](#dependencies). 
Refer to the [LLVM_TOOLS_SEARCH_PREFIX](#llvm_tools_search_prefix) section later in this document for approaches to manage the multiple Clang toolchain versions required for hipDNN.

Refer to the [Platform-Specific Instructions](#platform-specific-instructions) section for details on building under Windows.

Build and install hipDNN
========================

1. Clone the rocm-libraries repository.

   As a faster alternative to cloning the entire git repository, you can do a fast sparse-checkout of just the hipDNN project.

   ```bash
   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/hipdnn
   git checkout develop # or the branch you are starting from
   ```

   Alternatively, a traditional `git clone` can also be used (though only the `projects/hipdnn` folder is needed):

   ```bash
   git clone https://github.com/ROCm/rocm-libraries.git
   ```

2. Build hipDNN:

   ```bash
   cd rocm-libraries/projects/hipdnn
   mkdir build && cd build

   # Configure with Ninja (recommended)
   cmake -GNinja ..

   # Build and run all tests
   # Note that some tests may take several minutes to complete
   ninja check
   ```
   Refer to the [Build Targets](#build-targets) section below for additional build targets that can be used.

3. Install hipDNN:

   Refer to the [Build Configurations](#build-configurations) section below for details on setting the install path.

   ```bash
   sudo ninja install
   ```

Troubleshooting
===============

Common build issues
-------------------

- **Out of memory during build**
   ```bash
   # Reduce parallel jobs
   ninja -j4  # or even -j2 for systems with limited RAM
   ```

- **Docker GPU access issues**
   - Ensure ROCm is installed on the host system
   - Verify GPU is visible: `rocm-smi` or `rocminfo`
   - Check user is in `video` and `render` groups:
     ```bash
     sudo usermod -a -G video,render $USER
     # Log out and back in for changes to take effect
     ```

Verifying Installation

See [samples README](../samples/README.md) for detailed instructions on building test sample programs using hipDNN.
