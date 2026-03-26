# rocWMMA Community Samples

This directory contains community-contributed samples demonstrating advanced techniques, specialized use cases, and experimental approaches using the rocWMMA library.

## Purpose

Community samples extend beyond the core competency demonstrations in the official samples directory. They showcase:
- Advanced kernel fusion techniques (e.g., GEMM+GEMM, GEMM+activation functions)
- Specialized machine learning operations
- Performance optimization strategies
- Complex multi-technique examples
- Experimental or cutting-edge research applications

Unlike official samples, community samples have reduced review and maintenance requirements, allowing for faster contribution and more experimental code.

## Current Status

**This directory is currently empty but ready for contributions.**

As samples accumulate, they may be organized into subdirectories by technique or domain:
- `fusion/` - Kernel fusion examples (GEMM+GEMM, GEMM+activation, etc.)
- `ml-models/` - Machine learning specific applications
- `optimizations/` - Performance optimization techniques
- `advanced/` - Complex use cases combining multiple techniques
- `experimental/` - Cutting-edge research techniques

## Requirements

- **Minimum ROCm version**: 6.0+ recommended (community samples may work with earlier versions)
- **Minimum hipcc version**: 4.4+ (same as rocWMMA library)
- **Build system**: CMake 3.10+
- Samples should be compatible with at least one officially supported GPU architecture

## Contribution Guidelines

For complete contribution guidelines, see the [Community Samples section in CONTRIBUTING.md](../../CONTRIBUTING.md#community-samples). This README provides a quick reference.

### What Makes a Good Community Sample?

1. **Demonstrates rocWMMA Usage**: The sample must meaningfully use the rocWMMA API
2. **Provides Value**: Shows a technique, optimization, or use case not covered in official samples
3. **Code Quality**: Code should be readable, well-commented, and follow basic C++ best practices
4. **Builds Successfully**: Must compile with supported ROCm/hipcc versions
5. **Proper Licensing**: Must include MIT license header and be contributor's original work or properly attributed
6. **Security**: Does not introduce security vulnerabilities (e.g., buffer overflows, command injection, arbitrary code execution)

### What is NOT Required

Community samples have significantly reduced requirements compared to official samples. You are **NOT** required to provide:

- Unit tests or validation tests
- Benchmark tests or performance comparisons
- Support for all data types (int8, f16, f32, f64, bf16, f8, bf8)
- Support for all GPU architectures (gfx908, gfx90a, gfx942, gfx950, gfx11xx, gfx12xx)
- Support for all block sizes or fragment parameters
- API documentation in the API Reference Guide
- Extensive error handling or edge case coverage

Focus on demonstrating your technique clearly. It's acceptable if your sample only works on specific architectures or with specific data types.

## How to Contribute

### Step-by-Step Guide

1. **Start with the Template** (Recommended)
   - Copy `samples/community/template.cpp` as a starting point
   - The template includes proper license headers and comment structure
   - Fill in the description, requirements, and limitations sections

2. **Write Your Sample**
   - Place your `.cpp` file directly in `samples/community/` (flat structure initially)
   - Use meaningful variable names and add comments explaining non-obvious code
   - Focus on demonstrating your technique clearly rather than handling all edge cases
   - **Security note**: Avoid unsafe operations like:
     - Unchecked buffer accesses or unbounded loops
     - User-controlled format strings or command execution
     - Reading from uninitialized memory
     - Excessive memory allocations that could exhaust system resources

3. **Update Build Configuration**
   - Edit `samples/community/CMakeLists.txt`
   - Add your sample using the `add_community_sample()` function:
     ```cmake
     add_community_sample(your_sample_name ${CMAKE_CURRENT_SOURCE_DIR}/your_sample_name.cpp)
     ```

4. **Update Documentation**
   - Add an entry to the "Current Community Samples" section below
   - Use the provided template (see HTML comment in that section)
   - Include: sample name, file, description, requirements, and known limitations

5. **Test Your Sample**
   - Build with: `cmake -B build . -DROCWMMA_BUILD_COMMUNITY_SAMPLES=ON && cmake --build build`
   - Run your sample: `./build/projects/rocwmma/samples/community/your_sample_name`
   - Verify it produces expected output without errors on at least one GPU architecture

6. **Submit Pull Request**
   - Follow the standard PR process outlined in the main CONTRIBUTING.md
   - Provide a clear PR description explaining the technique demonstrated
   - Be responsive to basic code review feedback (readability, licensing, build issues)

### Contribution Checklist

Before submitting your PR, verify:

- [ ] Source file includes MIT license header
- [ ] File includes clear comment header explaining what the sample demonstrates
- [ ] Sample specifies requirements (ROCm version, GPU architectures, data types)
- [ ] Known limitations are documented
- [ ] Code builds successfully without errors on at least one GPU architecture
- [ ] CMakeLists.txt updated with `add_community_sample()` call
- [ ] README.md updated in "Current Community Samples" section
- [ ] PR description explains the technique and its value
- [ ] No obvious security issues (buffer overflows, unsafe operations)
- [ ] Sample can be run successfully: `./build/projects/rocwmma/samples/community/your_sample_name`

## Building Community Samples

Community samples are **disabled by default**. To build them:

```bash
cmake -B build . -DROCWMMA_BUILD_COMMUNITY_SAMPLES=ON
cmake --build build

# Or build only community samples target
cmake --build build --target rocwmma_community_samples
```

## Running Community Samples

After building, samples are located in:
```bash
./build/projects/rocwmma/samples/community/
```

Run a sample:
```bash
./build/projects/rocwmma/samples/community/sample_name
```

Most samples will display output indicating success or failure. Check the sample's documentation for specific usage instructions.

## Important Disclaimers

### As-Is Status

Community samples are provided **as-is** with the following caveats:

- AMD does not guarantee maintenance or updates for community samples
- Samples may become outdated as rocWMMA evolves
- Performance may not be optimal or may not represent best practices
- Code may only work on specific GPU architectures or with specific configurations
- Issues reported for community samples may be closed if they are low priority

### Support Expectations

- Community samples have **no official support commitment** from AMD
- The community and original authors are the primary support resources
- Users should adapt samples to their specific needs
- Bug reports and improvements from the community are welcome but not guaranteed to be addressed

### Quality Standards

While community samples have reduced requirements, they should still:
- Follow basic coding standards and be readable
- Include meaningful comments explaining the technique
- Build successfully without errors
- Not introduce security vulnerabilities (buffer overflows, command injection, unsafe memory access)
- Use appropriate error checking for GPU API calls (CHECK_HIP_ERROR, etc.)
- Clearly document any assumptions or prerequisites

## Sample Lifecycle

### Contribution → Review → Merge
- New samples undergo basic review for licensing, build issues, and code quality
- Review focuses on "does it compile and demonstrate something useful?"
- Higher scrutiny than regular code contributions is not required

### Community Maintenance
- Original authors are encouraged to maintain their samples
- Other community members can submit improvements
- Samples that break due to rocWMMA API changes may be deprecated if not fixed

### Graduation to Official Samples

Community samples that prove valuable and widely applicable may be promoted to official samples:

**Criteria for Graduation:**
- Demonstrates a broadly useful technique or common use case
- High code quality and comprehensive error handling
- Works across multiple GPU architectures and data types
- Well-documented with clear explanations
- Community validation and positive feedback

**Graduation Process:**
1. AMD maintainers identify candidate samples for promotion
2. Sample author is contacted (if identifiable) to discuss promotion
3. Sample undergoes full code review to official sample standards
4. Sample is enhanced to support all architectures/types (if needed)
5. Tests and benchmarks are added
6. Documentation is added to API Reference Guide
7. Original community version may be deprecated or removed after transition
8. AMD commits to ongoing maintenance as an official sample

Graduation is at AMD's discretion and depends on maintenance capacity. Not all valuable community samples will be promoted.

## Questions?

For questions about contributing community samples:
- Open an issue in the rocm-libraries repository
- Tag with "rocwmma" and "community-samples"
- Consult the main CONTRIBUTING.md for general contribution guidelines

---

## Current Community Samples

*This section will be populated as samples are contributed.*

<!-- Template for documenting samples:
### Sample Name
- **File**: `sample_file.cpp`
- **Description**: Brief explanation of what the sample demonstrates
- **Requirements**: Architecture, data type, or other requirements
- **Limitations**: Known limitations or caveats
- **Author**: [Optional] Original contributor
-->
