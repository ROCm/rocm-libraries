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

## Contribution Guidelines

### What Makes a Good Community Sample?

1. **Demonstrates rocWMMA Usage**: The sample must meaningfully use the rocWMMA API
2. **Provides Value**: Shows a technique, optimization, or use case not covered in official samples
3. **Code Quality**: Code should be readable, well-commented, and follow basic C++ best practices
4. **Builds Successfully**: Must compile with supported ROCm/hipcc versions
5. **Proper Licensing**: Must include MIT license header and be contributor's original work or properly attributed

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

1. **Write Your Sample**
   - Place your `.cpp` file directly in `samples/community/` (flat structure initially)
   - Include a clear comment header explaining:
     - What the sample demonstrates
     - Any architecture, data type, or configuration requirements
     - Known limitations
     - Author/contributor information (optional)
   - Add MIT license header to your file

2. **Update Build Configuration**
   - Edit `samples/community/CMakeLists.txt`
   - Add your sample using the `add_community_sample()` function:
     ```cmake
     add_community_sample(your_sample_name ${CMAKE_CURRENT_SOURCE_DIR}/your_sample_name.cpp)
     ```

3. **Update Documentation**
   - Add an entry to this README documenting your sample:
     - Sample name and file
     - Brief description of what it demonstrates
     - Requirements (architecture, data types, etc.)
     - Any known limitations

4. **Submit Pull Request**
   - Follow the standard PR process outlined in the main CONTRIBUTING.md
   - Provide a clear PR description explaining the technique demonstrated
   - Be responsive to basic code review feedback (readability, licensing, build issues)

## Building Community Samples

Community samples are **disabled by default**. To build them:

```bash
cmake -B build . -DROCWMMA_BUILD_COMMUNITY_SAMPLES=ON
cmake --build build

# Or build only community samples target
cmake --build build --target rocwmma_community_samples
```

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
- Not introduce security vulnerabilities

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
- High-value samples demonstrating important techniques may be considered for promotion
- Official samples require AMD commitment to ongoing maintenance
- Promotion involves more thorough review, testing, and documentation

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
