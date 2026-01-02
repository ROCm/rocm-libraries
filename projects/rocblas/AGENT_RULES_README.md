# rocBLAS Agent Development Rules

This directory contains agent development rules for AI coding assistants (Cursor, Cline, GitHub Copilot) working with the rocBLAS project.

## 📁 Files Overview

### Core Rules
- **`.cursorrules`** - Main rules file for Cursor/Cline (15 KB)
  - Complete project overview and architecture
  - Build and testing quickstart (Linux + Windows)
  - C++ code style guidelines
  - Development workflow and best practices

### Cursor-Specific Rules (`.cursor/rules/`)
- **`rocblas-architecture.mdc`** (3 KB) - `alwaysApply: true`
  - BLAS operation levels (1, 2, 3, Extensions)
  - Tensile integration
  - Data type support
  - Component organization

- **`building.mdc`** (4 KB) - `alwaysApply: false`
  - Build commands (install.sh, rmake.py)
  - Test execution (rtest.py)
  - Test levels (smoke, psdb, osdb, cqe)
  - Performance benchmarking

- **`cpp-style.mdc`** (6 KB) - `globs: ["*.cpp", "*.hpp", "*.h"]`
  - Naming conventions (snake_case)
  - BLAS API patterns
  - HIP/GPU code examples
  - MIT license header template

- **`testing.mdc`** (7 KB) - `globs: ["*.cpp", "*.hpp", "*.h", "clients/gtest/**"]`
  - Google Test patterns
  - YAML test configuration
  - Test categories and filters
  - Multi-OS testing framework

### GitHub Copilot
- **`.github/copilot-instructions.md`** (7 KB)
  - Condensed rules for GitHub Copilot
  - Quick reference for common patterns
  - API examples and build commands

### Documentation
- **`AGENT_RULES_README.md`** (this file)
  - Quick overview of all files
  - Getting started guide

## 🚀 Quick Start

### For Cursor/Cline Users

1. **Open any rocBLAS file** - Rules activate automatically
2. **Ask about the project:**
   - "Explain the rocBLAS architecture"
   - "How do I build rocBLAS on Linux?"
   - "What are the test levels?"
3. **Request code changes:**
   - "Add a new BLAS Level 1 operation"
   - "Create a test for gemm_strided_batched"
4. **Build and test:**
   - "Build rocBLAS with clients"
   - "Run smoke tests"

### For GitHub Copilot Users

GitHub Copilot will automatically read `.github/copilot-instructions.md` and provide context-aware suggestions following rocBLAS conventions.

## 📖 What Agents Know

### Project Structure
- Library components (blas1, blas2, blas3, blas_ex)
- Client structure (gtest, benchmarks, samples)
- Tensile integration for GEMM operations
- Documentation and scripts

### Build System
- **Linux:** `./install.sh` with various flags
- **Windows:** `python rmake.py` (with HIP_PATH set)
- Architecture flags (`--architecture auto`)
- Dependency management

### Testing
- **Test Levels:**
  - smoke (5-10 min) - Quick sanity
  - psdb (30-60 min) - PR validation
  - osdb (1.5-2 hrs) - Nightly regression
  - cqe (3-3.5 hrs) - Complete QE
- **Test Execution:** `rtest.py -t <level>`
- **Direct Testing:** `rocblas-test --gtest_filter=*pattern*`
- **YAML Configuration:** Parameterized test definitions

### Code Style
- **Naming:** snake_case for functions/variables
- **BLAS Convention:** `rocblas_<precision><operation>`
- **Precisions:** s=float, d=double, c=complex, z=complex double, h=half, bf16=bfloat16
- **Headers:** Full MIT license header
- **Include Guards:** `#pragma once`

### Testing Workflow
- **Test Orchestration:** `rtest.py` with multiple test levels
- **Direct Testing:** `rocblas-test` with gtest filters
- **YAML Configuration:** Parameterized test definitions
- **Benchmarking:** `rocblas-bench` for performance testing

## 🎯 Key Features

### 1. BLAS-Specific Conventions
Unlike hipDNN's CamelCase, rocBLAS uses snake_case to match traditional BLAS naming standards.

### 2. Dual Platform Support
Comprehensive documentation for both Linux (primary) and Windows development.

### 3. Test Hierarchy
Clear progression from quick smoke tests to comprehensive QE validation.

### 4. Performance Focus
Includes benchmarking guidance and Tensile kernel optimization.

## 🔧 Customization

### To Modify Rules:

1. **General changes:** Edit `.cursorrules`
2. **Architecture info:** Edit `.cursor/rules/rocblas-architecture.mdc`
3. **Build/test commands:** Edit `.cursor/rules/building.mdc`
4. **Code style:** Edit `.cursor/rules/cpp-style.mdc`
5. **Testing patterns:** Edit `.cursor/rules/testing.mdc`
6. **GitHub Copilot:** Edit `.github/copilot-instructions.md`

### To Add New Rules:

Create new `.mdc` files in `.cursor/rules/` with appropriate metadata:

```markdown
---
alwaysApply: false
globs: ["*.cpp", "*.hpp"]
---

# Your Rule Title

Your rule content here...
```

## 📊 Comparison to hipDNN

| Aspect | hipDNN | rocBLAS |
|--------|--------|---------|
| Naming | CamelCase | snake_case |
| Build | CMake + Ninja | install.sh / rmake.py |
| Testing | Direct binaries | rtest.py orchestration |
| Platform | Linux | Linux + Windows |
| Architecture | Plugin-based | BLAS levels |
| Special | Graph operations | Tensile kernels |

See `AGENT_RULES_COMPARISON.md` for detailed comparison.

## 💡 Tips for Agents

### When Suggesting Code:
1. ✅ Use snake_case naming
2. ✅ Follow `rocblas_<precision><operation>` pattern
3. ✅ Include full MIT license header
4. ✅ Add tests in `clients/gtest/`
5. ✅ Consider all data types (s, d, c, z, h, bf16)
6. ✅ Use `#pragma once` for headers

### When Building:
1. ✅ Use `./install.sh` on Linux
2. ✅ Use `python rmake.py` on Windows
3. ✅ Add `--architecture auto` for faster builds
4. ✅ Only build when explicitly requested

### When Testing:
1. ✅ Start with `rtest.py -t smoke`
2. ✅ Use appropriate test level for context
3. ✅ Use `--gtest_filter` for specific tests
4. ✅ Use YAML configs for parameterized testing

## 📚 Additional Resources

- **rocBLAS Documentation:** `docs/`
- **Build Guides:** `docs/install/`
- **Test Configuration:** `rtest.xml` - Test suite definitions
- **YAML Test Files:** `clients/gtest/*.yaml` - Parameterized test configurations

## 🤝 Contributing

### For Developers Updating Rules

When updating these rules:

1. Test with actual agent interactions
2. Update all relevant files (don't forget copilot-instructions.md)
3. Document changes in this README
4. Consider impact on both Cursor and GitHub Copilot

### For AI Agents Using These Rules

**Help improve these rules!** If you notice:

- ✅ **Inconsistencies:** Pattern in code differs from documented rule
- ✅ **Missing patterns:** Critical idioms not documented
- ✅ **Outdated examples:** Code examples don't match current practices
- ✅ **Gaps:** Questions these rules should answer but don't

**Suggest updates by saying:**
> "I notice [observation]. The rules in [file] section [section] should be updated to [suggestion]. This will help future agents [benefit]."

This creates a continuous improvement loop where the rules evolve with the codebase.

## ✅ Verification

To verify the rules are working:

1. **Open a rocBLAS file in Cursor**
2. **Ask:** "What are the BLAS operation levels?"
3. **Expected:** Agent should reference Level 1, 2, 3, and Extensions
4. **Ask:** "How do I build rocBLAS?"
5. **Expected:** Agent should provide platform-specific commands

## 📝 Version History

- **2026-01-02:** Initial creation based on hipDNN rules
  - Adapted naming conventions for BLAS
  - Added dual platform support
  - Documented rtest.py test orchestration
  - Documented test hierarchy and YAML configurations
  - Added Windows-specific guidance

## 🙏 Credits

These rules were inspired by the hipDNN project's agent development rules and adapted specifically for rocBLAS development workflows and testing infrastructure.

---

**Questions?** See `AGENT_RULES_SUMMARY.md` for detailed documentation.

