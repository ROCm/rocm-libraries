# Contributing to hipDNN

Thank you for your interest in contributing to hipDNN!

 We welcome contributions from the community to help make hipDNN better. This guide will help you understand the contribution process and requirements.

> hipDNN is a graph-based deep learning library that enables multi-operation fusion for improved performance on AMD GPUs.

We're excited to have you join our community of contributors!

## Before You Start

Before contributing, please review these essential documents to understand the project structure and goals:

- **[Design](./docs/Design.md)** - Understand hipDNN's architecture and component design
- **[Building](./docs/Building.md)** - Learn how to build hipDNN on your system
- **[How-To](./docs/HowTo.md)** - Using hipDNN components and common approaches for extending functionality
- **[TestingStrategy](./docs/testing/TestingStrategy.md)** - Understand our testing approach and requirements
- **[Roadmap](./docs/Roadmap.md)** - Check planned features and find contribution opportunities
- **[Coding Style and Naming Guidelines](./docs/CodingStyleAndNamingGuidelines.md)** - Follow our coding conventions for consistency

We encourage you to open a GitHub issue to discuss your planned contribution before starting work. This helps ensure your efforts align with project goals and prevents duplicate work.

## Contribution Requirements

All contributions must meet the following requirements before they can be merged:

### Code Quality Standards

- **Code Formatting**: All code must follow the format specified by the `.clang-format` file
  - Run `ninja format` to auto-format your code
  - Run `ninja check_format` to verify formatting compliance
- **Compiler Warnings**: Code must compile without warnings
- **Clang-tidy Compliance**: Code must be free of clang-tidy errors

### Testing Requirements

- **Unit Tests**: All new code must be covered by unit tests
- **Integration Tests**: Add integration tests where applicable to verify end-to-end functionality
- **ASAN Compliance**: All tests must run cleanly with AddressSanitizer enabled
  - Build with `cmake -DBUILD_ADDRESS_SANITIZER=ON ..`
  - Run tests with `ninja check` to verify ASAN compliance
  - Note: Some HIP-related tests may be skipped due to AddressSanitizer incompatibility
- **GPU Test Handling**:
  - Mark GPU-dependent tests with `SKIP_IF_NO_DEVICE()`
  - Tests must be skippable without GPU (warnings, not errors)
- **Test Coverage**:
  - Maintain overall 80% code coverage target
  - New code should not decrease existing coverage
  - Each component should maintain >80% coverage individually

See [Testing](./docs/Testing.md) for further details.

### Documentation Requirements

- **Update Documentation**: Update all relevant documentation to reflect your changes
- **Remove Stale Documentation**: Remove any documentation that becomes obsolete due to your changes
- **Clear PR details**: Write clear and descriptive pull request details to help reviewers understand the changes

## Architecture Considerations

When contributing to hipDNN, please keep these architectural principles in mind:

### Dependency Management

- **hipDNN Core** (backend, SDK, frontend) should remain very light on dependencies
  - Avoid adding new library dependencies to the backend if possible
  - No compiled libraries required for the frontend or SDK (should remain header-only projects)
  - Any new dependencies require discussion and strong justification

### Plugin Development

- Plugins are **separate projects** from hipDNN core
  - Plugins can have their own dependencies as needed
  - See [Plugin Development](./docs/PluginDevelopment.md) for further guidance

> [!NOTE]
> 📝 The MIOpen Legacy Plugin is currently an exception and will be migrated to its own repository in the future.

## Development Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/hipDNN.git
cd hipDNN
git remote add upstream https://github.com/ROCm/hipDNN.git
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Build Locally

Follow the remaining instructions in the [Quick Start Guide](./docs/Building.md#quick-start-guide) to build hipDNN.

### 4. Run All Required Checks

Before submitting your PR, ensure all checks pass:

```bash
# Format code
ninja format

# Check formatting
ninja check_format

# Run all tests
ninja check

# Run tests with ASAN
cmake -DBUILD_ADDRESS_SANITIZER=ON ..
ninja check

# Check code coverage (optional but recommended)
cmake -DCODE_COVERAGE=ON ..
ninja code_coverage
```

### 5. Create a Pull Request

- Push your changes to your fork
- Create a pull request against the main hipDNN repository
- Fill out the pull request template completely
- Ensure all CI checks pass

## Pull Request Checklist

When creating a pull request, ensure you can check all these boxes:

- [ ] I have added automated tests relevant to the introduced functionality
- [ ] I have sufficient test coverage for the changes, and code coverage hasn't decreased
- [ ] I have run the tests, and they are all passing locally
- [ ] I have run the tests with ASAN, and they are all passing locally
- [ ] I have added relevant documentation for the changes
- [ ] I have removed stale documentation that is no longer relevant
- [ ] I have run `make format` & `make check_format` to ensure proper formatting

## Getting Help

- **Questions**: Open a GitHub issue with your question
- **Discussion**: For design discussions or feature proposals, open an issue before starting work
- **Draft PRs**: Feel free to open a draft PR early to get feedback on your approach
- **CI Pipelines**: In your PR, ask members of the hipDNN team to run CI on your branch
- **Code Reviews**: Be responsive to code review feedback and make requested changes promptly

## Community Guidelines

- Be respectful and constructive in all interactions
- Help review other contributors' PRs when possible
- Share knowledge and help newcomers get started

Thank you for contributing to hipDNN!
