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
- **[Testing](./docs/TESTING.md)** - Run the checks relevant to your change
- **[Testing Strategy](./docs/TESTING_STRATEGY.md)** - Understand the detailed test model and ownership
- **[Roadmap](./docs/Roadmap.md)** - Check planned features and find contribution opportunities
- **[Coding Style and Naming Guidelines](./docs/CodingStyleAndNamingGuidelines.md)** - Follow our coding conventions for consistency

We encourage you to open a GitHub issue to discuss your planned contribution before starting work. This helps ensure your efforts align with project goals and prevents duplicate work.

## Contribution Requirements

All contributions must meet the following requirements before they can be merged:

### Feature Proposals & RFCs

- **RFC Process**: For large or complex feature changes, contributors must provide a Request for Comments (RFC) proposal ahead of development.
  - This proposal should be discussed and iterated upon with maintainers prior to beginning feature work and implementation.
  - RFCs should be submitted as a Markdown document added to `hipdnn/docs/rfcs` via a Pull Request (e.g., see [PR #3266](https://github.com/ROCm/rocm-libraries/pull/3266)).
- **Phased Implementation**: We strongly encourage landing changes in multiple phases.
- **Small PRs**: Please keep Pull Requests small and focused.
  - This makes reviews easier to digest.
  - It minimizes the potential for conflicts or large feature reverts if issues are discovered later.

### Code Quality Standards

- **Formatting and Naming**: Follow [Coding Style and Naming Guidelines](./docs/CodingStyleAndNamingGuidelines.md).
- **Build Checks**: Use the canonical formatting and static-analysis targets documented in [Building](./docs/Building.md).
- **Compiler Warnings**: Code must compile without warnings.

### Testing Requirements

Add focused tests that cover the behavior changed by your contribution. Use the [Testing guide](./docs/TESTING.md) to choose and run checks, and the [Testing Strategy](./docs/TESTING_STRATEGY.md) for detailed test-layer and validation responsibilities. Test names must follow the [Coding Style and Naming Guidelines](./docs/CodingStyleAndNamingGuidelines.md#11-test-naming-guidelines).

Coverage and sanitizer automation have known enforcement and platform gaps. The 80% coverage figure is an aspirational goal, not a verified required-status gate; see [Known Gaps](./docs/KNOWN_GAPS.md) for current limitations.

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

## Development Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/rocm-libraries.git
cd rocm-libraries
git remote add upstream https://github.com/ROCm/rocm-libraries.git
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Build Locally

Follow the remaining instructions in the [Quick Start Guide](./docs/Building.md#quick-start-guide) to build hipDNN.

### 4. Set Up Pre-commit Hooks

hipDNN uses pre-commit hooks to automatically validate code quality. See the [main contributing guide](../../CONTRIBUTING.md#pre-commit-hooks) for general pre-commit setup instructions.

#### Installing flatc (Required for hipDNN)

hipDNN requires `flatc` version **25.9.23** for the FlatBuffers schema compiler. Install it before setting up pre-commit:

**Linux:**
```bash
wget https://github.com/google/flatbuffers/releases/download/v25.9.23/Linux.flatc.binary.g++-13.zip
unzip Linux.flatc.binary.g++-13.zip
sudo mv flatc /usr/local/bin/
sudo chmod +x /usr/local/bin/flatc
rm Linux.flatc.binary.g++-13.zip
```

**Windows:**
1. Download: https://github.com/google/flatbuffers/releases/download/v25.9.23/Windows.flatc.binary.zip.
2. Extract `flatc.exe` and add it to your system PATH.

**Verify installation:**
```bash
flatc --version
# Should output: flatc version 25.9.23
```

After `flatc` is installed, set up pre-commit:
```bash
pip install pre-commit
pre-commit install
```

### 5. Run All Required Checks

Before submitting your PR:

- Configure, build, and run sanitizer or coverage variants with the canonical commands in [Building](./docs/Building.md).
- Select tests appropriate to the change using [Testing](./docs/TESTING.md).
- Follow the formatting and naming rules in [Coding Style and Naming Guidelines](./docs/CodingStyleAndNamingGuidelines.md).

### 5. Create a Pull Request

- Push your changes to your fork
- Create a pull request against the main hipDNN repository
- Fill out the pull request template completely
- Ensure all CI checks pass

## Pull Request Checklist

Before opening a pull request:

- [ ] I added focused automated tests for the behavior introduced or changed.
- [ ] I ran the relevant checks described in [Testing](./docs/TESTING.md).
- [ ] I updated relevant documentation and removed documentation made stale by the change.
- [ ] I followed the formatting and naming guidance in [Coding Style and Naming Guidelines](./docs/CodingStyleAndNamingGuidelines.md).

## Getting Help

- **Questions**: Open a GitHub issue with your question
- **Discussion**: For design discussions or feature proposals, open an issue before starting work
- **RFCs**: For large or complex features, submitting an RFC is a great way to get early feedback on your design
- **Draft PRs**: Feel free to open a draft PR early to get feedback on your approach
- **CI Pipelines**: In your PR, ask members of the hipDNN team to run CI on your branch
- **Code Reviews**: Be responsive to code review feedback and make requested changes promptly

## Community Guidelines

- Be respectful and constructive in all interactions
- Help review other contributors' PRs when possible
- Share knowledge and help newcomers get started

Thank you for contributing to hipDNN!
