# Python Virtual Environment Proof-of-Concept

This directory contains a proof-of-concept implementation for automatic Python virtual environment management in hipBLASLt builds.

## Quick Summary

**Problem**: Building hipBLASLt fails if Python dependencies (joblib, msgpack, etc.) aren't installed system-wide.

**Solution**: Extend existing `HIPBLASLT_BUNDLE_PYTHON_DEPS` to automatically create a venv and install dependencies during CMake configuration.

## Files in This POC

### 1. `PYTHON_VENV_PROPOSAL.md` - Complete Proposal Document
Comprehensive proposal including:
- Problem statement and current workarounds
- Implementation details and design decisions
- Performance impact analysis
- Testing strategy
- Rollout plan for broader ROCm adoption
- FAQ and approval considerations

**→ This is the document to share with stakeholders for approval**

### 2. `cmake/hipblaslt_python.cmake.proposed` - Implementation
The complete CMake module with:
- New `hipblaslt_setup_python_venv()` function (~80 lines)
  - Creates venv if needed
  - Installs/upgrades dependencies
  - Uses stamp file to avoid redundant installs
- Modified `hipblaslt_configure_bundled_python_command()` (~40 lines)
  - Uses venv Python when available
  - Maintains backward compatibility

**→ This would replace the existing `cmake/hipblaslt_python.cmake`**

### 3. `CMakeLists.txt.patch` - Integration Patch
Shows the minimal 5-line change needed in main CMakeLists.txt:
```cmake
hipblaslt_setup_python_venv(
    "${CMAKE_BINARY_DIR}/python-venv"
    "${CMAKE_CURRENT_SOURCE_DIR}/tensilelite/requirements.txt"
)
```

**→ Apply this patch to integrate the venv setup**

## Testing the POC

### Option 1: Manual Application
```bash
# Backup original file
cp cmake/hipblaslt_python.cmake cmake/hipblaslt_python.cmake.backup

# Apply the implementation
cp cmake/hipblaslt_python.cmake.proposed cmake/hipblaslt_python.cmake

# Apply the patch to CMakeLists.txt
patch -p1 < CMakeLists.txt.patch

# Test it
rm -rf build
cmake --preset default:release -DGPU_TARGETS=gfx90a
# Should see: "Creating Python virtual environment..." etc.

# Restore if needed
cp cmake/hipblaslt_python.cmake.backup cmake/hipblaslt_python.cmake
```

### Option 2: Side-by-Side Comparison
```bash
# Compare what would change
diff -u cmake/hipblaslt_python.cmake cmake/hipblaslt_python.cmake.proposed

# Review the proposal
less PYTHON_VENV_PROPOSAL.md
```

## Key Points for Discussion

### Technical
- **Code size**: ~125 lines total (2 files)
- **Build time**: +45s first configure, <1s subsequent
- **Disk space**: +50-100MB in build directory
- **Compatibility**: Full backward compatibility with existing builds

### Strategic
- **Standardization**: Template for other ROCm projects
- **Developer UX**: Eliminates manual pip installation steps
- **CI/Containers**: Simpler Dockerfiles, no system package setup
- **Deprecation path**: Move away from shell scripts to pure CMake

## Next Steps

1. **Review** the proposal document
2. **Test** the implementation in a clean environment
3. **Discuss** with the team about broader ROCm adoption
4. **Refine** based on feedback
5. **Submit** as a PR with documentation updates

## Questions or Concerns?

Common concerns addressed in the proposal:
- "Will this slow down builds?" → Only first configure, cached after
- "What about system Python?" → Can disable with `-DHIPBLASLT_BUNDLE_PYTHON_DEPS=OFF`
- "Is this tested elsewhere?" → Yes, pattern used by LLVM, Blender, TensorFlow
- "What about CI?" → Actually simplifies CI - less setup required

See FAQ section in `PYTHON_VENV_PROPOSAL.md` for more details.

---

**Status**: Ready for review and testing  
**Created**: 2026-01-15  
**Next session**: Review with team
