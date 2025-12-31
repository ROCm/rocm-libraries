# Operation Registry Design

## Overview

The **Operation Registry** is a general-purpose system for building, caching, and optimizing reusable IR operations in StinkyTofu. It provides a flexible infrastructure for managing operations like activation functions, global memory reads/writes, and MFMA instructions.

## Motivation

### The Problem

Before the Operation Registry, complex operations (like activation functions) were either:
1. **Built on-the-fly every time** - causing redundant optimization overhead
2. **Manually cached** - requiring developers to implement custom caching logic for each operation
3. **Not optimized** - leaving performance on the table

The Python 'Activation.py' demonstrated the value of build-once-optimize-once-cache-reuse pattern, but this pattern needed to be generalized and integrated into the C++ codebase.

### The Solution

The Operation Registry provides:
- **Centralized registration** - Single place to register all operations
- **Automatic caching** - Operations cache their results based on configuration
- **Optional optimization** - Operations can opt-in to optimization pipelines
- **Flexible configuration** - Each operation controls its own caching and optimization behavior
- **Statistics tracking** - Built-in performance monitoring

## Architecture

### Core Components

'''
+-------------------------------------------------------------+
|                    OperationRegistry                        |
|  - builderFactories: map<string, factory<OperationBuilder>> |
|  - cache: map<string, map<cacheKey, IRList>>                |
|  - stats: map<string, CacheStats>                           |
|  - cacheEnabled: bool                                        |
+-------------------------------------------------------------+
                              |
                              | delegates to
                              ?
                    +-----------------+
                    | OperationBuilder| (abstract)
                    |  - buildRaw()   |
                    |  - isCacheable()|
                    |  - shouldOptimize()|
                    |  - getOptimizationConfig()|
                    +-----------------+
                              |
                    +---------+---------+
                    |                   |
          +---------?--------+  +------?-----------+
          |ActivationBuilder |  |GlobalReadBuilder |
          |  (Concrete)      |  |   (Concrete)     |
          +------------------+  +------------------+
'''

### Key Classes

#### 1. 'OperationConfig' (Base Class)

Represents the configuration for an operation. Subclasses define operation-specific parameters.

'''cpp
class OperationConfig {
public:
    virtual std::string getCacheKey() const = 0;  // Generate unique cache key
    virtual bool validate() const = 0;             // Validate configuration
    virtual ~OperationConfig() = default;
};
'''

**Responsibilities:**
- Define operation-specific parameters
- Generate cache keys for deduplication
- Validate configurations before use

#### 2. 'OperationBuilder' (Base Class)

Builds IR instructions for an operation. Subclasses implement the actual IR generation logic.

'''cpp
class OperationBuilder {
public:
    virtual OperationResult buildRaw(const OperationConfig& config) = 0;
    virtual bool isCacheable() const { return true; }
    virtual bool shouldOptimize() const { return true; }
    virtual PipelineConfig getOptimizationConfig() const;
};
'''

**Responsibilities:**
- Generate raw IR instructions
- Declare caching/optimization preferences
- Provide optimization configuration

#### 3. 'OperationRegistry'

Central registry that manages all operations.

'''cpp
class OperationRegistry {
public:
    template<typename BuilderT>
    void registerOperation(const std::string& name);

    OperationResult build(const std::string& opName, const OperationConfig& config);

    void clearCache(const std::string& opName);
    void setCacheEnabled(bool enabled);
    CacheStats getStats(const std::string& opName) const;
};
'''

**Responsibilities:**
- Register operation builders
- Coordinate build -> optimize -> cache flow
- Track cache statistics

### Data Flow

'''
+---------------------------------------------------------------------+
| 1. User calls registry.build("activation", config)                 |
+----------------------------+----------------------------------------+
                             |
                             ?
                    +----------------+
                    | Generate Cache |
                    |      Key       |
                    +--------+-------+
                             |
                             ?
                    +----------------+
                    |  Cache Lookup  |?------- cacheEnabled?
                    +--------+-------+
                             |
                    +--------+--------+
                    |                 |
                 Hit|                 |Miss
                    ?                 ?
            +--------------+  +--------------+
            |Return cached |  |Call buildRaw()|
            |instructions  |  |               |
            +--------------+  +------+--------+
                                     |
                                     ?
                            +-----------------+
                            |shouldOptimize()?|
                            +--------+---------+
                                     |
                            +--------+--------+
                            |                 |
                         Yes|                 |No
                            ?                 ?
                   +-----------------+  +---------+
                   |Run Optimization |  |  Done   |
                   |    Pipeline     |  +---------+
                   +--------+---------+
                            |
                            ?
                   +-----------------+
                   |Store in Cache   |?--- if isCacheable()
                   +--------+---------+
                            |
                            ?
                   +-----------------+
                   |Return optimized |
                   |  instructions   |
                   +-----------------+
'''

## Design Decisions

### 1. Why Separate Config from Builder?

**Decision:** Configuration ('OperationConfig') is separate from builder logic ('OperationBuilder').

**Rationale:**
- **Serializable:** Configs can be easily serialized/deserialized for cache persistence
- **Testable:** Configs can be validated independently
- **Composable:** Same config can be used with different builders
- **Cache Key Generation:** Configs own their cache key logic

### 2. Why Virtual Methods for Caching/Optimization?

**Decision:** 'isCacheable()', 'shouldOptimize()', and 'getOptimizationConfig()' are virtual methods.

**Rationale:**
- **Flexibility:** Each operation decides its own caching/optimization strategy
- **Simplicity:** Simple operations (like ReLU) don't need optimization
- **Performance:** Complex operations (like GELU) opt-in to aggressive optimization
- **Evolution:** New operations can have different strategies without changing the registry

### 3. Why Pass StinkyTofu and StinkyIR to Builder?

**Decision:** 'OperationBuilder' receives both 'StinkyTofu' (low-level builder) and 'StinkyIR' (high-level builder).

**Rationale:**
- **Low-level operations:** Can use 'StinkyTofu' directly for maximum control
- **High-level operations:** Can use 'StinkyIR' for complex instruction sequences
- **Consistency:** Mirrors the existing StinkyTofu architecture

### 4. Why Not Automatic Register Remapping?

**Decision:** The registry does NOT automatically remap registers in cached results.

**Rationale:**
- **Correctness:** The Python 'Activation.py' performs deepcopy + remapping, which is expensive
- **Simplicity:** Cache keys include register usage patterns, so different register assignments use the same cached code
- **Performance:** Users can manually remap if needed, or accept the cached register assignment

**Alternative Considered:** Automatic remapping using 'VirtualRegisterRemapping' pass
- **Rejected because:** Too complex, breaks optimization assumptions, and Python's approach is already proven

### 5. Cache Key Strategy

**Decision:** Cache keys are operation-specific strings generated by 'OperationConfig::getCacheKey()'.

**Example:**
'''cpp
std::string ActivationConfig::getCacheKey() const {
    std::ostringstream oss;
    oss << static_cast<int>(activationType) << ":"
        << static_cast<int>(dataType) << ":"
        << params.size() << ":"
        << tempVgprs.size();
    return oss.str();
}
'''

**Rationale:**
- **Deduplication:** Different 'vgprIn'/'vgprOut' but same activation type -> same cache entry
- **Flexibility:** Each operation controls what goes into the cache key
- **Debuggability:** String keys are human-readable

## Operation Lifecycle

### 1. Registration

'''cpp
OperationRegistry registry(builder, ir, true);  // cacheEnabled = true
registry.registerOperation<ActivationBuilder>("activation");
'''

### 2. Building

'''cpp
ActivationConfig cfg(ActivationType::Gelu, ActivationDataType::F32,
                     vgprIn, vgprOut, {}, {tempVgpr});
auto result = registry.build("activation", cfg);
'''

### 3. Result Structure

'''cpp
struct OperationResult {
    std::vector<StinkyInstruction*> instructions;  // Generated IR
    bool fromCache;                                // Cache hit?
    size_t instructionCountBeforeOpt;              // Before optimization
    size_t instructionCountAfterOpt;               // After optimization
    int optimizationIterations;                    // Number of opt passes
};
'''

### 4. Statistics

'''cpp
auto stats = registry.getStats("activation");
std::cout << "Builds: " << stats.totalBuilds << "\n";
std::cout << "Hit rate: " << stats.hitRate() << "%\n";
'''

## Example: Activation Functions

### Configuration

'''cpp
class ActivationConfig : public OperationConfig {
    ActivationType activationType;
    ActivationDataType dataType;
    uint32_t vgprIn, vgprOut;
    std::vector<double> params;      // e.g., alpha for LeakyReLU
    std::vector<uint32_t> tempVgprs; // Temp registers for complex activations

    std::string getCacheKey() const override {
        // Cache based on type, datatype, param count, temp count
        // NOT on actual register numbers
    }

    bool validate() const override {
        // Check: GELU needs 1 temp, LeakyReLU needs alpha, etc.
    }
};
'''

### Builder

'''cpp
class ActivationBuilder : public OperationBuilder {
    OperationResult buildRaw(const OperationConfig& config) override {
        auto& cfg = static_cast<const ActivationConfig&>(config);

        std::vector<StinkyInstruction*> instructions;

        switch (cfg.activationType) {
            case ActivationType::Relu:
                instructions = ir.reluF32(cfg.vgprIn, cfg.vgprOut);
                break;
            case ActivationType::Gelu:
                instructions = ir.geluF32(cfg.vgprIn, cfg.vgprOut, cfg.tempVgprs[0]);
                break;
            // ... other activations
        }

        return {instructions, false, instructions.size(), 0, 0};
    }

    bool isCacheable() const override { return true; }
    bool shouldOptimize() const override { return true; }

    PipelineConfig getOptimizationConfig() const override {
        return ir.getActivationOptimizationConfig();  // Peephole + DCE + DuplicateElim
    }
};
'''

## Performance Characteristics

### Time Complexity

- **Registration:** O(1)
- **Cache Hit:** O(1) lookup + O(n) instruction copy
- **Cache Miss:** O(build) + O(optimize) + O(n) cache store
- **Optimization:** O(n2) in worst case (DCE), typically O(n)

### Space Complexity

- **Cache Storage:** O(unique_configs x avg_instruction_count)
- **Per-operation Overhead:** ~100 bytes (builderFactory + stats)

### Optimization Impact

Measured on 'gelu_f32' (13 instructions):
- **First call:** ~5ms (build + 3x optimization iterations)
- **Subsequent calls:** ~0.1ms (cache hit)
- **Speedup:** ~50x

## Extension Points

### Adding a New Operation

1. **Define Config:**
   '''cpp
   class GlobalReadConfig : public OperationConfig {
       bool useBuffer;
       int bpl;
       bool hi16;
       // ...
   };
   '''

2. **Define Builder:**
   '''cpp
   class GlobalReadBuilder : public OperationBuilder {
       OperationResult buildRaw(const OperationConfig& config) override;
   };
   '''

3. **Register:**
   '''cpp
   registry.registerOperation<GlobalReadBuilder>("global_read");
   '''

4. **Use:**
   '''cpp
   GlobalReadConfig cfg{true, 4, false};
   auto result = registry.build("global_read", cfg);
   '''

### Variant Operations

For operations with multiple implementation variants (e.g., buffer vs flat memory):

'''cpp
class VariantOperationBuilder : public OperationBuilder {
    std::map<std::string, std::unique_ptr<OperationVariant>> variants;

    OperationResult buildRaw(const OperationConfig& config) override {
        std::string variant = selectVariant(config);
        return variants[variant]->build(config);
    }
};
'''

## Limitations and Future Work

### Current Limitations

1. **No Persistent Cache:** Cache is in-memory only; lost on program restart
2. **No Register Remapping:** Cached code uses original register assignments
3. **Coarse-grained Cache Keys:** Cannot cache based on constants or immediate values
4. **Single-threaded:** Cache access is not thread-safe

### Future Enhancements

1. **Persistent Cache:**
   - Serialize cached IR to disk
   - Load on startup for instant availability
   - Versioning for cache invalidation

2. **Virtual Register Support:**
   - Use placeholder registers in cached code
   - Remap to actual registers on retrieval
   - Enables more cache hits

3. **Thread-safe Cache:**
   - Add mutex/RWLock for concurrent access
   - Per-operation locks for better parallelism

4. **Advanced Statistics:**
   - Time saved by caching
   - Memory usage per operation
   - Optimization effectiveness metrics

5. **Cache Eviction:**
   - LRU policy for memory-constrained scenarios
   - Per-operation cache size limits

## Relationship to Other Systems

### OptimizationPipeline

The Operation Registry **uses** the 'OptimizationPipeline':
- Each 'OperationBuilder' can provide its own 'PipelineConfig'
- Optimization happens automatically if 'shouldOptimize()' returns true
- Results are cached post-optimization

### StinkyIR

The Operation Registry **complements** 'StinkyIR':
- 'StinkyIR' provides high-level functions (like 'reluF32', 'geluF32')
- Operation Registry adds caching and optimization on top
- Developers can use 'StinkyIR' directly for simple cases, or registry for repeated operations

### Pattern-based Passes

The Operation Registry is **orthogonal** to pattern-based passes:
- Peephole/DCE/DuplicateElim optimize individual instructions
- Operation Registry caches entire operation sequences
- Together, they provide multi-level optimization

## References

- Python reference implementation: 'projects/hipblaslt/tensilelite/Tensile/Activation.py'
- Optimization pipeline design: 'docs/design/optimization-pipeline.md'
- StinkyIR design: 'docs/design/stinky-ir.md'

