# CPU Graph Executor Design Document

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture Components](#architecture-components)
4. [Plan Builder Pattern](#plan-builder-pattern)
5. [Plan Execution Pattern](#plan-execution-pattern)
6. [Signature Key System](#signature-key-system)
7. [Registry Mechanism](#registry-mechanism)
8. [Execution Flow](#execution-flow)
9. [Supported Operations](#supported-operations)
10. [Extension Guidelines](#extension-guidelines)

## Executive Summary

The CPU Graph Executor is a reference implementation system designed to execute computational graphs on CPU for testing and validation purposes within the hipDNN framework. It provides a flexible, extensible architecture for executing various deep learning operations including BatchNormalization, Convolution, and Pointwise operations.

### Purpose
- **Reference Implementation**: Provides ground-truth results for validating GPU implementations
- **Testing Infrastructure**: Enables comprehensive testing of graph execution
- **Extensibility**: Supports easy addition of new operations through a plugin-like architecture
- **Type Safety**: Uses C++ templates to ensure compile-time type checking

## System Overview

The CPU Graph Executor follows a modular architecture pattern with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                     CPU Graph Executor System                    │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Graph     │───>│  Signature   │───>│   Plan       │       │
│  │   Input     │    │   Key Gen    │    │   Builder    │       │
│  └─────────────┘    └──────────────┘    └──────────────┘       │
│         │                  │                     │               │
│         v                  v                     v               │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Graph     │    │   Registry   │    │   Plan       │       │
│  │   Wrapper   │    │   Lookup     │    │  Executor    │       │
│  └─────────────┘    └──────────────┘    └──────────────┘       │
│                                                  │               │
│                                                  v               │
│                                          ┌──────────────┐       │
│                                          │   Result     │       │
│                                          │   Tensors    │       │
│                                          └──────────────┘       │
└───────────────────────────────────────────────────────────────────┘
```

## Architecture Components

### Core Interfaces

```
                    ┌────────────────────────┐
                    │ IGraphNodePlanBuilder  │
                    └───────────┬────────────┘
                                │ builds
                                v
                    ┌────────────────────────┐
                    │ IGraphNodePlanExecutor │
                    └────────────────────────┘
```

#### IGraphNodePlanBuilder
- **Purpose**: Interface for creating execution plans for graph nodes
- **Methods**:
  - `isApplicable()`: Checks if builder can handle a specific node
  - `buildNodePlan()`: Creates an execution plan for the node

#### IGraphNodePlanExecutor
- **Purpose**: Interface for executing plans with tensor data
- **Methods**:
  - `execute()`: Executes the plan with provided tensor data

### Main Components

```
┌───────────────────────────────────────────────────────────┐
│              CpuReferenceGraphExecutor                     │
├───────────────────────────────────────────────────────────┤
│ - _planRegistry: PlanBuilderRegistry                       │
├───────────────────────────────────────────────────────────┤
│ + execute(graphBuffer, size, variantPack)                  │
│ - buildPlanForNode(graph, node)                           │
│ - buildSignatureKey(node, tensorMap, computeType)         │
│ - populateVariantPackWithMissingVirtualTensors(...)       │
└───────────────────────────────────────────────────────────┘
                           │
                           │ uses
                           v
┌───────────────────────────────────────────────────────────┐
│              PlanBuilderRegistry                           │
├───────────────────────────────────────────────────────────┤
│ - _registry: PlanRegistryMap                               │
│ - _initialized: bool                                       │
├───────────────────────────────────────────────────────────┤
│ + getPlanBuilder(key): IGraphNodePlanBuilder&             │
│ - initializeRegistry()                                     │
│ - initializePlanBuilders()                                │
│ - registerBuilder<T>()                                     │
└───────────────────────────────────────────────────────────┘
```

## Plan Builder Pattern

The Plan Builder pattern is used to create execution plans for different operation types:

```
┌────────────────────┐
│  Plan Builder      │
│  (Abstract)        │
└────────┬───────────┘
         │
         ├──────────────────────┬──────────────────────┬─────────────────────┐
         │                      │                      │                     │
┌────────v──────────┐  ┌───────v───────────┐  ┌──────v──────────┐  ┌───────v────────┐
│ BatchnormBwdPlan  │  │ ConvolutionFwdPlan│  │ PointwisePlan   │  │ Other Plans... │
│     Builder       │  │     Builder       │  │    Builder      │  │                │
└───────────────────┘  └───────────────────┘  └─────────────────┘  └────────────────┘
         │                      │                      │                     │
         │                      │                      │                     │
         v                      v                      v                     v
┌───────────────────┐  ┌───────────────────┐  ┌─────────────────┐  ┌────────────────┐
│ BatchnormBwdPlan  │  │ ConvolutionFwdPlan│  │ PointwisePlan   │  │ Other Plans... │
└───────────────────┘  └───────────────────┘  └─────────────────┘  └────────────────┘
```

### Template-Based Type Safety

Each plan builder uses templates to ensure type safety:

```cpp
template <typename InputDataType,
          typename ScaleBiasDataType,
          typename MeanVarianceDataType,
          typename ComputeDataType>
class BatchnormBwdPlan : public IGraphNodePlanExecutor
```

## Plan Execution Pattern

Plans are executed in a two-phase approach:

### Phase 1: Plan Creation
```
   Graph Node
       │
       v
   isApplicable() ──> Check tensor types
       │              Check node attributes
       │              Validate configuration
       v
   buildNodePlan() ──> Create specialized plan
       │              Configure parameters
       │              Return executor
       v
   Plan Executor
```

### Phase 2: Plan Execution
```
   Plan Executor
       │
       v
   execute(variantPack)
       │
       ├──> Create shallow tensors
       ├──> Invoke CPU reference implementation
       └──> Write results to output tensors
```

## Signature Key System

The Signature Key system provides unique identification for operation configurations:

```
┌─────────────────────────────────────────────────────────────┐
│                  PlanRegistrySignatureKey                    │
│                        (Variant)                             │
├─────────────────────────────────────────────────────────────┤
│  - BatchnormFwdInferenceSignatureKey                        │
│  - BatchnormBwdSignatureKey                                 │
│  - BatchnormTrainSignatureKey                               │
│  - ConvolutionFwdSignatureKey                               │
│  - ConvolutionBwdSignatureKey                               │
│  - ConvolutionWrwSignatureKey                               │
│  - PointwiseSignatureKey                                    │
└─────────────────────────────────────────────────────────────┘
```

### Signature Key Requirements

Each signature key must implement:
1. **Constructor**: Build from `Node` and `tensorMap`
2. **hashSelf()**: Generate unique hash
3. **Equality operator**: Compare keys
4. **getPlanBuilders()**: Return map of builders

### Signature Key Generation

```
Node Attributes + Tensor Types + Compute Type
                    │
                    v
            ┌──────────────┐
            │ Signature    │
            │ Key Builder  │
            └──────┬───────┘
                   │
                   v
            Unique Key Hash
                   │
                   v
            Registry Lookup
```

## Registry Mechanism

The registry provides a centralized mapping of signature keys to plan builders:

```
┌───────────────────────────────────────────────────────┐
│                 PlanBuilderRegistry                    │
├───────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────────────────────────────────────────┐    │
│  │         _registry (PlanRegistryMap)          │    │
│  ├──────────────────────────────────────────────┤    │
│  │  Key: PlanRegistrySignatureKey               │    │
│  │  Value: unique_ptr<IGraphNodePlanBuilder>    │    │
│  └──────────────────────────────────────────────┘    │
│                                                        │
│  Registration Process:                                 │
│  1. Initialize on first use (lazy initialization)     │
│  2. Each operation type registers its builders        │
│  3. Builders mapped to unique signature keys          │
│                                                        │
└───────────────────────────────────────────────────────┘
```

### Registry Initialization Flow

```
First getPlanBuilder() call
         │
         v
    Is initialized? ──No──> initializeRegistry()
         │                           │
        Yes                          v
         │                   registerBuilder<T>()
         │                   for each operation type
         │                           │
         v                           v
    Lookup builder              Populate _registry
         │                           │
         v                           │
    Return builder <─────────────────┘
```

## Execution Flow

The complete execution flow from graph input to results:

```
┌─────────────────┐
│  Graph Buffer   │
└────────┬────────┘
         │
         v
┌─────────────────────────────────────────────┐
│         GraphWrapper Creation                │
│  - Parse flatbuffer                         │
│  - Extract nodes and tensors                │
└────────┬────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────┐
│      For Each Node (Topological Order)      │
├─────────────────────────────────────────────┤
│                                             │
│  1. Build Signature Key                     │
│     - Extract node attributes               │
│     - Identify tensor types                 │
│     - Determine compute type                │
│                                             │
│  2. Registry Lookup                         │
│     - Find matching plan builder            │
│     - Verify applicability                  │
│                                             │
│  3. Build Execution Plan                    │
│     - Create specialized executor           │
│     - Configure parameters                  │
│                                             │
└────────┬────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────┐
│        Virtual Tensor Allocation            │
│  - Identify missing virtual tensors         │
│  - Allocate memory for intermediates        │
└────────┬────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────┐
│         Execute All Plans                   │
│  - Sequential execution                     │
│  - Pass variant pack to each executor       │
│  - Results written to output tensors        │
└─────────────────────────────────────────────┘
```

## Supported Operations

### BatchNormalization Operations

| Operation | Plan Builder | Signature Key | Description |
|-----------|-------------|---------------|-------------|
| BatchNorm Forward Inference | `BatchnormFwdInferencePlanBuilder` | `BatchnormFwdInferenceSignatureKey` | Inference-mode forward pass |
| BatchNorm Forward Training | `BatchnormTrainPlanBuilder` | `BatchnormTrainSignatureKey` | Training-mode forward pass with statistics |
| BatchNorm Backward | `BatchnormBwdPlanBuilder` | `BatchnormBwdSignatureKey` | Gradient computation |

### Convolution Operations

| Operation | Plan Builder | Signature Key | Description |
|-----------|-------------|---------------|-------------|
| Convolution Forward | `ConvolutionFwdPlanBuilder` | `ConvolutionFwdSignatureKey` | Forward convolution |
| Convolution Backward Data | `ConvolutionBwdPlanBuilder` | `ConvolutionBwdSignatureKey` | Data gradient computation |
| Convolution Backward Weights | `ConvolutionWrwPlanBuilder` | `ConvolutionWrwSignatureKey` | Weight gradient computation |

### Pointwise Operations

| Operation | Plan Builder | Signature Key | Description |
|-----------|-------------|---------------|-------------|
| Unary Operations | `PointwisePlanBuilder` | `PointwiseSignatureKey` | Element-wise unary operations |
| Binary Operations | `PointwisePlanBuilder` | `PointwiseSignatureKey` | Element-wise binary operations |
| Ternary Operations | `PointwisePlanBuilder` | `PointwiseSignatureKey` | Element-wise ternary operations |

## Extension Guidelines

### Adding a New Operation

To add support for a new operation, follow these steps:

#### 1. Create Plan Executor
```cpp
template <typename DataType>
class MyOperationPlan : public IGraphNodePlanExecutor {
public:
    void execute(const std::unordered_map<int64_t, void*>& variantPack) override {
        // Implementation
    }
};
```

#### 2. Create Plan Builder
```cpp
template <DataType DataTypeEnum>
class MyOperationPlanBuilder : public IGraphNodePlanBuilder {
public:
    bool isApplicable(...) const override { /* ... */ }
    std::unique_ptr<IGraphNodePlanExecutor> buildNodePlan(...) const override { /* ... */ }
};
```

#### 3. Create Signature Key
```cpp
class MyOperationSignatureKey {
public:
    MyOperationSignatureKey(const Node& node, const TensorMap& tensorMap);
    size_t hashSelf() const;
    bool operator==(const MyOperationSignatureKey& other) const;
    static PlanRegistryMap getPlanBuilders();
};
```

#### 4. Update Registry Variant
Add the new signature key to `PlanRegistrySignatureKey` variant:
```cpp
using PlanRegistrySignatureKey = std::variant<
    // ... existing keys ...
    MyOperationSignatureKey
>;
```

#### 5. Update Graph Executor
Add case to `buildSignatureKey()` method in `CpuReferenceGraphExecutor`:
```cpp
case NodeAttributes::MyOperationAttributes:
    return MyOperationSignatureKey(node, tensorMap);
```

### Best Practices

1. **Type Safety**: Use templates to ensure compile-time type checking
2. **Validation**: Implement thorough validation in `isApplicable()`
3. **Error Handling**: Provide clear error messages for unsupported configurations
4. **Testing**: Create comprehensive unit tests for new operations
5. **Documentation**: Document tensor requirements and operation semantics

### Utility Macros

The system provides utility macros for common validation tasks:

```cpp
CHECK_TENSOR_EXISTS(tensor_map, tensor_uid)
CHECK_TENSOR_TYPE(tensor_map, tensor_uid, datatype_enum)  
CHECK_OPTIONAL_TENSOR_EXISTS(tensor_map, optional_tensor_uid)
CHECK_OPTIONAL_TENSOR_TYPE(tensor_map, optional_tensor_uid, datatype_enum)
```

## Performance Considerations

While the CPU Graph Executor is primarily for reference and testing:

1. **Memory Management**: Virtual tensors are allocated on-demand
2. **Sequential Execution**: Operations execute in topological order
3. **Type Specialization**: Templates enable optimized implementations
4. **Shallow Tensors**: Avoid unnecessary data copies

## Testing Infrastructure

The CPU Graph Executor is extensively tested:

```
sdk/tests/test_utilities/cpu_graph_executor/
├── TestBatchnormBwdPlan.cpp
├── TestBatchnormFwdInferencePlan.cpp
├── TestConvolutionFwdPlan.cpp
├── TestPointwisePlan.cpp
├── TestCpuReferenceGraphExecutor.cpp
├── TestPlanRegistrySignatureKey.cpp
└── ... (additional test files)
```

## Conclusion

The CPU Graph Executor provides a robust, extensible framework for reference implementations of deep learning operations. Its modular architecture, type-safe design, and comprehensive testing infrastructure make it an essential component of the hipDNN validation ecosystem.

The system's clear separation of concerns through interfaces, use of the registry pattern for extensibility, and template-based type safety ensure maintainability and reliability while supporting the addition of new operations as the framework evolves.
