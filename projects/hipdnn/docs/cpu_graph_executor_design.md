# CPU Graph Executor Design Document

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture Components](#architecture-components)
4. [Plan Builders](#plan-builders)
5. [Signature Key System](#signature-key-system)
6. [Registry Mechanism](#registry-mechanism)
7. [Execution Flow](#execution-flow)
8. [Supported Operations](#supported-operations)
9. [Extension Guidelines](#extension-guidelines)

## Executive Summary

The CPU Graph Executor is a reference implementation system designed to execute computational graphs on CPU for testing and validation purposes within the hipDNN project. It provides a flexible, extensible architecture for executing various deep learning operations including BatchNormalization, Convolution, and Pointwise operations.

### Purpose
- **Reference Implementation**: Provides ground-truth results for validating GPU implementations
- **Testing Infrastructure**: Enables comprehensive testing of graph execution
- **Extensibility**: Supports easy addition of new operations through a plugin-like architecture
- **Type Safety**: Uses C++ templates to ensure compile-time type checking

## System Overview

The CPU Graph Executor follows a modular architecture pattern with clear separation of concerns:

``` 
┌───────────────────────────────────────────────────────────┐
│                  CPU Graph Executor System                │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Graph     │───>│  Signature   │───>│   Plan       │  │
│  │   Input     │    │   Key Gen    │    │   Builder    │  │
│  └─────────────┘    └──────────────┘    └──────────────┘  │
│                            │                     │        │
│                            v                     v        │
│                     ┌──────────────┐    ┌──────────────┐  │
│                     │   Registry   │    │   Plan       │  │
│                     │   Lookup     │    │  Executor    │  │
│                     └──────────────┘    └──────────────┘  │
│                                                  │        │
│                                                  v        │
│                                         ┌──────────────┐  │
│                                         │   Result     │  │
│                                         │   Tensors    │  │
│                                         └──────────────┘  │
└───────────────────────────────────────────────────────────┘
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
- **Purpose**: Interface for executing plans with tensor data. Each plan executor is specialized for a specific operation and data type combination.
- **Methods**:
  - `execute()`: Executes the plan with provided tensor data

### Main Components

```
┌───────────────────────────────────────────────────────────┐
│              CpuReferenceGraphExecutor                    │
├───────────────────────────────────────────────────────────┤
│ - _planRegistry: PlanBuilderRegistry                      │
├───────────────────────────────────────────────────────────┤
│ + execute(graphBuffer, size, variantPack)                 │
│ - buildPlanForNode(graph, node)                           │
│ - buildSignatureKey(node, tensorMap, computeType)         │
│ - populateVariantPackWithMissingVirtualTensors(...)       │
└───────────────────────────────────────────────────────────┘
                           │
                           │ uses
                           v
┌───────────────────────────────────────────────────────────┐
│              PlanBuilderRegistry                          │
├───────────────────────────────────────────────────────────┤
│ - _registry: PlanRegistryMap                              │
│ - _initialized: bool                                      │
├───────────────────────────────────────────────────────────┤
│ + getPlanBuilder(key): IGraphNodePlanBuilder&             │
│ - initializeRegistry()                                    │
│ - initializePlanBuilders()                                │
│ - registerBuilder<T>()                                    │
└───────────────────────────────────────────────────────────┘
```

## Plan Builders

Plan Builders are instantiated for each supported operation & data type combination. All plan builders are then stored in the PlanBuilderRegistry for lookup during graph execution.

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

### Plan Builder Template-Based Type Safety

Each plan builder uses templates to ensure type safety:

```cpp
template <typename InputDataType,
          typename ScaleBiasDataType,
          typename MeanVarianceDataType,
          typename ComputeDataType>
class BatchnormBwdPlan : public IGraphNodePlanExecutor
```

## Signature Key System

The Signature Key system provides unique identification for operation configurations:

```
┌─────────────────────────────────────────────────────────────┐
│                  PlanRegistrySignatureKey                   │
│                        (Variant)                            │
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
1. **Constructor from Node**: Build from `Node` and `tensorMap` for runtime graph parsing
2. **Constructor from Datatypes**: Build from base datatypes for direct instantiation
2. **hashSelf()**: Generate unique hash
3. **Equality operator**: Compare keys
4. **getPlanBuilders()**: Return map of supported builder types for this key

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
┌────────────────────────────────────────────────────┐
│                 PlanBuilderRegistry                │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │         _registry (PlanRegistryMap)          │  │
│  ├──────────────────────────────────────────────┤  │
│  │  Key: PlanRegistrySignatureKey               │  │
│  │  Value: unique_ptr<IGraphNodePlanBuilder>    │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  Registration Process:                             │
│  1. Initialize on first use (lazy initialization)  │
│  2. Each operation type registers its builders     │
│  3. Builders mapped to unique signature keys       │
│                                                    │
└────────────────────────────────────────────────────┘
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
    Lookup builder           Populate _registry
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
├─────────────────────────────────────────────┤
│  - Identify missing virtual tensors         │
│  - Allocate memory for intermediates        │
└────────┬────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────┐
│         Execute All Plans                   │
├─────────────────────────────────────────────┤
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
| Unary Operations | `PointwisePlanBuilder` | `PointwiseSignatureKey` | RELU_FWD, SIGMOID_FWD, TANH_FWD, ABS, NEG |
| Binary Operations | `PointwisePlanBuilder` | `PointwiseSignatureKey` | ADD, SUB, MUL, RELU_BWD, SIGMOID_BWD, TANH_BWD |
| Ternary Operations | `PointwisePlanBuilder` | `PointwiseSignatureKey` | None Supported Yet |

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
    MyOperationSignatureKey(DataType dataType1, DataType dataType2, /* ... */);
    MyOperationSignatureKey(const Node& node, const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap);
    size_t hashSelf() const;
    bool operator==(const MyOperationSignatureKey& other) const;
    static std::unordered_map<MyOperationSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              MyOperationSignatureKey> getPlanBuilders();
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

## Performance Considerations

While the CPU Graph Executor is primarily for reference and testing:

1. **Memory Management**: Virtual tensors are allocated on-demand
2. **Sequential Execution**: Operations execute in topological order
3. **Type Specialization**: Templates enable different typed implementations
