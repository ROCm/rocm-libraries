# Plugin SDK Design Document

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
2. [Current System Overview](#current-system-overview)
3. [Proposed Design](#proposed-design)
7. [Execution Plan](#execution-plan)

## Executive Summary


## Problem Statement


## Current System Overview

The hipDNN framework consists of a frontend (C++ Graph API), a backend (core runtime), and a plugin system. The backend manages prepares and dispatches execution to dynamically loaded plugins via a C-API interface.

![Current System Overview](../images/current_system_overview.png)

### MIOpen Plugin Architecture

The MIOpen Legacy Plugin serves as the kernel provider. It employs a modular C++ architecture, largely decoupled from the API layer.

*   **Dependency Injection Container (`MiopenContainer`):**
    This is the root object that manages the lifecycle and dependencies of all other components. It initializes the `EngineManager` and ensures that all necessary services are correctly injected.

*   **Engine Manager (`EngineManager`):**
    The central registry for execution engines. It orchestrates the selection of the appropriate engine for a given operation graph by querying its registered engines.

*   **Plan Builders (`IPlanBuilder`):**
    Each engine is associated with a set of Plan Builders. These components are responsible for:
    *   **Applicability:** Inspecting an operation graph to determine if the engine can execute it.
    *   **Resource Estimation:** Calculating the required workspace size.
    *   **Plan Construction:** Creating an executable `IPlan` object if the graph is supported.

*   **Plans (`IPlan`):**
    An `IPlan` represents a strategy for executing a specific operation graph. It encapsulates all the necessary logic and state to run the routine, abstracting the details from the higher-level engine management.

*   **C-API Interface:**
    A thin translation layer that exposes these internal C++ components to the backend via the required engine plugin C-API.

### Execution Flow

When the backend requests a graph execution, the flow within the plugin is as follows:

1.  **Ingestion:** The C-API bridge receives the raw graph handle and forwards it to the `MiopenContainer`.
2.  **Selection:** The `EngineManager` iterates through registered engines to find a candidate.
3.  **Compilation:** The selected Engine's `PlanBuilder` validates the graph and constructs an `IPlan`.
4.  **Execution:** The `IPlan` executes the operation, marshaling pointers from the backend's `VariantPack` to the underlying device kernels.

This architecture effectively separates the plugin interface from the engine implementation details. However, currently, this infrastructure is largely internal to the MIOpen plugin. The goal of the Plugin SDK is to standardize and provide these as reusable components for plugin development, so developers can focus on the implementations of the underlying kernels and libraries.

## Proposed Design


## Execution Plan
