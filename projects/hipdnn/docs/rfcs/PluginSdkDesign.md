# hipDNN - Plugin SDK Design Document

- Contributors: Mitch Ousdahl, Sam Reeder, Adam Dickin
- Original Implementation: Adam Dickin
- Last Update: Dec 5, 2025

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Current System Overview](#current-system-overview)
4. [Proposed Design](#proposed-design)
5. [Key Design Decisions](#5-key-design-decisions)
6. [Risks](#6-risks)
6. [Execution Plan](#7-execution-plan)
7. [Testing Plan](#8-testing-plan)

## 1. Executive Summary
hipDNN represents a system for the validation and execution of complex graphs of tensor operations that can be expressed as a directed acyclic graph (DAG).  Rather that directly include engines that can capably execute those graphs directly in hipDNN code, hipDNN has implemented a plugin mechanism wherein engine authors can fulfill a simple C-compatible API with the goal of generating a dynamic plugin library.  This allows the system the capability for flexibility without recompilation, allows for future growth, and allows for a robust ecosystem of engine providers.

This proposal describes two pieces of work.

The first is to split the **Plugin SDK** from our existing **SDK**, with the eventual goal of making the original SDK a **Data SDK** that describes the data contract between the frontend and backend of hipDNN.  The **Plugin SDK** will be solely to aid authors of plugins.

**❓Open Question - Will we have a separate `engine_plugin_sdk` and `heuristic_plugin_sdk`?**

The second part of the proposal is for the design of an optional framework that will assist a hipDNN plugin developer to quickly and reliably create a engine plugin for hipDNN.  

Currently, a basic shared-library C-compatible API for plugins has been defined by PluginApi.h.  While plugin developers will be free to implement their plugins using solely this API, a set of helpful classes can provide some basic structure that will allow developers to:
- bootstrap a new engine plugin quickly
- provide mechanical guidance and guardrails
- create an comfortable object-oriented abstraction

## 2. Problem Statement
The C-compatible engine plugin API is certainly suggestive of certain concepts and patterns, such as Engines, Workspaces, Engine Configs, and Execution Contexts, it can be somewhat opaque in how those plugin entry points relate to one-another.  What is the recommended life-cycle of an engine?  Or an execution plan?  

While the proposed design isn't meant to be proscriptive, it should work either as an direct implementation aid (using the classes directly) or a documentation aid (these classes suggest how an engine plugin is used by hipDNN)

### 2.1 Object Lifetimes
For the plugin, there's a comment on std::weak_ptr<MiopenContainer> in plugins/miopen_legacy_plugin/MiopenLegacyPlugin.cpp that describes why the weak_ptr / shared_ptr for the container exist.  

To summarize, it's so that if the plugin is opened more than once, the container can be shared, but the container gets cleaned up if that last plugin ref drops off.

However, as the thread note indicates, that means that multiple threads can be accessing the container, the engine manager, the engines, and the plan builders.  Since the plans are handed back up via handles, they should exist inside the execution context of a single thread.

### 2.2 Multi-Threading
.so files aren't reloaded by dlopen calls from different threads, so we isolate threads by putting state on the hipDNN handle, but it also means that anything in the plugin needs to effectively be stateless after it's constructed (except for plans which are handed off to the execution context).

For example, it would be bad if the plugin container or engine manager had any caching or any other state.  Read-only stuff is fine.

## 3. Current System Overview


## 4. Proposed Design

Key components:

### 4.1 EnginePluginContainer
This class encapsulates the lifecycle of an engine plugin from `hipdnnEnginePluginCreate()` to `hipdnnEnginePluginDestroy()`.  It works best as an instance that is shared amongst plugin invocations and attached to the plugin handle via shared_ptr.  A weak_ptr at global scope can be used to allow subsequent invocations of `hipdnnEnginePluginCreate()` to make an additional shared copy of the `EnginePluginContainer`.  When the last shared_ptr of this class is destroyed (via destruction of the plugin handle), the weak_ptr ensures that that the memory isn't hung on to.

Each plugin typically maintains one instance of `EnginePluginContainer`.

All plugin API methods that use an opaque plugin handle can access the container from the handle.

### 4.2 EngineManager
This represents the collection of all engines that can be found in the plugin.  This class is a private member of the EnginePluginContainer, and shares a lifespan with it.

### 4.3 EngineBase (IEngine)

### 4.4 PlanBuilderBase (IPlanBuilder)

### 4.5 PlanBase (IPlan)


## 5. Key Design Decisions

## 6. Risks

## 7. Execution Plan
- Create blank plugin_sdk
   - Changes needed in rocm-libraries to enable new components (empty new library, or w/e)
    - Wait for TheRock submodule bump
    - Update TheRock to add these
    - Update rocm-libraries to use the new TheRock CI hash to have the merged changes from step #1
    - Changes in rocm-libraries to use the new artifacts
- Migrate sdk/plugin code to plugin_sdk
    - establish hipdnn_plugin_sdk namespace
    - Modify consuming CMakeFiles.txt to add required plugin_sdk dependencies and remove unnecessary sdk transitive dependencies
    - Modify consuming code to use the utilities from the new namespaces and the new include location
- Write new plugin helpers
- Migrate existing MIOpen plugin to consume the new plugin helpers
- Migrate existing MIOpen unit tests
- Write a reference CPU plugin that consumes the new plugin helpers

## 8. Testing Plan
- Execute existing MIOpen plugin unit tests
- Execute existing MIOpen plugin integration tests
- Execute new plugin helper unit tests
- Multi-threaded testing