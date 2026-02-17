# Welcome to hipThreads

> **Caution:** This release is an *early-access* software technology preview. Running production workloads is *not* recommended.
>
> **hipThreads currently works only with ROCm 7.0.2.** Other ROCm versions (including newer ones) are not supported. See [Setup](md_docs_setup.html) for detailed installation instructions.

hipThreads introduce a GPU execution model that lets developers launch and coordinate work on AMD GPUs using an idiom they already know: the C++ Concurrency Support Library. Instead of beginning the journey by learning kernel configuration and grid/block semantics, a developer can write `hip::thread`, `hip::mutex`, and `hip::condition_variable` code that feels structurally similar to `std::thread`-driven CPU programs.

The goal is to lower the barrier to entry and make first contact with GPU compute feel like an incremental extension of existing C++ expertise, not a wholesale shift in mental model.

### Key Features

hipThreads bridge the gap between host-side CPU concurrency and device-side execution with a suite of familiar abstractions:

-   **Familiar C++ Concurrency Model**: Launch GPU work with `hip::thread` much like you would with `std::thread`.
-   **Persistent Execution Engine**: A long-lived scheduler kernel time-slices many logical threads, reducing launch overhead.
-   **Cooperative Multitasking**: Logical threads can pause with `hip::this_thread::pseudo_yield` to let others progress without costly preemption.
-   **Standard-Style Sync Primitives**: Use `hip::mutex`, `hip::lock_guard`, `hip::unique_lock`, and `hip::condition_variable` to manage access to shared device resources.
-   **Multi-Fiber Threads (Width)**: A single `hip::thread` can comprise multiple simultaneous fibers (e.g., one per hardware lane) to enable cooperative, SIMD-style work partitioning within a thread.

### Getting Started

1.  **[Standard API](group__standard__api.html)**: Browse the core user-facing classes and functions.
2.  **[Setup](md_docs_setup.html)**: Follow the guide for prerequisites, build, and installation instructions.
3.  **[Releases](md_docs_releases.html)**: View the latest changes and version history.

### Core Abstractions

The primary components you will interact with are:

| Class/Namespace                                       | Description                                                                                             |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [`hip::thread`](clasship_1_1internal_1_1thread.html)             | The core abstraction for creating and managing a GPU thread of execution.                               |
| [`hip::spin_mutex`](clasship_1_1spin__mutex.html)     | A fast, spinning mutex for protecting short critical sections.                                          |
| [`hip::lock_guard`](clasship_1_1lock__guard.html)     | An RAII wrapper to manage a `spin_mutex` for the lifetime of a scope.                                   |
| [`hip::condition_variable_any`](clasship_1_1condition__variable__any.html) | A mechanism for blocking a thread until notified by another.                                            |
| [`hip::this_thread`](this__thread_8h.html)       | A namespace for functions that query or control the calling thread (e.g., `get_id`, `pseudo_yield`). |

### Examples

To see `hipThreads` in action, explore the **[examples/](https://github.com/ROCm/hipThreads/tree/release/0.1.0/examples)** directory. It includes SAXPY, sparse matrix multiply, a ray tracer, and llama3.c — each demonstrating how a `std::thread`-based CPU application can be ported to the GPU with minimal code changes, highlighting key patterns for memory management, fiber-based execution, and thread creation.
