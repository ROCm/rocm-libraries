# Tensile Tuning Client Architecture

The tuning client is written to exercise Tensile kernels by allocating and initializing input and output memory and calling the kernels while recording their run time and reporting the speed (in GFlops) back to the Python code.

It is written in a modular, structured way which involves the cooperation of many different classes and class hierarchies. It uses the Tensile host runtime library (sometimes referred to as the 'new client') and several Boost libraries:

 - Algorithm
 - Any
 - Program options
 - Lexical Cast

This document assumes general familiarity with the Tensile host runtime library.

 ## Program Flow

The program flow is driven from the `main()` function in `Tensile/Source/client/main.cpp`.  There is a loop structure in `main` which drives the whole process:

    BenchmarkRuns: Allows repeating the whole process
        Problems: Loop over problem sizes
            Solutions: Loop over solutions (kernels)
                SolutionRuns: Allows repeating one solution
                    Warmups: Non-timed runs of the solution to warm up the GPU
                             and/or validate the correctness of the solution.
                    Syncs: Sets of timed runs which are each synced to the CPU.
                        Enqueues: Timed runs of the solution within a sync.

While the loop structure and the kernel launching (through the SolutionAdapter) is in `main`, the majority of the decision-making and functionality is implemented in modular classes.

There are a few objects which are known directly to the `main` function:

 - `ClientProblemFactory` which creates `Problem` objects for each of the sizes to be run.
 - `DataInitialization` and its primary subclass `TypedDataInitialization` which is responsible for allocating and initializing input and output buffers both on CPU and GPU.
 - `SolutionIterator` which is responsible for iterating through all of the appropriate solutions.

This is in addition to the `SolutionLibrary`, `HipSolutionAdapter`, etc. classes which are part of the main Tensile host runtime library.

The remainder of the functionality is implemented through two main class hierarchies, `RunListener` classes,and `ResultReporter` classes.

## `RunListener`

`RunListener` classes can listen for and respond to steps in the `main` loop hierarchy via a set of virtual functions which are called before and after each of the steps in this hierarchy.

The `RunListener` objects are generally managed by the `MetaRunListener` object which forwards each of its calls to each of its owned `RunListener` objects.

RunListener subclasses implement key functionality in a generic way:
- `ReferenceValidator` implements correctness checking.
- `BenchmarkTimer` performs timing with Hip events or via CPU timing and calculates speed in GFlops
- `HardwareMonitorListener` uses the HardwareMonitor class to monitor the GPU's clock frequency, temperature, fan speed, etc using the ROCm-SMI library.
- The `DataInitialization`, `SolutionIterator`, and `ResultReporter` (mentioned below) classes also inherit from `RunListener`.

The loop conditions of several of the `main` loops are driven by calls into the `MetaRunListener` to allow subclasses to determine loop counts:

### While loops

If any object returns `true`, we will repeat these loops.  The current implementation is just based on simple trip count parameters, but this could in theory be based on confidence intervals or some other dynamic criteria.

 - `BenchmarkRuns` is driven by `needMoreBenchmarkRuns()`
 - `SolutionRuns` is driven by `needMoreRunsInSolution()`

### For loops

These calls are passed to each listener and `MetaRunListener` returns the highest value returned by any object back to `main`.

- Warmups: `numWarmupRuns()`
- Syncs: `numSyncs()`
- Enqueues: `numEnqueuesPerSync()`

MetaRunListener also communicates this highest value to all the objects by calling `setNumWarmupRuns()`, `setNumSyncs()`, and `setNumEnqueuesPerSync()`.

### Pre/post loop events

Each of the iterations of each of the loops is surrounded by a pair of `pre` and `post` functions.  Many of these are sent parameters which allow these objects to follow the overall program structure:

 - `preBenchmarkRun()` / `postBenchmarkRun()`
 - `preProblem(problem)` / `postProblem()`
 - `preSolution(solution)` / `postSolution()`
 - etc.
 - `postEnqueues()` is sent Hip events corresponding to the enqueued kernels via `TimingEvents` objects.

The inner enqueue loop does not have its `pre-` and `post-` calls inside the loop due to potential interference with the timing.  Instead, these functions are called before and after the entire loop.

The implementation of each of these is meant to be relatively trivial in terms of execution time, especially in the inner loops.  More involved calculations and validations can be done in the `validate` functions:

 - `validateWarmups(inputs, startEvents, stopEvents)`: This is where `ReferenceValidator` validates the results if appropriate.
 - `validateEnqueues(inputs, startEvents, stopEvents)`: This is where `BenchmarkTimer` calculates the time and speed of the kernel.

 - `finalizeReport()` is called outside of the outer benchmark loop.

## `ResultReporter`

`ResultReporter` inherits from `RunListener` and instances are hooked up to receive the event calls described above.

Any of the `RunListener` classes as well as the outer `main` loop may provide pieces of information to be logged to the user or included in data files which are used by the Python code.  These are provided in key/value pairs, and routed through the `RunListener`'s `m_reporter` object, which is generally a `MetaResultReporter`, and which forwards these values to any number of child `ResultReporter` objects that format this information for the screen or to files.

Right now, to get around the limitation in C++ against overloaded virtual functions, the structure is as follows:

  - `ResultReporter` implements `report()` which is *not* virtual, and is overloaded for:
     - `std::string`
     - `uint64_t`
     - `int64_t` and `int`
     - `double`
     - `std::vector<size_t>`
  - These functions forward (respectively) to virtual functions:
     - `reportValue_string()`
     - `reportValue_uint()`
     - `reportValue_int()`
     - `reportValue_double()`
     - `reportValue_sizes()`
  - The `MetaResultReporter` implements these virtual functions, forwarding the calls to the child reporters.

The overloading to renamed functions is awkward and may be replaced with `boost::any` or a CRTP implementation at some point.

The keys are generally declared as string constants in the `ResultKey` namespace, in order to prevent misspellings from causing uncaught bugs.

### Error reporting

Kernel correctness validation is reported via the `ResultKey::Validation` key.  The values are "FAILED", "PASSED", and "NO_CHECK".

Any other errors can currently be reported by implementing the `RunListener::error()` function and returning a non-zero value.  In the gfx10 branch, this has been enhanced to allow the client to optionally exit early.

## Worker Class Design

### DataInitialization

`DataInitialization` and its child class `TypedDataInitialization` are responsible for allocating and initializing the buffers used as input and output for the kernels and for reference calculations.

Where possible, care is taken to minimize runtime overhead between kernel invocations and between different problem sizes:

 - Buffers are allocated once at the beginning of the process, for the largest problem size.
 - Mostly, initialization modes are independent of the actual tensor dimensions (they are constant, or random, or based on their linear index), and this allows the initialization to be done only once in the process lifetime.
    - Tensor dimension-based initialization is available for kernel debugging purposes.  This is slower as the input data must be regenerated for each problem size.

These classes make use of a custom subclass of `TypedContractionInputs`, `ManagedContractionInputs`, which manage the lifetime of the memory allocations through the use of `shared_ptr` members.  Since the 'managed' members must remain pointing to the beginning of the allocation (to free the memory), they are also used by the bounds checking code.

#### Initialization Modes

Different initialization modes are supported

### Bounds Checking

Bounds checking is a feature that is implemented across the `DataInitialization` and `ReferenceValidator` classes.  There are two main bounds checking modes, value-based and page-based.

## `ClientProblemFactory`

Creates all of the ContractionProblem objects for the process.  The problems can be specified in several different ways, in order to support higher-dimensional tensor contractions.

## `SolutionIterator`

Responsible for deciding which solution(s) will be run for a particular problem.  There are two major subclasses:

### `AllSolutionsIterator`

Iterates over all the solutions that are capable of solving the problem.  This is the main class to be used for Tensile tuning.

### `BestSolutionIterator`

This class just calls `findBestSolution()` on the solution library, in order to test the solution selection.  This can be used to evaluate the tuning results.

## `HardwareMonitorListener` and `HardwareMonitor`

Responsible for monitoring GPU temperatures and clock rates, via ROCm-SMI.  **Note**: Hardware monitoring can be disabled by setting the `hardware-monitor` program argument to `false`.

`HardwareMonitor` uses threads to asynchronously query ROCm-SMI.  It provides an interface to start/stop monitoring manually, or to monitor between Hip events, to limit query rates, and to specify the specific sensor(s) that are required.  It also examines the PCI id data to make sure we are monitoring the correct device in the case that there are multiple GPUs in the system.

`HardwareMonitorListener` implements the `RunListener` interface and uses `HardwareMonitor`'s asynchronous interface to monitor GPU properties while the GPU kernels are actually executing.  It then provides this information to any reporter classes via the `report()` functionality:

 - `const std::string TempEdge            = "temp-edge";`
 - `const std::string ClockRateSys        = "clock-sys"; // GPU clock in Mhz`
 - `const std::string ClockRateSOC        = "clock-soc"; // Soc clock in Mhz`
 - `const std::string ClockRateMem        = "clock-mem"; // Mem clock in Mhz`
 - `const std::string DeviceIndex         = "device-idx";`
 - `const std::string FanSpeedRPMs        = "fan-rpm";`
 - `const std::string HardwareSampleCount = "hardware-samples";`

## `MetaRunListener`

This object maintains `shared_ptr` references to the worker classes and forwards calls along to each worker.

## `ReferenceValidator`

When requested, performs validation to ensure the kernels generate the correct result.  This uses the `SolveCPU()` function in `Reference.hpp/.cpp` to calculate a reference result.  This necessarily slows down the tuning process, but the `num-elements-to-validate` parameter can help to speed this up.  Values:

 - -1: Validate all elements.
 - 0: Do no element validation.
 - >1: Validate up to this number of elements.  This speeds up the tuning process, because we don't calculate unneeded the reference values, and we don't perform the comparison.
        - In order to reduce the possibility that a systemic error is missed, we validate every `n` elements, where `n` the next prime number that is >= the total number of elements in the output tensor divided by the number of elements to validate.

`ReferenceValidator` is also responsible for performing the checks when the Bounds Check mode is `NaN`.  When this is the case, extra memory is allocated around each of the tensors (by DataInitialization).  The `managed` pointers remain pointing to the beginning of the allocated volume, but the `a/b/c/d` pointers point to an address inside this allocated volume, and all of the values around the valid locations are filled with `infinity` or `NaN`: Input values are filled with `NaN`, and output values are filled with `infinity` (integer types are filled with max/min values for their type).  Then, the `ReferenceValidator` verifies that the memory locations surrounding the output are still `infinity` after the solution has been run.
