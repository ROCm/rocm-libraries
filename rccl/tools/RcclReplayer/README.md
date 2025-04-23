# RCCL REPLAYER
Collective log replayer tool for RCCL.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [How It Works](#how-it-works)
4. [Installation](#installation)
5. [Usage](#usage)

## Introduction

Replayer is a dubugging tool designed to analyze and replay collective logs obtained from RCCL (ROCm Communication Collectives Library) runs. It can be a useful tool when trying to recreate problem situations (without as much setup), or as a user-directed utility to run collectives (by crafting their own 'logfile').

## Features

- Parses and validates collective logs from RCCL runs.
- Detects missing/faulty group calls and provides report.
- Replays collective calls based on the recorded data.
- Skips faulty group calls during replay.
- Supports various MPI ranks and GPU configurations.
- Supports multi-node environment.

*Note: RCCL Replayer executes collective calls with dummy data.*

## How It Works

Replayer operates in the following steps:

1. **Collective Log Collection:** During your RCCL runs, the collective logs are generated when NCCL_DEBUG=INFO and NCCL_DEBUG_SUBSYS=COLL enabled, capturing important information like hostname, deviceIdx, collective call type, number of elements used, data type, operation type, task number, and global rank number about collective communication patterns.

2. **Data Aggregation:** Replayer collects and pareses the collective logs. organizing them based on opCount (collective count in the group call), and global rank information.

3. **Group Call Validation:** After acquiring data from the collective logs and generating group calls, the replayer validates the results using two different methods. For Non-Send/Recv collectives, it checks if each MPI rank has the required number of collective tasks. For Send/Recv collectives, it verifies if they all have a matching pair.

4. **Replaying RCCL:** Based on the aggregated and validated data, Replayer will replay the collective logs to reproduce the RCCL runs from your application.

5. **Reporting and Skipping.** Replayer outputs the detected faulty group calls and skips them during replay. It provides a report showing which group calls were skipped and why and, at the end, summarizes how many group calls were replayed and how many were skipped.

## Installation

To build the replayer, follow these steps:
1. Navigate to the rccl_replayer directory.
2. Make sure 'MPI_DIR' is set to the path where your MPI installation is located.

```bash
    cd rccl/tools/rccl_replayer
    MPI_DIR=/path/to/mpi make
```

Depending on the MPI library used and your installation path, you may need to set the MPI_DIR path accordingly.


## Usage

First Collect per-rank logs from the run by adding the following environment variables:
This prevents any race-conditions that might cause ranks to interupt other ranks lines of output.

```bash
   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_DEBUG_FILE=some_name_here.%h.%p.log
```

Secondly, combine all the logs into a single file which will be the input to the replayer:

```bash
  cat some_name_here_*.log > some_name_here.log
```

After successfully building the replayer, you can run it using the following command:

```bash
    mpirun -np <numProcesses> ./rcclReplayer </path/to/logfile> <numGpusPerMpiRank>
```

Replace <numProcesses> with the number of MPI processes you want to run during the replay, </path/to/logfile> with the path to the collective log file generated during your RCCL runs, and <numGpusPerMpiRank> with the number of GPUs per MPI rank used in your application.

Depending on the MPI library you use, you may need to modify the mpirun command accordingly.

### Multi-Node Environment:

If multiple nodes were used for your application, you can also replay the collective logs using multiple nodes. See the following command:

```bash
     mpirun --hostfile <path/to/hostfile.txt> -np <numProcesses> ./rcclReplayer </path/to/logfile> <numGpusPerMpiRank>
```

### SLURM:

For systems using SLURM, you can use the following command to replay the collective logs:

```bash
    srun -N <numNodes> -n <numProcesses> ./rcclReplayer </path/to/logfile> <numGpusPerMpiRank>
```

Replace <numNodes> with the number of nodes used in your application.
