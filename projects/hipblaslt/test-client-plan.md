# Goal
Improve the speed of the tensilelite common tests, and of the standalone test done through that path.

# Problem
The current testing is done through using code written to check both correctness AND performance through benchmarking. The benchmark requirements places additional contrainst on the numerical tesing which make it harder to improve performance. E.g. it include hard barriers and forces gpu->host drains before it continues to the next problem.

I'm attaching data for the two slowest tensilelite common tests on gfx942: dtv.yaml and cgemm.yaml. I'm only including the run side of things since this is what needs to live on the GPU-node.
 
Here is the breakdown of the top times, with the gpu work also included:
 
dtv.yaml: 84s total
	- 23.8s waiting for gpu buffers to be reset between solutions
	- 29.0s for client setup
	- 11.5s to do the gpu vs reference comparison
	-  3.6s for reference gemm calc
	-  2.3s for logging and perf calc
	-  0.5s for actual gpu work
 
cgemm: 41s total
	- 12.5s for client setup
	- 10.6s for the slow reference gemm
	-  5.2s for the gpu vs reference comparison
	-  2.0s for logging and perf calc
	-  0.3s for actual gpu work.

So for the run step, both of these yaml files spend less that 1% of their total wall clock time on the GPU. We need to seriously improve that.


# Some more info 

The yamls are set up as:
```
BenchmarkProblems:
	- case 1
	- ...
	- case N
```
The subentries of each run in parallel. with N sequential runs; N=9 for cgemm and N=25 for dtv).

The CMS yamls take much longer per kernel in their build step because each case contains a single kernel, and Tensile is set up to parallelize only *within* a case, not across them.

None of these times actually measure GPU time, we don't know exactly how long the kernel takes to solve from these traces as they're CPU only, kernel_solving only captures dispatch, dtv/cgemm's 10us per dispatch it too low. It's likely that this the GPU execution time is being misattributed to the validate_gpu_readback entry for validate_warmup ' 
# Initial Thoughts
- client setup seems to be repeated N times, if that can be shared it's an immediate saving of (N-1)/N (22.9 s for dtv and 12 s for cgem, 27-30% speedup).
- gpu buffer readbacks should not block the way they do now, we should use a ring buffer on the gpu(s) to pipeline the {reseting, solving, transfer, validation} loop.
- The reference gemm calc should be made faster, or moved to GPU, and results should either be cached or runs internally re-ordered to not duplicate reference gemm calcs.
- Logging and perf calc can be removed entirely. Numerical correctness should done on an all-or-nothing basis. We don't need to know which entries are wrong, we just need to know that they are. A separate path can be made available for letting the user figure out which values are wrong as useful debug info, but for CI we only need a pass-fail.
- New approach takes the pre-compiled kernels (as first pass) and pipelines everything (loading, data generation, reference solution, data transfers, gpu work, cleanup, validation).
- Looking at the times from the dtv run, the 

# Outstanding questions
- How much of the GPU can we claim for ourselves? How much of its RAM
- Can we get rid of ClientExecutionLockPath?
- Do we want to still run them as individual yaml files, or can we group them together into a single stream of tests to do?
- How to handle the A2A fusions, and any other kernels that requires multiple-gpus