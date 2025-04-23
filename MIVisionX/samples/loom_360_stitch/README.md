## Loom 360 Stitch - Radeon Loom 360 Stitch Samples

MIVisionX samples using [LoomShell](https://github.com/ROCm/MIVisionX/tree/master/utilities/loom_shell#radeon-loomshell)

[![Loom Stitch](https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/LOOM_LOGO_250X125.png)](https://youtu.be/E8pPU04iZjw)

**Note:** 

* To run the samples we need to put MIVisionX executables and libraries into the system path

``` 
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
```

* To get help on loom_shell, use `-help` option

``` 
loom_shell -help
```

### Sample - 1

usage:

* Get Data for the stitch

``` 
cd sample-1/
python loomStitch-sample1-get-data.py
```

* Run Loom Shell Script to generate the 360 Image

``` 
loom_shell loomStitch-sample1.txt
```

* Expected Output

``` 
loom_shell loomStitch-sample1.txt 
loom_shell 0.9.8 [loomsl 0.9.8]
... processing commands from loomStitch-sample1.txt
..ls_context context[1] created
..lsCreateContext: created context context[0]
..lsSetOutputConfig: successful for context[0]
..lsSetCameraConfig: successful for context[0]
OK: OpenVX using GPU device#0 (gfx906+sram-ecc) [OpenCL 2.0 ] [SvmCaps 0 0]
..lsInitialize: successful for context[0] (1380.383 ms)
..cl_mem mem[2] created
..cl_context opencl_context[1] created
..lsGetOpenCLContext: get OpenCL context opencl_context[0] from context[0]
OK: loaded cam00.bmp
OK: loaded cam01.bmp
OK: loaded cam02.bmp
OK: loaded cam03.bmp
..lsSetCameraBuffer: set OpenCL buffer mem[0] for context[0]
..lsSetOutputBuffer: set OpenCL buffer mem[1] for context[0]
OK: run: executed for 100 frames
OK: run: Time: 0.919 ms (min); 1.004 ms (avg); 1.238 ms (max); 1.212 ms (1st-frame) of 100 frames
OK: created LoomOutputStitch.bmp
> stitch graph profile
 COUNT,tmp(ms),avg(ms),min(ms),max(ms),DEV,KERNEL
 100, 0.965, 1.005, 0.918, 1.237,CPU,GRAPH
 100, 0.959, 0.999, 0.915, 1.234,GPU,com.amd.loomsl.warp
 100, 0.955, 0.994, 0.908, 1.232,GPU,com.amd.loomsl.merge
OK: OpenCL buffer usage: 324221600, 9/9
..lsReleaseContext: released context context[0]
... exit from loomStitch-sample1.txt
```

**Note:** The stitched output image is saved as **LoomOutputStitch.bmp**

### Sample - 2

usage:

* Get Data for the stitch

``` 
cd sample-2/
python loomStitch-sample2-get-data.py
```

* Run Loom Shell Script to generate the 360 Image

``` 
loom_shell loomStitch-sample2.txt
```

### Sample - 3

usage:

* Get Data for the stitch

``` 
cd sample-3/
python loomStitch-sample3-get-data.py
```

* Run Loom Shell Script to generate the 360 Image

``` 
loom_shell loomStitch-sample3.txt
```
