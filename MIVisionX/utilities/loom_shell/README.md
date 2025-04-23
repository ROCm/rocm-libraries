# Radeon LoomShell

<p align="center"><img src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/LOOM_LOGO_250X125.png" /></p>

## DESCRIPTION
LoomShell is an interpreter that enables stitching 360 degree videos using a script. It provides direct access to 
Live Stitch API by encapsulating the calls to enable rapid prototyping.

## Command-line Usage
    % loom_shell.exe [-v] [-help] [script.lss]

## Available Commands
    ~ context creation
        ls_context context;
        context = lsCreateContext();
        lsReleaseContext(&context);
    ~ rig and image configuration
        rig_params rig_par = {yaw,pitch,roll,d};
        camera_params cam_par = { {yaw,pitch,roll,tx,ty,tz},{hfov,haw,r_crop,du0,dv0,lens_type,k1,k2,k3} };
        lsSetOutputConfig(context,format,width,height);
        lsSetCameraConfig(context,rows,columns,format,width,height);
        lsSetOverlayConfig(context,rows,columns,format,width,height);
        lsSetCameraParams(context,index,&cam_par);
        lsSetOverlayParams(context,index,&cam_par);
        lsSetRigParams(context,&rig_par);
        showOutputConfig(context);
        showCameraConfig(context);
        showOverlayConfig(context);
        showRigParams(context);
    ~ import/export configuration
        showConfiguration(context,"exportType");
        lsExportConfiguration(context,"exportType","fileName");
        lsImportConfiguration(context,"importType","fileName");
    ~ LoomIO configuration
        lsSetCameraModule(context,"module","kernelName","kernelArguments");
        lsSetOutputModule(context,"module","kernelName","kernelArguments");
        lsSetOverlayModule(context,"module","kernelName","kernelArguments");
        lsSetViewingModule(context,"module","kernelName","kernelArguments");
        showCameraModule(context);
        showOutputModule(context);
        showOverlayModule(context);
        showViewingModule(context);
    ~ initialize and schedule
        lsInitialize(context);
        lsScheduleFrame(context);
        lsWaitForCompletion(context);
        run(context,frameCount);
        runParallel(contextArray,contextCount,frameCount);
    ~ image I/O configuration (not supported with LoomIO)
        lsSetCameraBufferStride(context,stride);
        lsSetOutputBufferStride(context,stride);
        lsSetOverlayBufferStride(context,stride);
        lsSetCameraBuffer(context,&buf[#]|NULL);
        lsSetOutputBuffer(context,&buf[#]|NULL);
        lsSetOverlayBuffer(context,&buf[#]|NULL);
        showCameraBufferStride(context);
        showOutputBufferStride(context);
        showOverlayBufferStride(context);
    ~ OpenCL/OpenVX contexts
        cl_context opencl_context;
        vx_context openvx_context;
        lsGetOpenCLContext(context,&opencl_context);
        lsGetOpenVXContext(context,&openvx_context);
    ~ OpenCL buffers
        cl_mem buf[count];
        createBuffer(opencl_context,size,&buf[#]);
        releaseBuffer(&buf[#]);
    ~ load/save OpenCL buffers
        saveBufferToImage(buf[#],"image.bmp",format,width,height,stride);
        loadBufferFromImage(buf[#],"image.bmp",format,width,height,stride);
        saveBufferToMultipleImages(buf[#],"image%02d.bmp",rows,columns,format,width,height,stride);
        loadBufferFromMultipleImages(buf[#],"image%02d.bmp",rows,columns,format,width,height,stride);
        loadBuffer(buf[#],"image.bin"[,offset]); // default: 0
        saveBuffer(buf[#],"image.bin"[,flags]); // flags: 0:write 1:append, deault: 0
    ~ OpenVX/OpenVX contexts (advanced)
        createOpenCLContext("platform","device",&opencl_context);
        createOpenVXContext(&openvx_context);
        lsSetOpenCLContext(context,opencl_context);
        lsSetOpenVXContext(context,openvx_context);
        releaseOpenCLContext(&opencl_context);
        releaseOpenVXContext(&openvx_context);
    ~ attributes (advanced)
        setGlobalAttribute(offset,value);
        showGlobalAttributes(offset,count);
        saveGlobalAttributes(offset,count,"attr.txt");
        loadGlobalAttributes(offset,count,"attr.txt");
        setAttribute(context,offset,value);
        showAttributes(context,offset,count);
        saveAttributes(context,offset,count,"attr.txt");
        loadAttributes(context,offset,count,"attr.txt");
    ~ components (advanced)
        showExpCompGains(context,num_entries);
        loadExpCompGains(context,num_entries,\"gains.txt\");
        saveExpCompGains(context,num_entries,\"gains.txt\");
    ~ miscellaneous
        help
        include "script.lss"
        exit
        quit

| Parameter       | Description
| ----------------|------------
| format          | buffer format: VX_DF_IMAGE_RGB, VX_DF_IMAGE_UYVY, VX_DF_IMAGE_YUYV, VX_DF_IMAGE_RGBX
| width           | buffer width in pixel units
| height          | buffer height in pixel units
| rows            | number of image tile rows inside the buffer (veritical direction)
| columns         | number of image tile columns inside the buffer (horizontal direction)
| index           | camera or overlay index
| yaw             | yaw in degrees
| pitch           | pitch in degrees
| roll            | roll in degrees
| d               | reserved (should be zero)
| tx              | reserved (should be zero)
| ty              | reserved (should be zero)
| tz              | reserved (should be zero)
| lens_type       | ptgui_lens_rectilinear, ptgui_lens_fisheye_ff, ptgui_lens_fisheye_circ, adobe_lens_rectilinear, or adobe_lens_fisheye
| haw             | horizontal active pixel count
| hfov            | horizontal field of view in degrees
| k1,k2,k3        | lens distortion correction parameters
| du0,dv0         | optical center correction in pixel units
| r_crop          | crop radius in pixel units
| importType      | supported values: "pts"
| exportType      | supported values: "pts", "loom_shell"
| frameCount      | number of frames to process (for live capture, use 0)
| size            | size of a buffer in bytes
| stride          | stride of an image inside buffer in bytes
| platform        | OpenCL platform name or platform index (default: "0")
| device          | OpenCL device name or device index (default: "0")
| module          | LoomIO plug-in: OpenVX module name
| kernelName      | LoomIO plug-in: OpenVX kernel name
| kernelArguments | LoomIO plug-in: custom kernel arguments
| offset          | start index of an attribute
| count           | number of attributes
| value           | value of attribute
| contextCount    | number of stitch instances in context[] allocated using "ls_context context[N];"

## Example #1: Simple Example
Let's consider a 360 rig that has 3 1080p cameras with Circular FishEye lenses. 
The below example demonstrates how to stitch images from these cameras into a 4K Equirectangular buffer.

    # define camera orientation and lens parameters
    camera_params cam1_par = { { 120,0,90,0,0,0},{176,1094,547,0,-37,ptgui_lens_fisheye_circ,-0.1719,0.1539,1.0177} };
    camera_params cam2_par = { {   0,0,90,0,0,0},{176,1094,547,0,-37,ptgui_lens_fisheye_circ,-0.1719,0.1539,1.0177} };
    camera_params cam3_par = { {-120,0,90,0,0,0},{176,1094,547,0,-37,ptgui_lens_fisheye_circ,-0.1719,0.1539,1.0177} };

    # create a live stitch instance and initialize
    ls_context context;
    context = lsCreateContext();
    lsSetOutputConfig(context,VX_DF_IMAGE_RGB,3840,1920);
    lsSetCameraConfig(context,3,1,VX_DF_IMAGE_RGB,1920,1080*3);
    lsSetCameraParams(context, 0, &cam1_par);
    lsSetCameraParams(context, 1, &cam2_par);
    lsSetCameraParams(context, 2, &cam3_par);
    lsInitialize(context);

    # Get OpenCL context and create OpenCL buffers for input and output
    cl_context opencl_context;
    cl_mem buf[2];
    lsGetOpenCLContext(context,&opencl_context);
    createBuffer(opencl_context,3*1920*1080*3, &buf[0]);
    createBuffer(opencl_context,3*3840*1920  , &buf[1]);

    # load CAM00.bmp, CAM01.bmp, and CAM02.bmp (1920x1080 each) into buf[0]
    loadBufferFromMultipleImages(buf[0],"CAM%02d.bmp",3,1,VX_DF_IMAGE_RGB,1920,1080*3);

    # set input and output buffers and stitch a frame
    lsSetCameraBuffer(context, &buf[0]);
    lsSetOutputBuffer(context, &buf[1]);
    run(context, 1);

    # save the stitched output into "output.bmp"
    saveBufferToImage(buf[1],"output.bmp",VX_DF_IMAGE_RGB,3840,1920);

    # release resources
    releaseBuffer(&buf[0]);
    releaseBuffer(&buf[1]);
    lsReleaseContext(&context);

## Example #2: Stitching Workflow using PTGui Pro Tool for Camera Calibration
It is easy to import camera parameters from PTGui Pro project file (.pts) into loom_shell. 
In this example, let's consider a 360 rig that has 16 1080p cameras.

### Step 1: Calibrate cameras
Save test input images from all cameras into BMP files: "CAM00.bmp", "CAM01.bmp", "CAM02.bmp", ..., and "CAM15.bmp". 
Align these test input images using PTGui Pro and save the project into "myrig.pts" (it should be in ASCII text format).

### Step 2: Use the below script to generate stitched 4K output
    # create context, configure, and initialize
    ls_context context;
    context = lsCreateContext();
    lsSetOutputConfig(context, VX_DF_IMAGE_RGB, 3840, 1920);
    lsSetCameraConfig(context, 16, 1, VX_DF_IMAGE_RGB, 1920, 1080*16);
    lsImportConfiguration(context, "pts", "myrig.pts");
    showConfiguration(context, "loom_shell");
    lsInitialize(context);
    
    # create buffers for input and output
    cl_context opencl_context;
    cl_mem mem[2];
    lsGetOpenCLContext(context, &opencl_context);
    createBuffer(opencl_context, 3*1920*1080*16, &buf[0]);
    createBuffer(opencl_context, 3*3840*1920, &buf[1]);
    
    # load input images into buf[0]
    loadBufferFromMultipleImages(buf[0], "CAM%02d.bmp", 16, 1, VX_DF_IMAGE_RGB, 1920, 1080*16);
    
    # process camera inputs from buf[0] into stitched output in buf[1]
    lsSetCameraBuffer(context, &buf[0]);
    lsSetCameraBuffer(context, &buf[1]);
    run(context, 1);
    
    # save the output
    saveBufferToImage(buf[1], "output.bmp", VX_DF_IMAGE_RGB, 3840, 1920);
    
    # release all resources
    releaseBuffer(&buf[0]);
    releaseBuffer(&buf[1]);
    lsReleaseContext(&context);
    
## Example #3: Real-time Live Stitch using LoomIO
This example makes use of a 3rd party LoomIO plug-ins for live camera capture and display.

    # create context, configure, and initialize
    ls_context context;
    context = lsCreateContext();
    lsSetOutputConfig(context, VX_DF_IMAGE_RGB, 3840, 1920);
    lsSetCameraConfig(context, 16, 1, VX_DF_IMAGE_RGB, 1920, 1080*16);
    lsImportConfiguration(context, "pts", "myrig.pts");
    lsSetCameraModule(context, "vx_loomio_bm", "com.amd.loomio_bm.capture", "30,0,0,16");
    lsSetOutputModule(context, "vx_loomio_bm", "com.amd.loomio_bm.display", "30,0,0");
    lsInitialize(context);
    
    # process live from camera until aborted by input capture plug-in
    run(context, 0);
    
    # release the context
    lsReleaseContext(&context);

## Example #4: Converting script into standalone C application
It is easy to convert a well written LoomShell script into a standalone C application using the following steps :
1. Convert the shell script comments into C - style inline comments and keep them inside main() function
2. Use "amdovx-modules/utils/loom_shell/loom_shell_util.h" for wrapper utility functions, such as, loadBuffer()
3. Add "amdovx-modules/utils/loom_shell/loom_shell_util.cpp" to project for wrapper utility function implementations

Below is the C code generated from script in Example#3.

    #include "loom_shell_util.h"
    int main()
    {
        // create context, configure, and initialize
        ls_context context;
        context = lsCreateContext();
        lsSetOutputConfig(context, VX_DF_IMAGE_RGB, 3840, 1920);
        lsSetCameraConfig(context, 16, 1, VX_DF_IMAGE_RGB, 1920, 1080 * 16);
        lsImportConfiguration(context, "pts", "myrig.pts");
        lsSetCameraModule(context, "vx_loomio_bm", "com.amd.loomio_bm.capture", "30,0,0,16");
        lsSetOutputModule(context, "vx_loomio_bm", "com.amd.loomio_bm.display", "30,0,0");
        lsInitialize(context);

        // process live from camera until aborted by input capture plug-in
        run(context, 0);

        // release the context
        lsReleaseContext(&context);

        return 0;
    }
