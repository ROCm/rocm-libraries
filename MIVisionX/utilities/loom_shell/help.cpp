/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "loom_shell.h"

void CLoomShellParser::help(bool detailed)
{
	if (detailed) {
		Message("\n");
		Message("# Radeon LoomShell\n");
		Message("\n");
		Message("## DESCRIPTION\n");
		Message("LoomShell is an interpreter that enables stitching 360 degree videos using a script. It provides direct access to \n");
		Message("Live Stitch API by encapsulating the calls to enable rapid prototyping.\n");
		Message("\n");
		Message("## Command-line Usage\n");
		Message("    %% loom_shell.exe [-v] [-help] [script.lss]\n");
	}
	Message("\n");
	Message("## Available Commands\n");
	Message("    ~ context creation\n");
	Message("        ls_context context;\n");
	Message("        context = lsCreateContext();\n");
	Message("        lsReleaseContext(&context);\n");
	Message("    ~ rig and image configuration\n");
	Message("        rig_params rig_par = {yaw,pitch,roll,d};\n");
	Message("        camera_params cam_par = {{yaw,pitch,roll,tx,ty,tz},{hfov,haw,r_crop,du0,dv0,lens_type,k1,k2,k3}};\n");
	Message("        lsSetOutputConfig(context,format,width,height);\n");
	Message("        lsSetCameraConfig(context,rows,columns,format,width,height);\n");
	Message("        lsSetOverlayConfig(context,rows,columns,format,width,height);\n");
	Message("        lsSetCameraParams(context,index,&cam_par);\n");
	Message("        lsSetOverlayParams(context,index,&cam_par);\n");
	Message("        lsSetRigParams(context,&rig_par);\n");
	Message("        showOutputConfig(context);\n");
	Message("        showCameraConfig(context);\n");
	Message("        showOverlayConfig(context);\n");
	Message("        showRigParams(context);\n");
	Message("    ~ import/export configuration\n");
	Message("        showConfiguration(context,\"exportType\");\n");
	Message("        lsExportConfiguration(context,\"exportType\",\"fileName\");\n");
	Message("        lsImportConfiguration(context,\"importType\",\"fileName\");\n");
	Message("    ~ LoomIO configuration\n");
	Message("        lsSetCameraModule(context,\"module\",\"kernelName\",\"kernelArguments\");\n");
	Message("        lsSetOutputModule(context,\"module\",\"kernelName\",\"kernelArguments\");\n");
	Message("        lsSetOverlayModule(context,\"module\",\"kernelName\",\"kernelArguments\");\n");
	Message("        lsSetViewingModule(context,\"module\",\"kernelName\",\"kernelArguments\");\n");
	Message("        showCameraModule(context);\n");
	Message("        showOutputModule(context);\n");
	Message("        showOverlayModule(context);\n");
	Message("        showViewingModule(context);\n");
	Message("    ~ initialize and schedule\n");
	Message("        lsInitialize(context);\n");
	Message("        lsScheduleFrame(context);\n");
	Message("        lsWaitForCompletion(context);\n");
	Message("        run(context,frameCount);\n");
	Message("        runParallel(contextArray,contextCount,frameCount);\n");
	Message("    ~ image I/O configuration (not supported with LoomIO)\n");
	Message("        lsSetCameraBufferStride(context,stride);\n");
	Message("        lsSetOutputBufferStride(context,stride);\n");
	Message("        lsSetOverlayBufferStride(context,stride);\n");
	Message("        lsSetCameraBuffer(context,&buf[#]|NULL);\n");
	Message("        lsSetOutputBuffer(context,&buf[#]|NULL);\n");
	Message("        lsSetOverlayBuffer(context,&buf[#]|NULL);\n");
	Message("        showCameraBufferStride(context);\n");
	Message("        showOutputBufferStride(context);\n");
	Message("        showOverlayBufferStride(context);\n");
	Message("    ~ OpenCL/OpenVX contexts\n");
	Message("        cl_context opencl_context;\n");
	Message("        vx_context openvx_context;\n");
	Message("        lsGetOpenCLContext(context,&opencl_context);\n");
	Message("        lsGetOpenVXContext(context,&openvx_context);\n");
	Message("    ~ OpenCL buffers\n");
	Message("        cl_mem buf[count];\n");
	Message("        createBuffer(opencl_context,size,&buf[#]);\n");
	Message("        releaseBuffer(&buf[#]);\n");
	Message("    ~ load/save OpenCL buffers\n");
	Message("        saveBufferToImage(buf[#],\"image.bmp\",format,width,height,stride);\n");
	Message("        loadBufferFromImage(buf[#],\"image.bmp\",format,width,height,stride);\n");
	Message("        saveBufferToMultipleImages(buf[#],\"image%%02d.bmp\",rows,columns,format,width,height,stride);\n");
	Message("        loadBufferFromMultipleImages(buf[#],\"image%%02d.bmp\",rows,columns,format,width,height,stride);\n");
	Message("        loadBuffer(buf[#],\"image.bin\");\n");
	Message("        saveBuffer(buf[#],\"image.bin\");\n");
	Message("    ~ OpenVX/OpenVX contexts (advanced)\n");
	Message("        createOpenCLContext(\"platform\",\"device\",&opencl_context);\n");
	Message("        createOpenVXContext(&openvx_context);\n");
	Message("        lsSetOpenCLContext(context,opencl_context);\n");
	Message("        lsSetOpenVXContext(context,openvx_context);\n");
	Message("        releaseOpenCLContext(&opencl_context);\n");
	Message("        releaseOpenVXContext(&openvx_context);\n");
	Message("    ~ attributes (advanced)\n");
	Message("        setGlobalAttribute(offset,value);\n");
	Message("        showGlobalAttributes(offset,count);\n");
	Message("        saveGlobalAttributes(offset,count,\"attr.txt\");\n");
	Message("        loadGlobalAttributes(offset,count,\"attr.txt\");\n");
	Message("        setAttribute(context,offset,value);\n");
	Message("        showAttributes(context,offset,count);\n");
	Message("        saveAttributes(context,offset,count,\"attr.txt\");\n");
	Message("        loadAttributes(context,offset,count,\"attr.txt\");\n");
	Message("    ~ components (advanced)\n");
	Message("        showExpCompGains(context,num_entries);\n");
	Message("        loadExpCompGains(context,num_entries,\"gains.txt\");\n");
	Message("        saveExpCompGains(context,num_entries,\"gains.txt\");\n");
	Message("        loadBlendWeights(context,\"blend-weights.raw\");\n");
	Message("    ~ miscellaneous\n");
	Message("        help\n");
	Message("        include \"script.lss\"\n");
	Message("        exit\n");
	Message("        quit\n");
	Message("\n");
	Message("| Parameter       | Description\n");
	Message("| ----------------|------------\n");
	Message("| format          | buffer format: VX_DF_IMAGE_RGB, VX_DF_IMAGE_UYVY, VX_DF_IMAGE_YUYV, VX_DF_IMAGE_RGBX\n");
	Message("| width           | buffer width in pixel units\n");
	Message("| height          | buffer height in pixel units\n");
	Message("| rows            | number of image tile rows inside the buffer (veritical direction)\n");
	Message("| columns         | number of image tile columns inside the buffer (horizontal direction)\n");
	Message("| index           | camera or overlay index\n");
	Message("| yaw             | yaw in degrees\n");
	Message("| pitch           | pitch in degrees\n");
	Message("| roll            | roll in degrees\n");
	Message("| d               | reserved (should be zero)\n");
	Message("| tx              | reserved (should be zero)\n");
	Message("| ty              | reserved (should be zero)\n");
	Message("| tz              | reserved (should be zero)\n");
	Message("| lens_type       | ptgui_lens_rectilinear, ptgui_lens_fisheye_ff, ptgui_lens_fisheye_circ, adobe_lens_rectilinear, or adobe_lens_fisheye\n");
	Message("| haw             | horizontal active pixel count\n");
	Message("| hfov            | horizontal field of view in degrees\n");
	Message("| k1,k2,k3        | lens distortion correction parameters\n");
	Message("| du0,dv0         | optical center correction in pixel units\n");
	Message("| r_crop          | crop radius in pixel units\n");
	Message("| importType      | supported values: \"pts\"\n");
	Message("| exportType      | supported values: \"pts\", \"loom_shell\", \"gdf\"\n");
	Message("| frameCount      | number of frames to process (for live capture, use 0)\n");
	Message("| size            | size of a buffer in bytes\n");
	Message("| stride          | stride of an image inside buffer in bytes\n");
	Message("| platform        | OpenCL platform name or platform index (default: \"0\")\n");
	Message("| device          | OpenCL device name or device index (default: \"0\")\n");
	Message("| module          | LoomIO plug-in: OpenVX module name\n");
	Message("| kernelName      | LoomIO plug-in: OpenVX kernel name\n");
	Message("| kernelArguments | LoomIO plug-in: custom kernel arguments\n");
	Message("| offset          | start index of an attribute\n");
	Message("| count           | number of attributes\n");
	Message("| value           | value of attribute\n");
	Message("| contextCount    | number of stitch instances in context[] allocated using \"ls_context context[N];\"\n");
	if (detailed) {
		Message("\n");
		Message("## Example #1: Simple Example\n");
		Message("Let's consider a 360 rig that has 3 1080p cameras with Circular FishEye lenses. \n");
		Message("The below example demonstrates how to stitch images from these cameras into a 4K Equirectangular buffer.\n");
		Message("\n");
		Message("    # define camera orientation and lens parameters\n");
		Message("    camera_params cam1_par = {{ 120,0,90,0,0,0},{176,1094,547,0,-37,ptgui_lens_fisheye_circ,-0.1719,0.1539,1.0177}};\n");
		Message("    camera_params cam2_par = {{   0,0,90,0,0,0},{176,1094,547,0,-37,ptgui_lens_fisheye_circ,-0.1719,0.1539,1.0177}};\n");
		Message("    camera_params cam3_par = {{-120,0,90,0,0,0},{176,1094,547,0,-37,ptgui_lens_fisheye_circ,-0.1719,0.1539,1.0177}};\n");
		Message("\n");
		Message("    # create a live stitch instance and initialize\n");
		Message("    ls_context context;\n");
		Message("    context = lsCreateContext();\n");
		Message("    lsSetOutputConfig(context,VX_DF_IMAGE_RGB,3840,1920);\n");
		Message("    lsSetCameraConfig(context,3,1,VX_DF_IMAGE_RGB,1920,1080*3);\n");
		Message("    lsSetCameraParams(context, 0, &cam1_par);\n");
		Message("    lsSetCameraParams(context, 1, &cam2_par);\n");
		Message("    lsSetCameraParams(context, 2, &cam3_par);\n");
		Message("    lsInitialize(context);\n");
		Message("\n");
		Message("    # Get OpenCL context and create OpenCL buffers for input and output\n");
		Message("    cl_context opencl_context;\n");
		Message("    cl_mem buf[2];\n");
		Message("    lsGetOpenCLContext(context,&opencl_context);\n");
		Message("    createBuffer(opencl_context,3*1920*1080*3, &buf[0]);\n");
		Message("    createBuffer(opencl_context,3*3840*1920  , &buf[1]);\n");
		Message("\n");
		Message("    # load CAM00.bmp, CAM01.bmp, and CAM02.bmp (1920x1080 each) into buf[0]\n");
		Message("    loadBufferFromMultipleImages(buf[0],\"CAM%%02d.bmp\",3,1,VX_DF_IMAGE_RGB,1920,1080*3);\n");
		Message("\n");
		Message("    # set input and output buffers and stitch a frame\n");
		Message("    lsSetCameraBuffer(context, &buf[0]);\n");
		Message("    lsSetOutputBuffer(context, &buf[1]);\n");
		Message("    run(context, 1);\n");
		Message("\n");
		Message("    # save the stitched output into \"output.bmp\"\n");
		Message("    saveBufferToImage(buf[1],\"output.bmp\",VX_DF_IMAGE_RGB,3840,1920);\n");
		Message("\n");
		Message("    # release resources\n");
		Message("    releaseBuffer(&buf[0]);\n");
		Message("    releaseBuffer(&buf[1]);\n");
		Message("    lsReleaseContext(&context);\n");
		Message("\n");
		Message("## Example #2: Stitching Workflow using PTGui Pro Tool for Camera Calibration\n");
		Message("It is easy to import camera parameters from PTGui Pro project file (.pts) into loom_shell. \n");
		Message("In this example, let's consider a 360 rig that has 16 1080p cameras.\n");
		Message("\n");
		Message("### Step 1: Calibrate cameras\n");
		Message("Save test input images from all cameras into BMP files: \"CAM00.bmp\", \"CAM01.bmp\", \"CAM02.bmp\", ..., and \"CAM15.bmp\". \n");
		Message("Align these test input images using PTGui Pro and save the project into \"myrig.pts\" (it should be in ASCII text format).\n");
		Message("\n");
		Message("### Step 2: Use the below script to generate stitched 4K output\n");
		Message("    # create context, configure, and initialize\n");
		Message("    ls_context context;\n");
		Message("    context = lsCreateContext();\n");
		Message("    lsSetOutputConfig(context, VX_DF_IMAGE_RGB, 3840, 1920);\n");
		Message("    lsSetCameraConfig(context, 16, 1, VX_DF_IMAGE_RGB, 1920, 1080*16);\n");
		Message("    lsImportConfiguration(context, \"pts\", \"myrig.pts\");\n");
		Message("    showConfiguration(context, \"loom_shell\");\n");
		Message("    lsInitialize(context);\n");
		Message("    \n");
		Message("    # create buffers for input and output\n");
		Message("    cl_context opencl_context;\n");
		Message("    cl_mem buf[2];\n");
		Message("    lsGetOpenCLContext(context, &opencl_context);\n");
		Message("    createBuffer(opencl_context, 3*1920*1080*16, &buf[0]);\n");
		Message("    createBuffer(opencl_context, 3*3840*1920, &buf[1]);\n");
		Message("    \n");
		Message("    # load input images into buf[0]\n");
		Message("    loadBufferFromMultipleImages(buf[0], \"CAM%%02d.bmp\", 16, 1, VX_DF_IMAGE_RGB, 1920, 1080*16);\n");
		Message("    \n");
		Message("    # process camera inputs from buf[0] into stitched output in buf[1]\n");
		Message("    lsSetCameraBuffer(context, &buf[0]);\n");
		Message("    lsSetOutputBuffer(context, &buf[1]);\n");
		Message("    run(context, 1);\n");
		Message("    \n");
		Message("    # save the output\n");
		Message("    saveBufferToImage(buf[1], \"output.bmp\", VX_DF_IMAGE_RGB, 3840, 1920);\n");
		Message("    \n");
		Message("    # release all resources\n");
		Message("    releaseBuffer(&buf[0]);\n");
		Message("    releaseBuffer(&buf[1]);\n");
		Message("    lsReleaseContext(&context);\n");
		Message("    \n");
		Message("## Example #3: Real-time Live Stitch using LoomIO\n");
		Message("This example makes use of a 3rd party LoomIO plug-ins for live camera capture and display.\n");
		Message("\n");
		Message("    # create context, configure, and initialize\n");
		Message("    ls_context context;\n");
		Message("    context = lsCreateContext();\n");
		Message("    lsSetOutputConfig(context, VX_DF_IMAGE_RGB, 3840, 1920);\n");
		Message("    lsSetCameraConfig(context, 16, 1, VX_DF_IMAGE_RGB, 1920, 1080*16);\n");
		Message("    lsImportConfiguration(context, \"pts\", \"myrig.pts\");\n");
		Message("    lsSetCameraModule(context, \"vx_loomio_bm\", \"com.amd.loomio_bm.capture\", \"30,0,0,16\");\n");
		Message("    lsSetOutputModule(context, \"vx_loomio_bm\", \"com.amd.loomio_bm.display\", \"30,0,0\");\n");
		Message("    lsInitialize(context);\n");
		Message("    \n");
		Message("    # process live from camera until aborted by input capture plug-in\n");
		Message("    run(context, 0);\n");
		Message("    \n");
		Message("    # release the context\n");
		Message("    lsReleaseContext(&context);\n");
		Message("\n");
		Message("## Example #4: Converting script into standalone C application\n");
		Message("It is easy to convert a well written LoomShell script into a standalone C application using the following steps :\n");
		Message("1. Convert the shell script comments into C - style inline comments and keep them inside main() function\n");
		Message("2. Use \"amdovx-modules/utils/loom_shell/loom_shell_util.h\" for wrapper utility functions, such as, loadBuffer()\n");
		Message("3. Add \"amdovx-modules/utils/loom_shell/loom_shell_util.cpp\" to project for wrapper utility function implementations\n");
		Message("\n");
		Message("Below is the C code generated from script in Example#3.\n");
		Message("\n");
		Message("    #include \"loom_shell_util.h\"\n");
		Message("    int main()\n");
		Message("    {\n");
		Message("        // create context, configure, and initialize\n");
		Message("        ls_context context;\n");
		Message("        context = lsCreateContext();\n");
		Message("        lsSetOutputConfig(context, VX_DF_IMAGE_RGB, 3840, 1920);\n");
		Message("        lsSetCameraConfig(context, 16, 1, VX_DF_IMAGE_RGB, 1920, 1080 * 16);\n");
		Message("        lsImportConfiguration(context, \"pts\", \"myrig.pts\");\n");
		Message("        lsSetCameraModule(context, \"vx_loomio_bm\", \"com.amd.loomio_bm.capture\", \"30,0,0,16\");\n");
		Message("        lsSetOutputModule(context, \"vx_loomio_bm\", \"com.amd.loomio_bm.display\", \"30,0,0\");\n");
		Message("        lsInitialize(context);\n");
		Message("\n");
		Message("        // process live from camera until aborted by input capture plug-in\n");
		Message("        run(context, 0);\n");
		Message("\n");
		Message("        // release the context\n");
		Message("        lsReleaseContext(&context);\n");
		Message("\n");
		Message("        return 0;\n");
		Message("    }\n");
	}
	Message("\n");
}
