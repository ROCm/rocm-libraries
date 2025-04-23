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

#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifndef ENABLE_OPENCL
#define ENABLE_OPENCL 1
#endif
#if ENABLE_OPENCL
#if __APPLE__
#include <opencl.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif
#if _WIN32
#include <windows.h>
#include <process.h>
#include <time.h>
#define strdup _strdup
#pragma comment(lib, "OpenCL.lib")
#else
#include <chrono>
#endif

#define MAX_KERNEL_ARG  64

// local variables
static cl_context       context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;
static char             device_name[1024];

#ifdef _WIN32
typedef LARGE_INTEGER coclock_t;
coclock_t coclock() { LARGE_INTEGER t; QueryPerformanceCounter(&t); return t; }
float coclock2sec(coclock_t tstart, coclock_t tend) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return (float)(tend.QuadPart - tstart.QuadPart) / (float)f.QuadPart; }
#else
typedef int64_t coclock_t;
coclock_t coclock() { return std::chrono::high_resolution_clock::now().time_since_epoch().count(); }
float coclock2sec(coclock_t tstart, coclock_t tend)
{
    return (float)(tend - tstart) / (float)(std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num);
}
#endif

//! \brief The macro for fread error checking and reporting.
#define ERROR_CHECK_FREAD_(call,value) {size_t retVal = (call); if(retVal != (size_t)value) { fprintf(stderr,"ERROR: fread call expected to return [ %d elements ] but returned [ %d elements ] at " __FILE__ "#%d\n", (int)value, (int)retVal, __LINE__); return -1; }  }

void show_usage(const char * program)
{
    printf("Usage: %s [platform-options] [-I<include-dir>] [[-D<name>=<value>] ...] <kernel.[cl|elf]>\n", program);
    printf("          [kernel-arguments] <arguments> <num_work_items>[/<work_group_size>]\n");
    printf("\n");
    printf("   [platform-options]\n");
    printf("       -v                    verbose\n");
    printf("       -gpu                  use GPU device (default)\n");
    printf("       -cpu                  use CPU device\n");
    printf("       -device <name>|#<num> use specified device\n");
    printf("       -bo <string>          OpenCL build option\n");
    printf("\n");
    printf("   [kernel-options]\n");
    printf("       -k <kernel-name>      kernel name\n");
    printf("       -p                    use persistence flag\n");
    printf("       -r[link] <exec-count> execution count\n");
    printf("       -w <msec>             waiting time\n");
    printf("       -dumpcl               dump OpenCL code after pre-processing\n");
    printf("       -dumpilisa            dump IL and ISA of kernel and show ISA statistics\n");
    printf("       -dumpelf              dump ELF binary\n");
    printf("\n");
    printf("   The <arguments> shall be given in the order as required by the kernel.\n");
    printf("     For value arguments use   iv#<int/float>[,<int/float>...] or iv:<file> (e.g., iv#10.2,10,0x10)\n");
    printf("     For local memory use      lm#<local-memory-size> (e.g., lm#8192)\n");
    printf("     For input buffer use      if[#<buffer-size>]:[<file>][#[[<checksum>][/<file>[@<offset>#<end>]]]] (e.g., if:input.bin\n");
    printf("     For output (or RW) buffer of[#<buffer-size>]:[#]<file>[@<ofile>][#[[<checksum>][/[+<float-tolerance>]<file>[@<offset>#<end>]]]] (e.g., of#16384:output.bin)\n");
    printf("     For input image  use      ii#<width>x<height>,<stride>,<u8/s16/u16/bgra/rgba/argb>:<file> (e.g., ii#1920x1080,7680,bgra:screen1920x1080.rgb\n");
    printf("     For output image  use     oi#<width>x<height>,<stride>,<u8/s16/u16/bgra/rgba/argb>:<file> (e.g., oi#1920x1080,7680,bgra:screen1920x1080.rgb\n");
}

static int initialize(int use_gpu, int verbose, const char * devname)
{
    cl_int result;
    size_t size;

    // create OpenCL context
    cl_platform_id platform_id;
    if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) { fprintf(stderr, "ERROR: clGetPlatformIDs(1,*,0) failed\n"); return -1; }
    cl_context_properties ctxprop[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id,
        0, 0
    };
    device_type = CL_DEVICE_TYPE_CPU;
    if (use_gpu) device_type = CL_DEVICE_TYPE_GPU;
    context = clCreateContextFromType(ctxprop, device_type, NULL, NULL, NULL);
    if (!context) { fprintf(stderr, "ERROR: clCreateContextFromType(%s) failed\n", use_gpu ? "GPU" : "CPU"); return -1; }

    // get the list of GPUs
    result = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
    num_devices = (int)(size / sizeof(cl_device_id));
    if (result != CL_SUCCESS || num_devices < 1) { fprintf(stderr, "ERROR: clGetContextInfo() failed\n"); return -1; }
    device_list = new cl_device_id[num_devices + 16];
    if (!device_list) { fprintf(stderr, "ERROR: new cl_device_id[] failed\n"); return -1; }
    result = clGetContextInfo(context, CL_CONTEXT_DEVICES, size, device_list, NULL);
    if (result != CL_SUCCESS) { fprintf(stderr, "ERROR: clGetContextInfo() failed\n"); return -1; }
    result = clReleaseContext(context);
    if (result != CL_SUCCESS) { fprintf(stderr, "ERROR: clReleaseContext() => %d\n", result); return -1; }
    if (verbose) {
        for (int i = 0; i < num_devices; i++) {
            device_name[0] = 0;
            clGetDeviceInfo(device_list[i], CL_DEVICE_NAME, sizeof(device_name), device_name, 0);
            printf("OK: DEVICE #%2d [%s]\n", i, device_name);
        }
    }
    // pick a device
    int selected_device = 0;
    if (devname && devname[0])
    {
        selected_device = -1;
        if (devname[0] == '#')
        {
            sscanf(&devname[1], "%d", &selected_device);
        }
        else
        {
            for (int i = 0; i < num_devices; i++)
            {
                device_name[0] = 0; clGetDeviceInfo(device_list[i], CL_DEVICE_NAME, sizeof(device_name), device_name, 0);
                if (strcmp(devname, device_name) == 0)
                {
                    selected_device = i;
                    break;
                }
            }
        }
        if (selected_device < 0 || selected_device >= num_devices) { fprintf(stderr, "ERROR: requested device [%s] not found\n", devname); return -1; }
    }
    device_list[0] = device_list[selected_device];
    num_devices = 1;
    device_list[num_devices] = 0;
    // get device name
    device_name[0] = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_NAME, sizeof(device_name), device_name, 0);
    // re-initialize the context
    context = clCreateContext(ctxprop, num_devices, device_list, NULL, NULL, NULL);
    if (!context) { fprintf(stderr, "ERROR: clCreateContext(device:%s) failed\n", device_name); return -1; }
    printf("OK: Using %s device#%d [%s]\n", use_gpu ? "GPU" : "CPU", selected_device, device_name);

    // dump device info
    if (verbose)
    {
        char s[8192]; cl_int info, iarr[3]; cl_long infl;
        printf("\nOpenCL Device Information:\n");
        s[0] = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_NAME, sizeof(s), s, 0); printf("  DEVICE_NAME              : %s\n", s);
        info = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(info), &info, 0); printf("  MAX_CLOCK_FREQUENCY      : %d MHz\n", info);
        s[0] = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_VENDOR, sizeof(s), s, 0); printf("  DEVICE_VENDOR            : %s\n", s);
        s[0] = 0; clGetDeviceInfo(device_list[0], CL_DRIVER_VERSION, sizeof(s), s, 0); printf("  DRIVER_VERSION           : %s\n", s);
        s[0] = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_VERSION, sizeof(s), s, 0); printf("  DEVICE_VERSION           : %s\n", s);
        info = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(info), &info, 0); printf("  MAX_COMPUTE_UNITS        : %d\n", info);
        info = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(info), &info, 0); printf("  MAX_WORK_ITEM_DIMENSIONS : %d\n", info);
        info = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(iarr), iarr, 0); printf("  MAX_WORK_ITEM_SIZES      : %d %d %d\n", iarr[0], iarr[1], iarr[2]);
        info = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(info), &info, 0); printf("  MAX_WORK_GROUP_SIZE      : %d\n", info);
        info = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_ADDRESS_BITS, sizeof(info), &info, 0); printf("  ADDRESS_BITS             : %d bits\n", info);
        info = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(info), &info, 0); printf("  MEM_BASE_ADDR_ALIGN      : %d bits\n", info);
        infl = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(infl), &infl, 0); printf("  MAX_MEM_ALLOC_SIZE       : %d MB\n", (int)(infl >> 20));
        infl = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(infl), &infl, 0); printf("  GLOBAL_MEM_SIZE          : %d MB\n", (int)(infl >> 20));
        info = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(info), &info, 0); printf("  GLOBAL_MEM_CACHE_TYPE    : %s\n", info == CL_NONE ? "NONE" : (info == CL_READ_ONLY_CACHE ? "READ ONLY" : (info == CL_READ_WRITE_CACHE ? "READ WRITE" : "????")));
        info = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(info), &info, 0); printf("  GLOBAL_MEM_CACHELINE_SIZE: %d bytes\n", info);
        infl = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(infl), &infl, 0); printf("  GLOBAL_MEM_CACHE_SIZE    : %d KB\n", (int)(infl >> 10));
        info = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_LOCAL_MEM_TYPE, sizeof(info), &info, 0); printf("  LOCAL_MEM_TYPE           : %s\n", info == CL_LOCAL ? "LOCAL" : (info == CL_GLOBAL ? "GLOBAL" : "????"));
        infl = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(infl), &infl, 0); printf("  LOCAL_MEM_SIZE           : %d KB\n", (int)(infl >> 10));
        if (verbose & 2) {
            s[0] = 0; clGetDeviceInfo(device_list[0], CL_DEVICE_EXTENSIONS, sizeof(s), s, 0); printf("  DEVICE_EXTENSIONS        : ");
            for (int i = 0; s[i]; i++) if (s[i] != ' ') printf("%c", s[i]); else                               printf("\n                             ");
            printf("\n");
        }
    }

    return 0;
}

static int shutdown()
{
    // release resources
    if (cmd_queue) clReleaseCommandQueue(cmd_queue);
    if (context) clReleaseContext(context);
    if (device_list) delete device_list;

    // reset all variables
    cmd_queue = 0;
    context = 0;
    device_list = 0;
    num_devices = 0;
    device_type = 0;

    return 0;
}

static int read_clfile(char * fname, char * source, char * incdir)
{
    { for (char * p = fname; *p; p++) if (*p == '\\') *p = '/'; } // replace backslash in the source filename to forward slash
    FILE * fp = fopen(fname, "r"); if (!fp) { fprintf(stderr, "ERROR: unable to open '%s'\n", fname); return -1; }
    char line[8192];
    char srcdir[4096];
    strcpy(srcdir, fname);
    { char * p = 0; for (char * q = srcdir; *q; q++) if (*q == '/' || *q == '\\') p = q; if (p) *p = 0; else srcdir[0] = 0; }
    sprintf(source + strlen(source), "#line 1 \"%s\"\n", fname);
    for (int cpos = (int)strlen(source), lpos = 1; fgets(line, sizeof(line) - 1, fp); lpos++)
    {
        if (strstr(line, "#include \"") == line)
        {
            char iname[128];
            if (sscanf(line, "#include \"%[^\"]s", iname) != 1) { fprintf(stderr, "ERROR: %s#%d: unsupported #include statement: %s\n", fname, lpos, line); fclose(fp); return -1; }
            FILE * fi = fopen(iname, "rb");
            if (!fi && *srcdir) { sprintf(line, "%s/%s", srcdir, iname); fi = fopen(line, "rb"); }
            if (!fi &&  incdir) { sprintf(line, "%s/%s", incdir, iname); fi = fopen(line, "rb"); }
            if (!fi) { fprintf(stderr, "ERROR: %s#%d: unable to open '%s'\n", fname, lpos, iname); fclose(fp); return -1; }
            sprintf(line, "#line 1 \"%s\"\n", iname);
            strcpy(source + cpos, line); cpos += (int)strlen(line); if (source[cpos - 1] != '\n') strcpy(source + cpos, "\n");
            while (fgets(line, sizeof(line) - 1, fi))
            {
                int n = (int)strlen(line); if (n >= 2 && line[n - 2] == '\r' && line[n - 1] == '\n') { line[n - 2] = '\n'; line[n - 1] = 0; }
                strcpy(source + cpos, line); cpos += (int)strlen(line); if (source[cpos - 1] != '\n') strcpy(source + cpos, "\n");
            }
            fclose(fi);
            sprintf(line, "#line %d \"%s\"\n", lpos + 1, fname);
            strcpy(source + cpos, line); cpos += (int)strlen(line); if (source[cpos - 1] != '\n') strcpy(source + cpos, "\n");
        }
        else
        {
            strcpy(source + cpos, line); cpos += (int)strlen(line); if (source[cpos - 1] != '\n') strcpy(source + cpos, "\n");
        }
    }

    fclose(fp);
    return 0;
}

int isa_statistics(const char * kernel_name, const char * device_name_, const char * isa_file)
{
    FILE * fp = fopen(isa_file, "r"); if (!fp) { return -1; }
    char line[2048];
    int num_gprs = -1, scratch = 0, cur_clause = -1, num_ops[4][8] = { 0 }, loop = 0, codelen = -1;
    int NumVgprs = -1, NumSgprs = -1, ScratchSize = -1, num_op_s = 0, num_op_v = 0, num_op_tbuf_load = 0, num_op_tbuf_store = 0, num_op_ds_read = 0, num_op_ds_write = 0, num_op_branch = 0;
    while (fgets(line, sizeof(line), fp))
    {
        if (loop>0 && line[loop * 4 - 4 + 0] >= '0' && line[loop * 4 - 4 + 0] <= '9') {
            if (!strstr(line, " ENDLOOP ")) { loop--; continue; }
        }
        if (strstr(line, "SQ_PGM_RESOURCES:NUM_GPRS") == line) sscanf(line, "%*s%*s%d", &num_gprs);
        else if (strstr(line, "NumVgprs") == line) sscanf(line, "%*s%*s%d", &NumVgprs);
        else if (strstr(line, "NumSgprs") == line) sscanf(line, "%*s%*s%d", &NumSgprs);
        else if (strstr(line, "CodeLen") == line) sscanf(line, "%*s%*s%d", &codelen);
        else if (strstr(line, "codeLenInByte") == line) sscanf(line, "%*s%*s%d", &codelen);
        else if (strstr(line, "_SCRATCH")) scratch = 1;
        else if (strstr(line, "ScratchSize") == line) sscanf(line, "%*s%*s%d", &ScratchSize);
        else if (line[loop * 4 + 0] >= '0' && line[loop * 4 + 0] <= '9') {
            int clause = -1; char type[1024]; sscanf(line, "%d%s", &clause, type);
            cur_clause = 0;
            if (!strncmp(type, "ALU", 3)) cur_clause = 1;
            else if (!strcmp(type, "TEX:")) cur_clause = 2;
            else if (!strncmp(type, "MEM_", 4)) cur_clause = 3;
            else if (!strcmp(type, "WAIT_ACK:")) cur_clause = 4;
            else if (!strcmp(type, "JUMP")) cur_clause = 5;
            else if (!strcmp(type, "LOOP_DX10")) { cur_clause = 6; loop++; }
            else if (!strcmp(type, "LOOP_NO_AL")) { cur_clause = 6; loop++; }
            else if (!strcmp(type, "VTX:")) cur_clause = 7;
            num_ops[0][cur_clause]++;
        }
        else if (strstr(line, "  s_branch") == line) { num_op_s++; num_op_branch++; }
        else if (strstr(line, "  s_cbranch") == line) { num_op_s++; num_op_branch++; }
        else if (strstr(line, "  s_") == line) num_op_s++;
        else if (strstr(line, "  v_") == line) num_op_v++;
        else if (strstr(line, "  tbuffer_load") == line) num_op_tbuf_load++;
        else if (strstr(line, "  tbuffer_store") == line) num_op_tbuf_store++;
        else if (strstr(line, "  ds_read") == line) num_op_ds_read++;
        else if (strstr(line, "  ds_write") == line) num_op_ds_write++;
        else if (line[loop * 4 + 0] == ' ')
        {
            char word0[256], word1[256]; sscanf(line, "%s%s", word0, word1);
            if (word0[0] >= '0' && word0[0] <= '9' && (word1[1] == ':' || !strcmp(word1, "VFETCH"))) {
                num_ops[1][cur_clause]++; strcpy(word0, word1);
            }
            if (word0[1] == ':') num_ops[2][cur_clause]++;
            if (cur_clause == 1 && strstr(line, ": MOV ")) num_ops[3][cur_clause]++;
        }
    }
    fclose(fp);
    if (loop)printf("WARNING: stats failed with loop = %d\n", loop);
    if (NumVgprs >= 0) {
        printf("OK: ISA [%s] [%s] [GPRS-VS %d %d] [OPS-VS %d %d] [JUML %d] [TBUF-RW %d %d] [DS-RW %d %d] [CodeLen %d] [ScratchSize %d]\n",
            kernel_name, device_name_,
            NumVgprs, NumSgprs,
            num_op_v, num_op_s, num_op_branch,
            num_op_tbuf_load, num_op_tbuf_store,
            num_op_ds_read, num_op_ds_write,
            codelen,
            ScratchSize);
    }
    else {
        printf("OK: ISA [%s] [NUM_GPRS %d] [ALU %d OP:%d/%d(%2.0f%%) MOV:%d(%2.0f%%)] [TEX %d %d] [VTX %d %d] [MEM %d] [WAIT_ACK %d] [JUMP %d] [LOOP %d] [OTHER %d] [CodeLen %d] %s\n",
            kernel_name,
            num_gprs,
            num_ops[0][1], num_ops[2][1], num_ops[1][1], (float)num_ops[2][1] * (100.0f / 5.0f) / (float)num_ops[1][1], num_ops[3][1], (float)num_ops[3][1] * 100.0f / (float)num_ops[2][1],
            num_ops[0][2], num_ops[1][2],
            num_ops[0][7], num_ops[1][7],
            num_ops[0][3],
            num_ops[0][4],
            num_ops[0][5],
            num_ops[0][6],
            num_ops[0][0],
            codelen,
            scratch ? " [*SCRATCH*SPILLS?]" : "");
    }
    return 0;
}

#ifdef _WIN32
int
create_time_prefix(char * buffer, int length)
{
    SYSTEMTIME current;
    memset(&current, 0, sizeof(current));

    GetLocalTime(&current);
    int res = _snprintf_s(buffer, length, _TRUNCATE, "_ek_%04d%02d%02d_%02d%02d%02d_%03d",
        current.wYear, current.wMonth, current.wDay, current.wHour, current.wMinute, current.wSecond, current.wMilliseconds);
    if (res == -1) { fprintf(stderr, "ERROR: Buffer not big enough to hold prefix\n"); return -1; }
    else return 0;
}

int
rename_and_cleanup_dumpfiles(char *prefix, char *kernel_name, char *device_name_)
{

    WIN32_FIND_DATA find_data;
    memset(&find_data, 0, sizeof(find_data));

    const int filename_len = 2048;
    char *filemask = (char*)malloc(filename_len);

    // First, try and find the .il file.
    int len = _snprintf_s(filemask, filename_len, _TRUNCATE, "%s_*.il", prefix);
    if (len == -1) { fprintf(stderr, "ERROR: Filemask too long for buffer.\n"); return -1; }

    HANDLE search = FindFirstFile(filemask, &find_data);

    if (search == INVALID_HANDLE_VALUE) {
        if (GetLastError() != ERROR_FILE_NOT_FOUND)  { fprintf(stderr, "ERROR: FindFirstFile failed\n"); return -1; } //Failed for some other reason than file not found.
        // We don't always expect a .il file, so not finding it is fine.
    }
    else {
        char *new_filename = (char*)malloc(filename_len);
        len = _snprintf_s(new_filename, filename_len, _TRUNCATE, "%s_%s.il", kernel_name, device_name_);
        if (len == -1) { fprintf(stderr, "ERROR: Filename too long for buffer.\n"); return -1; }
        BOOL move_res = MoveFileEx(find_data.cFileName, new_filename, MOVEFILE_REPLACE_EXISTING);
        free(new_filename);
        if (!move_res) { fprintf(stderr, "ERROR: Could not rename %s to %s\n", find_data.cFileName, new_filename); return -1; }
    }

    FindClose(search);

    // Now fine the .isa file
    len = _snprintf_s(filemask, filename_len, _TRUNCATE, "%s_*.isa", prefix);
    if (len == -1) { fprintf(stderr, "ERROR: Filemask too long for buffer.\n"); return -1; }
    search = FindFirstFile(filemask, &find_data);

    if (search == INVALID_HANDLE_VALUE) {
        if (GetLastError() != ERROR_FILE_NOT_FOUND)  { fprintf(stderr, "ERROR: FindFirstFile failed\n"); return -1; } //Failed for some other reason than file not found.
        else { fprintf(stderr, "ERROR: Could not find .isa file.\n"); return -1; }
    }
    else {
        char *new_filename = (char*)malloc(filename_len);
        len = _snprintf_s(new_filename, filename_len, _TRUNCATE, "%s_%s.isa", kernel_name, device_name_);
        if (len == -1) { fprintf(stderr, "ERROR: Filename too long for buffer.\n"); return -1; }
        BOOL move_res = MoveFileEx(find_data.cFileName, new_filename, MOVEFILE_REPLACE_EXISTING);
        free(new_filename);
        if (!move_res) { fprintf(stderr, "ERROR: Could not rename %s to %s\n", find_data.cFileName, new_filename); return -1; }
    }

    FindClose(search);

    // Now cleanup remaining files so we don't leave a mess
    len = _snprintf_s(filemask, filename_len, _TRUNCATE, "%s_*", prefix);
    if (len == -1) { fprintf(stderr, "ERROR: Filemask too long for buffer.\n"); return -1; }
    search = FindFirstFile(filemask, &find_data);

    BOOL found = TRUE;
    while (found && search != INVALID_HANDLE_VALUE) {
        BOOL delete_res = DeleteFile(find_data.cFileName);
        if (!delete_res) { fprintf(stderr, "ERROR: Failed to delete %s.\n", find_data.cFileName); return -1; }
        found = FindNextFile(search, &find_data);
    }

    FindClose(search);
    free(filemask);

    return 0;
}
#endif

int
#ifdef _WIN32
__cdecl
#endif
main(int argc, char * argv[])
{
    coclock_t liStart, liStop; float time_sec;

    // get the program name
#if _WIN32
    const char * program = "runcl.exe";
#else
    const char * program = "runcl";
#endif
    --argc; ++argv;

    // allocate memory for kernel source
    int sourcesize = 64 * 1024 * 1024;
    char * source = (char *)calloc(sourcesize, sizeof(char)); if (!source) { fprintf(stderr, "ERROR: calloc(%d) failed\n", sourcesize); return !!- 1; }

    // process command-line switches
    const char * use_device = 0;
    int use_gpu = 1;
    int verbose = 0;
    char * incdir = 0;
#ifdef _WIN32
    char message[1024]; message[0] = 0;
#endif
    char * kernel_name_alias = 0;
    char build_options[4096]; build_options[0] = 0;
    for (; (argc > 0) && (argv[0][0] == '-'); --argc, ++argv)
    {
        if (strncmp(argv[0], "-D", 2) == 0)
        {
            char * def = argv[0] + 2;
            if (!*def && argv[1]) {
                def = argv[1];
                argc -= 1;
                argv += 1;
            }
            char * p = def;
            for (; *p && *p != '='; p++);
            if (*p != '=') { fprintf(stderr, "ERROR: -D switch requires the following format: -D<name>=<value>\n"); return !!- 1; }
            *p = ' ';
            strcat(source, "#define ");
            strcat(source, def);
            strcat(source, "\n");
        }
        else if (strncmp(argv[0], "-cl-std=", 8) == 0)
        {
            if (build_options[0]) strcat(build_options, " ");
            strcat(build_options, argv[0]);
        }
        else if (strcmp(argv[0], "-bo") == 0)
        {
            if (build_options[0]) strcat(build_options, " ");
            strcat(build_options, argv[1]);
            argc -= 1;
            argv += 1;
        }
        else if (strcmp(argv[0], "-alias") == 0)
        {
            kernel_name_alias = argv[1];
            argc -= 1;
            argv += 1;
        }
        else if (strncmp(argv[0], "-I", 2) == 0) incdir = argv[0] + 2;
        else if (strcmp(argv[0], "-gpu") == 0) use_gpu = 1;
        else if (strcmp(argv[0], "-cpu") == 0) use_gpu = 0;
        else if (strcmp(argv[0], "-gpud") == 0) use_gpu = 2;
        else if (strcmp(argv[0], "-gpue") == 0) use_gpu = 3;
        else if (strcmp(argv[0], "-device") == 0) {
            use_device = argv[1];
            --argc;
            ++argv;
        }
        else if (strncmp(argv[0], "-v", 2) == 0) { verbose = 1; if (argv[0][2]) verbose = atoi(&argv[0][2]); }
        else { fprintf(stderr, "ERROR: invalid option: %s\n", argv[0]); show_usage(program); return !!- 1; }
    }

    bool ignoreCompareResult = false;
    {
#ifdef _WIN32
        char value[64] = { 0 };
        DWORD env_res = GetEnvironmentVariable("RUNVX_IGNORE_COMPARE_ERROR", value, sizeof(value)-1);
        if(env_res > 0) {
            ignoreCompareResult = atoi(value) != 0 ? true : false;
        }
#else
        const char * value = getenv("RUNVX_IGNORE_COMPARE_ERROR");
        if(value) {
            ignoreCompareResult = atoi(value) != 0 ? true : false;
        }
#endif
    }

    // read the kernel core source
    char kernel_name[1024]; strcpy(kernel_name, "");
    int repeat = 0;
    int dumpelf = 0;
    int dumpilisa = 0;
    const int dumpilisa_prefix_len = 1024;
    char dumpilisa_prefix[dumpilisa_prefix_len];
    int sourceelf = 0;
    char * sourceelffile = argv[0];
    int msec = 0;
    int link = 0;
    int use_persistent = 0;
    if (argc > 0)
    {
        char * p = strstr(argv[0], ".elf");
        if (p && !p[4])
        { // check to make sure .elf exists otherwise switch to using .cl file with -dumpelf switch
            for (char * q = argv[0]; *q; q++) if (*q == '/' || *q == '\\') sourceelffile = q + 1;
            FILE * fp = fopen(sourceelffile, "rb"); if (!fp) { strcpy(p, ".cl"); dumpelf = 1; }
            else fclose(fp);
        }
        p = strstr(argv[0], ".cl");
        if (!p || p[3])
        {
            if ((p = strstr(argv[0], ".elf")) != 0 && !strcmp(p, ".elf")) {
                sourceelf = 1;
                memset(source, 0, sourcesize);
            }
            else {
                fprintf(stderr, "ERROR: invalid <kernel>.cl/.elf file name: '%s'\n", argv[0]);
                return !!- 1;
            }
        }
        // get kernel name
        char c = *p, *q = argv[0];
        *p = 0; for (p = q; *p; p++) if (*p == '/' || *p == '\\') q = p + 1;
        strcpy(kernel_name, q);
        *p = c;
        if (sourceelf)
        {
            FILE * fp = fopen(sourceelffile, "rb"); if (!fp) { fprintf(stderr, "ERROR: unable to open '%s'\n", sourceelffile); return !!- 1; }
            char * elfsrc = source + 2 * sizeof(size_t);
            size_t elfsize = fread(elfsrc, 1, sizeof(elfsrc), fp);
            fclose(fp);
            ((size_t *)source)[0] = 1;
            ((size_t *)source)[1] = elfsize;
        }
        else
        {
            if (kernel_name_alias)
            { // change kernel name to aliased name
                sprintf(source + strlen(source), "#define %s %s\n", kernel_name, kernel_name_alias);
                strcpy(kernel_name, kernel_name_alias);
            }
            // read source
            if (read_clfile(argv[0], source, incdir)) return !!- 1;
        }
        --argc; ++argv;
        while (argc > 0)
        {
            if (strcmp(argv[0], "-k") == 0)
            {
                strcpy(kernel_name, argv[1]);
                argc -= 2;
                argv += 2;
            }
            else if ((strcmp(argv[0], "-r") == 0) || (strcmp(argv[0], "-rlink") == 0))
            {
                if (strcmp(argv[0], "-rlink") == 0) link = -1;
                repeat = atoi(argv[1]);
                if (repeat < 0) { fprintf(stderr, "ERROR: invalid <exec-count> (shall be 0 or more)\n"); return !!- 1; }
                argc -= 2;
                argv += 2;
            }
            else if (strcmp(argv[0], "-dumpilisa") == 0)
            {
                dumpilisa = 1;
#ifdef _WIN32
                // Since this environment variable is commonly used to conduct experiments, we will append
                // to the existing one if it exists, otherwise we create it.
                int res = create_time_prefix(dumpilisa_prefix, dumpilisa_prefix_len);
                if (res != 0) return !!res;
                const int existing_len = 31 * 1024;
                char *existing = (char*)calloc(1, existing_len);
                DWORD env_res = GetEnvironmentVariable("AMD_OCL_BUILD_OPTIONS_APPEND", existing, existing_len);
                if (env_res >= existing_len - 1) { fprintf(stderr, "ERROR: Existing AMD_OCL_BUILD_OPTIONS_APPEND too long. \n"); return !!- 1; }
                char *new_contents = (char *)malloc(existing_len + 1024);
                if (res == 0) {
                    // Assume environment variable doesn't exist (Why else would we get an error?)
                    res = _snprintf_s(new_contents, existing_len + 1024, _TRUNCATE, "-save-temps=%s", dumpilisa_prefix);
                    if (res == 1) { fprintf(stderr, "ERROR: Created AMD_OCL_BUILD_OPTIONS_APPEND too long. \n"); return !!- 1; }
                }
                else {
                    res = _snprintf_s(new_contents, existing_len + 1024, _TRUNCATE, "%s -save-temps=%s", existing, dumpilisa_prefix);
                    if (res == 1) { fprintf(stderr, "ERROR: Created AMD_OCL_BUILD_OPTIONS_APPEND too long. \n"); return !!- 1; }
                }
                SetEnvironmentVariable("AMD_OCL_BUILD_OPTIONS_APPEND", new_contents);
                free(new_contents);
                free(existing);
#else
                strcpy(dumpilisa_prefix, "_ek_");
                const char * value = getenv("AMD_OCL_BUILD_OPTIONS_APPEND");
                char buffer[8192];
                if (value) sprintf(buffer, "%s -save-temps=%s", value, dumpilisa_prefix);
                else sprintf(buffer, "-save-temps=%s", dumpilisa_prefix);
                setenv("AMD_OCL_BUILD_OPTIONS_APPEND", buffer, 1);
#endif

                argc -= 1;
                argv += 1;
            }
            else if (strncmp(argv[0], "-dumpelf", 8) == 0)
            {
                dumpelf = 1;
                if ((argv[0][8] >= '0') && (argv[0][8] <= '9')) dumpelf = argv[0][8] - '0';
                argc -= 1;
                argv += 1;
            }
            else if (strcmp(argv[0], "-dumpcl") == 0)
            {
                if (!sourceelf)
                {
                    char fname[2048]; sprintf(fname, "%s.cl.txt", kernel_name);
                    FILE * fp = fopen(fname, "wb"); if (!fp) { fprintf(stderr, "ERROR: dumpcl: unable to create '%s'\n", fname); return !!- 1; }
                    fwrite(source, 1, strlen(source), fp);
                    fclose(fp);
                    if (verbose)printf("OK: dumpcl: saved OpenCL source in '%s'\n", fname);
                }
                argc -= 1;
                argv += 1;
            }
            else if (strcmp(argv[0], "-w") == 0)
            {
                msec = atoi(argv[1]);
                if (msec < 0) { fprintf(stderr, "ERROR: invalid <msec> (shall be 0 or more)\n"); return !!- 1; }
                argc -= 2;
                argv += 2;
            }
            else if (strcmp(argv[0], "-p") == 0)
            {
                if (verbose)printf("OK: using CL_MEM_USE_PERSISTENT_MEM_AMD\n");
                use_persistent = 1;
                argc -= 1;
                argv += 1;
            }
            else break;
        }
    }
    else if (!verbose)
    {
        show_usage(program);
        return !!- 1;
    }

    // image formats supported
    struct {
        const char * name;
        cl_image_format fmt;
    } imgfmt[] = {
        { "BGRA", { CL_BGRA, CL_UNORM_INT8 } },
        { "RGBA", { CL_RGBA, CL_UNORM_INT8 } },
        { "ARGB", { CL_RGBA, CL_UNORM_INT8 } },
        { "U8", { CL_INTENSITY, CL_UNORM_INT8 } },
        { "S16", { CL_INTENSITY, CL_SNORM_INT16 } },
        { "U16", { CL_INTENSITY, CL_UNORM_INT16 } },
    };
    // load all arguments
    void * arg[MAX_KERNEL_ARG];
    int size[MAX_KERNEL_ARG];
    int type[MAX_KERNEL_ARG];
    struct { int isvalid, width, height, row_pitch, fmt; } image[MAX_KERNEL_ARG]; memset(image, 0, sizeof(image));
    char * name[MAX_KERNEL_ARG];
    int skip[MAX_KERNEL_ARG]; memset(skip, 0, sizeof(skip));
    int check[MAX_KERNEL_ARG]; memset(check, 0, sizeof(check));
    int checksum[MAX_KERNEL_ARG]; memset(checksum, 0, sizeof(checksum));
    char * comparefile[MAX_KERNEL_ARG]; memset(comparefile, 0, sizeof(comparefile));
    int cmpskipa[MAX_KERNEL_ARG]; memset(cmpskipa, 0, sizeof(cmpskipa));
    int cmpsizea[MAX_KERNEL_ARG]; memset(cmpsizea, 0, sizeof(cmpsizea));
    float compareErrorLimit[MAX_KERNEL_ARG] = { 0 };
    int narg;
    size_t nworkitems[3] = { 0 }, workgroupsize[3] = { 0 }, ndim = 0, nwgdim = 0;
    for (narg = 0; narg < argc; narg++) { // discard arguments after --
        if (strcmp(argv[narg], "--") == 0) {
            argc = narg;
            break;
        }
    }
    for (narg = 0; argc > 0; ++narg, --argc, ++argv)
    {
        if (strncmp(argv[0], "iv#", 3) == 0)
        {
            type[narg] = 0;
            name[narg] = (char *) "<immediate>";
            size[narg] = (int)strlen(argv[0] + 2)*(int)sizeof(cl_int) / 2;
            arg[narg] = calloc(size[narg], 1); if (!arg[narg]) { fprintf(stderr, "ERROR: calloc(%d) failed\n", size[narg]); return !!- 1; }
            char * p = argv[0] + 3;
            int i;
            for (i = 0; *p; i++)
            {
                int isfloat = 0;
                int j; for (j = 0; p[j]; j++) {
                    if (p[j] == '.') isfloat = 1;
                    if (p[j] == ',') break;
                }
                if (!isfloat) {
                    if (sscanf(p, "%i", ((int *)arg[narg]) + i) != 1) { fprintf(stderr, "ERROR: invalid integer value passed for argument #%d\n", narg); return !!- 1; }
                }
                else {
                    if (sscanf(p, "%f", ((float *)arg[narg]) + i) != 1) { fprintf(stderr, "ERROR: invalid floating-point value passed for argument #%d\n", narg); return !!- 1; }
                }
                p += j + (p[j] != 0);
            }
            if (i < 1) { fprintf(stderr, "ERROR: no integer/float value passed for argument #%d\n", narg); return !!- 1; }
            size[narg] = i*sizeof(cl_int);
        }
        else if (strncmp(argv[0], "iv:", 3) == 0)
        {
            type[narg] = 0;
            name[narg] = argv[0] + 3;
            FILE * fp = fopen(name[narg], "rb"); if (!fp) { fprintf(stderr, "ERROR: unable to open '%s'\n", name[narg]); return !!- 1; }
            fseek(fp, 0L, SEEK_END); size[narg] = ftell(fp); fseek(fp, 0L, SEEK_SET);
            if (size[narg] < 1) { fprintf(stderr, "ERROR: invalid size value passed/derived for argument #%d\n", narg); return !!- 1; }
            arg[narg] = calloc(size[narg], 1); if (!arg[narg]) { fprintf(stderr, "ERROR: calloc(%d) failed\n", size[narg]); return !!- 1; }
            ERROR_CHECK_FREAD_(fread(arg[narg], size[narg], 1, fp),1); fclose(fp);
        }
        else if (strncmp(argv[0], "lm#", 3) == 0)
        {
            type[narg] = 1;
            size[narg] = 0;
            sscanf(argv[0] + 3, "%i", &size[narg]);
            name[narg] = (char *) "<local-memory>";
            if (size[narg] < 1) { fprintf(stderr, "ERROR: invalid size value passed for argument #%d\n", narg); return !!- 1; }
            arg[narg] = 0;
        }
        else if (argv[0][0] == 'i' && (argv[0][2] == '#' || argv[0][2] == ':') && (argv[0][1] == 'f' || argv[0][1] == 'i'))
        {
            type[narg] = 2;
            size[narg] = 0;
            char * p = argv[0] + 3;
            if (argv[0][1] == 'i') {
                if (argv[0][2] != '#'){ fprintf(stderr, "ERROR: image format not specified for argument #%d\n", narg); return !!- 1; }
                image[narg].fmt = -1;
                if (sscanf(argv[0] + 3, "%ix%i,%i,%i", &image[narg].width, &image[narg].height, &image[narg].row_pitch, &image[narg].fmt) < 3) { fprintf(stderr, "ERROR: invalid image description passed for argument #%d\n", narg); return !!- 1; }
                if ((image[narg].fmt == -1) && strstr(argv[0] + 3, ",bgra")) image[narg].fmt = 0;
                if ((image[narg].fmt == -1) && strstr(argv[0] + 3, ",rgba")) image[narg].fmt = 1;
                if ((image[narg].fmt == -1) && strstr(argv[0] + 3, ",argb")) image[narg].fmt = 2;
                if ((image[narg].fmt == -1) && strstr(argv[0] + 3, ",u8")) image[narg].fmt = 3;
                if ((image[narg].fmt == -1) && strstr(argv[0] + 3, ",s16")) image[narg].fmt = 4;
                if ((image[narg].fmt == -1) && strstr(argv[0] + 3, ",u16")) image[narg].fmt = 5;
                size[narg] = (image[narg].height > 0 ? image[narg].height : 1) * image[narg].row_pitch;
                if (image[narg].fmt < 0 || image[narg].fmt >= (int)(sizeof(imgfmt) / sizeof(imgfmt[0]))) {
                    fprintf(stderr, "ERROR: invalid image format passed for argument #%d (shall be 0..%d as shown below)\n", narg, (int)(sizeof(imgfmt) / sizeof(imgfmt[0])) - 1);
                    for (int i = 0; i < (int)(sizeof(imgfmt) / sizeof(imgfmt[0])); i++)
                        printf("    %2d - %s\n", i, imgfmt[i].name);
                    return !!- 1;
                }
                image[narg].isvalid = 1;
                while (*p && *p != ':')p++;
                if (*p != ':'){ fprintf(stderr, "ERROR: invalid format for argument #%d\n", narg); return !!- 1; }
                ++p;
                for (char * q = p; *q; q++)
                { // look for skip field
                    if (*q == ':') { *q = 0; sscanf(q + 1, "%i", &skip[narg]); break; }
                }
            }
            else
                if (argv[0][2] == '#')
                {
                    if (sscanf(argv[0] + 3, "%i", &size[narg]) != 1) { fprintf(stderr, "ERROR: invalid size value passed for argument #%d\n", narg); return !!- 1; }
                    while (*p && *p != ':')p++;
                    if (*p != ':'){ fprintf(stderr, "ERROR: invalid format for argument #%d\n", narg); return !!- 1; }
                    ++p;
                    for (char * q = p; *q; q++)
                    { // look for skip field
                        if (*q == ':') { *q = 0; sscanf(q + 1, "%i", &skip[narg]); break; }
                    }
                }
            if (*p) {
                char * q = strstr(p, "#"); if (q) {
                    *q = 0;
                    if (q[1] == '/') {
                        if (q[2] == '+') { // use /+0.001/ to specify error tolerance
                            compareErrorLimit[narg] = (float)atof(&q[3]);
                            q++; while (q[1] && q[1] != '/') q++;
                        }
                        comparefile[narg] = q + 2;
                        char * pp = strstr(q + 2, "@"); if (pp) {
                            *pp = 0;
                            cmpskipa[narg] = atoi(pp + 1);
                            if ((pp = strstr(pp + 1, "#")) != 0) {
                                cmpsizea[narg] = atoi(pp + 1);
                            }
                        }
                    }
                    else {
                        check[narg] = 1;
                        if (q[1] >= '0' && q[1] <= '9') {
                            check[narg] = 2;
                            checksum[narg] = 0;
                            sscanf(q + 1, "%i", &checksum[narg]);
                        }
                        else if (q[1]) { fprintf(stderr, "ERROR: invalid checksum value passed for argument #%d\n", narg); return !!- 1; }
                    }
                }
            }
            name[narg] = p;
            if (size[narg] < 1) { fprintf(stderr, "ERROR: invalid size value passed/derived for argument #%d\n", narg); return !!- 1; }
            arg[narg] = calloc(size[narg], 1); if (!arg[narg]) { fprintf(stderr, "ERROR: calloc(%d) failed\n", size[narg]); return !!- 1; }
            if(!strncmp(p, "init=", 5)) {
                const char * q = p + 5;
                int v = atoi(q);
                if(strstr(q, ".")) sscanf(q, "%f", (float *)&v);
                int * buf = (int *)arg[narg];
                for(int i = 0; i < size[narg]/4; i++)
                    buf[i] = v;
            }
            else if(!strncmp(p, "rand=", 5)) {
                const char * q = p + 5;
                double vmin = 0.0f, vmax = 1.0f;
                int seed = 0;
                if(sscanf(q, "%lf,%lf,%d", &vmin, &vmax, &seed) == 3) {
                    srand(seed);
                }
                double vrange = vmax - vmin;
                float * buf = (float *)arg[narg];
                for(int i = 0; i < size[narg]/4; i++) {
                    float v = (float)(vmin + vrange * (rand()*(1.0 / RAND_MAX)));
                    buf[i] = v;
                }
            }
            else if (*p) {
                FILE * fp = 0;
                fp = fopen(p, "rb"); if (!fp) { fprintf(stderr, "ERROR: unable to open '%s'\n", p); return !!- 1; } 
                if (size[narg] < 1) { fseek(fp, 0L, SEEK_END); size[narg] = ftell(fp); fseek(fp, 0L, SEEK_SET); }
                if (skip[narg] > 0) fseek(fp, (long)(size[narg] * skip[narg]), SEEK_SET);
                ERROR_CHECK_FREAD_(fread(arg[narg], size[narg], 1, fp),1);
                fclose(fp);
            }
        }
        else if (argv[0][0] == 'o' && (argv[0][2] == '#' || argv[0][2] == ':') && (argv[0][1] == 'f' || argv[0][1] == 'i'))
        {
            type[narg] = 3;
            size[narg] = 0;
            char * p = argv[0] + 3;
            if (argv[0][1] == 'i') {
                if (argv[0][2] != '#'){ fprintf(stderr, "ERROR: image format not specified for argument #%d\n", narg); return !!- 1; }
                image[narg].fmt = -1;
                if (sscanf(argv[0] + 3, "%ix%i,%i,%i", &image[narg].width, &image[narg].height, &image[narg].row_pitch, &image[narg].fmt) < 3) { fprintf(stderr, "ERROR: invalid image description passed for argument #%d\n", narg); return !!- 1; }
                if ((image[narg].fmt == -1) && strstr(argv[0] + 3, ",bgra")) image[narg].fmt = 0;
                if ((image[narg].fmt == -1) && strstr(argv[0] + 3, ",rgba")) image[narg].fmt = 1;
                if ((image[narg].fmt == -1) && strstr(argv[0] + 3, ",argb")) image[narg].fmt = 2;
                if ((image[narg].fmt == -1) && strstr(argv[0] + 3, ",u8")) image[narg].fmt = 3;
                if ((image[narg].fmt == -1) && strstr(argv[0] + 3, ",s16")) image[narg].fmt = 4;
                if ((image[narg].fmt == -1) && strstr(argv[0] + 3, ",u16")) image[narg].fmt = 5;
                size[narg] = (image[narg].height > 0 ? image[narg].height : 1) * image[narg].row_pitch;
                if (image[narg].fmt < 0 || image[narg].fmt >= (int)(sizeof(imgfmt) / sizeof(imgfmt[0]))) {
                    fprintf(stderr, "ERROR: invalid image format passed for argument #%d (shall be 0..%d as shown below)\n", narg, (int)(sizeof(imgfmt) / sizeof(imgfmt[0])) - 1);
                    for (int i = 0; i < (int)(sizeof(imgfmt) / sizeof(imgfmt[0])); i++)
                        printf("    %2d - %s\n", i, imgfmt[i].name);
                    return !!- 1;
                }
                image[narg].isvalid = 1;
                while (*p && *p != ':')p++;
                if (*p != ':'){ fprintf(stderr, "ERROR: invalid format for argument #%d\n", narg); return !!- 1; }
                ++p;
                for (char * q = p; *q; q++)
                { // look for skip field
                    if (*q == ':') { *q = 0; sscanf(q + 1, "%i", &skip[narg]); break; }
                }
            }
            else
                if (argv[0][2] == '#')
                {
                    if (sscanf(argv[0] + 3, "%i", &size[narg]) != 1) { fprintf(stderr, "ERROR: invalid size value passed for argument #%d\n", narg); return !!- 1; }
                    while (*p && *p != ':')p++;
                    if (*p != ':'){ fprintf(stderr, "ERROR: invalid format for argument #%d\n", narg); return !!- 1; }
                    ++p;
                    for (char * q = p; *q; q++)
                    { // look for skip field
                        if (*q == ':') { *q = 0; sscanf(q + 1, "%i", &skip[narg]); break; }
                    }
                }
            int noload = 0;
            if (p[0] == '#' || p[0] == '@') { noload = 1; p++; }
            if (*p) {
                char * q = strstr(p, "#"); if (q) {
                    *q = 0;
                    if (q[1] == '/') {
                        if (q[2] == '+') { // use /+0.001/ to specify error tolerance
                            compareErrorLimit[narg] = (float)atof(&q[3]);
                            q++; while (q[1] && q[1] != '/') q++;
                        }
                        comparefile[narg] = q + 2;
                        char * pp = strstr(q + 2, "@"); if (pp) {
                            *pp = 0;
                            cmpskipa[narg] = atoi(pp + 1);
                            if ((pp = strstr(pp + 1, "#")) != 0) {
                                cmpsizea[narg] = atoi(pp + 1);
                            }
                        }
                    }
                    else {
                        check[narg] = 1;
                        if (q[1] >= '0' && q[1] <= '9') {
                            check[narg] = 2;
                            checksum[narg] = 0;
                            sscanf(q + 1, "%i", &checksum[narg]);
                        }
                        else if (q[1]) { fprintf(stderr, "ERROR: invalid checksum value passed for argument #%d\n", narg); return !!- 1; }
                    }
                }
            }
            name[narg] = p;
            arg[narg] = 0;
            char * q = strstr(name[narg], "@"); if (q) {
                *q = 0;
                name[narg] = q + 1;
            }
            FILE * fp = fopen(p, "rb");
            if (fp)
            {
                if (size[narg] < 1) { fseek(fp, 0L, SEEK_END); size[narg] = ftell(fp); fseek(fp, 0L, SEEK_SET); }
                if (size[narg] < 1) { fprintf(stderr, "ERROR: invalid size value passed/derived for argument #%d\n", narg); return !!- 1; }
                arg[narg] = calloc(size[narg], 1); if (!arg[narg]) { fprintf(stderr, "ERROR: calloc(%d) failed\n", size[narg]); return !!- 1; }
                if (skip[narg] > 0) fseek(fp, (long)(size[narg] * skip[narg]), SEEK_SET);
                if (!noload) { ERROR_CHECK_FREAD_(fread(arg[narg], size[narg], 1, fp), 1); }
                fclose(fp);
            }
            if (!arg[narg])
            {
                if (size[narg] < 1) { fprintf(stderr, "ERROR: invalid size value passed/derived for argument #%d\n", narg); return !!- 1; }
                arg[narg] = calloc(size[narg], 1); if (!arg[narg]) { fprintf(stderr, "ERROR: calloc(%d) failed\n", size[narg]); return !!- 1; }
            }
        }
        else if (argc == 1 && argv[0][0] >= '0' && argv[0][1] <= '9')
        {
            char * p = argv[0];
            int i;
            for (i = 0; (i < 3) && *p; i++)
            {
                int j; for (j = 0; p[j]; j++) {
                    if (p[j] == ',' || p[j] == '/') break;
                }
                int k;
                if (sscanf(p, "%i", &k) != 1 || k < 1) { fprintf(stderr, "ERROR: invalid integer value passed for <num_work_items>\n"); return !!- 1; }
                nworkitems[i] = k;
                p += j + (p[j] != 0);
                if (p[-1] == '/') { i++; break; }
            }
            ndim = i;
            for (; i < 3; i++) nworkitems[i] = 1;
            for (i = 0; i < 3; i++) workgroupsize[i] = 1;
            if (p[-1] == '/')
            {
                for (i = 0; (i < 3) && *p; i++)
                {
                    int j; for (j = 0; p[j]; j++) {
                        if (p[j] == ',') break;
                    }
                    int k;
                    if (sscanf(p, "%i", &k) != 1 || k < 1) { fprintf(stderr, "ERROR: invalid integer value passed for <work_group_size>\n"); return !!- 1; }
                    workgroupsize[i] = k;
                    p += j + (p[j] != 0);
                }
                nwgdim = i;
                for (; i < 3; i++) workgroupsize[i] = 1;
                if (nwgdim != ndim) { fprintf(stderr, "ERROR: dimensions of <work_group_size> shall match <num_work_items>\n"); return !!- 1; }
            }
            break;
        }
        else { fprintf(stderr, "ERROR: invalid argument '%s' (use 'iv{#<values>|:<filename>}' 'i{f|i}[#size]:<filename>' 'o{f|i}[#size]:<filename>' 'lm#size'\n", argv[0]); return !!- 1; }
    }
    if ((narg > 0) && !ndim)
    {
        fprintf(stderr, "ERROR: missing <num_work_items>[/<work_group_size>]\n");
        return !!- 1;
    }

    // OpenCL initialization
    if (initialize(use_gpu, verbose, use_device)) return !!- 1;
    if (!kernel_name[0]) return 0;

    // compile kernel
    cl_int err = 0;
    const char * slist[2] = { source, 0 };
    liStart = coclock();
    cl_program prog = 0;
    if (sourceelf)
    { // use pre-compiled binary
        size_t * lengths = ((size_t *)source);
        if ((cl_int)lengths[0] != num_devices) { fprintf(stderr, "ERROR: number of devices in .elf and this machine doesn't match\n"); return !!- 1; }
        const unsigned char ** binaries = new const unsigned char *[num_devices];
        size_t binsize = (1 + num_devices)*sizeof(size_t);
        for (cl_int i = 0; i < num_devices; i++) {
            binaries[i] = ((unsigned char *)source) + binsize;
            binsize += lengths[1 + i];
        }
        if ((int)binsize > sourcesize) { fprintf(stderr, "ERROR: binary size exceeded internal buffer size of %d bytes\n", sourcesize); return !!- 1; }
        prog = clCreateProgramWithBinary(context, num_devices, device_list, &lengths[1], binaries, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateProgramWithBinary() => %d\n", err); return !!- 1; }
        dumpilisa = 0;
    }
    else
    { // compile from OpenCL source
        prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateProgramWithSource() => %d\n", err); return !!- 1; }
    }

    err = clBuildProgram(prog, num_devices, device_list, build_options[0] ? build_options : NULL, NULL, NULL);
    liStop = coclock();
    { // show warnings/errors
        cl_int errr;
        static char log[65536 * 16]; memset(log, 0, sizeof(log));
        errr = clGetProgramBuildInfo(prog, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        if (!errr) // if(errr) { fprintf(stderr, "ERROR: clGetProgramBuildInfo(%p,%p,CL_PROGRAM_BUILD_LOG,%d,%p,0) => %d\n", prog, device_list[0], (int)sizeof(log), log, errr); return !!-1; }
            if (err || strstr(log, "warning:") || strstr(log, "error:")) fprintf(stderr, "<<<<\n%s\n>>>>\n", log);
    }
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clBuildProgram('%s') => %d (%s)\n", build_options, err, kernel_name); return !!- 1; }
    time_sec = coclock2sec(liStart, liStop);
    printf("OK: COMPILATION on %s took %8.4f sec for %s\n", use_gpu ? "GPU" : "CPU", time_sec, kernel_name);
    if (dumpelf)
    { // dump program binary
        cl_uint ndevs;
        err = clGetProgramInfo(prog, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &ndevs, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clGetProgramInfo(*,CL_PROGRAM_NUM_DEVICES,...) => %d\n", err); return !!- 1; }
        size_t * sizelist = ((size_t *)source);
        sizelist[0] = (size_t)ndevs;
        unsigned char ** binlist = new unsigned char *[ndevs];
        err = clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, ndevs*sizeof(size_t), &sizelist[1], NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clGetProgramInfo(*,CL_PROGRAM_BINARY_SIZES,...) => %d\n", err); return !!- 1; }
        size_t binsize = (1 + ndevs)*sizeof(size_t);
        for (size_t i = 0; i < ndevs; i++) {
            binlist[i] = ((unsigned char *)source) + binsize;
            binsize += sizelist[1 + i];
        }
        if ((int)binsize > sourcesize) { fprintf(stderr, "ERROR: binary size exceeded internal buffer size of %d bytes\n", sourcesize); return !!- 1; }
        err = clGetProgramInfo(prog, CL_PROGRAM_BINARIES, ndevs*sizeof(unsigned char *), binlist, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clGetProgramInfo(*,CL_PROGRAM_BINARIES,...) => %d\n", err); return !!- 1; }
        char fname[2048]; sprintf(fname, "%s.elf", kernel_name);
        FILE * fp;
        if ((fp = fopen(fname, "wb")) == NULL) { fprintf(stderr, "ERROR: unable to create '%s'\n", fname); return !!- 1; }
        char * elfsrc = source + 2 * sizeof(size_t);
        size_t elfsize = binsize - 2 * sizeof(size_t);
        fwrite(elfsrc, 1, elfsize, fp);
        fclose(fp);
        printf("OK: Saved compiled binary into '%s'\n", fname);
    }
    if (dumpilisa)
    { // dump ISA statistics
#ifdef _WIN32
        int res = rename_and_cleanup_dumpfiles(dumpilisa_prefix, kernel_name, device_name);
        if (res != 0) return !!res;
        char isa_file[1024];
        sprintf(isa_file, "%s_%s.isa", kernel_name, device_name);
        if (isa_statistics(kernel_name, device_name, isa_file) < 0) {
            fprintf(stderr, "ERROR: unable to open '%s'\n", isa_file);
            return !!- 1;
        }
#else
        char isa_file[4096];
        sprintf(isa_file, "%s_0_%s_&__OpenCL_%s_kernel.isa", dumpilisa_prefix, device_name, kernel_name);
        if (isa_statistics(kernel_name, device_name, isa_file) < 0) {
            sprintf(isa_file, "%s_0_%s_%s.isa", dumpilisa_prefix, device_name, kernel_name);
            if (isa_statistics(kernel_name, device_name, isa_file) < 0) {
                fprintf(stderr, "ERROR: unable to open '%s'\n", isa_file);
                return !!- 1;
            }
        }
#endif
    }

    // terminate if number of arguments is zero
    if (narg < 1) return 0;

    // get the kernel object
    cl_kernel kernel = clCreateKernel(prog, kernel_name, &err);
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel(%p:%s) => %d\n", prog, kernel_name, err); return !!-1; }
    clReleaseProgram(prog);

    // display kernel info
    size_t wgsize[3] = { 0 };
    err = clGetKernelWorkGroupInfo(kernel, device_list[0], CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(wgsize), wgsize, NULL);
    if(err == CL_SUCCESS) printf("OK: kernel %s info reqd_work_group_size(%d,%d,%d)\n", kernel_name, (int)wgsize[0], (int)wgsize[1], (int)wgsize[1]);
    err = clGetKernelWorkGroupInfo(kernel, device_list[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgsize[0]), wgsize, NULL);
    if(err == CL_SUCCESS) printf("OK: kernel %s info work_group_size(%d)\n", kernel_name, (int)wgsize[0]);
    cl_ulong memsize = 0;
    err = clGetKernelWorkGroupInfo(kernel, device_list[0], CL_KERNEL_LOCAL_MEM_SIZE, sizeof(memsize), &memsize, NULL);
    if(err == CL_SUCCESS) printf("OK: kernel %s info local_mem_size(%d)\n", kernel_name, (int)memsize);
    err = clGetKernelWorkGroupInfo(kernel, device_list[0], CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(memsize), &memsize, NULL);
    if(err == CL_SUCCESS) printf("OK: kernel %s info local_private_size(%d)\n", kernel_name, (int)memsize);

    // create command queue for the first device
#if defined(CL_VERSION_2_0)
    cmd_queue = clCreateCommandQueueWithProperties(context, device_list[0], NULL, &err);
#else
    cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err);
#endif
    if (!cmd_queue) { fprintf(stderr, "ERROR: clCreateCommandQueue() failed\n"); return !!- 1; }

    // create and set arguments 
    cl_mem mem[MAX_KERNEL_ARG];
    for (int i = 0; i < narg; i++)
    {
        size_t asize = 0;
        void * abufp = 0;
        if (type[i] == 0) {
            if (verbose) {
                if (size[i] != sizeof(cl_int)) printf("  ARG# %2d -- immediate -- 0x%08x (%12d) bytes\n", i, size[i], size[i]);
                else printf("  ARG# %2d -- immediate    -- 0x%08x (%12d) [%12.6g]\n", i, *((int*)arg[i]), *((int*)arg[i]), *((float*)arg[i]));
            }
            asize = (size_t)size[i];
            abufp = arg[i];
        }
        else if (type[i] == 1) {
            if (verbose) printf("  ARG# %2d -- local mem -- 0x%08x (%12d) bytes\n", i, size[i], size[i]);
            asize = (size_t)size[i];
            abufp = 0;
        }
        else {
            if (verbose) printf("  ARG# %2d -- R%c buffer -- 0x%08x (%12d) bytes\n", i, (type[i] & 1) ? 'W' : ' ', size[i], size[i]);
            cl_int flags = CL_MEM_READ_WRITE;
#if CL_MEM_USE_PERSISTENT_MEM_AMD
            if (use_persistent) flags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
#endif
            if (image[i].isvalid) {
                cl_image_desc desc = { (unsigned int)(image[i].height == 0 ? CL_MEM_OBJECT_IMAGE1D : CL_MEM_OBJECT_IMAGE2D), (unsigned int)image[i].width, (unsigned int)image[i].height, 0, 1u, 0, 0, 0, 0, NULL };
                mem[i] = clCreateImage(context, flags, &imgfmt[image[i].fmt].fmt, &desc, NULL, &err);
                if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateImage(fmt:%d-%s:%x,%x;%dx%d;stride:%d;size:%d) => %d for argument #%d\n", image[i].fmt, imgfmt[image[i].fmt].name, imgfmt[image[i].fmt].fmt.image_channel_order, imgfmt[image[i].fmt].fmt.image_channel_data_type, image[i].width, image[i].height, image[i].row_pitch, size[i], err, i); return !!- 1; }
                size_t origin[3] = { 0, 0, 0 };
                size_t region[3] = { (size_t)image[i].width, image[i].height > 0 ? (size_t)image[i].height : 1u, 1u };
                err = clEnqueueWriteImage(cmd_queue, mem[i], CL_TRUE, origin, region, image[i].row_pitch, 0, arg[i], 0, 0, 0);
                if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteImage(fmt:%d-%s:%x,%x;%dx%d;stride:%d;size:%d) => %d for argument #%d\n", image[i].fmt, imgfmt[image[i].fmt].name, imgfmt[image[i].fmt].fmt.image_channel_order, imgfmt[image[i].fmt].fmt.image_channel_data_type, image[i].width, image[i].height, image[i].row_pitch, size[i], err, i); return !!- 1; }
            }
            else {
                mem[i] = clCreateBuffer(context, flags, size[i], NULL, &err);
                if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer(size:%d) => %d for argument #%d\n", size[i], err, i); return !!- 1; }
                err = clEnqueueWriteBuffer(cmd_queue, mem[i], 1, 0, size[i], arg[i], 0, 0, 0);
                if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer(size:%d) => %d for argument #%d\n", size[i], err, i); return !!- 1; }
            }
            asize = sizeof(cl_mem);
            abufp = &mem[i];
        }
        err = clSetKernelArg(kernel, i, asize, abufp);
        if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clSetKernelArg(%s,%d,%d,%p) => %d for argument #%d\n", kernel_name, i, (int)asize, abufp, err, i); return !!- 1; }
    }

    // dump kernel workgroun info
    if (verbose)
    {
        size_t wgsize[3];
        wgsize[0] = wgsize[1] = wgsize[2] = 0;
        clGetDeviceInfo(device_list[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wgsize[0]), wgsize, NULL);
        printf("clGetDeviceInfo(*,             CL_DEVICE_MAX_WORK_GROUP_SIZE,*) => %d\n", (int)wgsize[0]);
        wgsize[0] = wgsize[1] = wgsize[2] = 0;
        clGetKernelWorkGroupInfo(kernel, device_list[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgsize[0]), wgsize, NULL);
        printf("clGetKernelWorkGroupInfo(*,        CL_KERNEL_WORK_GROUP_SIZE,*) => %d\n", (int)wgsize[0]);
        wgsize[0] = wgsize[1] = wgsize[2] = 0;
        clGetKernelWorkGroupInfo(kernel, device_list[0], CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(wgsize), wgsize, NULL);
        printf("clGetKernelWorkGroupInfo(*,CL_KERNEL_COMPILE_WORK_GROUP_SIZE,*) => { %d, %d, %d }\n", (int)wgsize[0], (int)wgsize[1], (int)wgsize[2]);
        cl_ulong device_local_mem_type = 0;
        clGetDeviceInfo(device_list[0], CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_ulong), &device_local_mem_type, NULL);
        printf("clGetDeviceInfo(*,                  CL_DEVICE_LOCAL_MEM_TYPE,*) => %d\n", (int)device_local_mem_type);
        cl_ulong device_local_mem_size = 0;
        clGetDeviceInfo(device_list[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &device_local_mem_size, NULL);
        printf("clGetDeviceInfo(*,                  CL_DEVICE_LOCAL_MEM_SIZE,*) => %d\n", (int)device_local_mem_size);
        cl_ulong kernel_local_mem_size = 0;
        clGetKernelWorkGroupInfo(kernel, device_list[0], CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &kernel_local_mem_size, NULL);
        printf("clGetKernelWorkGroupInfo(*,         CL_KERNEL_LOCAL_MEM_SIZE,*) => %d\n", (int)kernel_local_mem_size);
        printf("global_work_size = { %d, %d, %d }\n", (int)nworkitems[0], (int)nworkitems[1], (int)nworkitems[2]);
        printf("local_work_size  = { %d, %d, %d }\n", (int)workgroupsize[0], (int)workgroupsize[1], (int)workgroupsize[2]);
        fflush(stdout);
    }

    // execute the kernel
    clFinish(cmd_queue);
    liStart = coclock();
    err = clEnqueueNDRangeKernel(cmd_queue, kernel, (cl_uint)ndim, NULL, nworkitems, nwgdim ? workgroupsize : NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueNDRangeKernel(%s)=>%d failed\n", kernel_name, err); return !!- 1; }
    // wait for completion
    err = clFinish(cmd_queue);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clFinish(%s)=>%d failed\n", kernel_name, err); return !!- 1; }
    liStop = coclock();
    time_sec = coclock2sec(liStart, liStop);
    printf("OK: RUN SUCCESSFUL on %s work:{%d", use_gpu ? "GPU" : "CPU", (int)nworkitems[0]); for (int i = 1; i < (int)ndim; i++) printf(",%d", (int)nworkitems[i]);
    if (nwgdim) { printf("}/{%d", (int)workgroupsize[0]); for (int i = 1; i < (int)nwgdim; i++) printf(",%d", (int)workgroupsize[i]); }
    printf("} [%9.5f sec/exec] %s (1st execution)\n", time_sec, kernel_name);
    if (repeat > 0)
    { // repeated execution of the kernel for performance measurement
        clFinish(cmd_queue);
        liStart = coclock();
        for(int i = 0; i < repeat; i++)
        { // schedule kernel for execution specified number of times
            err = clEnqueueNDRangeKernel(cmd_queue, kernel, (cl_uint)ndim, NULL, nworkitems, nwgdim ? workgroupsize : NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueNDRangeKernel(%s)=>%d failed\n", kernel_name, err); return !!- 1; }
        }
        // wait for completion
        clFinish(cmd_queue);
        if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clFinish(%s) => %d\n", kernel_name, err); return !!- 1; }
        liStop = coclock();
        time_sec = coclock2sec(liStart, liStop);
        printf("OK: RUN SUCCESSFUL on %s work:{%d", use_gpu ? "GPU" : "CPU", (int)nworkitems[0]); for (int i = 1; i < (int)ndim; i++) printf(",%d", (int)nworkitems[i]);
        if (nwgdim) { printf("}/{%d", (int)workgroupsize[0]); for (int i = 1; i < (int)nwgdim; i++) printf(",%d", (int)workgroupsize[i]); }
        printf("} [%9.5f sec/exec] %s (%d iterations in %9.5f sec)\n", time_sec / (float)repeat, kernel_name, repeat, time_sec);
    }

    // save output
    int status = 0;
    for (int i = 0; i < narg; i++)
    {
        if (type[i] == 2 || type[i] == 3)
        {
            if (image[i].isvalid) {
                size_t origin[3] = { 0, 0, 0 };
                size_t region[3] = { (size_t)image[i].width, image[i].height > 0 ? (size_t)image[i].height : 1u, 1u };
                err = clEnqueueReadImage(cmd_queue, mem[i], CL_TRUE, origin, region, image[i].row_pitch, 0, arg[i], 0, 0, 0);
                if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueReadImage(fmt:%d-%s:%x,%x;%dx%d;stride:%d;size:%d) => %d for argument #%d\n", image[i].fmt, imgfmt[image[i].fmt].name, imgfmt[image[i].fmt].fmt.image_channel_order, imgfmt[image[i].fmt].fmt.image_channel_data_type, image[i].width, image[i].height, image[i].row_pitch, size[i], err, i); return !!-1; }
            }
            else {
                err = clEnqueueReadBuffer(cmd_queue, mem[i], 1, 0, size[i], arg[i], 0, 0, 0);
                if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueReadBuffer(size:%d) => %d for argument #%d\n", size[i], err, i); return !!- 1; }
            }
        }
        if (type[i] == 3)
        {
            FILE * fp = fopen(name[i], "wb");
            if (!fp) { fprintf(stderr, "ERROR: Unable to create '%s' for storing output of argument #%d\n", name[i], i); return !!- 1; }
            fwrite(arg[i], size[i], 1, fp);
            fclose(fp);
            if (verbose)printf("OK: Saved output of argument #%d in '%s'\n", i, name[i]);
        }
        if (check[i])
        {
            int sum = 0xc0de8576;
            int n = size[i];
            for (unsigned int * p = (unsigned int *)arg[i]; (n -= 4) >= 0; p++)
                if (*p)sum += *p + (sum << 3) + ((unsigned int)sum >> 3);
            if (check[i] == 2) {
                if (sum != checksum[i]) {
                    fprintf(stderr, "ERROR: CHECKSUM of argument %s#%d is 0x%08x (shall be 0x%08x) **** MISMATCH\n", kernel_name, i, sum, checksum[i]);
                    status = -1;
                }
                else printf("OK: CHECKSUM of argument %s#%d is 0x%08x ** MATCHED\n", kernel_name, i, sum);
            }
            else printf("OK: CHECKSUM of argument %s#%d is 0x%08x\n", kernel_name, i, sum);
        }
        if (comparefile[i] && comparefile[i][0])
        {
            FILE * fp = fopen(comparefile[i], "rb");
            if (!fp) { fprintf(stderr, "ERROR: unable to open '%s' for argument #%d\n", comparefile[i], i); status = -1; }
            else {
                if (skip[i] > 0) fseek(fp, (long)(size[i] * skip[i]), SEEK_SET);
                static char buf[65536]; char * pbuf = (char *)arg[i];
                int cb = size[i];
                int cmpskip = cmpskipa[i];
                int cmpsize = cmpsizea[i];
                if (cmpsize == 0) cmpsize = cb;
                float maxMagFile = 0, maxMagMem = 0, sumSqrErr = 0, maxErr = 0;
                int maxErrPos = 0, sumCount = 0;
                for (int n; (cb > 0) && (n = (int)fread(buf, 1, (cb > sizeof(buf)) ? sizeof(buf) : cb, fp)) > 0; cb -= n, pbuf += n, cmpskip -= n, cmpsize -= n)
                {
                    if (compareErrorLimit[i] > 0) {
                        int j; for (j = 0; j < n; j += 4) {
                            if ((j - cmpskip) < 0) continue;
                            if ((cmpsize - j) <= 0) continue;
                            float diff = *(float *)&buf[j] - *(float *)&pbuf[j]; sumSqrErr += diff * diff;
                            float magFile = fabsf(*(float *)&buf[j]); maxMagFile = (magFile > maxMagFile) ? magFile : maxMagFile;
                            float magMem = fabsf(*(float *)&pbuf[j]); maxMagMem = (magMem  > maxMagMem) ? magMem : maxMagMem;
                            diff = (diff < 0) ? -diff : diff;
                            if(diff > maxErr) {
                                maxErr = diff;
                                maxErrPos = sumCount;
                            }
                            sumCount++;
                        }
                    }
                    else {
                        int j; for (j = 0; j < n; j++) {
                            if ((j - cmpskip) < 0) continue;
                            if ((cmpsize - j) <= 0) continue;
                            if (buf[j] != pbuf[j]) break;
                        }
                        if (j < n) {
                            fprintf(stderr, "ERROR: FILE COMPARE for argument %s#%d at position %d (0x%02x(mem) != 0x%02x(file)) **** MISMATCH of %s with %s\n", kernel_name, i, size[i] - cb + j, pbuf[j] & 0xff, buf[j] & 0xff, name[i] ? name[i] : "unsaved", comparefile[i]);
                            cb = -1;
                            break;
                        }
                    }
                }
                fclose(fp);
                if (compareErrorLimit[i] > 0 && sumCount > 0) {
                    float mag = (maxMagFile > maxMagMem) ? maxMagFile : maxMagMem; mag = (mag > 1) ? mag : 1;
                    float avgerr = sqrtf(sumSqrErr) / (sumCount * mag);
                    if (avgerr > compareErrorLimit[i]) {
                        printf("ERROR: floating-point compare failed with avg-err-%g max-err=%g@%d in %s for argument %s#%d\n", avgerr, maxErr, maxErrPos*4, comparefile[i], kernel_name, i);
                        if(!ignoreCompareResult)
                            status = -1;
                    }
                    else printf("OK: FILE COMPARE for argument #%d with %s ** MATCHED (with avg-err=%g max-err=%g@%d)\n", i, comparefile[i], avgerr, maxErr, maxErrPos*4);
                }
                else if (cb < 0) {
                    if(!ignoreCompareResult)
                        status = -1;
                }
                else if (cmpsize > 0) printf("WARNING: EOF at position %d in %s for argument %s#%d\n", size[i] - cb, comparefile[i], kernel_name, i);
                else printf("OK: FILE COMPARE for argument #%d with %s ** MATCHED\n", i, comparefile[i]);
            }
        }
    }
    fflush(stdout);
    // OpenCL shutdown
    if(shutdown()) return !!-1;

    return !!status;
}
#else
int
#if _WIN32
__cdecl
#endif
main(int argc, char * argv[])
{
    printf("ERROR: This version of RUNCL was not built with OpenCL\n");
    return 1;
}
#endif
