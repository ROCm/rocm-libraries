# AMD RunCL
RunCL is a command-line tool to build, execute, and debug OpenCL programs, with a simple, easy-to-use interface.

## RunCL Usage

    Usage: runcl [platform-options] [-I<include-dir>] [[-D<name>=<value>] ...]
                 <kernel.[cl|elf]> [kernel-arguments] 
                 <arguments> <num_work_items>[/<work_group_size>]
    
       [platform-options]
           -v                    verbose
           -gpu                  use GPU device (default)
           -cpu                  use CPU device
           -device <name>|#<num> use specified device
           -bo <string>          OpenCL build option
    
       [kernel-options]
           -k <kernel-name>      kernel name
           -p                    use persistence flag
           -r[link] <exec-count> execution count
           -w <msec>             waiting time
           -dumpcl               dump OpenCL code after pre-processing
           -dumpilisa            dump ISA of kernel and show ISA statistics
           -dumpelf              dump ELF binary
    
       The <arguments> shall be given in the order as required by the kernel.
         For value arguments use   
             iv#<int/float>[,<int/float>...] or 
             iv:<file> (e.g., iv#10.2,10,0x10)
         For local memory use      
             lm#<local-memory-size> (e.g., lm#8192)
         For input buffer use      
             if[#<buffer-size>]:[<file>][#[[<checksum>][/<file>[@<offset>#<end>]]]]
             (e.g., if:input.bin)
         For output (or RW) buffer 
             of[#<buffer-size>]:[#]<file>[@<ofile>][#[[<checksum>][/[+<float-tolerance>]<file>[@<offset>#<end>]]]] 
             (e.g., of#16384:output.bin)
         For input image  use      
             ii#<width>x<height>,<stride>,<u8/s16/u16/bgra/rgba/argb>:<file> 
             (e.g., ii#1920x1080,7680,bgra:screen1920x1080.rgb)
         For output image  use     
             oi#<width>x<height>,<stride>,<u8/s16/u16/bgra/rgba/argb>:<file> 
             (e.g., oi#1920x1080,7680,bgra:screen1920x1080.rgb

## Example

    % cat subtract.cl
    __kernel __attribute__((reqd_work_group_size(64, 1, 1)))
    void subtract(
        __global float * a, 
        __global float * b, 
        __global float * c, 
        uint count)
    {
        uint id = get_global_id(0);
        if(id < count) {
            c[id] = a[id] - b[id];
        }
    }
    % runcl subtract.cl if#4000:a.f32 if#4000:b.f32 of#4000:#out.f32 iv#1000 1024,1,1/64,1,1
    OK: Using GPU device#0 [...]
    OK: COMPILATION on GPU took   0.1268 sec for subtract
    OK: kernel subtract info reqd_work_group_size(64,1,1)
    OK: kernel subtract info work_group_size(256)
    OK: kernel subtract info local_mem_size(0)
    OK: kernel subtract info local_private_size(0)
    OK: RUN SUCCESSFUL on GPU work:{1024,1,1}/{64,1,1} [  0.00025 sec/exec] subtract (1st execution)
