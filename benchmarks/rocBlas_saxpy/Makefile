C = hipcc
CFLAGS = -std=c++14 -lrocblas
LDFLAGS = -L/opt/rocm/rocblas/lib

all: saxpy

saxpy: saxpy_rocblas.cpp
	$(C) $(LDFLAGS) $(CFLAGS) -o saxpy.exe saxpy_rocblas.cpp 
clean:
	rm -f saxpy.exe
