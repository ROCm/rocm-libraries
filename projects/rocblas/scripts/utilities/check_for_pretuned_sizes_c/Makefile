# Makefile assumes rocBLAS is installed in /opt/rocm

ROCM_INSTALL_DIR=/opt/rocm
ROCBLAS_INSTALL_DIR=$(ROCM_INSTALL_DIR)
ROCBLAS_INCLUDE=$(ROCBLAS_INSTALL_DIR)/include
ROCBLAS_LIB_PATH=$(ROCBLAS_INSTALL_DIR)/lib
HIP_INCLUDE=$(ROCM_INSTALL_DIR)/include

CXX_FLAGS=-I$(ROCBLAS_INCLUDE) -I$(HIP_INCLUDE) -std=c++17 -DROCM_USE_FLOAT16
CXX=$(ROCM_INSTALL_DIR)/bin/amdclang++
LD_FLAGS=-L$(ROCBLAS_LIB_PATH) -lrocblas
LD=$(ROCM_INSTALL_DIR)/bin/amdclang++

OBJ=check-for-pretuned-sizes.o
EXE=check-for-pretuned-sizes

%.o: %.cpp
	$(CXX) -x hip -c -o $@ $< $(CXX_FLAGS)

$(EXE) : $(OBJ)
	$(LD) $(OBJ) $(LD_FLAGS) -o $@

clean:
	rm -f $(EXE) $(OBJ)
