# ########################################################################
# Copyright (C) 2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# ------------------------------------------------------------------------
#
# Usage:
#
# Assuming one runs the following Tensile command:
#
#   Tensile/bin/Tensile <yaml> tensile-out
#
# One can invoke make from the directory containing this Makefile as follows:
#
# make co TENSILE_OUT=tensile-out
#
# where TENSILE_OUT is the relative path to the tensile output directory.
#
# The Makefile auto-detects .co and .s files anywhere under the problem's
# 00_Final directory, so it is not sensitive to changes in intermediate
# path structure (e.g. whether a caches/<hash>/ level is present or not).
#
# To target a specific set of assembly files instead of auto-detecting,
# pass SRCFILES explicitly:
#
# make co TENSILE_OUT=tensile-out \
#         SRCFILES="tensile-out/1_BenchmarkProblems/Cijk_.../00_Final/source/build_tmp/SOURCE/assembly/*.s"
#
# ARCH and WAVEFRONTSIZE can also be overridden:
#
# make co TENSILE_OUT=tensile-out ARCH="gfx942:xnack-" WAVEFRONTSIZE="-mno-wavefrontsize64"
#
# For 64-wavefront systems:   make co TENSILE_OUT=tensile-out ARCH="gfx942"  WAVE=64
# For 32-wavefront systems:   make co TENSILE_OUT=tensile-out ARCH="gfx1100" WAVE=32
#
# In addition, we provide AS and ASM_ARGS, and LD and LINK_ARGS as customization
# points for the assemble and link step respectively.
# ------------------------------------------------------------------------

ROCM_PATH ?= /opt/rocm
AS := $(ROCM_PATH)/bin/amdclang++
LDD := $(ROCM_PATH)/bin/amdclang++
WAVE ?= 64
WAVEFRONTSIZE ?= $(if $(WAVE:32=),-mwavefrontsize64, -mno-wavefrontsize64)

TRUE16_FEATURE :=
ifneq ($(filter gfx1250,$(TARGET_ARCH)),)
	TRUE16_FEATURE += -Xclangas -target-feature -Xclangas +real-true16
endif

.SECONDEXPANSION:
co: COFILES = $(shell find ${TENSILE_OUT}/1_BenchmarkProblems/*/00_Final -name '*.co' -path '*/library/*')
co: $$(COFILES)

.SECONDEXPANSION:
${TENSILE_OUT}/1_BenchmarkProblems%.co: PROBLEM  = $(firstword $(subst /, ,$*))
                                        SRCFILES ?= $(shell find ${TENSILE_OUT}/1_BenchmarkProblems/$(PROBLEM)/00_Final -path '*/assembly/*.s')
                                        OBJFILES = $(patsubst %.s,%.o,$(SRCFILES))
${TENSILE_OUT}/1_BenchmarkProblems%.co: $$(OBJFILES)
	@echo relinking $(PROBLEM) $(lastword $(subst /, ,$@))
	@$(LDD) -target amdgcn-amd-amdhsa -Xlinker --build-id=sha1 ${TRUE16_FEATURE} ${LINK_ARGS} -o $@ $^

%.o: PROBLEM = $(word 3, $(subst /, ,$*))
     COFILE = $(shell find ${TENSILE_OUT}/1_BenchmarkProblems/$(PROBLEM)/00_Final -name '*.co' -path '*/library/*' | head -1)
     ARCH ?= $(subst .co,,$(subst TensileLibrary_,,$(lastword $(subst /, ,$(COFILE)))))
%.o: %.s
	@echo rebuilding $(lastword $(subst /, ,$@)) for $(ARCH)
	$(AS) -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=$(ARCH) ${WAVEFRONTSIZE} ${ASM_ARGS} ${TRUE16_FEATURE} -c -o $@ $^
