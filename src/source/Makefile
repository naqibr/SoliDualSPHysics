# =================== SoliDualSPHysics GPU/CPU ===================

.DEFAULT_GOAL := all

# Build toggles (YES/NO)
USE_DEBUG                    ?= NO
USE_FAST_MATH                ?= YES
USE_NATIVE_CPU_OPTIMIZATIONS ?= YES
COMPILE_VTKLIB               ?= YES
COMPILE_NUMEXLIB             ?= YES
COMPILE_CHRONO               ?= YES
COMPILE_WAVEGEN              ?= YES
COMPILE_MOORDYN              ?= YES

# ------------------- Output -------------------
EXECNAME        := SoliDualSPHysics_linux64

# Where to place the executable (can be overridden)
EXECS_DIRECTORY ?= ../../bin/linux
ifeq ($(strip $(EXECS_DIRECTORY)),)
  EXECS_DIRECTORY := ../../bin/linux
endif

# Where to place .o files
BUILD_DIR ?= ../build/SoliDualSPHysics

# ------------------- Toolchain -------------------
CXX   ?= g++
CC    ?= gcc
NVCC  ?= nvcc
override LD := $(CXX)

# ------------------- CUDA selection -------------------
CUDAVER ?= 12

# CUDA root resolution:
# 1) nvcc on PATH, 2) CUDA_HOME, 3) /usr/local/cuda
NVCC_PATH  := $(shell which $(NVCC) 2>/dev/null)
ifeq ($(NVCC_PATH),)
  CUDA_ROOT ?= $(CUDA_HOME)
else
  CUDA_ROOT := $(abspath $(dir $(dir $(NVCC_PATH))))
endif
CUDA_ROOT ?= /usr/local/cuda

# ------------------- Codegen (SM) -------------------
ifeq ($(CUDAVER),12)
  # RTX 4000 Ada (compute capability 8.9 => sm_89)
  GENCODE := -gencode=arch=compute_89,code=\"sm_89,compute_89\"
endif

# ------------------- Flags -------------------
# Common preprocessor/includes
CPPFLAGS := -I./ -I$(CUDA_ROOT)/include -D_WITHGPU

# C++ compile flags
ifeq ($(USE_DEBUG),YES)
  CXXFLAGS := -std=c++11 -O0 -g -Wall -fopenmp
else
  CXXFLAGS := -std=c++11 -O3 -fopenmp
endif
ifeq ($(USE_FAST_MATH),YES)
  CXXFLAGS += -ffast-math
endif
ifeq ($(USE_NATIVE_CPU_OPTIMIZATIONS),YES)
  CXXFLAGS += -march=native
endif

# Generate .d dep files for C++ (headers trigger rebuild)
CXXFLAGS += -MMD -MP

# NVCC compile flags
NVCCFLAGS := $(GENCODE) -ccbin $(CXX)
ifeq ($(USE_DEBUG),YES)
  NVCCFLAGS += -O0 -g
else
  NVCCFLAGS += -O3
endif
ifeq ($(USE_FAST_MATH),YES)
  NVCCFLAGS += -use_fast_math
endif
ifeq ($(CUDAVER),11)
  NVCCFLAGS += -Wno-deprecated-gpu-targets
endif

# Linker flags & libraries
LDFLAGS := -fopenmp -L$(CUDA_ROOT)/lib64
# Static cudart combo for CUDA 12:
LDLIBS  := -lcudart_static -lcudadevrt -lculibos -lpthread -ldl -lrt -lgomp

# ------------------- External libs -------------------
LIBS_DIRECTORIES := -L./ -L../lib/linux_gcc
JLIBS := $(LIBS_DIRECTORIES)
ifeq ($(COMPILE_VTKLIB),YES)
  JLIBS += -ljvtklib_64
else
  CPPFLAGS += -DDISABLE_VTKLIB
endif
ifeq ($(COMPILE_NUMEXLIB),YES)
  JLIBS += -ljnumexlib_64
else
  CPPFLAGS += -DDISABLE_NUMEXLIB
endif
ifeq ($(COMPILE_CHRONO),YES)
  JLIBS += -ldsphchrono -lChronoEngine
else
  CPPFLAGS += -DDISABLE_CHRONO
endif
ifeq ($(COMPILE_WAVEGEN),YES)
  JLIBS += -ljwavegen_64
else
  CPPFLAGS += -DDISABLE_WAVEGEN
endif
ifeq ($(COMPILE_MOORDYN),YES)
  JLIBS += -ldsphmoordyn_64
else
  CPPFLAGS += -DDISABLE_MOORDYN
endif

# ------------------- Sources/Objects (logical names) -------------------
OBJXML          = JXml.o tinystr.o tinyxml.o tinyxmlerror.o tinyxmlparser.o
OBJSPHMOTION    = JMotion.o JMotionList.o JMotionMov.o JMotionObj.o JMotionPos.o JDsMotion.o
OBCOMMON        = Functions.o FunGeo3d.o FunSphKernelsCfg.o JAppInfo.o JBinaryData.o JCfgRunBase.o JDataArrays.o JException.o JLinearValue.o JLog2.o JObject.o JOutputCsv.o JRadixSort.o JRangeFilter.o JReadDatafile.o JSaveCsv2.o JTimeControl.o randomc.o
OBCOMMONDSPH    = JDsphConfig.o JDsPips.o JPartDataBi4.o JPartDataHead.o JPartFloatBi4.o JPartOutBi4Save.o JCaseCtes.o JCaseEParms.o JCaseParts.o JCaseProperties.o JCaseUserVars.o JCaseVtkOut.o
OBSPH           = JArraysCpu.o JCellDivCpu.o JSphCfgRun.o JComputeMotionRef.o JDsDcell.o JDsDamping.o JDsExtraData.o JDsGaugeItem.o JDsGaugeSystem.o JDsPartsOut.o JDsSaveDt.o JSphShifting.o JSph.o JDsAccInput.o JSphCpu.o JDsInitialize.o JFtMotionSave.o JSphMk.o JDsPartsInit.o JDsFixedDt.o JDsViscoInput.o JDsOutputTime.o JDsTimers.o JWaveSpectrumGpu.o main.o
OBSPHSINGLE     = JCellDivCpuSingle.o JPartsLoad4.o JSphCpuSingle.o
OBCOMMONGPU     = FunctionsCuda.o JObjectGpu.o
OBSPHGPU        = JArraysGpu.o JDebugSphGpu.o JCellDivGpu.o JSphGpu.o JDsGpuInfo.o
OBSPHSINGLEGPU  = JCellDivGpuSingle.o JSphGpuSingle.o
OBCUDA          = JCellDivGpu_ker.o JCellDivGpuSingle_ker.o JDsPips_ker.o JDsGauge_ker.o JReduSum_ker.o JSphShifting_ker.o JDsAccInput_ker.o JSphGpu_ker.o JSphGpuSimple_ker.o JWaveOrder2_ker.o JSphGpu_DefStruc_ker.o JSphGpu_ExpressionParser.o
OBWAVERZ        = JMLPistonsGpu.o JRelaxZonesGpu.o
OBWAVERZCUDA    = JRelaxZone_ker.o
OBCHRONO        = JChronoObjects.o
OBMOORDYN       = JDsMooredFloatings.o JDsFtForcePoints.o
OBINOUT         = JSphCpu_InOut.o JSphCpuSingle_InOut.o JSphInOut.o JSphInOutZone.o JSphInOutGridData.o JSphInOutPoints.o JSphInOutVel.o JSphInOutVelAwas.o JSphInOutZsurf.o JSimpleNeigs.o
OBINOUTGPU      = JSphGpuSingle_InOut.o
OBMDBC          = JPartNormalData.o
OBDeformStruc   = JSphDeformStruc.o JSphCpu_ExpressionParser.o

OBJECTS = $(OBJXML) $(OBJSPHMOTION) $(OBCOMMON) $(OBCOMMONDSPH) $(OBSPH) $(OBSPHSINGLE) \
          $(OBCOMMONGPU) $(OBSPHGPU) $(OBSPHSINGLEGPU) $(OBCUDA) \
          $(OBWAVERZ) $(OBWAVERZCUDA) $(OBCHRONO) $(OBMOORDYN) \
          $(OBINOUT) $(OBINOUTGPU) $(OBMDBC) $(OBDeformStruc)

# Map to BUILD_DIR (out-of-tree)
OBJECTS := $(addprefix $(BUILD_DIR)/,$(OBJECTS))

# Optional: dependency files (for C++; CUDA ones may not create .d but that's OK)
DEPS := $(OBJECTS:.o=.d)
-include $(DEPS)

# ------------------- Rules -------------------
.PHONY: all clean
.SUFFIXES:
.DELETE_ON_ERROR:

all: $(EXECS_DIRECTORY)/$(EXECNAME)
ifeq ($(USE_DEBUG),YES)
	@echo " --- Compiled Debug GPU version ---"
else
	@echo " --- Compiled Release GPU version ---"
endif

# Ensure directories exist
$(BUILD_DIR):
	@mkdir -p $@

$(EXECS_DIRECTORY):
	@mkdir -p $@

# Link rule
$(EXECS_DIRECTORY)/$(EXECNAME): $(OBJECTS) | $(BUILD_DIR) $(EXECS_DIRECTORY)
	$(CXX) $(LDFLAGS) $(OBJECTS) $(JLIBS) $(LDLIBS) -o $@

# Compile rules: .o in BUILD_DIR, sources in src dir
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

clean:
	$(RM) -rf $(BUILD_DIR) $(EXECS_DIRECTORY)/$(EXECNAME) $(EXECNAME)_debug
