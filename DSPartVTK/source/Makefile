# ================= ToVtk v5.2.029 (09-10-2022) =================

# Toggles
USE_DEBUG ?= NO
USE_FAST_MATH ?= YES
USE_NATIVE_CPU_OPTIMIZATIONS ?= NO

# Output
EXECNAME        := DSPartVTK_linux64
EXECS_DIRECTORY := ../bin/

# Toolchain
CXX ?= g++

# Include / lib dirs
LIBS_DIRECTORIES := -L./ -L../../src/lib/linux_gcc

# Flags
CPPFLAGS := -I./
ifeq ($(USE_DEBUG),YES)
  CXXFLAGS := -std=c++0x -O0 -g -Wall
else
  CXXFLAGS := -std=c++0x -O3
endif
ifeq ($(USE_FAST_MATH),YES)
  CXXFLAGS += -ffast-math
endif
ifeq ($(USE_NATIVE_CPU_OPTIMIZATIONS),YES)
  CXXFLAGS += -march=native
endif
CXXFLAGS += -fopenmp

LDFLAGS :=
LDFLAGS += -fopenmp
JLIBS := $(LIBS_DIRECTORIES) -ljvtklib_64
LDLIBS := -lm

# Sources / objects
OBJXML   = JXml.o tinystr.o tinyxml.o tinyxmlerror.o tinyxmlparser.o
OBCOMMON = Functions.o FunctionsGeo3d.o JBinaryData.o JDataArrays.o JException.o JObject.o \
           JOutputCsv.o JPartDataBi4.o JPartDataHead.o JRangeFilter.o JSpaceCtes.o \
           JSpaceEParms.o JSpaceParts.o
OBCODE   = JCfgRun.o main.o

OBJECTS = $(OBJXML) $(OBCOMMON) $(OBCODE)

# Rules
.PHONY: all clean
.SUFFIXES:
.DELETE_ON_ERROR:

all: $(EXECS_DIRECTORY)/$(EXECNAME)
ifeq ($(USE_DEBUG),NO)
	@echo "  --- Compiled Release CPU version ---"
else
	@echo "  --- Compiled Debug CPU version ---"
	mv $(EXECS_DIRECTORY)/$(EXECNAME) $(EXECS_DIRECTORY)/$(EXECNAME)_debug
endif
	@rm -f *.o

$(EXECS_DIRECTORY):
	@mkdir -p $@

$(EXECS_DIRECTORY)/$(EXECNAME): $(EXECS_DIRECTORY) $(OBJECTS)
	$(CXX) $(LDFLAGS) $(OBJECTS) $(JLIBS) $(LDLIBS) -o $@

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	$(RM) -f *.o $(EXECS_DIRECTORY)/$(EXECNAME) $(EXECS_DIRECTORY)/$(EXECNAME)_debug
