cmake_minimum_required(VERSION 2.8)

PROJECT(To_VTK)

# CPU Objects

set(OBJXML JXml.cpp tinystr.cpp tinyxml.cpp tinyxmlerror.cpp tinyxmlparser.cpp)
set(OBCOMMON Functions.cpp FunctionsGeo3d.cpp JBinaryData.cpp JDataArrays.cpp JException.cpp JObject.cpp JOutputCsv.cpp JPartDataBi4.cpp JPartDataHead.cpp JRangeFilter.cpp JSpaceCtes.cpp JSpaceEParms.cpp JSpaceParts.cpp)
set(OBCODE JCfgRun.cpp main.cpp)

# Static libraries linker path

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message("Using libraries for gcc")
    LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/linux_gcc)
elseif(MSVC)
  message("Windows version")
  
  if(MSVC_VERSION VERSION_EQUAL 1800)
    # MSVC 2013
    LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/vs2013)
  elseif(MSVC_VERSION VERSION_EQUAL 1900)
    # MSVC 2015
    LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/vs2015)
  elseif(MSVC_VERSION VERSION_GREATER_EQUAL 1930)
    # MSVC 2022
    LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../lib/vs)
  endif()
endif()

# Binaries

add_executable(ToVTK4_linux64 ${OBJXML} ${OBCOMMON} ${OBCODE})
install(TARGETS	ToVTK4_linux64 DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../bin)

# Linker flags

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(ToVTK4_linux64 jvtklib_64)
  set_target_properties(ToVTK4_linux64 PROPERTIES COMPILE_FLAGS "-use_fast_math -O3")
  
elseif(MSVC)
  if(MSVC_VERSION VERSION_EQUAL 1800)
    # MSVC 2013
    target_link_libraries(ToVTK4_linux64 LibJVtkLib_x64_v120_Release)
  elseif(MSVC_VERSION VERSION_EQUAL 1900)
    # MSVC 2015
    target_link_libraries(ToVTK4_linux64 LibJVtkLib_x64_v140_Release)
  elseif(MSVC_VERSION VERSION_GREATER_EQUAL 1930)
    # MSVC 2022
    target_link_libraries(ToVTK4_linux64 LibJVtkLib_x64_v143_Release)
  endif()
  
  SET(CUDA_PROPAGATE_HOST_FLAGS OFF CACHE BOOL "Propagate C/CXX Flags and friends to the host compiler in NVCC via -Xompile  " FORCE)
  
  foreach(CPP_FLAGS CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${CPP_FLAGS} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${CPP_FLAGS} "${${CPP_FLAGS}}")
    endif(${CPP_FLAGS} MATCHES "/MD")
  endforeach(CPP_FLAGS)
endif()
 
