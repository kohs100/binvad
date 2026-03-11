cmake_minimum_required(VERSION 3.14)
project(binvad CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(BINVAD_ORT_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "Prebuilt ONNX Runtime root (contains include/ and lib/)")

set(_ORT_CANDIDATE_INCLUDE_DIRS)
set(_ORT_CANDIDATE_LIB_DIRS)

# 1) Prebuilt ORT in third_party (preferred)
list(APPEND _ORT_CANDIDATE_INCLUDE_DIRS
  "${BINVAD_ORT_DIR}/include"
  "${BINVAD_ORT_DIR}"
)
list(APPEND _ORT_CANDIDATE_LIB_DIRS
  "${BINVAD_ORT_DIR}/lib"
  "${BINVAD_ORT_DIR}/lib64"
  "${BINVAD_ORT_DIR}"
  # Backward compatibility: source-built ORT output layout.
  "${BINVAD_ORT_DIR}/build/Linux/Release"
  "${BINVAD_ORT_DIR}/build/Linux/Release/lib"
)

# 2) Optional ONNXRUNTIME_DIR override
if (DEFINED ONNXRUNTIME_DIR)
  list(APPEND _ORT_CANDIDATE_INCLUDE_DIRS
    "${ONNXRUNTIME_DIR}/include"
    "${ONNXRUNTIME_DIR}"
  )
  list(APPEND _ORT_CANDIDATE_LIB_DIRS
    "${ONNXRUNTIME_DIR}/lib"
    "${ONNXRUNTIME_DIR}/lib64"
    "${ONNXRUNTIME_DIR}"
  )
endif()

# 3) System fallbacks
list(APPEND _ORT_CANDIDATE_INCLUDE_DIRS /usr/include /usr/local/include /opt/homebrew/include)
list(APPEND _ORT_CANDIDATE_LIB_DIRS /usr/lib /usr/local/lib /opt/homebrew/lib)

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS ${_ORT_CANDIDATE_INCLUDE_DIRS}
  PATH_SUFFIXES "" onnxruntime/core/session
)

find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
  PATHS ${_ORT_CANDIDATE_LIB_DIRS}
)

if (NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime not found.\n"
    "Place prebuilt ONNX Runtime under ${BINVAD_ORT_DIR}.\n"
    "Expected include: ${BINVAD_ORT_DIR}/include/onnxruntime/core/session/onnxruntime_cxx_api.h\n"
    "Expected lib: ${BINVAD_ORT_DIR}/lib/libonnxruntime.so (or .a)\n"
    "Or set -DONNXRUNTIME_DIR=/path/to/onnxruntime-install"
  )
endif()

add_library(onnxruntime_external INTERFACE)
target_include_directories(onnxruntime_external INTERFACE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(onnxruntime_external INTERFACE ${ONNXRUNTIME_LIBRARY})
target_link_libraries(onnxruntime_external INTERFACE Threads::Threads)

add_executable(binvad src/main.cpp)
target_link_libraries(binvad PRIVATE onnxruntime_external)

if (UNIX AND NOT APPLE)
  get_filename_component(_ORT_LIB_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
  set_target_properties(binvad PROPERTIES BUILD_RPATH "${_ORT_LIB_DIR}")
endif()

message(STATUS "ONNXRUNTIME_INCLUDE_DIR=${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNXRUNTIME_LIBRARY=${ONNXRUNTIME_LIBRARY}")
