cmake_minimum_required(VERSION 3.18)
project(rgrid CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

add_library(rgrid STATIC src/grid.cpp src/codecs.cpp src/field.cpp
                         src/interp.cpp)
target_include_directories(rgrid PUBLIC include)

option(RGRID_BUILD_PYTHON "Build Python bindings" ON)
option(RGRID_BUILD_TESTS "Build C++ tests" ON)
option(RGRID_WITH_ZFP "Enable ZFP fixed-rate codec (C3)" ON)

include(FetchContent)

if(RGRID_WITH_ZFP)
  set(BUILD_UTILITIES
      OFF
      CACHE BOOL "" FORCE)
  set(BUILD_TESTING
      OFF
      CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES
      OFF
      CACHE BOOL "" FORCE)
  set(ZFP_WITH_OPENMP
      OFF
      CACHE BOOL "" FORCE)
  FetchContent_Declare(
    zfp
    GIT_REPOSITORY https://github.com/LLNL/zfp.git
    GIT_TAG 1.0.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(zfp)
  target_link_libraries(rgrid PUBLIC zfp::zfp)
  target_compile_definitions(rgrid PUBLIC RGRID_WITH_ZFP=1)
endif()

if(RGRID_BUILD_TESTS)
  enable_testing()
  add_executable(test_smoke tests/test_smoke.cpp)
  target_link_libraries(test_smoke PRIVATE rgrid)
  add_test(NAME test_smoke COMMAND test_smoke)
endif()

if(RGRID_BUILD_PYTHON)
  find_package(
    Python 3.11
    COMPONENTS Interpreter Development.Module
    REQUIRED)
  # Prefer nanobind's CMake config if installed as a pip package.
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR
    RESULT_VARIABLE NB_RC)
  if(NB_RC EQUAL 0 AND NB_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
    find_package(nanobind CONFIG REQUIRED)
  else()
    # Fallback when nanobind isn't pip-installed: fetch from GitHub.
    FetchContent_Declare(
      nanobind
      GIT_REPOSITORY https://github.com/wjakob/nanobind.git
      GIT_TAG v2.5.0
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(nanobind)
  endif()

  nanobind_add_module(_rgrid python/bindings.cpp)
  target_link_libraries(_rgrid PRIVATE rgrid)
  set_target_properties(_rgrid PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                          ${CMAKE_BINARY_DIR}/python_pkg/rgrid)
  # Copy Python facade
  configure_file(${CMAKE_SOURCE_DIR}/python/rgrid/__init__.py
                 ${CMAKE_BINARY_DIR}/python_pkg/rgrid/__init__.py COPYONLY)
endif()
