CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
PROJECT(amba_arm)

set(CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#include dependencies
include(cmake/ambacv22_dependencies.cmake)

#include_directories
message(STATUS "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")

add_subdirectory(inference)