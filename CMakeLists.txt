cmake_minimum_required (VERSION 2.6)

project (CNN)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
SET (CMAKE_C_COMPILER "/usr/bin/clang")

include_directories("${PROJECT_SOURCE_DIR}/CNN")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("/usr/local/include/eigen3")

add_executable(play_ground ${PROJECT_SOURCE_DIR}/play_ground.cpp)
add_executable(cnn_test ${PROJECT_SOURCE_DIR}/main.cpp)
