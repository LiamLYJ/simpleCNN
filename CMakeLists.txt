cmake_minimum_required (VERSION 2.6)

# project (cnn_test)
# set (CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Debug)
# SET (CMAKE_C_COMPILER "/usr/bin/clang")

# include_directories("${PROJECT_SOURCE_DIR}/CNN")
# include_directories("${PROJECT_SOURCE_DIR}")

# # add_executable(cnn_test ${PROJECT_SOURCE_DIR}/example.cpp)
# add_executable(cnn_test ${PROJECT_SOURCE_DIR}/main.cpp)

project (play_ground)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
SET (CMAKE_C_COMPILER "/usr/bin/clang")

include_directories("${PROJECT_SOURCE_DIR}/CNN")
include_directories("${PROJECT_SOURCE_DIR}")

add_executable(play_ground ${PROJECT_SOURCE_DIR}/play_ground.cpp)
