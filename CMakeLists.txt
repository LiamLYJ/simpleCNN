cmake_minimum_required (VERSION 2.6)

# project (cnn_test)
# set (CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Debug)

# include_directories("${PROJECT_SOURCE_DIR}/CNN")
# include_directories("${PROJECT_SOURCE_DIR}")

# add_executable(cnn_test ${PROJECT_SOURCE_DIR}/example.cpp)

project (npy_test)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories("${PROJECT_SOURCE_DIR}")

add_executable(npy_test ${PROJECT_SOURCE_DIR}/play_ground.cpp)

