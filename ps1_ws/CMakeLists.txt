CMAKE_MINIMUM_REQUIRED(VERSION 3.10.2)
 
PROJECT(ps0_ws)
 
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
add_compile_options(-std=c++14)
add_definitions(-std=c++14)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) 

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 

set(ALL_TARGET_LIBRARIES "")

include_directories("/usr/local/include/eigen3")
include_directories(include)

find_package(OpenCV 3.4.16 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND ALL_TARGET_LIBRARIES ${OpenCV_LIBRARIES})

file(GLOB_RECURSE ALL_SRCS "${PROJECT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE NODE_SRCS "${PROJECT_SOURCE_DIR}/src/apps/*_app.cpp")
file(GLOB_RECURSE COMPILE_SRCS "${PROJECT_SOURCE_DIR}/*CompilerId.cpp")
list(REMOVE_ITEM ALL_SRCS ${NODE_SRCS} ${COMPILE_SRCS})

add_executable(affine_camera src/apps/affine_camera_app.cpp ${ALL_SRCS})
target_link_libraries(affine_camera ${ALL_TARGET_LIBRARIES})

add_executable(single_view_geometry src/apps/single_view_geometry_app.cpp ${ALL_SRCS})
target_link_libraries(single_view_geometry ${ALL_TARGET_LIBRARIES})

