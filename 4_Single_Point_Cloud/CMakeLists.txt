cmake_minimum_required(VERSION 2.8)
project(4_Single_Point_Cloud)

find_package(OpenCV 3 REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable(single_point_cloud vis.cc)

target_link_libraries(single_point_cloud
  ${OpenCV_LIBS})

install(TARGETS single_point_cloud
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
