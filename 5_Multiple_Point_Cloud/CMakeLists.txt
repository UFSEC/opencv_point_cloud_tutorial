cmake_minimum_required(VERSION 2.8)
project(5_Multiple_Point_Cloud)

find_package(OpenCV 3 REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable(multiple_point_cloud multiple_point_cloud.cc)

target_link_libraries(multiple_point_cloud
  ${OpenCV_LIBS})

install(TARGETS multiple_point_cloud
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
