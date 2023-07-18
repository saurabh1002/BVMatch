cmake_minimum_required(VERSION 2.8.3)
project(loop_rift)

add_compile_options(-std=c++17)
add_compile_options(-std=gnu++17)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PCL REQUIRED)

include_directories(
  include
	${PCL_INCLUDE_DIRS}
)

add_library(bvft STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/bvftdescriptors.cpp)
target_link_libraries(bvft PUBLIC ${OpenCV_LIBS})

add_executable(match_two_scan ${CMAKE_CURRENT_SOURCE_DIR}/src/match_two_scan.cpp) 
target_link_libraries(match_two_scan ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} python2.7 bvft)

add_executable(generate_descriptors ${CMAKE_CURRENT_SOURCE_DIR}/src/generate_descriptors.cpp)    #change the verticle 
target_link_libraries(generate_descriptors ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS}  python2.7 bvft)
