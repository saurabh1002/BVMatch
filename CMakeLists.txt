cmake_minimum_required(VERSION 2.8.3)
project(loop_rift)

add_compile_options(-std=c++17)
add_compile_options(-std=gnu++17)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  include
	${PCL_INCLUDE_DIRS}
)

add_library(bvft STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/bvftdescriptors.cpp)
target_link_libraries(bvft PUBLIC ${OpenCV_LIBS})

add_library(utils STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp)
target_link_libraries(utils PUBLIC ${OpenCV_LIBS} ${PCL_LIBRARIES} Eigen3::Eigen python2.7)

add_executable(match_two_scan ${CMAKE_CURRENT_SOURCE_DIR}/src/match_two_scan.cpp) 
target_link_libraries(match_two_scan ${PCL_LIBRARIES} ${OpenCV_LIBS} utils bvft)

add_executable(generate_descriptors ${CMAKE_CURRENT_SOURCE_DIR}/src/generate_descriptors.cpp)
target_link_libraries(generate_descriptors ${OpenCV_LIBS} utils bvft)
