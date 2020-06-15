cmake_minimum_required(VERSION 2.8)
project(demo)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors")

#add_definitions(-DYOLO416) # or define YOLO416 for yolo416 model
add_definitions(-DYOLO608) # or define YOLO416 for yolo416 model

#add wrapper
set(TRT_WRAPPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tensorRTWrapper/code)
add_subdirectory(${TRT_WRAPPER_DIR})
include_directories(${TRT_WRAPPER_DIR}/include)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories("/opt/ros/kinetic/include")

#add opencv
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})

link_directories("/opt/ros/kinetic/lib")

#build runYolov3
add_executable(runYolov3 main.cpp eval.cpp dataReader.cpp)
add_executable(convert_engine convert_engine.cpp eval.cpp dataReader.cpp)
add_executable(main_test main_test.cpp eval.cpp dataReader.cpp)

target_link_libraries(runYolov3 
	TrtNet 
	${OpenCV_LIBS}
	roscpp
	roslib
	rosconsole
	roscpp_serialization
	xmlrpcpp
	rosconsole_log4cxx
	rosconsole_backend_interface
	# fastcdr
	# slog
	# mmpa
	# c_sec
	# fastrtps
	cpp_common
	rostime
	image_transport
	cv_bridge
	message_filters
	class_loader
	rospack
	)


target_link_libraries(convert_engine TrtNet ${OpenCV_LIBS})
target_link_libraries(main_test TrtNet ${OpenCV_LIBS})

install(TARGETS runYolov3 DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)
install(TARGETS convert_engine DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)
install(TARGETS main_test DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)
