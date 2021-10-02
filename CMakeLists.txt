cmake_minimum_required(VERSION 3.16)

project(ideepcolor
	LANGUAGES CXX)
	
find_package(OpenCV REQUIRED)
	
file(GLOB SOURCE_LIST CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.*")

add_library("ideepcolor" SHARED ${SOURCE_LIST})

message("Using:")
message("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message("OpenCV_LIBS: ${OpenCV_LIBS}")

target_include_directories("ideepcolor" PRIVATE "/mechsrc/colorize/NumCpp/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries("ideepcolor" PRIVATE ${OpenCV_LIBS})
