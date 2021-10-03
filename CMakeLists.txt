cmake_minimum_required(VERSION 3.16)

project(ideepcolor LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
	
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
	
file(GLOB SOURCE_LIST CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.*")

#add_library("ideepcolor" SHARED ${SOURCE_LIST})
add_executable("ideepcolor" "${CMAKE_SOURCE_DIR}/main.cpp" ${SOURCE_LIST})

message("Using:")
message("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message("OpenCV_LIBS: ${OpenCV_LIBS}")
message("TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message("TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_include_directories("ideepcolor" PRIVATE "/mechsrc/colorize/NumCpp/include" ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries("ideepcolor" PRIVATE ${OpenCV_LIBS} ${TORCH_LIBRARIES})
