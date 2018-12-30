cmake_minimum_required(VERSION 3.2)
project(yass CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

add_subdirectory(glfw++)

add_executable(yass src/main.cpp src/glfw.cpp)

target_link_libraries(yass glfw++ glfw Vulkan::Vulkan)

