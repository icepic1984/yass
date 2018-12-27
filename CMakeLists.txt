cmake_minimum_required(VERSION 3.2)
project(yass CXX)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(yass src/main.cpp)

target_link_libraries(yass Vulkan::Vulkan glfw)
