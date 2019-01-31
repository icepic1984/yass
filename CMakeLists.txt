cmake_minimum_required(VERSION 3.2)
project(yass CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

add_subdirectory(glfw++)

add_executable(yass src/main.cpp)
add_executable(copy_image src/copy_image.cpp)
add_executable(double_buffer src/double_buffer.cpp)

# Build shader
add_custom_command(TARGET yass POST_BUILD
    COMMAND glslangValidator -V ${CMAKE_CURRENT_LIST_DIR}/src/shaders/shader.vert
    COMMAND glslangValidator -V ${CMAKE_CURRENT_LIST_DIR}/src/shaders/shader.frag)

target_compile_options(yass PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(yass glfw++ glfw Vulkan::Vulkan)
target_link_libraries(copy_image glfw++ glfw Vulkan::Vulkan)
target_link_libraries(double_buffer glfw++ glfw Vulkan::Vulkan)

