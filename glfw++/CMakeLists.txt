
add_library(glfw++ "")

target_sources(glfw++ PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include/glfw.hpp PRIVATE src/glfw.cpp)

target_include_directories(glfw++ PUBLIC include/)

target_link_libraries(glfw++ PUBLIC Vulkan::Vulkan glfw)

