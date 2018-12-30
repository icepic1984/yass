add_library(glfw++ INTERFACE)

target_include_directories(glfw++ INTERFACE include/)

target_link_libraries(glfw++ INTERFACE Vulkan::Vulkan glfw)

