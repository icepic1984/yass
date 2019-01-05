#include <glfw.hpp>

namespace glfw {

vk::UniqueSurfaceKHR
Window::createWindowSurface(const vk::UniqueInstance& instance)
{
    std::cout << instance.get() << std::endl;
    VkSurfaceKHR surface;
    glfwCreateWindowSurface(instance.get(), m_window.get(), nullptr, &surface);

    vk::ObjectDestroy<vk::Instance, vk::DispatchLoaderStatic> deleter(
        instance.get());
    return vk::UniqueSurfaceKHR(surface, deleter);
}

Window createWindow(int width, int height, const std::string& title)
{
    return Window{
        glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr)};
}

} // namespace glfw
