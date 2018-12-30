#include <glfw.hpp>

namespace glfw {

vk::SurfaceKHR Window::createWindowSurface(const vk::UniqueInstance& instance)
{
    VkSurfaceKHR surface;
    glfwCreateWindowSurface(instance.get(), m_window.get(), nullptr, &surface);
    return vk::SurfaceKHR(surface);
    //return vk::UniqueSurfaceKHR(vk::SurfaceKHR(surface));
}

Window createWindow(int width, int height, const std::string& title)
{
    return Window{
        glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr)};
}

} // namespace glfw
