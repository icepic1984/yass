#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>

namespace glfw { // Begin of namespace glfw

class Window {
public:
    Window() : m_window(nullptr, glfwDestroyWindow)
    {
    }

    Window(std::nullptr_t) : m_window(nullptr, glfwDestroyWindow)

    {
    }

    Window(GLFWwindow* window) : m_window(window, glfwDestroyWindow)
    {
    }

    bool operator==(const Window& rhs) const
    {
        return m_window == rhs.m_window;
    }

    bool operator!=(const Window& rhs) const
    {
        return m_window != rhs.m_window;
    }

    explicit operator bool() const
    {
        return m_window != nullptr;
    }

    bool operator!() const
    {
        return m_window == nullptr;
    }

    operator GLFWwindow*() const
    {
        return m_window.get();
    }

    vk::UniqueSurfaceKHR createWindowSurface(const vk::UniqueInstance&);

private:
    std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> m_window;
};

Window createWindow(int width, int height, const std::string& title = "");

} // End of namespace glfw
