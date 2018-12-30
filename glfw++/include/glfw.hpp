#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cstddef>
#include <utility>
#include <iostream>

namespace glfw { // Begin of namespace glfw

template <typename Type> class UniqueHandleTraits;

template <typename Type>
class UniqueHandle : public UniqueHandleTraits<Type>::deleter {
private:
    using Deleter = typename UniqueHandleTraits<Type>::deleter;

public:
    explicit UniqueHandle(Type const& value = Type(),
                          Deleter const& deleter = Deleter())
        : Deleter(deleter), m_value(value)
    {
    }

    UniqueHandle(UniqueHandle const&) = delete;

    UniqueHandle(UniqueHandle&& other)
        : Deleter(std::move(static_cast<Deleter&>(other))),
          m_value(other.release())
    {
    }

    ~UniqueHandle()
    {
        if (m_value)
            this->destroy(m_value);
    }

    UniqueHandle& operator=(UniqueHandle const&) = delete;

    UniqueHandle& operator=(UniqueHandle&& other)
    {
        reset(other.release());
        *static_cast<Deleter*>(this) = std::move(static_cast<Deleter&>(other));
        return *this;
    }

    explicit operator bool() const
    {
        return m_value.operator bool();
    }

    Type const* operator->() const
    {
        return &m_value;
    }

    Type* operator->()
    {
        return &m_value;
    }

    Type const& operator*() const
    {
        return m_value;
    }

    Type& operator*()
    {
        return m_value;
    }

    const Type& get() const
    {
        return m_value;
    }

    Type& get()
    {
        return m_value;
    }

    void reset(Type const& value = Type())
    {
        if (m_value != value) {
            if (m_value)
                this->destroy(m_value);
            m_value = value;
        }
    }

    Type release()
    {
        Type value = m_value;
        m_value = nullptr;
        return value;
    }

    void swap(UniqueHandle<Type>& rhs)
    {
        std::swap(m_value, rhs.m_value);
        std::swap(static_cast<Deleter&>(*this), static_cast<Deleter&>(rhs));
    }

private:
    Type m_value;
};

class Window {
public:
    Window() : m_window(nullptr)
    {
    }

    Window(std::nullptr_t) : m_window(nullptr)

    {
    }

    Window(GLFWwindow* window) : m_window(window)
    {
    }

    Window& operator=(GLFWwindow* window)
    {
        m_window = window;
        return *this;
    }

    Window& operator=(std::nullptr_t)
    {
        m_window = nullptr;
        return *this;
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
        return m_window;
    }

private:
    GLFWwindow* m_window;
};

struct WindowDeleter {
    void destroy(Window window)
    {
        std::cout << "Destory" << std::endl;
        if (window) {
            glfwDestroyWindow(window);
        }
    }
};

template<> class UniqueHandleTraits<Window> {
public:
    using deleter = WindowDeleter;
};
using UniqueWindow = UniqueHandle<Window>;

} // End of namespace glfw
