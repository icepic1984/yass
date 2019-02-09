#include <chrono>

using namespace std::chrono_literals;

template <typename T> class Timer {
public:
    Timer() : m_start(std::chrono::system_clock::now())
    {
    }

    T elapsed()
    {
        return std::chrono::duration_cast<T>(std::chrono::system_clock::now() -
                                             m_start);
    }

    void reset()
    {
        m_start = std::chrono::system_clock::now();
    }

private:
    std::chrono::time_point<std::chrono::system_clock> m_start;
};
